"""Durable lifecycle store for managed background terminal processes.

The in-memory :mod:`tools.process_registry` remains the execution owner and
public interface.  This module gives that existing rail the minimum durable
coordination it needs across Hairball restarts: one process identity row, a
short renewable poller lease, an idempotency key, and a completion outbox.

All rows live in Hairball's existing ``state.db``.  Secrets are never accepted
by this API; callers must pass a force-redacted command and non-secret backend
reconnect descriptor.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hairball_constants import get_hairball_home


_DB_LOCK = threading.Lock()
_MAX_DELIVERY_ATTEMPTS = 8


_SCHEMA = """
CREATE TABLE IF NOT EXISTS background_processes (
    process_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL DEFAULT '',
    session_key TEXT NOT NULL DEFAULT '',
    command_redacted TEXT NOT NULL DEFAULT '',
    cwd TEXT,
    pid INTEGER,
    pid_scope TEXT NOT NULL DEFAULT 'sandbox',
    remote_start_time TEXT,
    process_token TEXT,
    reconnect_json TEXT,
    paths_json TEXT,
    started_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    state TEXT NOT NULL DEFAULT 'running',
    exit_code INTEGER,
    completion_reason TEXT,
    termination_source TEXT,
    watcher_json TEXT,
    artifact_path TEXT,
    artifact_sha256 TEXT,
    idempotency_key TEXT,
    lease_owner TEXT,
    lease_expires_at REAL,
    heartbeat_at REAL,
    event_json TEXT,
    delivery_state TEXT NOT NULL DEFAULT 'pending',
    delivery_attempts INTEGER NOT NULL DEFAULT 0,
    delivery_claim TEXT,
    delivery_claimed_at REAL,
    delivered_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_background_process_idempotency
    ON background_processes(idempotency_key)
    WHERE idempotency_key IS NOT NULL AND idempotency_key != '';
CREATE INDEX IF NOT EXISTS idx_background_process_recovery
    ON background_processes(state, updated_at);
CREATE INDEX IF NOT EXISTS idx_background_process_delivery
    ON background_processes(delivery_state, updated_at);
"""


def _db_path() -> Path:
    return get_hairball_home() / "state.db"


def _connect() -> sqlite3.Connection:
    from hairball_state import apply_wal_with_fallback

    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    apply_wal_with_fallback(conn, db_label="state.db (background_processes)")
    conn.executescript(_SCHEMA)
    return conn


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _decode_row(row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    result = dict(row)
    for key in ("reconnect_json", "paths_json", "watcher_json", "event_json"):
        raw = result.pop(key, None)
        result[key.removesuffix("_json")] = json.loads(raw) if raw else {}
    return result


class BackgroundProcessStore:
    """SQLite-backed ownership and delivery state for ``ProcessRegistry``."""

    def claim_spawn(
        self, record: Dict[str, Any]
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Atomically reserve an idempotent dispatch before its side effect.

        Returns ``(True, None)`` to the one caller allowed to launch the
        command.  A competing/replayed caller gets ``(False, existing_row)``
        and must reuse that process identity without launching anything.

        Non-idempotent calls deliberately bypass the reservation; their
        historical behavior is unchanged.
        """
        key = str(record.get("idempotency_key") or "")
        if not key:
            return True, None
        now = time.time()
        try:
            with _DB_LOCK, _connect() as conn:
                conn.execute(
                    """INSERT INTO background_processes (
                           process_id, task_id, session_key, command_redacted,
                           cwd, pid, pid_scope, remote_start_time, process_token,
                           reconnect_json, paths_json, started_at, updated_at,
                           state, watcher_json, artifact_path, idempotency_key,
                           heartbeat_at, delivery_state
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                 'dispatching', ?, ?, ?, ?, 'delivered')""",
                    (
                        record["process_id"],
                        record.get("task_id", ""),
                        record.get("session_key", ""),
                        record.get("command_redacted", ""),
                        record.get("cwd"),
                        record.get("pid"),
                        record.get("pid_scope", "sandbox"),
                        record.get("remote_start_time"),
                        record.get("process_token"),
                        _json(record.get("reconnect", {})),
                        _json(record.get("paths", {})),
                        float(record.get("started_at") or now),
                        now,
                        _json(record.get("watcher", {})),
                        record.get("artifact_path"),
                        key,
                        now,
                    ),
                )
            return True, None
        except sqlite3.IntegrityError:
            # The partial unique index is the cross-process arbitration point.
            # Read the winner after the failed insert transaction has closed.
            return False, self.find_by_idempotency_key(key)

    def upsert_running(self, record: Dict[str, Any]) -> None:
        now = time.time()
        with _DB_LOCK, _connect() as conn:
            conn.execute(
                """INSERT INTO background_processes (
                       process_id, task_id, session_key, command_redacted, cwd,
                       pid, pid_scope, remote_start_time, process_token,
                       reconnect_json, paths_json, started_at, updated_at, state,
                       watcher_json, artifact_path, idempotency_key, heartbeat_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running',
                             ?, ?, ?, ?)
                   ON CONFLICT(process_id) DO UPDATE SET
                       task_id=excluded.task_id,
                       session_key=excluded.session_key,
                       command_redacted=excluded.command_redacted,
                       cwd=excluded.cwd,
                       pid=excluded.pid,
                       pid_scope=excluded.pid_scope,
                       remote_start_time=excluded.remote_start_time,
                       process_token=excluded.process_token,
                       reconnect_json=excluded.reconnect_json,
                       paths_json=excluded.paths_json,
                       updated_at=excluded.updated_at,
                       state='running',
                       watcher_json=excluded.watcher_json,
                       artifact_path=excluded.artifact_path,
                       idempotency_key=excluded.idempotency_key,
                       heartbeat_at=excluded.heartbeat_at""",
                (
                    record["process_id"],
                    record.get("task_id", ""),
                    record.get("session_key", ""),
                    record.get("command_redacted", ""),
                    record.get("cwd"),
                    record.get("pid"),
                    record.get("pid_scope", "sandbox"),
                    record.get("remote_start_time"),
                    record.get("process_token"),
                    _json(record.get("reconnect", {})),
                    _json(record.get("paths", {})),
                    float(record.get("started_at") or now),
                    now,
                    _json(record.get("watcher", {})),
                    record.get("artifact_path"),
                    record.get("idempotency_key") or None,
                    now,
                ),
            )

    def get(self, process_id: str) -> Optional[Dict[str, Any]]:
        with _DB_LOCK, _connect() as conn:
            row = conn.execute(
                "SELECT * FROM background_processes WHERE process_id=?",
                (process_id,),
            ).fetchone()
        return _decode_row(row)

    def find_by_idempotency_key(self, key: str) -> Optional[Dict[str, Any]]:
        if not key:
            return None
        with _DB_LOCK, _connect() as conn:
            row = conn.execute(
                "SELECT * FROM background_processes WHERE idempotency_key=?",
                (key,),
            ).fetchone()
        return _decode_row(row)

    def list_recoverable(self) -> list[Dict[str, Any]]:
        with _DB_LOCK, _connect() as conn:
            rows = conn.execute(
                """SELECT * FROM background_processes
                   WHERE state='running'
                   ORDER BY started_at, process_id"""
            ).fetchall()
        return [_decode_row(row) or {} for row in rows]

    def claim_lease(self, process_id: str, owner: str, ttl: float = 15.0) -> bool:
        now = time.time()
        with _DB_LOCK, _connect() as conn:
            cur = conn.execute(
                """UPDATE background_processes
                   SET lease_owner=?, lease_expires_at=?, heartbeat_at=?, updated_at=?
                   WHERE process_id=? AND state='running'
                     AND (lease_owner IS NULL OR lease_owner=?
                          OR lease_expires_at IS NULL OR lease_expires_at < ?)""",
                (owner, now + max(1.0, ttl), now, now, process_id, owner, now),
            )
            return cur.rowcount == 1

    def heartbeat(self, process_id: str, owner: str, ttl: float = 15.0) -> bool:
        now = time.time()
        with _DB_LOCK, _connect() as conn:
            cur = conn.execute(
                """UPDATE background_processes
                   SET lease_expires_at=?, heartbeat_at=?, updated_at=?
                   WHERE process_id=? AND state='running' AND lease_owner=?""",
                (now + max(1.0, ttl), now, now, process_id, owner),
            )
            return cur.rowcount == 1

    def release_lease(self, process_id: str, owner: str) -> bool:
        with _DB_LOCK, _connect() as conn:
            cur = conn.execute(
                """UPDATE background_processes
                   SET lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                   WHERE process_id=? AND lease_owner=?""",
                (time.time(), process_id, owner),
            )
            return cur.rowcount == 1

    def mark_finished(
        self,
        process_id: str,
        *,
        exit_code: Optional[int],
        completion_reason: str,
        termination_source: str,
        artifact_path: str = "",
        artifact_sha256: str = "",
        event: Optional[Dict[str, Any]] = None,
    ) -> bool:
        now = time.time()
        with _DB_LOCK, _connect() as conn:
            cur = conn.execute(
                """UPDATE background_processes
                   SET state='finished', exit_code=?, completion_reason=?,
                       termination_source=?, artifact_path=?, artifact_sha256=?,
                       event_json=?, delivery_state=CASE
                           WHEN ? = 1 THEN 'delivered' ELSE 'pending' END,
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                   WHERE process_id=? AND state IN ('dispatching', 'running')""",
                (
                    exit_code,
                    completion_reason,
                    termination_source,
                    artifact_path or None,
                    artifact_sha256 or None,
                    _json(event) if event is not None else None,
                    1 if event is None else 0,
                    now,
                    process_id,
                ),
            )
            return cur.rowcount == 1

    def restore_pending_events(self, target_queue) -> int:
        with _DB_LOCK, _connect() as conn:
            rows = conn.execute(
                """SELECT process_id, event_json FROM background_processes
                   WHERE state='finished' AND delivery_state='pending'
                     AND event_json IS NOT NULL
                   ORDER BY updated_at, process_id"""
            ).fetchall()
        count = 0
        for process_id, raw in rows:
            event = json.loads(raw)
            if isinstance(event, dict):
                event["restored"] = True
                event.setdefault("session_id", process_id)
                target_queue.put(event)
                count += 1
        return count

    def claim_delivery(self, process_id: str, consumer: str) -> Optional[str]:
        now = time.time()
        claim = f"{consumer}:{os.getpid()}:{uuid.uuid4().hex}"
        with _DB_LOCK, _connect() as conn:
            row = conn.execute(
                "SELECT delivery_attempts FROM background_processes WHERE process_id=?",
                (process_id,),
            ).fetchone()
            if row is None:
                return ""  # Legacy in-memory completion.
            if int(row[0] or 0) >= _MAX_DELIVERY_ATTEMPTS:
                conn.execute(
                    """UPDATE background_processes SET delivery_state='dropped',
                              updated_at=? WHERE process_id=? AND delivery_state='pending'""",
                    (now, process_id),
                )
                return None
            cur = conn.execute(
                """UPDATE background_processes
                   SET delivery_claim=?, delivery_claimed_at=?,
                       delivery_attempts=delivery_attempts+1, updated_at=?
                   WHERE process_id=? AND delivery_state='pending'
                     AND (delivery_claim IS NULL OR delivery_claimed_at < ?)""",
                (claim, now, now, process_id, now - 300),
            )
            return claim if cur.rowcount == 1 else None

    def complete_delivery(self, process_id: str, claim: str) -> bool:
        if not claim:
            return True
        now = time.time()
        with _DB_LOCK, _connect() as conn:
            cur = conn.execute(
                """UPDATE background_processes
                   SET delivery_state='delivered', delivered_at=?, updated_at=?,
                       delivery_claim=NULL, delivery_claimed_at=NULL
                   WHERE process_id=? AND delivery_state='pending'
                     AND delivery_claim=?""",
                (now, now, process_id, claim),
            )
            return cur.rowcount == 1

    def release_delivery(self, process_id: str, claim: str) -> bool:
        if not claim:
            return True
        with _DB_LOCK, _connect() as conn:
            cur = conn.execute(
                """UPDATE background_processes
                   SET delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=?
                   WHERE process_id=? AND delivery_state='pending'
                     AND delivery_claim=?""",
                (time.time(), process_id, claim),
            )
            return cur.rowcount == 1

    def forget(self, process_ids: Iterable[str]) -> None:
        ids = [str(item) for item in process_ids if item]
        if not ids:
            return
        with _DB_LOCK, _connect() as conn:
            conn.executemany(
                "DELETE FROM background_processes WHERE process_id=?",
                [(item,) for item in ids],
            )


background_process_store = BackgroundProcessStore()
