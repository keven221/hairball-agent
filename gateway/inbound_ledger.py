"""Durable, profile-local idempotency for gateway inbound events.

Platform adapters already normalize inbound payloads to ``MessageEvent`` and
most of them expose the platform's stable message/delivery identifier as
``message_id``.  Several adapters also keep a short-lived in-memory dedupe
cache, but that cache disappears on restart.  This ledger closes that restart
gap at the shared adapter seam without guessing from message text.

Only events with a stable platform identifier participate.  Events without an
identifier keep the historical at-least-once behaviour.  Ledger failures also
fail open: losing dedupe is preferable to losing a user's message.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from hairball_constants import get_hairball_home

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
STALE_AFTER_SECONDS = 7 * 24 * 60 * 60
RETENTION_SECONDS = 14 * 24 * 60 * 60
MAX_ROWS = 20_000

_DB_LOCK = threading.RLock()
_PLATFORM_IDS_KEY = "_hairball_platform_message_ids"
_CLAIM_IDS_KEY = "_hairball_inbound_claim_ids"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gateway_inbound_events (
    event_id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    scope_id TEXT,
    chat_id TEXT NOT NULL,
    thread_id TEXT,
    platform_message_id TEXT NOT NULL,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    owner_pid INTEGER,
    owner_started_at INTEGER,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_gateway_inbound_state_updated
    ON gateway_inbound_events(state, updated_at);
"""


@dataclass(frozen=True)
class InboundClaim:
    """Result of atomically claiming every stable identity on one event."""

    claimed_ids: tuple[str, ...] = ()
    duplicate_ids: tuple[str, ...] = ()
    had_identity: bool = False

    @property
    def should_process(self) -> bool:
        return not self.had_identity or bool(self.claimed_ids)


def _profile_home(event: Any) -> Path:
    source = getattr(event, "source", None)
    profile = str(getattr(source, "profile", None) or "").strip()
    if profile:
        try:
            from hairball_cli.profiles import get_profile_dir

            return Path(get_profile_dir(profile))
        except Exception:
            logger.debug("Could not resolve inbound profile %r", profile, exc_info=True)
    return get_hairball_home()


def _connect(path: Path) -> sqlite3.Connection:
    from hairball_state import apply_wal_with_fallback

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path), timeout=10, isolation_level=None, check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    apply_wal_with_fallback(conn, db_label="state.db (gateway inbound ledger)")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.executescript(_SCHEMA)
    return conn


def _owner_stamp() -> tuple[int, int | None]:
    pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _owner_alive(pid: Any, started_at: Any) -> bool:
    try:
        owner_pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        from gateway.status import get_process_start_time

        current_started = get_process_start_time(owner_pid)
    except Exception:
        current_started = None
    if current_started is None:
        try:
            os.kill(owner_pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    if started_at is None:
        return True
    try:
        return int(current_started) == int(started_at)
    except (TypeError, ValueError):
        return True


def _metadata(event: Any) -> dict[str, Any]:
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        try:
            event.metadata = metadata
        except Exception:
            pass
    return metadata


def _unique_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def event_message_ids(event: Any) -> list[str]:
    """Return every stable platform id represented by an event/batch."""
    metadata = _metadata(event)
    explicit = metadata.get("gateway_event_id") or metadata.get("dedupe_key")
    batch_ids = metadata.get(_PLATFORM_IDS_KEY)
    if not isinstance(batch_ids, (list, tuple, set)):
        batch_ids = []
    source = getattr(event, "source", None)
    primary = getattr(event, "message_id", None) or getattr(source, "message_id", None)
    update_id = getattr(event, "platform_update_id", None)
    fallback_update = f"update:{update_id}" if primary in (None, "") and update_id is not None else ""
    return _unique_strings([explicit, primary, fallback_update, *batch_ids])


def merge_event_message_ids(target: Any, incoming: Any) -> None:
    """Preserve all stable ids when platform/base batching merges events.

    Claims already made before a busy-session merge are carried too, so the
    eventual processing task completes every obligation rather than leaving
    the later fragments permanently in ``claimed`` state.
    """
    target_meta = _metadata(target)
    incoming_meta = _metadata(incoming)
    target_meta[_PLATFORM_IDS_KEY] = _unique_strings(
        [*event_message_ids(target), *event_message_ids(incoming)]
    )
    target_claims = target_meta.get(_CLAIM_IDS_KEY)
    incoming_claims = incoming_meta.get(_CLAIM_IDS_KEY)
    target_meta[_CLAIM_IDS_KEY] = _unique_strings([
        *(target_claims if isinstance(target_claims, (list, tuple, set)) else []),
        *(incoming_claims if isinstance(incoming_claims, (list, tuple, set)) else []),
    ])


def _event_key(event: Any, platform_message_id: str) -> tuple[str, dict[str, str]]:
    source = getattr(event, "source", None)
    platform_raw = getattr(source, "platform", "")
    platform = str(getattr(platform_raw, "value", platform_raw) or "unknown")
    fields = {
        "profile": str(getattr(source, "profile", None) or "default"),
        "platform": platform,
        "scope_id": str(getattr(source, "scope_id", None) or ""),
        "chat_id": str(getattr(source, "chat_id", None) or ""),
        "thread_id": str(getattr(source, "thread_id", None) or ""),
        "platform_message_id": platform_message_id,
        "revision": str(_metadata(event).get("event_revision") or ""),
    }
    raw = "\x1f".join(fields.values())
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:32], fields


def claim_inbound_event(event: Any) -> InboundClaim:
    """Atomically claim stable identities, returning duplicate disposition."""
    if bool(getattr(event, "internal", False)):
        return InboundClaim()
    message_ids = event_message_ids(event)
    if not message_ids:
        return InboundClaim()

    path = _profile_home(event) / "state.db"
    now = time.time()
    pid, started_at = _owner_stamp()
    claimed: list[str] = []
    duplicates: list[str] = []

    with _DB_LOCK, _connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for platform_message_id in message_ids:
                event_id, fields = _event_key(event, platform_message_id)
                row = conn.execute(
                    "SELECT state, attempts, owner_pid, owner_started_at "
                    "FROM gateway_inbound_events WHERE event_id=?",
                    (event_id,),
                ).fetchone()
                if row is None:
                    conn.execute(
                        """INSERT INTO gateway_inbound_events(
                               event_id, platform, scope_id, chat_id, thread_id,
                               platform_message_id, state, attempts, created_at,
                               updated_at, owner_pid, owner_started_at
                           ) VALUES (?, ?, ?, ?, ?, ?, 'claimed', 1, ?, ?, ?, ?)""",
                        (
                            event_id, fields["platform"], fields["scope_id"],
                            fields["chat_id"], fields["thread_id"],
                            fields["platform_message_id"], now, now, pid, started_at,
                        ),
                    )
                    claimed.append(event_id)
                    continue

                state = str(row["state"] or "")
                attempts = int(row["attempts"] or 0)
                live_owner = _owner_alive(row["owner_pid"], row["owner_started_at"])
                reclaimable = state in {"failed", "cancelled"} or (
                    state in {"claimed", "processing"} and not live_owner
                )
                if reclaimable and attempts < MAX_ATTEMPTS:
                    conn.execute(
                        """UPDATE gateway_inbound_events
                           SET state='claimed', attempts=attempts+1, updated_at=?,
                               owner_pid=?, owner_started_at=?, last_error=NULL
                           WHERE event_id=?""",
                        (now, pid, started_at, event_id),
                    )
                    claimed.append(event_id)
                else:
                    duplicates.append(event_id)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    if claimed:
        metadata = _metadata(event)
        prior = metadata.get(_CLAIM_IDS_KEY)
        metadata[_CLAIM_IDS_KEY] = _unique_strings([
            *(prior if isinstance(prior, (list, tuple, set)) else []), *claimed,
        ])
    _prune(path, now)
    return InboundClaim(tuple(claimed), tuple(duplicates), had_identity=True)


def _claim_ids(event: Any) -> list[str]:
    raw = _metadata(event).get(_CLAIM_IDS_KEY)
    return _unique_strings(raw if isinstance(raw, (list, tuple, set)) else [])


def has_inbound_claim(event: Any) -> bool:
    """Return whether *event* owns durable inbound ledger rows."""
    return bool(_claim_ids(event))


def mark_inbound_processing(event: Any) -> None:
    _set_state(event, "processing", from_states=("claimed",))


def finish_inbound_event(event: Any, outcome: Any, error: str = "") -> None:
    value = str(getattr(outcome, "value", outcome) or "").lower()
    state = "completed" if value == "success" else "cancelled" if value == "cancelled" else "failed"
    _set_state(
        event,
        state,
        error=error,
        from_states=("claimed", "processing"),
    )


def _set_state(
    event: Any,
    state: str,
    *,
    error: str = "",
    from_states: tuple[str, ...] = (),
) -> None:
    claim_ids = _claim_ids(event)
    if not claim_ids:
        return
    path = _profile_home(event) / "state.db"
    now = time.time()
    marks = ",".join("?" for _ in claim_ids)
    state_filter = ""
    params: list[Any] = [state, now, str(error or "")[:500] or None, *claim_ids]
    if from_states:
        allowed = ",".join("?" for _ in from_states)
        state_filter = f" AND state IN ({allowed})"
        params.extend(from_states)
    with _DB_LOCK, _connect(path) as conn:
        conn.execute(
            f"""UPDATE gateway_inbound_events
                SET state=?, updated_at=?, last_error=?
                WHERE event_id IN ({marks}){state_filter}""",
            params,
        )


def _prune(path: Path, now: float | None = None) -> None:
    now = time.time() if now is None else now
    cutoff = now - RETENTION_SECONDS
    try:
        with _DB_LOCK, _connect(path) as conn:
            conn.execute(
                "DELETE FROM gateway_inbound_events "
                "WHERE state IN ('completed', 'failed', 'cancelled') AND updated_at < ?",
                (cutoff,),
            )
            total = int(conn.execute("SELECT COUNT(*) FROM gateway_inbound_events").fetchone()[0])
            excess = max(0, total - MAX_ROWS)
            if excess:
                conn.execute(
                    """DELETE FROM gateway_inbound_events WHERE event_id IN (
                           SELECT event_id FROM gateway_inbound_events
                           WHERE state IN ('completed', 'failed', 'cancelled')
                           ORDER BY CASE state WHEN 'completed' THEN 0
                                              WHEN 'failed' THEN 1
                                              WHEN 'cancelled' THEN 2
                                              ELSE 3 END,
                                    updated_at ASC LIMIT ?
                       )""",
                    (excess,),
                )
            conn.execute(
                """UPDATE gateway_inbound_events
                   SET state='failed', updated_at=?, last_error='stale inbound owner'
                   WHERE state IN ('claimed', 'processing') AND updated_at < ?""",
                (now, now - STALE_AFTER_SECONDS),
            )
    except Exception:
        logger.debug("Gateway inbound ledger prune failed", exc_info=True)


def read_inbound_event(event: Any, platform_message_id: str | None = None) -> dict[str, Any] | None:
    """Return a ledger row for diagnostics/tests; never used for dispatch."""
    ids = [platform_message_id] if platform_message_id else event_message_ids(event)
    if not ids:
        return None
    event_id, _ = _event_key(event, str(ids[0]))
    path = _profile_home(event) / "state.db"
    with _DB_LOCK, _connect(path) as conn:
        row = conn.execute(
            "SELECT * FROM gateway_inbound_events WHERE event_id=?", (event_id,),
        ).fetchone()
    return dict(row) if row is not None else None
