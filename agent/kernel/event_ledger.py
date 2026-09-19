"""Append-only kernel event ledger.

The ledger is the kernel's single source of truth. Every surface — chat,
workspace, sidebar, agent tree, gateway, metrics — is a *projection* of this
stream, which is what makes the class of regression the design doc calls out
structurally impossible: clearing a chat's live buffer cannot delete the
workspace's evidence, because the workspace never owned that data in the first
place.

Two rules follow from that and are enforced here:

- **SSE/WebSocket is notification, not truth.** ``subscribe()`` fires only after
  the durable write has committed. A dropped connection is recovered by reading
  from a ``seq`` cursor, never by replaying an in-memory buffer.
- **Durable terminals happen once.** A turn gets exactly one
  completed/failed/cancelled; so does a started tool call. A second one is an
  :class:`InvariantViolation`, not a warning.

Cursor semantics (exclusive ``after_seq`` + ``limit``) are ADAPTed from Pi's
``SessionEntryCursorOptions`` / ``getEntries`` in
``packages/agent/src/harness/session/jsonl-storage.ts`` (MIT, ``dd6bea41``).
Storage is Hairball's existing SQLite ``SessionDB`` — the design doc explicitly
forbids introducing a second database for this.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import replace
from typing import Any, Callable, Iterable, Optional, Sequence

from agent.kernel.contracts import (
    EPHEMERAL_KINDS,
    EVENT_SCHEMA_VERSION,
    KERNEL_PROTOCOL_VERSION,
    KERNEL_V2,
    TERMINAL_TOOL_KINDS,
    TERMINAL_TURN_KINDS,
    TOOL_SCOPED_KINDS,
    ArtifactRef,
    EventKind,
    InvariantViolation,
    KernelEvent,
    LedgerUnavailable,
)

logger = logging.getLogger(__name__)

#: Callback invoked with each durably-appended event. Must not raise.
Subscriber = Callable[[KernelEvent], None]


class EventLedger:
    """Durable append-only event log for one Hairball state database.

    A single ledger instance serves many sessions; ``session_id`` scopes both
    the sequence space and the invariant bookkeeping.
    """

    def __init__(
        self,
        db: Any,
        *,
        persist_ephemeral: bool = False,
        enforce_invariants: bool = True,
    ) -> None:
        """Wrap *db* (a :class:`hairball_state.SessionDB`) as a ledger.

        ``persist_ephemeral=False`` keeps token-level deltas out of SQLite. They
        are pure notification — the ``assistant.message.completed`` event carries
        the full text — so persisting them would multiply ledger volume by the
        length of every response for no recoverable fact. Live subscribers
        receive them either way.
        """
        self._db = db
        self._persist_ephemeral = persist_ephemeral
        self._enforce_invariants = enforce_invariants
        self._lock = threading.RLock()
        self._subscribers: list[Subscriber] = []
        # Per-session terminal bookkeeping so a double terminal is caught at
        # append time rather than discovered later during replay.
        self._terminal_turns: dict[str, set[str]] = {}
        self._terminal_tools: dict[str, set[str]] = {}
        self._started_tools: dict[str, set[str]] = {}
        self._migrated = False

    # ── lifecycle ──

    def ensure_ready(self) -> None:
        """Create the ledger tables if this database has never hosted one."""
        if self._migrated:
            return
        try:
            self._db.apply_kernel_ledger_migration()
        except Exception as exc:  # pragma: no cover - surfaced to caller
            raise LedgerUnavailable(f"kernel ledger migration failed: {exc}") from exc
        self._migrated = True

    def open_session(
        self,
        session_id: str,
        *,
        kernel_version: str = KERNEL_V2,
        prompt_epoch: str = "",
        tool_schema_epoch: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Pin *session_id* to a kernel version and return the pinned row.

        Pinning is first-writer-wins. If the session was already pinned to a
        different kernel, the existing pin is returned unchanged — a live
        session must never be moved between kernels, only forked into a new
        branch.
        """
        self.ensure_ready()
        row = self._db.pin_kernel_session_version(
            session_id,
            kernel_version=kernel_version,
            prompt_epoch=prompt_epoch,
            tool_schema_epoch=tool_schema_epoch,
            event_schema_version=EVENT_SCHEMA_VERSION,
            protocol_version=KERNEL_PROTOCOL_VERSION,
            metadata_json=json.dumps(metadata or {}, ensure_ascii=False, default=str),
        )
        pinned = str(row.get("kernel_version") or "")
        if pinned != kernel_version:
            logger.info(
                "kernel session %s already pinned to %s; requested %s ignored",
                session_id,
                pinned,
                kernel_version,
            )
        # Rehydrate invariant state so a resumed process does not re-emit a
        # terminal for a turn that already finished before the restart.
        self._rehydrate_terminals(session_id)
        return row

    def session_version(self, session_id: str) -> Optional[dict[str, Any]]:
        self.ensure_ready()
        return self._db.get_kernel_session_version(session_id)

    def _rehydrate_terminals(self, session_id: str) -> None:
        rows = self._db.read_kernel_events(
            session_id,
            kinds=sorted(
                str(k) for k in (TERMINAL_TURN_KINDS | TERMINAL_TOOL_KINDS)
            )
            + [str(EventKind.TOOL_CALL_STARTED)],
        )
        turns: set[str] = set()
        tools: set[str] = set()
        started: set[str] = set()
        for row in rows:
            kind = str(row.get("kind") or "")
            if kind in {str(k) for k in TERMINAL_TURN_KINDS}:
                turns.add(str(row.get("turn_id") or ""))
            elif kind == str(EventKind.TOOL_CALL_STARTED):
                started.add(str(row.get("tool_call_id") or ""))
            else:
                tools.add(str(row.get("tool_call_id") or ""))
        with self._lock:
            self._terminal_turns[session_id] = turns
            self._terminal_tools[session_id] = tools
            self._started_tools[session_id] = started

    # ── append ──

    def append(self, event: KernelEvent) -> KernelEvent:
        """Durably append one event and return it stamped with its ``seq``.

        Ephemeral kinds skip SQLite (unless configured otherwise) and are
        returned with ``seq=0`` to make it obvious to callers that they are not
        a recoverable fact.
        """
        return self.append_many([event])[0]

    def append_many(self, events: Sequence[KernelEvent]) -> list[KernelEvent]:
        """Append a batch in one transaction, preserving caller order.

        Batching matters for a parallel tool wave: the ``seq`` order of the
        whole batch is decided once, so a projector replaying the ledger sees a
        deterministic interleaving even though the tools completed out of order.
        """
        if not events:
            return []
        self.ensure_ready()

        for event in events:
            self._validate(event)

        durable: list[KernelEvent] = []
        rows: list[dict[str, Any]] = []
        passthrough: list[tuple[int, KernelEvent]] = []
        for index, event in enumerate(events):
            if event.is_ephemeral and not self._persist_ephemeral:
                passthrough.append((index, event))
                continue
            rows.append({**event.to_json(), "payload_json": event.payload_json(),
                         "tool_call_id": event.tool_call_id})
            durable.append(event)

        stamped: list[KernelEvent] = []
        if rows:
            session_id = rows[0]["session_id"]
            if any(row["session_id"] != session_id for row in rows):
                raise InvariantViolation(
                    "append_many received events for multiple sessions; "
                    "sequence numbers are per-session"
                )
            last_seq = self._db.append_kernel_events(session_id, rows)
            first_seq = last_seq - len(rows) + 1
            stamped = [
                replace(event, seq=first_seq + offset)
                for offset, event in enumerate(durable)
            ]
            self._record_terminals(stamped)

        # Reassemble in caller order so the returned list lines up 1:1 with the
        # input, ephemeral events included.
        result: list[KernelEvent] = [None] * len(events)  # type: ignore[list-item]
        for index, event in passthrough:
            result[index] = event
        stamped_iter = iter(stamped)
        for index, event in enumerate(events):
            if result[index] is None:
                result[index] = next(stamped_iter)

        for event in result:
            self._notify(event)
        return result

    def _validate(self, event: KernelEvent) -> None:
        if not event.session_id:
            raise InvariantViolation(f"{event.kind} event has no session_id")
        if event.kind in TOOL_SCOPED_KINDS and not event.tool_call_id:
            raise InvariantViolation(
                f"{event.kind} event must carry payload['tool_call_id']"
            )
        if not self._enforce_invariants:
            return
        with self._lock:
            if event.kind in TERMINAL_TURN_KINDS:
                seen = self._terminal_turns.setdefault(event.session_id, set())
                if event.turn_id and event.turn_id in seen:
                    raise InvariantViolation(
                        f"turn {event.turn_id} already reached a terminal state; "
                        f"refusing second terminal {event.kind}"
                    )
            elif event.kind in TERMINAL_TOOL_KINDS:
                seen_tools = self._terminal_tools.setdefault(event.session_id, set())
                if event.tool_call_id in seen_tools:
                    raise InvariantViolation(
                        f"tool call {event.tool_call_id} already reached a terminal "
                        f"state; refusing second terminal {event.kind}"
                    )

    def _record_terminals(self, events: Iterable[KernelEvent]) -> None:
        with self._lock:
            for event in events:
                if event.kind in TERMINAL_TURN_KINDS and event.turn_id:
                    self._terminal_turns.setdefault(event.session_id, set()).add(
                        event.turn_id
                    )
                elif event.kind in TERMINAL_TOOL_KINDS and event.tool_call_id:
                    self._terminal_tools.setdefault(event.session_id, set()).add(
                        event.tool_call_id
                    )
                elif event.kind == EventKind.TOOL_CALL_STARTED and event.tool_call_id:
                    self._started_tools.setdefault(event.session_id, set()).add(
                        event.tool_call_id
                    )

    # ── read ──

    def read(
        self,
        session_id: str,
        *,
        after_seq: int = 0,
        limit: Optional[int] = None,
        turn_id: Optional[str] = None,
        kinds: Optional[Iterable[EventKind | str]] = None,
    ) -> list[KernelEvent]:
        """Read events after the exclusive cursor *after_seq*, in seq order."""
        self.ensure_ready()
        kind_list = [str(k) for k in kinds] if kinds else None
        rows = self._db.read_kernel_events(
            session_id,
            after_seq=after_seq,
            limit=limit,
            turn_id=turn_id,
            kinds=kind_list,
        )
        return [row_to_event(row) for row in rows]

    def max_seq(self, session_id: str) -> int:
        """The highest durable seq for *session_id* (0 when empty)."""
        self.ensure_ready()
        return self._db.kernel_events_max_seq(session_id)

    def unterminated_tool_calls(self, session_id: str) -> set[str]:
        """Tool calls that started but never reached a terminal event.

        A non-empty result after a turn settles means the turn leaked a tool
        call — the exact condition that leaves a provider history with an
        assistant tool call and no matching result.
        """
        self.ensure_ready()
        started: set[str] = set()
        terminal: set[str] = set()
        for event in self.read(session_id):
            if event.kind == EventKind.TOOL_CALL_STARTED:
                started.add(event.tool_call_id)
            elif event.kind in TERMINAL_TOOL_KINDS:
                terminal.add(event.tool_call_id)
        return started - terminal

    # ── artifacts ──

    def put_artifact(self, artifact: ArtifactRef) -> ArtifactRef:
        """Record an artifact and emit ``artifact.created``.

        The artifact row is written before the event so a subscriber reacting to
        ``artifact.created`` can always resolve the reference.
        """
        self.ensure_ready()
        payload = artifact.to_json()
        metadata = payload.pop("metadata", {})
        self._db.record_kernel_artifact(
            {
                **payload,
                "metadata_json": json.dumps(metadata, ensure_ascii=False, default=str),
            }
        )
        return artifact

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactRef]:
        self.ensure_ready()
        row = self._db.get_kernel_artifact(artifact_id)
        if not row:
            return None
        return _artifact_from_row(row)

    def list_artifacts(self, session_id: str) -> list[ArtifactRef]:
        self.ensure_ready()
        return [
            _artifact_from_row(row)
            for row in self._db.list_kernel_artifacts(session_id)
        ]

    # ── idempotency ──

    def claim_once(
        self,
        scope: str,
        key: str,
        *,
        session_id: str = "",
        result: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Return ``None`` if this caller won the claim, else the prior record."""
        self.ensure_ready()
        prior = self._db.claim_kernel_idempotency_key(
            scope,
            key,
            session_id=session_id,
            result_json=json.dumps(result or {}, ensure_ascii=False, default=str),
        )
        if prior is None:
            return None
        try:
            prior["result"] = json.loads(prior.get("result_json") or "{}")
        except (TypeError, ValueError):
            prior["result"] = {}
        return prior

    # ── checkpoints ──

    def load_checkpoint(self, session_id: str, projector: str) -> tuple[int, dict[str, Any]]:
        self.ensure_ready()
        row = self._db.get_kernel_projector_checkpoint(session_id, projector)
        if not row:
            return 0, {}
        try:
            state = json.loads(row.get("state_json") or "{}")
        except (TypeError, ValueError):
            state = {}
        return int(row.get("last_seq") or 0), state

    def save_checkpoint(
        self,
        session_id: str,
        projector: str,
        *,
        last_seq: int,
        state: Optional[dict[str, Any]] = None,
    ) -> None:
        self.ensure_ready()
        self._db.save_kernel_projector_checkpoint(
            session_id,
            projector,
            last_seq=int(last_seq),
            state_json=json.dumps(state or {}, ensure_ascii=False, default=str),
        )

    # ── live notification ──

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        """Register *callback* for post-commit notification.

        Returns an unsubscribe callable. Subscriber exceptions are swallowed and
        logged: a failing UI bridge must never abort the turn that is feeding it.
        """
        with self._lock:
            self._subscribers.append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(callback)
                except ValueError:
                    pass

        return _unsubscribe

    def _notify(self, event: KernelEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                logger.debug("kernel ledger subscriber failed", exc_info=True)


# ── row conversion ────────────────────────────────────────────────────────────


def row_to_event(row: dict[str, Any]) -> KernelEvent:
    """Rebuild a :class:`KernelEvent` from a ``kernel_events`` row."""
    try:
        payload = json.loads(row.get("payload_json") or "{}")
    except (TypeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {"value": payload}
    # The row stores tool_call_id in its own indexed column; keep the payload
    # authoritative so readers only ever look in one place.
    tool_call_id = str(row.get("tool_call_id") or "")
    if tool_call_id and "tool_call_id" not in payload:
        payload["tool_call_id"] = tool_call_id
    return KernelEvent(
        kind=EventKind(str(row["kind"])),
        session_id=str(row.get("session_id") or ""),
        event_id=str(row.get("event_id") or ""),
        seq=int(row.get("seq") or 0),
        timestamp=float(row.get("timestamp") or 0.0),
        branch_id=str(row.get("branch_id") or ""),
        turn_id=str(row.get("turn_id") or ""),
        agent_id=str(row.get("agent_id") or ""),
        schema_version=int(row.get("schema_version") or EVENT_SCHEMA_VERSION),
        causation_id=str(row.get("causation_id") or ""),
        correlation_id=str(row.get("correlation_id") or ""),
        payload=payload,
    )


def _artifact_from_row(row: dict[str, Any]) -> ArtifactRef:
    try:
        metadata = json.loads(row.get("metadata_json") or "{}")
    except (TypeError, ValueError):
        metadata = {}
    return ArtifactRef(
        artifact_id=str(row.get("artifact_id") or ""),
        session_id=str(row.get("session_id") or ""),
        kind=str(row.get("kind") or ""),
        byte_count=int(row.get("byte_count") or 0),
        digest=str(row.get("digest") or ""),
        path=str(row.get("path") or ""),
        media_type=str(row.get("media_type") or "text/plain"),
        summary=str(row.get("summary") or ""),
        truncated=bool(row.get("truncated")),
        created_at=float(row.get("created_at") or 0.0),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


# ── in-memory ledger (tests, shadow mode) ─────────────────────────────────────


class InMemoryLedger(EventLedger):
    """Ledger backed by a dict instead of SQLite.

    Used by state-machine and projector tests, and by shadow-mode v2 runs that
    must not touch the user's real ``state.db``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(_MemoryStore(), **kwargs)
        self._migrated = True


class _MemoryStore:
    """Minimal in-memory stand-in for the SessionDB surface the ledger uses."""

    def __init__(self) -> None:
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._versions: dict[str, dict[str, Any]] = {}
        self._checkpoints: dict[tuple[str, str], dict[str, Any]] = {}
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._idempotency: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = threading.Lock()

    def apply_kernel_ledger_migration(self) -> None:
        return None

    def kernel_ledger_ready(self) -> bool:
        return True

    def append_kernel_events(self, session_id: str, events: list[dict[str, Any]]) -> int:
        with self._lock:
            bucket = self._events.setdefault(session_id, [])
            known = {row["event_id"] for row in bucket}
            next_seq = (bucket[-1]["seq"] + 1) if bucket else 1
            last_seq = next_seq - 1
            for event in events:
                if event["event_id"] in known:
                    continue
                row = dict(event)
                row["session_id"] = session_id
                row["seq"] = next_seq
                bucket.append(row)
                known.add(event["event_id"])
                last_seq = next_seq
                next_seq += 1
            return last_seq

    def read_kernel_events(
        self,
        session_id: str,
        *,
        after_seq: int = 0,
        limit: Optional[int] = None,
        turn_id: Optional[str] = None,
        kinds: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                dict(row)
                for row in self._events.get(session_id, [])
                if row["seq"] > after_seq
                and (turn_id is None or row.get("turn_id") == turn_id)
                and (kinds is None or row.get("kind") in set(kinds))
            ]
        rows.sort(key=lambda row: row["seq"])
        return rows[:limit] if limit is not None else rows

    def kernel_events_max_seq(self, session_id: str) -> int:
        with self._lock:
            bucket = self._events.get(session_id) or []
            return bucket[-1]["seq"] if bucket else 0

    def get_kernel_session_version(self, session_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._versions.get(session_id)
            return dict(row) if row else None

    def pin_kernel_session_version(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        import time as _time

        with self._lock:
            if session_id not in self._versions:
                now = _time.time()
                self._versions[session_id] = {
                    "session_id": session_id,
                    "created_at": now,
                    "updated_at": now,
                    **kwargs,
                }
            return dict(self._versions[session_id])

    def get_kernel_projector_checkpoint(
        self, session_id: str, projector: str
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._checkpoints.get((session_id, projector))
            return dict(row) if row else None

    def save_kernel_projector_checkpoint(
        self, session_id: str, projector: str, *, last_seq: int, state_json: str = "{}"
    ) -> None:
        import time as _time

        with self._lock:
            key = (session_id, projector)
            prior = self._checkpoints.get(key)
            prior_seq = int(prior.get("last_seq") or 0) if prior else 0
            self._checkpoints[key] = {
                "session_id": session_id,
                "projector": projector,
                "last_seq": max(prior_seq, int(last_seq)),
                "updated_at": _time.time(),
                "state_json": state_json,
            }

    def record_kernel_artifact(self, artifact: dict[str, Any]) -> None:
        with self._lock:
            self._artifacts[artifact["artifact_id"]] = dict(artifact)

    def get_kernel_artifact(self, artifact_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._artifacts.get(artifact_id)
            return dict(row) if row else None

    def list_kernel_artifacts(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                dict(row)
                for row in self._artifacts.values()
                if row.get("session_id") == session_id
            ]
        rows.sort(key=lambda row: (row.get("created_at") or 0, row["artifact_id"]))
        return rows

    def claim_kernel_idempotency_key(
        self, scope: str, key: str, *, session_id: str = "", result_json: str = "{}"
    ) -> Optional[dict[str, Any]]:
        import time as _time

        with self._lock:
            existing = self._idempotency.get((scope, key))
            if existing is not None:
                return dict(existing)
            self._idempotency[(scope, key)] = {
                "scope": scope,
                "key": key,
                "session_id": session_id,
                "result_json": result_json,
                "created_at": _time.time(),
            }
            return None
