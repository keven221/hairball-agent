"""Two-list pending-input inbox for a live agent turn.

Lists are ``next-step`` (steer / inject) and ``next-turn`` (follow-up).
Mutations are normalized splices: the durable record is committed, then the
live projection updates. Message ids are unique across both lists.

Presentation of claimed ``next-step`` text:
- opening wake user still last: insert a separate ``user`` message before it
  (official claim order is next-step, then the queued turn)
- after a completed assistant/tool sequence: append a real ``user`` message
- OpenAI-style adjacent users are flattened only at request time by
  ``repair_message_sequence``; display and persist stay two rows
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

InboxTarget = Literal["next-turn", "next-step"]

_TARGETS: tuple[InboxTarget, InboxTarget] = ("next-turn", "next-step")


def new_inbox_message(content: str, *, message_id: str | None = None) -> dict[str, str]:
    text = str(content or "")
    return {
        "id": str(message_id or uuid.uuid4().hex),
        "role": "user",
        "content": text,
    }


def new_followup_message(
    content: str,
    *,
    message_id: str | None = None,
    model: str = "",
    source: str = "queue",
    enqueued_at: float | None = None,
) -> dict[str, Any]:
    """Session next-turn row. Extra keys stay off the model-visible path."""
    message: dict[str, Any] = new_inbox_message(content, message_id=message_id)
    message["model"] = str(model or "")
    message["source"] = str(source or "queue")
    if enqueued_at is not None:
        message["enqueued_at"] = enqueued_at
    return message


def join_user_content(existing: Any, extra: str) -> Any:
    """Join parked next-step text into an opening user message (no user+user)."""
    text = str(extra or "").strip()
    if not text:
        return existing
    if existing is None:
        return text
    if isinstance(existing, str):
        base = existing.strip()
        if not base:
            return text
        if base == text or base.endswith(text):
            return existing
        return f"{base}\n\n{text}"
    try:
        blocks = list(existing)
        last = blocks[-1] if blocks else None
        if isinstance(last, dict) and str(last.get("text") or "").endswith(text):
            return blocks
        blocks.append({"type": "text", "text": text})
        return blocks
    except Exception:
        return text


def claim_next_step_as_user_messages(agent: Any) -> list[dict[str, Any]]:
    """Claim all next-step rows as one user message. Does not touch next-turn."""
    drain = getattr(agent, "_drain_pending_steer", None)
    if not callable(drain):
        return []
    text = drain()
    if not text or not str(text).strip():
        return []
    return [{"role": "user", "content": str(text).strip()}]


def sync_opening_user_persist(agent: Any, extra: str) -> None:
    """Keep the typed wake persist override unchanged.

    Claimed next-step is a separate user row. Joining it into the override
    rewrote the visible ``继续`` bubble (session 56888beda99a).
    """
    return None


def notify_next_step_claimed(
    agent: Any,
    text: str,
    *,
    merged: bool = False,
) -> None:
    """Tell the live surface that next-step input became model-visible."""
    body = str(text or "").strip()
    if not body:
        return
    notify = getattr(agent, "_on_next_step_claimed", None)
    if not callable(notify):
        return
    try:
        notify(body, merged=merged)
    except TypeError:
        notify(body)


def claim_next_step_into_messages(
    agent: Any,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Claim next-step input as its own user message at a step boundary.

    Official ``Inbox.claim('next-turn')`` returns next-step rows first, then
    the queued wake. When the opening wake user is already last, insert the
    claimed row immediately before it. After assistant/tool output, append.
    """
    claimed = claim_next_step_as_user_messages(agent)
    if not claimed:
        return []
    user_message = claimed[0]
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
        messages.insert(len(messages) - 1, user_message)
    else:
        messages.append(user_message)
    return claimed


def snapshot_inbox(inbox: Any) -> dict[str, list[dict[str, Any]]]:
    """Compact persistable projection of both inbox lists."""
    if inbox is None:
        return {"next_step": [], "next_turn": []}
    return {
        "next_step": [dict(row) for row in getattr(inbox, "next_step", ()) or ()],
        "next_turn": [dict(row) for row in getattr(inbox, "next_turn", ()) or ()],
    }


class TurnInbox:
    """Replay-once projection of inbox splices."""

    def __init__(
        self,
        events: Iterable[Mapping[str, Any]] | None = None,
        *,
        on_inserted: Callable[[Mapping[str, Any]], None] | None = None,
        on_discarded: Callable[[Mapping[str, Any]], None] | None = None,
        on_claimed: Callable[[Mapping[str, Any], int], None] | None = None,
        commit: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self._state: dict[InboxTarget, list[dict[str, Any]]] = {
            "next-turn": [],
            "next-step": [],
        }
        self._log: list[Mapping[str, Any]] = []
        self._on_inserted = on_inserted
        self._on_discarded = on_discarded
        self._on_claimed = on_claimed
        self._commit = commit
        for event in events or ():
            if str(event.get("type") or "") != "agent.inbox.spliced":
                continue
            data = event.get("data")
            if not isinstance(data, Mapping):
                raise ValueError("invalid persisted inbox splice")
            try:
                self._apply(data)
            except Exception as exc:
                raise ValueError("invalid persisted inbox splice") from exc

    @property
    def next_turn(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._state["next-turn"])

    @property
    def next_step(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._state["next-step"])

    @property
    def has_pending(self) -> bool:
        return bool(self._state["next-turn"] or self._state["next-step"])

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._log)

    def clear(self) -> None:
        self.splice("next-step", 0, len(self._state["next-step"]), [])
        self.splice("next-turn", 0, len(self._state["next-turn"]), [])

    def claim(self, target: InboxTarget, turn: int) -> list[dict[str, Any]]:
        claimed = self._mutate("next-step", 0, len(self._state["next-step"]), [], False)
        if target == "next-turn":
            claimed.extend(self._mutate("next-turn", 0, 1, [], False))
        if self._on_claimed is not None:
            for message in claimed:
                self._on_claimed(message, int(turn))
        return claimed

    def append(self, target: InboxTarget, message: Mapping[str, Any]) -> None:
        self.splice(target, len(self._state[target]), 0, [dict(message)])

    def prepend(self, target: InboxTarget, message: Mapping[str, Any]) -> None:
        self.splice(target, 0, 0, [dict(message)])

    def replace(self, message_id: str, new_message: Mapping[str, Any]) -> bool:
        location = self._locate(message_id)
        if location is None:
            return False
        target, index = location
        self.splice(target, index, 1, [dict(new_message)])
        return True

    def remove(self, message_id: str) -> bool:
        location = self._locate(message_id)
        if location is None:
            return False
        target, index = location
        self.splice(target, index, 1, [])
        return True

    def splice(
        self,
        target: InboxTarget,
        start: int,
        delete_count: int,
        inserted: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return self._mutate(target, start, delete_count, [dict(row) for row in inserted], True)

    def next_step_text(self) -> str | None:
        texts = [str(row.get("content") or "") for row in self._state["next-step"]]
        if not texts:
            return None
        return "\n".join(texts)

    def replace_next_step_text(self, text: str | None) -> None:
        self.splice("next-step", 0, len(self._state["next-step"]), [])
        if text:
            self.append("next-step", new_inbox_message(str(text)))

    def _locate(self, message_id: str) -> tuple[InboxTarget, int] | None:
        for target in _TARGETS:
            for index, message in enumerate(self._state[target]):
                if str(message.get("id") or "") == message_id:
                    return target, index
        return None

    def _mutate(
        self,
        target: InboxTarget,
        start: int,
        delete_count: int,
        inserted: list[dict[str, Any]],
        discard_removed: bool,
    ) -> list[dict[str, Any]]:
        inbox = self._state[target]
        try:
            truncated_start = int(start)
        except (TypeError, ValueError):
            truncated_start = 0
        offset = truncated_start
        actual_start = (
            max(len(inbox) + offset, 0) if offset < 0 else min(offset, len(inbox))
        )
        try:
            truncated_delete = int(delete_count)
        except (TypeError, ValueError):
            truncated_delete = 0
        actual_delete = min(max(truncated_delete, 0), len(inbox) - actual_start)
        if actual_delete == 0 and not inserted:
            return []
        splice: dict[str, Any] = {
            "target": target,
            "start": actual_start,
            "inserted": inserted,
        }
        if actual_delete:
            splice["removedCount"] = actual_delete
            if discard_removed:
                splice["outcome"] = "canceled"
        self._validate(splice)
        committed = self._commit(splice) if self._commit is not None else splice
        if not isinstance(committed, Mapping):
            committed = splice
        record = {"type": "agent.inbox.spliced", "data": dict(committed)}
        self._log.append(record)
        removed = inbox[actual_start : actual_start + actual_delete]
        applied = list(committed.get("inserted") or inserted)
        inbox[actual_start : actual_start + actual_delete] = applied
        if discard_removed and self._on_discarded is not None:
            for message in removed:
                self._on_discarded(message)
        if self._on_inserted is not None:
            for message in applied:
                self._on_inserted(message)
        return list(removed)

    def _apply(self, splice: Mapping[str, Any]) -> list[dict[str, Any]]:
        self._validate(splice)
        target = str(splice.get("target") or "")
        if target not in _TARGETS:
            raise ValueError("invalid inbox splice")
        inbox = self._state[target]  # type: ignore[index]
        start = int(splice.get("start") or 0)
        removed_count = int(splice.get("removedCount") or 0)
        inserted = [dict(row) for row in list(splice.get("inserted") or [])]
        removed = inbox[start : start + removed_count]
        inbox[start : start + removed_count] = inserted
        return list(removed)

    def _validate(self, splice: Mapping[str, Any]) -> None:
        target = str(splice.get("target") or "")
        if target not in _TARGETS:
            raise ValueError("invalid inbox splice")
        inbox = self._state[target]  # type: ignore[index]
        start = splice.get("start")
        removed_count = splice.get("removedCount") or 0
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or start < 0
            or start > len(inbox)
            or not isinstance(removed_count, int)
            or isinstance(removed_count, bool)
            or removed_count < 0
            or start + removed_count > len(inbox)
        ):
            raise ValueError("invalid inbox splice")
        inserted = [dict(row) for row in list(splice.get("inserted") or [])]
        candidate = inbox[:start] + inserted + inbox[start + removed_count :]
        other = self._state["next-step"] if target == "next-turn" else self._state["next-turn"]
        seen: set[str] = set()
        for message in [*candidate, *other]:
            message_id = str(message.get("id") or "")
            if not message_id:
                raise ValueError('message "" is already pending')
            if message_id in seen:
                raise ValueError(f'message "{message_id}" is already pending')
            seen.add(message_id)


def load_inbox_snapshot(data: Mapping[str, Any] | None) -> TurnInbox:
    """Rebuild a live inbox from a persisted snapshot. Does not wake a turn."""
    inbox = TurnInbox()
    if not isinstance(data, Mapping):
        return inbox
    for target, key in (("next-step", "next_step"), ("next-turn", "next_turn")):
        for row in data.get(key) or []:
            if not isinstance(row, Mapping):
                continue
            payload = dict(row)
            if not str(payload.get("id") or "").strip():
                payload["id"] = uuid.uuid4().hex
            if "content" not in payload:
                continue
            inbox.append(target, payload)  # type: ignore[arg-type]
    return inbox


def load_durable_turn_inbox(
    session_id: str,
    *,
    db: Any = None,
    snapshot: Mapping[str, Any] | None = None,
) -> TurnInbox:
    """Replay one session inbox from the kernel ledger.

    ``snapshot`` is a one-way compatibility import for pre-ledger UI/gateway
    records. Once any durable inbox splice exists, the ledger is authoritative.
    Each mutation commits its normalized splice before the live projection
    changes, matching :class:`TurnInbox`'s commit contract.
    """
    sid = str(session_id or "").strip()
    if not sid:
        return load_inbox_snapshot(snapshot)

    from agent.kernel.contracts import EventKind, KernelEvent
    from agent.kernel.event_ledger import EventLedger

    owns_db = db is None

    def _open_db():
        if db is not None:
            return db
        from hairball_state import SessionDB

        return SessionDB()

    shared_ledger = EventLedger(db) if db is not None else None

    reader_db = _open_db()
    try:
        ledger = shared_ledger or EventLedger(reader_db)
        persisted = ledger.read(
            sid,
            kinds=[EventKind.AGENT_INBOX_SPLICED],
        )
    finally:
        if owns_db:
            close = getattr(reader_db, "close", None)
            if callable(close):
                close()

    events = [
        {"type": str(event.kind), "data": dict(event.payload)}
        for event in persisted
    ]

    def _commit(splice: dict[str, Any]) -> Mapping[str, Any]:
        writer_db = _open_db()
        try:
            ledger = shared_ledger or EventLedger(writer_db)
            committed = ledger.append(
                KernelEvent.build(
                    EventKind.AGENT_INBOX_SPLICED,
                    sid,
                    payload=splice,
                )
            )
            return dict(committed.payload)
        finally:
            if owns_db:
                close = getattr(writer_db, "close", None)
                if callable(close):
                    close()

    inbox = TurnInbox(events, commit=_commit)
    if events or not isinstance(snapshot, Mapping):
        return inbox

    # Upgrade legacy snapshots in place. Appending through the normal mutation
    # path writes the durable events before exposing the imported projection.
    for target, key in (("next-step", "next_step"), ("next-turn", "next_turn")):
        for row in snapshot.get(key) or []:
            if not isinstance(row, Mapping):
                continue
            payload = dict(row)
            if not str(payload.get("id") or "").strip():
                payload["id"] = uuid.uuid4().hex
            if "content" not in payload:
                continue
            inbox.append(target, payload)  # type: ignore[arg-type]
    return inbox
