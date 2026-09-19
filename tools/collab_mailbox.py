"""Parent-side collab mailbox: doorbell wait + structured envelopes.

reference V2 ``wait_agent`` wakes on input-queue *activity* without carrying the
FINAL_ANSWER body. Hairball mirrors that with an in-process mailbox keyed by
parent session.

Single Rail: content re-enters via ``collab_collect`` (envelope drain). Every
``async_delegation`` completion posts ``agent_completed`` here and is
``suppress_inject`` — inject consumers ack only, never synthesize a parent turn.
This module must never double-inject the same completion into a turn.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

_lock = threading.Lock()
_cond = threading.Condition(_lock)

# When wake_check is provided, poll the condition in short slices so parent
# steer/interrupt can interrupt wait without posting a mailbox envelope.
_WAKE_POLL_S = 0.05

# parent_key -> list of activity event ids (doorbell tokens, FIFO)
_activity: Dict[str, List[str]] = {}
# event_id -> envelope
_envelopes: Dict[str, Dict[str, Any]] = {}
# event_id -> delivery claim state for mailbox-side drains (not inject)
_claims: Dict[str, Dict[str, Any]] = {}


def _reset_for_tests() -> None:
    with _cond:
        _activity.clear()
        _envelopes.clear()
        _claims.clear()


# Kinds that may wake collab_wait. Spawn envelopes are recorded but must NOT
# doorbell — otherwise parallel spawn wakes wait immediately and the model
# collects before children finish (live bug 2026-07-21 session e745a2a6d1ad).
_WAIT_WAKE_KINDS = frozenset({"agent_completed"})


def post_activity(
    parent_key: str,
    *,
    kind: str,
    agent_id: str = "",
    payload: Optional[Dict[str, Any]] = None,
    doorbell: Optional[bool] = None,
) -> str:
    """Post an envelope; optionally enqueue a wait doorbell.

    ``doorbell`` defaults to True for ``agent_completed``, False otherwise.
    """
    key = str(parent_key or "")
    kind_s = str(kind or "activity")
    event_id = f"mb-{uuid.uuid4().hex[:12]}"
    envelope = {
        "event_id": event_id,
        "parent_key": key,
        "kind": kind_s,
        "agent_id": str(agent_id or ""),
        "payload": dict(payload or {}),
        "posted_at": time.time(),
        "delivery_state": "pending",
    }
    ring = kind_s in _WAIT_WAKE_KINDS if doorbell is None else bool(doorbell)
    with _cond:
        _envelopes[event_id] = envelope
        if ring:
            _activity.setdefault(key, []).append(event_id)
            _cond.notify_all()
    # Only completion envelopes carry a body that can re-enter the parent
    # context.  Spawn activity is a status marker, not a mailbox message, so
    # counting it would make the Single Rail's enqueue/deliver accounting lie.
    if kind_s == "agent_completed":
        try:
            from agent.kernel.native_events import emit_mailbox_enqueued

            emit_mailbox_enqueued(
                key,
                envelope_id=event_id,
                envelope_kind=kind_s,
                agent_id=envelope["agent_id"],
                payload=envelope["payload"],
            )
        except Exception:
            pass
    return event_id


def wait_activity(
    parent_key: str,
    timeout_ms: int,
    *,
    wake_check: Optional[Callable[[], Optional[str]]] = None,
    wake_kinds: Optional[frozenset] = None,
) -> Dict[str, Any]:
    """Block until a wake-kind doorbell arrives, wake_check fires, or timeout.

    Never returns message body. Non-wake kinds left in the queue are skipped
    (spawn must not abort a wait for completions). ``wake_check`` may return a
    status string (e.g. ``interrupted``) without consuming a doorbell.
    """
    key = str(parent_key or "")
    kinds = wake_kinds if wake_kinds is not None else _WAIT_WAKE_KINDS
    timeout_s = max(0.0, float(timeout_ms) / 1000.0)
    deadline = time.monotonic() + timeout_s
    with _cond:
        while True:
            if wake_check is not None:
                try:
                    reason = wake_check()
                except Exception:
                    reason = None
                if reason:
                    return {"status": str(reason), "event_id": None}
            pending = _activity.get(key) or []
            if pending:
                # Prefer first wake-kind event; drop stale non-wake tokens.
                chosen = None
                keep: List[str] = []
                for eid in pending:
                    env = _envelopes.get(eid) or {}
                    kind = str(env.get("kind") or "")
                    if chosen is None and (not kinds or kind in kinds):
                        chosen = eid
                    else:
                        keep.append(eid)
                if chosen is not None:
                    _activity[key] = keep
                    return {
                        "status": "woke",
                        "event_id": chosen,
                        # Explicitly omit payload — doorbell only.
                    }
                # Only non-wake leftovers — clear them so we don't spin.
                _activity[key] = []
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"status": "timeout", "event_id": None}
            slice_s = remaining
            if wake_check is not None:
                slice_s = min(remaining, _WAKE_POLL_S)
            _cond.wait(timeout=slice_s)


def peek_envelope(event_id: str) -> Optional[Dict[str, Any]]:
    """Read envelope metadata (for tests / list). Does not claim delivery."""
    with _lock:
        env = _envelopes.get(str(event_id or ""))
        if env is None:
            return None
        return {k: v for k, v in env.items() if k != "payload"} | {
            "has_payload": bool(env.get("payload")),
        }


def get_envelope(event_id: str) -> Optional[Dict[str, Any]]:
    """Return a shallow copy of the full envelope including payload."""
    with _lock:
        env = _envelopes.get(str(event_id or ""))
        if env is None:
            return None
        out = dict(env)
        out["payload"] = dict(env.get("payload") or {})
        return out


def list_envelopes(
    parent_key: str,
    *,
    kind: str = "",
    agent_id: str = "",
    include_delivered: bool = False,
) -> List[Dict[str, Any]]:
    """List envelopes for a parent (newest last). Includes payload copies."""
    key = str(parent_key or "")
    rows: List[Dict[str, Any]] = []
    with _lock:
        for eid, env in _envelopes.items():
            if env.get("parent_key") != key:
                continue
            if kind and env.get("kind") != kind:
                continue
            if agent_id and env.get("agent_id") != agent_id:
                continue
            if not include_delivered and env.get("delivery_state") == "delivered":
                continue
            row = dict(env)
            row["payload"] = dict(env.get("payload") or {})
            rows.append(row)
    rows.sort(key=lambda r: float(r.get("posted_at") or 0.0))
    return rows


def claim_mailbox_delivery(event_id: str, consumer: str) -> Optional[str]:
    """Claim one mailbox envelope for a non-inject consumer (anti double-drain)."""
    eid = str(event_id or "")
    if not eid:
        return None
    claim_id = f"{consumer}:{uuid.uuid4().hex}"
    with _lock:
        env = _envelopes.get(eid)
        if env is None:
            return None
        if env.get("delivery_state") == "delivered":
            return None
        existing = _claims.get(eid)
        if existing and existing.get("state") == "claimed":
            return None
        _claims[eid] = {"claim_id": claim_id, "state": "claimed", "consumer": consumer}
        return claim_id


def complete_mailbox_delivery(event_id: str, claim_id: str) -> bool:
    delivered: Optional[Dict[str, Any]] = None
    with _lock:
        env = _envelopes.get(str(event_id or ""))
        existing = _claims.get(str(event_id or ""))
        if env is None or not existing:
            return False
        if existing.get("claim_id") != claim_id:
            return False
        if existing.get("state") == "delivered" or env.get("delivery_state") == "delivered":
            return False
        env["delivery_state"] = "delivered"
        existing["state"] = "delivered"
        delivered = dict(env)
    # Emit only after the state transition above commits.  A failed/contended
    # claim must never look like a parent received a result.
    try:
        from agent.kernel.native_events import emit_mailbox_delivered

        emit_mailbox_delivered(
            delivered.get("parent_key"),
            envelope_id=delivered.get("event_id"),
            envelope_kind=delivered.get("kind"),
            agent_id=delivered.get("agent_id"),
        )
    except Exception:
        pass
    return True


def release_mailbox_delivery(event_id: str, claim_id: str) -> bool:
    with _lock:
        existing = _claims.get(str(event_id or ""))
        if not existing or existing.get("claim_id") != claim_id:
            return False
        if existing.get("state") == "delivered":
            return False
        _claims.pop(str(event_id or ""), None)
        return True
