"""Durable transport adapter for Hairball's Plan/Build flow.

The plan file is the model-facing source of truth.  This append-only snapshot
store exists only so Hairball's independent UI/CLI/TUI/gateway processes can
show and approve the same revision.  It does not execute steps or journal tool
side effects.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

from agent.plan_execution import replay_plan_state


@dataclass(frozen=True)
class StoredPlan:
    plan: dict[str, Any] | None
    seq: int
    transition: str = ""
    duplicate: bool = False


class PlanStoreConflict(RuntimeError):
    def __init__(
        self,
        expected_seq: int,
        current_seq: int,
        reason: str = "stale_kind_seq",
    ) -> None:
        super().__init__(
            f"plan store conflict: expected seq {expected_seq}, current {current_seq} ({reason})"
        )
        self.expected_seq = int(expected_seq)
        self.current_seq = int(current_seq)
        self.reason = str(reason)


class PlanEventStore:
    """Persist UI transport snapshots on Hairball's existing kernel ledger."""

    SNAPSHOT_KIND = "plan.snapshot.v1"

    def __init__(self, db, *, owns_db: bool = False) -> None:
        self.db = db
        self._owns_db = bool(owns_db)
        self.db.apply_kernel_ledger_migration()

    def __enter__(self) -> "PlanEventStore":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_db:
            close = getattr(self.db, "close", None)
            if callable(close):
                close()
            self._owns_db = False

    @staticmethod
    def _event_id(session_id: str, operation_id: str) -> str:
        sid_hash = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:16]
        op_hash = hashlib.sha256(str(operation_id).encode("utf-8")).hexdigest()[:24]
        return f"plan:{sid_hash}:{op_hash}"

    @staticmethod
    def _decode_event(row: Mapping[str, Any]) -> StoredPlan:
        try:
            payload = json.loads(str(row.get("payload_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid durable plan snapshot: {exc}") from exc
        raw = payload.get("plan") if isinstance(payload, Mapping) else None
        return StoredPlan(
            plan=copy.deepcopy(dict(raw)) if isinstance(raw, Mapping) else None,
            seq=int(row.get("seq") or 0),
            transition=str(payload.get("transition") or ""),
        )

    def load(self, session_id: str) -> StoredPlan:
        rows = self.db.read_kernel_events(str(session_id), kinds=[self.SNAPSHOT_KIND])
        if not rows:
            return StoredPlan(plan=None, seq=0)
        latest = self._decode_event(rows[-1])
        return StoredPlan(
            plan=replay_plan_state(rows),
            seq=latest.seq,
            transition=latest.transition,
        )

    def history(self, session_id: str) -> list[StoredPlan]:
        rows = self.db.read_kernel_events(str(session_id), kinds=[self.SNAPSHOT_KIND])
        return [self._decode_event(row) for row in rows]

    def save_plan(
        self,
        session_id: str,
        plan: Mapping[str, Any] | None,
        *,
        expected_seq: int,
        transition: str,
        operation_id: str,
        actor: str = "",
        now: float | None = None,
    ) -> StoredPlan:
        sid = str(session_id or "").strip()
        operation = str(operation_id or "").strip()
        if not sid:
            raise ValueError("session_id is required")
        if not operation:
            raise ValueError("operation_id is required")
        payload = {
            "schema_version": 2,
            "transition": str(transition or "snapshot"),
            "operation_id": operation,
            "actor": str(actor or ""),
            "plan": copy.deepcopy(dict(plan)) if isinstance(plan, Mapping) else None,
        }
        event = {
            "event_id": self._event_id(sid, operation),
            "timestamp": float(time.time() if now is None else now),
            "kind": self.SNAPSHOT_KIND,
            "schema_version": 2,
            "payload_json": json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        result = self.db.append_kernel_event_if_kind_seq(
            sid,
            kind=self.SNAPSHOT_KIND,
            expected_kind_seq=int(expected_seq or 0),
            event=event,
        )
        if not result.get("ok"):
            raise PlanStoreConflict(
                int(expected_seq or 0),
                int(result.get("current_seq") or 0),
                str(result.get("reason") or "stale_kind_seq"),
            )
        stored = self._decode_event(result["event"])
        if result.get("duplicate"):
            # The ledger returns the original event for an idempotent replay.
            # Callers, however, consume a current state projection.  Returning
            # that historical snapshot after later transitions would move a UI
            # mirror backwards even though no new event was appended.
            latest = self.load(sid)
            return StoredPlan(
                plan=latest.plan,
                seq=latest.seq,
                transition=latest.transition,
                duplicate=True,
            )
        return StoredPlan(
            plan=stored.plan,
            seq=stored.seq,
            transition=stored.transition,
            duplicate=False,
        )

    def clear_plan(
        self,
        session_id: str,
        *,
        expected_seq: int,
        operation_id: str,
        actor: str = "user",
        now: float | None = None,
    ) -> StoredPlan:
        return self.save_plan(
            session_id,
            None,
            expected_seq=expected_seq,
            transition="cleared",
            operation_id=operation_id,
            actor=actor,
            now=now,
        )

    def inherit_plan(
        self,
        source_session_id: str,
        target_session_id: str,
        *,
        operation_id: str,
        reason: str,
        now: float | None = None,
    ) -> StoredPlan:
        """Carry the exact Plan snapshot across one logical-session rotation.

        Compression may mint a continuation ``session_id`` even though the
        logical conversation, reviewed artifact, approval, and active execution
        ownership have not changed.  The continuation therefore receives the
        exact source snapshot.  A real branch must not call this method: branch
        sessions intentionally do not inherit approval or live execution state.
        """

        source_id = str(source_session_id or "").strip()
        target_id = str(target_session_id or "").strip()
        if not source_id or not target_id:
            raise ValueError("source and target session ids are required")
        if source_id == target_id:
            return self.load(target_id)

        source = self.load(source_id)
        target = self.load(target_id)
        if source.plan is None:
            return target
        if target.plan is not None:
            if target.plan == source.plan:
                return target
            raise PlanStoreConflict(
                target.seq,
                target.seq,
                "target_plan_exists",
            )
        return self.save_plan(
            target_id,
            source.plan,
            expected_seq=target.seq,
            transition=f"session_rebound:{str(reason or 'continuation')}",
            operation_id=str(operation_id),
            actor="system",
            now=now,
        )


__all__ = ["PlanEventStore", "PlanStoreConflict", "StoredPlan"]
