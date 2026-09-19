"""Session-scoped adapter for Hairball Plan/Build state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent.plan_execution import (
    begin_build,
    build_plan,
    cancel_execution,
    project_plan_event,
    record_build_outcome,
    request_plan_approval,
    resolve_plan_approval,
)
from agent.plan_store import PlanEventStore, PlanStoreConflict, StoredPlan


class PlanRuntime:
    """Persist plan-file review state without scheduling model work."""

    def __init__(
        self,
        store: PlanEventStore,
        *,
        session_id: str,
        surface: str,
        actor: str,
    ) -> None:
        self.store = store
        self.session_id = str(session_id or "").strip()
        self.surface = str(surface or "").strip().lower()
        self.actor = str(actor or "")
        if not self.session_id:
            raise ValueError("session_id is required")

    def load(self) -> StoredPlan:
        return self.store.load(self.session_id)

    def _save(
        self,
        current: StoredPlan,
        plan: dict[str, Any] | None,
        *,
        transition: str,
        operation_id: str,
        now: float | None = None,
    ) -> StoredPlan:
        return self.store.save_plan(
            self.session_id,
            plan,
            expected_seq=current.seq,
            transition=transition,
            operation_id=operation_id,
            actor=self.actor,
            now=now,
        )

    def capture(
        self,
        content: str,
        steps: Sequence[Any] | None,
        *,
        plan_file: str,
        operation_id: str,
        now: float | None = None,
    ) -> StoredPlan:
        current = self.load()
        plan = build_plan(
            content,
            steps,
            previous=current.plan,
            plan_file=plan_file,
            now=now,
        )
        if current.plan == plan:
            return current
        return self._save(
            current,
            plan,
            transition="captured" if current.plan is None else "revised",
            operation_id=operation_id,
            now=now,
        )

    # Compatibility name used by product adapters; semantics are file capture,
    # not a second plan manager.
    def propose(
        self,
        content: str,
        steps: Sequence[Any] | None,
        *,
        operation_id: str,
        plan_file: str = "",
        now: float | None = None,
    ) -> StoredPlan:
        path = str(plan_file or (self.load().plan or {}).get("plan_file") or "")
        return self.capture(
            content,
            steps,
            plan_file=path,
            operation_id=operation_id,
            now=now,
        )

    def request_approval(
        self,
        *,
        timeout: float | None,
        request_id: str,
        operation_id: str,
        now: float | None = None,
    ) -> StoredPlan:
        current = self.load()
        plan = request_plan_approval(
            current.plan or {},
            session_id=self.session_id,
            surface=self.surface,
            timeout=timeout,
            request_id=request_id,
            now=now,
        )
        return self._save(
            current,
            plan,
            transition="approval_requested",
            operation_id=operation_id,
            now=now,
        )

    def resolve_approval(
        self,
        request_id: str,
        decision: str,
        *,
        expected_revision: int,
        expected_digest: str,
        resolution_id: str,
        operation_id: str,
        now: float | None = None,
    ) -> StoredPlan:
        current = self.load()
        plan = resolve_plan_approval(
            current.plan or {},
            request_id,
            decision,
            actor=self.actor,
            expected_revision=expected_revision,
            expected_digest=expected_digest,
            resolution_id=resolution_id,
            now=now,
        )
        return self._save(
            current,
            plan,
            transition="approved" if plan.get("status") == "approved" else "approval_rejected",
            operation_id=operation_id,
            now=now,
        )

    def record_build(
        self,
        *,
        execution_id: str,
        success: bool,
        step_outcomes: Any = None,
        output: Any,
        error: str,
        operation_id: str,
        now: float | None = None,
    ) -> StoredPlan:
        current = self.load()
        if current.plan is None:
            return current
        plan = record_build_outcome(
            current.plan,
            execution_id=execution_id,
            success=success,
            step_outcomes=step_outcomes,
            output=output,
            error=error,
            now=now,
        )
        if plan == current.plan:
            return current
        build = plan.get("build") if isinstance(plan.get("build"), dict) else {}
        transition = (
            "build_finished"
            if str(build.get("status") or "") == "finished"
            else "build_failed"
        )
        try:
            return self._save(
                current,
                plan,
                transition=transition,
                operation_id=operation_id,
                now=now,
            )
        except PlanStoreConflict as conflict:
            latest = self.load()
            if latest.plan is None:
                return latest
            reconciled = record_build_outcome(
                latest.plan,
                execution_id=execution_id,
                success=success,
                step_outcomes=step_outcomes,
                output=output,
                error=error,
                now=now,
            )
            if reconciled == latest.plan:
                return latest
            raise conflict

    def begin_build(
        self,
        *,
        execution_id: str,
        selected_step_ids: Sequence[str] | None = None,
        operation_id: str,
        now: float | None = None,
    ) -> StoredPlan:
        current = self.load()
        plan = begin_build(
            current.plan or {},
            execution_id=execution_id,
            selected_step_ids=selected_step_ids,
            now=now,
        )
        if plan == current.plan:
            return current
        try:
            return self._save(
                current,
                plan,
                transition="build_started",
                operation_id=operation_id,
                now=now,
            )
        except PlanStoreConflict as conflict:
            latest = self.load()
            reconciled = begin_build(
                latest.plan or {},
                execution_id=execution_id,
                selected_step_ids=selected_step_ids,
                now=now,
            )
            if reconciled == latest.plan:
                return latest
            raise conflict

    def release_build(
        self,
        *,
        execution_id: str,
        reason: str,
        operation_id: str,
        now: float | None = None,
    ) -> StoredPlan:
        """Release a claim when approved execution was never dispatched."""

        from agent.plan_execution import release_build_claim

        current = self.load()
        plan = release_build_claim(
            current.plan or {},
            execution_id=execution_id,
            reason=reason,
            now=now,
        )
        if plan == current.plan:
            return current
        try:
            return self._save(
                current,
                plan,
                transition="build_deferred",
                operation_id=operation_id,
                now=now,
            )
        except PlanStoreConflict as conflict:
            latest = self.load()
            reconciled = release_build_claim(
                latest.plan or {},
                execution_id=execution_id,
                reason=reason,
                now=now,
            )
            if reconciled == latest.plan:
                return latest
            raise conflict

    def cancel(
        self,
        *,
        execution_id: str = "",
        reason: str,
        operation_id: str,
        now: float | None = None,
    ) -> StoredPlan:
        current = self.load()
        if current.plan is None:
            return current
        plan = cancel_execution(
            current.plan,
            execution_id=execution_id,
            reason=reason,
            now=now,
        )
        if plan == current.plan:
            return current
        try:
            return self._save(
                current,
                plan,
                transition="cancelled",
                operation_id=operation_id,
                now=now,
            )
        except PlanStoreConflict as conflict:
            latest = self.load()
            if latest.plan is None:
                return latest
            reconciled = cancel_execution(
                latest.plan,
                execution_id=execution_id,
                reason=reason,
                now=now,
            )
            if reconciled == latest.plan:
                return latest
            raise conflict

    def clear(self, *, operation_id: str, now: float | None = None) -> StoredPlan:
        current = self.load()
        return self.store.clear_plan(
            self.session_id,
            expected_seq=current.seq,
            operation_id=operation_id,
            actor=self.actor,
            now=now,
        )

    def project(self) -> dict[str, Any]:
        current = self.load()
        return project_plan_event(current.plan or {}, surface=self.surface)


__all__ = ["PlanRuntime"]
