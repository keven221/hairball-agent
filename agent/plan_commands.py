"""Shared Hairball Plan/Build controls for every product surface."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Mapping

from agent.plan_execution import (
    PlanContractError,
    plan_exit_succeeded,
    project_plan_event,
    resolve_plan_file,
    validate_plan_execution,
)
from agent.plan_runtime import PlanRuntime
from agent.plan_store import PlanEventStore, StoredPlan
from agent.plan_handoff import normalize_plan_context_mode


logger = logging.getLogger(__name__)

PLAN_MODE_INSTRUCTION_TEMPLATE = """<system-reminder>
Plan mode is active. Preserve read-only working-tree and system semantics.

- NEVER create, edit, delete, or rename working-tree files.
- NEVER run state-changing commands or mutate services/external state.
- The only writable artifact is the Hairball plan file described below.
- To request human review, call `plan_exit` after the file is complete.

## What the plan is

The plan is an execution spec, not a design essay. A competent implementer who
never saw this conversation must be able to execute the file top to bottom
without making a design decision. Detail exists to remove decisions, not to
look thorough. Completeness wins when brevity and decision-completeness conflict.

## Plan file

{plan_info}

Update it incrementally as facts are learned. The file is the source of truth
for review, compression, resume, handoff, and execution.

## Ground every claim

- Discoverable facts (paths, callers, signatures, configuration, current
  behavior) MUST be verified by reading/searching the real workspace. Mark any
  fact that could not be verified as `unverified — confirm first`.
- Preferences and tradeoffs cannot be derived from code. Use `clarify` for
  load-bearing choices, with mutually exclusive options and a recommendation.
- Every question must change the plan or settle a real choice. Batch related
  questions; never ask what read-only exploration can answer.

## Workflow

1. Understand the literal request and inspect the relevant code.
2. Explore independent areas in parallel with read-only explorer subagents when
   useful; reuse existing symbols and conventions before proposing new ones.
3. Commit to one approach, then pressure-test it against callers, shared state,
   neighboring surfaces, failure paths, and existing tests.
4. Write a decision-complete plan and re-read every critical target before
   requesting approval.

## Required plan contents

- **Context** — the literal ask, why it is needed, and the intended end state.
- **Approach** — ordered behavior-level steps. Each step names exact targets,
  existing symbols to reuse, dependencies, edge/error behavior, and observable
  success. For renames/removals/signature changes, list every caller to update.
- **Critical files & anchors** — only the few files that disambiguate the work.
- **Verification** — exact commands and at least one end-to-end new-behavior
  check with concrete input and expected observable output.
- **Assumptions & contingencies** — only user-overridable decisions; pre-decide
  the fallback for every load-bearing assumption that may prove false.

Do not pad the file with decision-free Non-Goals, Alternatives, Future Work, or
mechanical cleanup steps. Never refer to choices as "discussed above"; the
implementer will only have the file.

Before approval, apply this test: an engineer with only the plan can execute
every step, handle its failures, and tell whether it worked without inventing
anything. If not, deepen the plan first.

The turn ends only by using `clarify` for a necessary choice or successfully
calling `plan_exit`. Do not request approval in prose. Keep going until the plan
is decision-complete.
</system-reminder>"""

BUILD_SWITCH_INSTRUCTION_TEMPLATE = """<system-reminder>
Plan mode has ended and the approved plan is ready for execution.

You MUST read `{plan_file}` before executing. That file is the authoritative
source of truth; visible or compressed conversation context is secondary. If it
cannot be read, report the exact path and error instead of guessing.

After reading it, execute the plan step by step with the full Hairball toolset
under the user's normal safety and approval policy. Initialize/update the
existing todo tracker when available, verify each step before moving on, and
continue until the approved work is complete. Preserve every working capability
outside the approved scope.
</system-reminder>"""

# Compatibility constants for callers that import them before a concrete
# session path is known. Runtime callers receive the rendered variants.
PLAN_MODE_INSTRUCTION = PLAN_MODE_INSTRUCTION_TEMPLATE.format(
    plan_info="The concrete path is supplied when the Plan turn starts."
)
EXECUTE_MODE_INSTRUCTION = BUILD_SWITCH_INSTRUCTION_TEMPLATE.format(
    plan_file="the approved Hairball plan file"
)


def render_plan_mode_instruction(plan_file: str | Path, *, exists: bool) -> str:
    path = str(plan_file)
    info = (
        f"A plan file exists at {path}. Read it and make incremental edits."
        if exists
        else f"No plan file exists yet. Create the plan at {path} with `write_file`."
    )
    return PLAN_MODE_INSTRUCTION_TEMPLATE.format(plan_info=info)


def render_build_switch_instruction(plan_file: str | Path) -> str:
    return BUILD_SWITCH_INSTRUCTION_TEMPLATE.format(plan_file=str(plan_file))


def evaluate_plan_turn_result(
    result: Mapping[str, Any] | None,
    *,
    response: str = "",
) -> dict[str, Any]:
    """Return presentation-only Build outcome; never judge task completeness."""

    outcome = dict(result or {})
    raw_step_outcomes = outcome.get("plan_step_outcomes")
    step_outcomes: dict[str, str] = {}
    if isinstance(raw_step_outcomes, Mapping):
        for step_id, raw in raw_step_outcomes.items():
            status = raw.get("status") if isinstance(raw, Mapping) else raw
            step_outcomes[str(step_id or "")] = str(status or "pending").strip().lower()
    elif isinstance(raw_step_outcomes, list):
        for raw in raw_step_outcomes:
            if isinstance(raw, Mapping) and str(raw.get("id") or "").strip():
                step_outcomes[str(raw.get("id")).strip()] = str(
                    raw.get("status") or "pending"
                ).strip().lower()

    cancelled = bool(
        outcome.get("interrupted")
        or outcome.get("cancelled")
        or outcome.get("canceled")
    )
    if cancelled:
        reason = str(
            outcome.get("error")
            or outcome.get("interrupt_reason")
            or outcome.get("cancel_reason")
            or "user_interrupted"
        )
        return {
            "success": False,
            "ready": False,
            "state": "cancelled",
            "assessment": {},
            "step_outcomes": step_outcomes,
            "error": reason,
            "response": str(response or ""),
        }
    unfinished = [
        step_id
        for step_id, status in step_outcomes.items()
        if status not in {"completed", "done", "skipped"}
    ]
    hard_error = bool(
        outcome.get("is_error")
        or outcome.get("failed")
        or outcome.get("error")
        or outcome.get("completed") is False
        or not step_outcomes
        or unfinished
    )
    error = str(outcome.get("error") or "")
    if not step_outcomes and not error:
        error = "selected plan step outcomes are missing"
    elif unfinished and not error:
        error = "selected plan steps remain unfinished: " + ", ".join(unfinished)
    return {
        "success": not hard_error,
        "ready": not hard_error,
        "state": "finished" if not hard_error else "failed",
        "assessment": {},
        "step_outcomes": step_outcomes,
        "error": error or ("agent build failed" if hard_error else ""),
        "response": str(response or ""),
    }


def extract_plan_steps(text: str) -> list[dict[str, Any]]:
    """Build a UI checklist projection from Markdown list items."""

    steps: list[dict[str, Any]] = []
    for line in str(text or "").splitlines():
        match = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(?:\[[ xX]\]\s*)?(.+)$", line)
        if not match:
            continue
        body = match.group(1).strip()
        if len(body) >= 3:
            steps.append({"text": body, "status": "pending"})
    return steps[:80]


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "\x1f".join(str(part or "") for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:24]}"


def _open_runtime(
    session_id: str,
    *,
    surface: str,
    actor: str,
    db=None,
    db_path: str | Path | None = None,
) -> tuple[PlanEventStore, PlanRuntime]:
    owns_db = db is None
    if db is None:
        from hairball_state import SessionDB

        db = SessionDB(db_path=Path(db_path) if db_path is not None else None)
    store = PlanEventStore(db, owns_db=owns_db)
    return store, PlanRuntime(
        store,
        session_id=str(session_id),
        surface=str(surface),
        actor=str(actor),
    )


def _project(stored: StoredPlan | None, *, surface: str) -> dict[str, Any] | None:
    if stored is None or not isinstance(stored.plan, Mapping):
        return None
    projection = project_plan_event(stored.plan, surface=surface)
    projection["store_seq"] = int(stored.seq)
    projection["transition"] = str(stored.transition or "")
    return projection


def _status_message(plan: Mapping[str, Any] | None) -> str:
    if not plan:
        return "No Hairball plan is active."
    approval = plan.get("approval") if isinstance(plan.get("approval"), Mapping) else {}
    lines = [
        f"Plan {plan.get('plan_id')} · revision {plan.get('revision')} · {plan.get('status')}",
        f"file: {plan.get('plan_file')}",
        f"digest: {plan.get('digest')}",
    ]
    if approval:
        lines.append(f"approval: {approval.get('request_id')} · {approval.get('status')}")
    for index, step in enumerate(plan.get("todos") or [], start=1):
        lines.append(f"{index}. [{step.get('status') or 'pending'}] {step.get('text') or ''}")
    return "\n".join(lines)


def _build_context(
    session_id: str,
    plan: Mapping[str, Any],
    *,
    resume: bool,
    operation_seed: Any,
    execution_id: str = "",
    context_mode: str = "fresh",
) -> dict[str, Any]:
    approval = plan.get("approval") if isinstance(plan.get("approval"), Mapping) else {}
    stable_execution_id = str(execution_id or "").strip() or _stable_id(
        "plan-build", session_id, plan.get("plan_id"), plan.get("revision"),
        plan.get("digest"), operation_seed,
    )
    return {
        "kind": "execute",
        "message": f"Execute the approved plan at {plan.get('plan_file')}.",
        "agent_mode": "build",
        "system_message": render_build_switch_instruction(str(plan.get("plan_file") or "")),
        "plan_id": str(plan.get("plan_id") or ""),
        "plan_revision": int(plan.get("revision") or 0),
        "plan_digest": str(plan.get("digest") or ""),
        "plan_approval_request_id": str(approval.get("request_id") or ""),
        "plan_execution_id": stable_execution_id,
        "resume": bool(resume),
        "context_mode": str(context_mode),
        "synthetic": True,
        "synthetic_kind": "plan-approved",
        "plan": dict(plan),
        "validated_plan": dict(plan),
    }


def load_claimed_plan_steps(
    session_id: str,
    execution_id: str,
    *,
    db=None,
    db_path: str | Path | None = None,
) -> list[dict[str, str]]:
    """Load the exact step claim that was durably acquired before dispatch."""

    bound_id = str(execution_id or "").strip()
    if not bound_id:
        return []
    store, runtime = _open_runtime(
        session_id,
        surface="agent",
        actor="system",
        db=db,
        db_path=db_path,
    )
    try:
        plan = runtime.load().plan or {}
        build = plan.get("build") if isinstance(plan.get("build"), Mapping) else {}
        if (
            str(build.get("status") or "") != "running"
            or str(build.get("execution_id") or "") != bound_id
        ):
            return []
        selected = set(str(value or "") for value in build.get("selected_step_ids") or [])
        return [
            {
                "id": str(step.get("id") or ""),
                "content": str(step.get("text") or ""),
                "status": "in_progress",
            }
            for step in plan.get("steps") or []
            if isinstance(step, Mapping) and str(step.get("id") or "") in selected
        ]
    finally:
        store.close()


def prepare_transport_plan_steps(
    steps: Any,
    selected_step_ids: Any = None,
) -> list[dict[str, str]]:
    """Validate proxy-carried canonical Plan rows for protected Todo scope.

    The owning gateway has already validated and claimed the Plan.  A remote
    execution host receives those exact ids/text rows as transport metadata so
    it can return per-step facts without inventing a second approval ledger.
    """

    if not isinstance(steps, list) or not steps:
        raise PlanContractError(
            "transport_plan_steps_required",
            "proxy Plan execution requires canonical plan steps",
        )
    rows: list[dict[str, str]] = []
    known: set[str] = set()
    for raw in steps:
        if not isinstance(raw, Mapping):
            raise PlanContractError(
                "invalid_transport_plan_step",
                "proxy Plan step must be an object",
            )
        step_id = str(raw.get("id") or "").strip()
        text = str(raw.get("text") or raw.get("content") or "").strip()
        if not step_id or not text or step_id in known:
            raise PlanContractError(
                "invalid_transport_plan_step",
                "proxy Plan steps require unique non-empty ids and text",
            )
        known.add(step_id)
        rows.append({"id": step_id, "content": text, "status": "in_progress"})
    if selected_step_ids is None:
        selected = set(known)
    else:
        if not isinstance(selected_step_ids, list) or not selected_step_ids:
            raise PlanContractError(
                "invalid_transport_plan_selection",
                "proxy Plan selection must be a non-empty list",
            )
        selected = {str(value or "").strip() for value in selected_step_ids}
        unknown = sorted(value for value in selected if value not in known)
        if "" in selected or unknown:
            raise PlanContractError(
                "unknown_plan_step",
                "proxy selected Plan step does not exist",
                unknown_step_ids=unknown or ["<empty>"],
            )
    return [row for row in rows if row["id"] in selected]


def migrate_plan_to_session(
    source_session_id: str,
    target_session_id: str,
    *,
    reason: str,
    db=None,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Preserve canonical Plan state across a continuation-id rotation.

    This is only for the same logical conversation (currently legacy context
    compression).  Independent branches deliberately start without inherited
    approval/execution authority.
    """

    source_id = str(source_session_id or "").strip()
    target_id = str(target_session_id or "").strip()
    if not source_id or not target_id or source_id == target_id:
        return None
    owns_db = db is None
    if db is None:
        from hairball_state import SessionDB

        db = SessionDB(db_path=Path(db_path) if db_path is not None else None)
    store = PlanEventStore(db, owns_db=owns_db)
    try:
        stored = store.inherit_plan(
            source_id,
            target_id,
            operation_id=_stable_id(
                "plan-session-rebind", source_id, target_id, reason
            ),
            reason=str(reason or "continuation"),
        )
        return _project(stored, surface="agent")
    finally:
        store.close()


def begin_plan_turn(
    session_id: str,
    message: str,
    *,
    surface: str,
    db=None,
) -> dict[str, Any]:
    """Create one Plan turn context without interpreting user text as a command."""

    path = resolve_plan_file(session_id, db=db)
    path.parent.mkdir(parents=True, exist_ok=True)
    instruction = render_plan_mode_instruction(path, exists=path.is_file())
    logger.info(
        "plan_mode_enter session=%s surface=%s file=%s exists=%s",
        str(session_id),
        str(surface),
        str(path),
        path.is_file(),
    )
    return {
        "kind": "prompt",
        "message": str(message or "").strip(),
        "agent_mode": "plan",
        "system_message": instruction,
        "plan_file": str(path),
    }


def handle_plan_command(
    session_id: str,
    argument: str,
    *,
    surface: str,
    actor: str,
    db=None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve one surface-neutral ``/plan`` command."""

    raw = str(argument or "").strip()
    parts = raw.split()
    command = parts[0].lower() if parts else "status"
    controls = {"status", "show", "approve", "reject", "execute", "resume", "cancel", "clear"}

    if command not in controls:
        return begin_plan_turn(session_id, raw, surface=surface, db=db)

    store, runtime = _open_runtime(
        session_id,
        surface=surface,
        actor=actor,
        db=db,
        db_path=db_path,
    )
    try:
        if command in {"status", "show"}:
            plan = _project(runtime.load(), surface=surface)
            return {"kind": "status", "plan": plan, "message": _status_message(plan)}

        if command in {"approve", "reject"}:
            if len(parts) < 4:
                raise PlanContractError(
                    "approval_precondition_required",
                    f"usage: /plan {command} <request_id> <revision> <digest>",
                )
            request_id, revision_text, digest = parts[1:4]
            try:
                revision = int(revision_text)
            except ValueError as exc:
                raise PlanContractError(
                    "approval_precondition_required",
                    f"usage: /plan {command} <request_id> <revision> <digest>",
                ) from exc
            resolution_id = _stable_id(
                "plan-resolution", session_id, command, request_id, revision, digest, actor
            )
            stored = runtime.resolve_approval(
                request_id,
                command,
                expected_revision=revision,
                expected_digest=digest,
                resolution_id=resolution_id,
                operation_id=f"plan-command:{resolution_id}",
            )
            plan = _project(stored, surface=surface)
            if command == "approve" and plan:
                context_mode = normalize_plan_context_mode(
                    parts[4] if len(parts) > 4 else None,
                    default="fresh",
                )
                context = _build_context(
                    session_id,
                    plan,
                    resume=False,
                    operation_seed=resolution_id,
                    context_mode=context_mode,
                )
                claimed = runtime.begin_build(
                    execution_id=context["plan_execution_id"],
                    operation_id=f"plan-command:begin:{context['plan_execution_id']}",
                )
                context["plan"] = _project(claimed, surface=surface)
                return context
            if command == "reject" and plan:
                feedback = " ".join(parts[4:]).strip()
                prompt = (
                    "Refine the plan using this review feedback: " + feedback
                    if feedback
                    else "Refine the plan after the user's review, then request approval again."
                )
                context = begin_plan_turn(
                    session_id,
                    prompt,
                    surface=surface,
                    db=store.db,
                )
                context["plan"] = plan
                context["review_feedback"] = feedback
                return context
            return {"kind": "status", "plan": plan, "message": _status_message(plan)}

        if command in {"execute", "resume"}:
            if len(parts) < 5:
                raise PlanContractError(
                    "execution_precondition_required",
                    f"usage: /plan {command} <plan_id> <revision> <digest> <request_id>",
                )
            plan_id, revision_text, digest, request_id = parts[1:5]
            try:
                revision = int(revision_text)
            except ValueError as exc:
                raise PlanContractError(
                    "execution_precondition_required",
                    f"usage: /plan {command} <plan_id> <revision> <digest> <request_id>",
                ) from exc
            current = runtime.load()
            validated = validate_plan_execution(
                current.plan or {},
                plan_id=plan_id,
                expected_revision=revision,
                expected_digest=digest,
                approval_request_id=request_id,
            )
            context = _build_context(
                session_id,
                validated,
                resume=command == "resume",
                operation_seed=(command, current.seq),
                context_mode=normalize_plan_context_mode(
                    parts[5] if len(parts) > 5 else None,
                    default="keep" if command == "resume" else "fresh",
                ),
            )
            claimed = runtime.begin_build(
                execution_id=context["plan_execution_id"],
                operation_id=f"plan-command:begin:{context['plan_execution_id']}",
            )
            context["plan"] = _project(claimed, surface=surface)
            return context

        if command == "cancel":
            current = runtime.load()
            if current.plan is None:
                return {"kind": "status", "plan": None, "message": "No Hairball plan is active."}
            build = current.plan.get("build")
            execution_id = (
                str(build.get("execution_id") or "")
                if isinstance(build, Mapping) and str(build.get("status") or "") == "running"
                else ""
            )
            stored = runtime.cancel(
                execution_id=execution_id,
                reason="user_cancelled",
                operation_id=_stable_id("plan-cancel", session_id, current.seq, actor),
            )
            plan = _project(stored, surface=surface)
            return {"kind": "status", "plan": plan, "message": _status_message(plan)}

        current = runtime.load()
        if current.plan is not None:
            runtime.clear(
                operation_id=_stable_id("plan-clear", session_id, current.seq, actor)
            )
        return {"kind": "status", "plan": None, "message": "Hairball plan cleared."}
    finally:
        store.close()


def cancel_active_plan_turn(
    session_id: str,
    execution_id: str,
    *,
    surface: str,
    actor: str,
    reason: str,
    operation_id: str,
    db=None,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Persist an immediate exact-ID cancellation from an interrupt surface."""

    bound_execution_id = str(execution_id or "").strip()
    if not bound_execution_id:
        raise PlanContractError(
            "missing_execution_id",
            "plan execution id is required for active-turn cancellation",
        )
    store, runtime = _open_runtime(
        session_id,
        surface=surface,
        actor=actor,
        db=db,
        db_path=db_path,
    )
    try:
        stored = runtime.cancel(
            execution_id=bound_execution_id,
            reason=reason,
            operation_id=operation_id,
        )
        return _project(stored, surface=surface)
    finally:
        store.close()


def finalize_plan_turn(
    session_id: str,
    context: Mapping[str, Any],
    *,
    response: str,
    result: Mapping[str, Any] | None,
    surface: str,
    actor: str,
    operation_id: str,
    db=None,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Persist plan-file or Build presentation state after the Agent turn."""

    kind = str(context.get("kind") or "")
    store, runtime = _open_runtime(
        session_id,
        surface=surface,
        actor=actor,
        db=db,
        db_path=db_path,
    )
    try:
        if kind == "prompt":
            plan_file = Path(
                str(context.get("plan_file") or resolve_plan_file(session_id, db=store.db))
            )
            if not plan_file.is_file():
                logger.warning(
                    "plan_turn_finished_without_file session=%s surface=%s file=%s",
                    str(session_id),
                    str(surface),
                    str(plan_file),
                )
                return _project(runtime.load(), surface=surface)
            content = plan_file.read_text(encoding="utf-8").strip()
            if not content:
                raise PlanContractError("empty_plan_content", "plan file is empty")
            stable = str(operation_id or _stable_id("plan-turn", session_id, content))
            captured = runtime.capture(
                content,
                extract_plan_steps(content),
                plan_file=str(plan_file),
                operation_id=f"plan-command:capture:{stable}",
            )
            if not plan_exit_succeeded(result):
                logger.info(
                    "plan_turn_remains_draft session=%s surface=%s file=%s revision=%s",
                    str(session_id),
                    str(surface),
                    str(plan_file),
                    int((captured.plan or {}).get("revision") or 0),
                )
                return _project(captured, surface=surface)
            pending = runtime.request_approval(
                timeout=None,
                request_id=_stable_id(
                    "plan-approval",
                    session_id,
                    (captured.plan or {}).get("digest"),
                ),
                operation_id=f"plan-command:approval:{stable}",
            )
            return _project(pending, surface=surface)

        if kind != "execute":
            return _project(runtime.load(), surface=surface)

        if bool((result or {}).get("execution_not_dispatched")):
            execution_id = str(context.get("plan_execution_id") or operation_id).strip()
            stored = runtime.release_build(
                execution_id=execution_id,
                reason=str(
                    (result or {}).get("error")
                    or (result or {}).get("reason")
                    or "execution_not_dispatched"
                ),
                operation_id=f"plan-command:defer:{operation_id}",
            )
            return _project(stored, surface=surface)

        decision = evaluate_plan_turn_result(result, response=response)
        execution_id = str(context.get("plan_execution_id") or operation_id).strip()
        if decision["state"] == "cancelled":
            stored = runtime.cancel(
                execution_id=execution_id,
                reason=str(decision["error"] or "user_interrupted"),
                operation_id=f"plan-command:cancel:{operation_id}",
            )
            return _project(stored, surface=surface)
        stored = runtime.record_build(
            execution_id=execution_id,
            success=bool(decision["success"]),
            step_outcomes=decision["step_outcomes"],
            output={"final_response": str(response or "")[:12000]},
            error=str(decision["error"] or ""),
            operation_id=f"plan-command:build:{operation_id}",
        )
        return _project(stored, surface=surface)
    finally:
        store.close()


__all__ = [
    "BUILD_SWITCH_INSTRUCTION_TEMPLATE",
    "EXECUTE_MODE_INSTRUCTION",
    "PLAN_MODE_INSTRUCTION",
    "PLAN_MODE_INSTRUCTION_TEMPLATE",
    "begin_plan_turn",
    "cancel_active_plan_turn",
    "evaluate_plan_turn_result",
    "extract_plan_steps",
    "finalize_plan_turn",
    "handle_plan_command",
    "load_claimed_plan_steps",
    "migrate_plan_to_session",
    "prepare_transport_plan_steps",
    "render_build_switch_instruction",
    "render_plan_mode_instruction",
]
