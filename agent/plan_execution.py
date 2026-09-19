"""Hairball Plan -> Build contract.

Hairball keeps revision/digest/approval fields solely as a transport adapter:
its UI, CLI, TUI, and messaging gateway cannot all block inside one tool call,
so the exact reviewed plan is approved asynchronously.  Those fields never
schedule plan steps, suppress model retries, or deduplicate side effects.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

PLAN_SCHEMA_VERSION = 2
PLAN_STEP_STATUSES = frozenset(
    {"pending", "in_progress", "completed", "failed", "skipped", "cancelled"}
)
PLAN_EXECUTABLE_STEP_STATUSES = frozenset(
    {"pending", "in_progress", "failed", "cancelled"}
)
PLAN_SUCCESS_STEP_STATUSES = frozenset({"completed", "skipped"})
_AGENT_MODES = frozenset({"plan", "build"})
_MODE_ALIASES = {
    "": "build",
    "default": "build",
    "normal": "build",
    "discover": "build",
    "execute": "build",
    "execution": "build",
    "action": "build",
    "verify": "build",
    "verification": "build",
    "review": "build",
    "complete": "build",
    "planning": "plan",
}

# The execution-time Plan context follows the tool worker through Hairball's
# existing ContextVar propagation.  It is deliberately not a model argument:
# direct tool/MCP callers cannot forge permission to write a Plan artifact.
_PLAN_TOOL_CONTEXT: ContextVar[tuple[str, str]] = ContextVar(
    "hairball_plan_tool_context",
    default=("build", ""),
)


class PlanContractError(RuntimeError):
    """Stable machine-readable Plan contract error."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details)

    def to_dict(self) -> dict[str, Any]:
        return {"error": str(self), "code": self.code, **self.details}


def _now(value: float | None) -> float:
    return float(time.time() if value is None else value)


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _slug(value: Any, *, fallback: str = "session") -> str:
    compact = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "")).strip("-").lower()
    return compact[:64] or fallback


def normalize_plan_title(title: Any) -> dict[str, str]:
    """Normalize a model-supplied review title without treating it as a path."""

    raw = str(title or "").strip()
    if not raw:
        raise PlanContractError("missing_plan_title", "plan title is required")
    if "/" in raw or "\\" in raw or ".." in raw:
        raise PlanContractError(
            "invalid_plan_title",
            "plan title must not contain path separators or '..'",
        )
    stem = re.sub(r"\.md$", "", raw, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", "-", stem)
    normalized = re.sub(r"[^A-Za-z0-9_-]", "", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    if not normalized:
        raise PlanContractError(
            "invalid_plan_title",
            "plan title must contain a letter, number, underscore, or hyphen",
        )
    return {"title": normalized, "file_name": f"{normalized}.md"}


def resolve_plan_title(
    content: str,
    plan_file: str | Path,
    supplied_title: Any = None,
) -> str:
    """Resolve a usable title from explicit input, heading, filename, then default."""

    candidates: list[Any] = [supplied_title]
    heading = re.search(r"^[ \t]*#[ \t]+(.+?)[ \t]*$", str(content or ""), re.MULTILINE)
    if heading:
        candidates.append(heading.group(1))
    candidates.extend((Path(str(plan_file or "plan.md")).stem, "plan"))
    for candidate in candidates:
        try:
            return normalize_plan_title(candidate)["title"]
        except PlanContractError:
            continue
    return "plan"


def _digest(content: str, plan_file: str) -> str:
    payload = json.dumps(
        {"content": str(content), "plan_file": str(plan_file)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_plan_file(
    session_id: str,
    *,
    db: Any = None,
    cwd: str | Path | None = None,
) -> Path:
    """Return the stable plan file for one session.

    Plans are session artifacts, not project source files.  Keep them below the
    active Hairball profile so entering Plan mode never dirties the workspace.
    The path remains stable across restarts, cwd changes, and mutable titles.
    """

    sid = str(session_id or "").strip()
    if not sid:
        raise PlanContractError("missing_session_id", "session_id is required")
    row: Mapping[str, Any] = {}
    getter = getattr(db, "get_session", None)
    if callable(getter):
        try:
            candidate = getter(sid)
            if isinstance(candidate, Mapping):
                row = candidate
        except Exception:
            logger.debug("plan session metadata lookup failed", exc_info=True)

    # ``cwd`` remains in the signature for transport compatibility.  Artifact
    # identity deliberately does not depend on it: the same session may move
    # between CLI, desktop, gateway, and a renamed or relocated checkout.
    del cwd
    from hairball_constants import get_hairball_home

    base = get_hairball_home() / "artifacts" / "plans" / _slug(sid)

    created = row.get("started_at")
    try:
        created_ms = int(float(created) * 1000)
    except (TypeError, ValueError):
        # Direct-agent and early transport calls may not have a session row
        # yet.  Keep the path deterministic instead of producing a different
        # filename on every call; the session slug still isolates files.
        created_ms = 0
    # Session titles are mutable presentation.  Artifact identity is not: a
    # rename must never make a pending/reviewed plan disappear.
    return base / f"{created_ms}-plan.md"


def normalize_agent_mode(raw: Any, *, default: str = "build") -> str:
    fallback = _MODE_ALIASES.get(str(default or "build").strip().lower(), str(default))
    value = str(raw or "").strip().lower()
    value = _MODE_ALIASES.get(value, value or fallback)
    if value not in _AGENT_MODES:
        raise PlanContractError("invalid_agent_mode", f"unsupported agent mode: {value}")
    return value


def transition_mode(
    state: Mapping[str, Any],
    command: Any,
    *,
    actor: str,
    epoch: int,
    now: float | None = None,
) -> dict[str, Any]:
    """Apply one explicit Plan/Build switch without changing conversation history."""

    result = _copy(dict(state or {}))
    current = normalize_agent_mode(result.get("mode"))
    current_epoch = int(result.get("mode_epoch") or 0)
    next_epoch = int(epoch)
    if next_epoch <= current_epoch:
        raise PlanContractError(
            "stale_mode_epoch",
            "mode transition epoch is stale",
            expected_after=current_epoch,
            received=next_epoch,
        )
    target = normalize_agent_mode(command)
    timestamp = _now(now)
    result.update(
        {
            "mode": target,
            "mode_epoch": next_epoch,
            "mode_transition": {
                "from": current,
                "to": target,
                "actor": str(actor or ""),
                "at": timestamp,
            },
            "updated_at": timestamp,
        }
    )
    logger.info(
        "plan_mode_transition from=%s to=%s epoch=%s actor=%s",
        current,
        target,
        next_epoch,
        str(actor or ""),
    )
    return result


def _normalize_steps(steps: Sequence[Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    if steps is None:
        return rows
    if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
        raise PlanContractError("invalid_plan_steps", "plan steps must be a sequence")
    for index, raw in enumerate(steps, start=1):
        source = dict(raw) if isinstance(raw, Mapping) else {}
        text = str(
            source.get("text")
            or source.get("content")
            or source.get("title")
            or raw
            or ""
        ).strip()
        if not text:
            continue
        base = _slug(source.get("id") or text, fallback=f"step-{index}")
        step_id = base
        suffix = 2
        while step_id in used:
            step_id = f"{base}-{suffix}"
            suffix += 1
        used.add(step_id)
        rows.append({"id": step_id, "text": text, "status": "pending"})
    return rows


def _normalize_step_status(value: Any, *, default: str = "pending") -> str:
    raw = str(value or default).strip().lower()
    aliases = {
        "active": "in_progress",
        "running": "in_progress",
        "done": "completed",
        "complete": "completed",
        "succeeded": "completed",
        "success": "completed",
        "blocked": "failed",
        "error": "failed",
        "canceled": "cancelled",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in PLAN_STEP_STATUSES else default


def _synchronize_plan_steps(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy Plan rows and refresh the read-only Todo projection.

    ``steps`` is the reducer input.  ``todos`` remains on disk for compatibility
    with already-released UI/session readers, but no lifecycle function mutates
    it independently anymore.
    """

    result = _copy(dict(plan or {}))
    raw_steps = result.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raw_steps = result.get("todos")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raw_steps = []
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for index, raw in enumerate(raw_steps, start=1):
        source = dict(raw) if isinstance(raw, Mapping) else {"text": str(raw or "")}
        text = str(
            source.get("text")
            or source.get("content")
            or source.get("title")
            or ""
        ).strip()
        if not text:
            continue
        base = str(source.get("id") or "").strip() or _slug(
            text, fallback=f"step-{index}"
        )
        step_id = base
        suffix = 2
        while step_id in used:
            step_id = f"{base}-{suffix}"
            suffix += 1
        used.add(step_id)
        source.update(
            {
                "id": step_id,
                "text": text,
                "status": _normalize_step_status(source.get("status")),
            }
        )
        rows.append(source)
    result["steps"] = rows
    result["todos"] = _copy(rows)
    return result


def _plan_has_unfinished_steps(plan: Mapping[str, Any]) -> bool:
    return any(
        _normalize_step_status(step.get("status")) not in PLAN_SUCCESS_STEP_STATUSES
        for step in plan.get("steps") or []
        if isinstance(step, Mapping)
    )


def build_plan(
    content: str,
    steps: Sequence[Any] | None,
    *,
    previous: Mapping[str, Any] | None = None,
    plan_id: str | None = None,
    plan_file: str | Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Capture the current plan file as a reviewable transport projection."""

    body = str(content or "").strip()
    path = str(plan_file or (previous or {}).get("plan_file") or "").strip()
    if not body:
        raise PlanContractError("empty_plan_content", "plan file is empty")
    if not path:
        raise PlanContractError("missing_plan_file", "plan file path is required")
    prior = _synchronize_plan_steps(previous or {})
    digest = _digest(body, path)
    if prior and str(prior.get("digest") or "") == digest:
        return _copy(prior)
    timestamp = _now(now)
    revision = int(prior.get("revision") or 0) + 1
    normalized_steps = _normalize_steps(steps)
    stable_id = str(
        plan_id
        or prior.get("plan_id")
        or f"plan-{hashlib.sha256(path.encode('utf-8')).hexdigest()[:20]}"
    )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": stable_id,
        "revision": revision,
        "digest": digest,
        "plan_file": path,
        "title": resolve_plan_title(body, path),
        "content": body,
        "steps": normalized_steps,
        "todos": _copy(normalized_steps),
        "status": "draft",
        "mode": "plan",
        "can_approve": False,
        "can_execute": False,
        "approval": None,
        "build": None,
        "created_at": float(prior.get("created_at") or timestamp),
        "updated_at": timestamp,
    }


def revise_plan(
    plan: Mapping[str, Any],
    *,
    content: str,
    steps: Sequence[Any] | None,
    now: float | None = None,
) -> dict[str, Any]:
    return build_plan(
        content,
        steps,
        previous=plan,
        plan_file=str(plan.get("plan_file") or ""),
        now=now,
    )


def request_plan_approval(
    plan: Mapping[str, Any],
    *,
    session_id: str,
    surface: str,
    timeout: float | None,
    request_id: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Project ``plan_exit`` into Hairball's async approval surfaces."""

    if not isinstance(plan, Mapping) or not plan:
        raise PlanContractError("no_plan", "no plan file is available for approval")
    canonical = _synchronize_plan_steps(plan)
    if not canonical.get("steps"):
        raise PlanContractError(
            "plan_steps_required",
            "the plan must contain at least one executable step before approval",
        )
    ttl = None if timeout is None else float(timeout)
    if ttl is not None and ttl <= 0:
        raise PlanContractError("invalid_approval_timeout", "approval timeout must be positive")
    existing = canonical.get("approval")
    stable_id = str(request_id or f"plan-approval-{uuid.uuid4().hex[:16]}")
    if isinstance(existing, Mapping) and str(existing.get("status") or "") == "pending":
        if str(existing.get("request_id") or "") == stable_id:
            return _copy(plan)
        raise PlanContractError(
            "approval_already_pending",
            "a plan approval request is already pending",
            request_id=str(existing.get("request_id") or ""),
        )
    timestamp = _now(now)
    result = _copy(canonical)
    result["approval"] = {
        "request_id": stable_id,
        "session_id": str(session_id or ""),
        "surface": str(surface or ""),
        "status": "pending",
        "revision": int(result.get("revision") or 0),
        "digest": str(result.get("digest") or ""),
        "requested_at": timestamp,
        "expires_at": None if ttl is None else timestamp + ttl,
    }
    result.update(
        {
            "status": "awaiting_approval",
            "mode": "plan",
            "can_approve": True,
            "can_execute": False,
            "updated_at": timestamp,
        }
    )
    logger.info(
        "plan_exit_requested session=%s surface=%s plan=%s revision=%s request=%s",
        str(session_id or ""),
        str(surface or ""),
        str(result.get("plan_file") or ""),
        int(result.get("revision") or 0),
        stable_id,
    )
    return result


def require_matching_plan(
    plan: Mapping[str, Any],
    *,
    expected_revision: int | None,
    expected_digest: str | None,
) -> None:
    if expected_revision is None or not str(expected_digest or "").strip():
        raise PlanContractError(
            "approval_precondition_required",
            "plan approval requires the displayed revision and digest",
        )
    if int(expected_revision) != int(plan.get("revision") or 0):
        raise PlanContractError("plan_revision_mismatch", "plan revision does not match")
    if str(expected_digest) != str(plan.get("digest") or ""):
        raise PlanContractError("plan_digest_mismatch", "plan digest does not match")


def resolve_plan_approval(
    plan: Mapping[str, Any],
    request_id: str,
    decision: str,
    *,
    actor: str,
    expected_revision: int | None,
    expected_digest: str | None,
    resolution_id: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Resolve the Plan-to-Build Yes/No approval question."""

    if not isinstance(plan, Mapping) or not plan:
        raise PlanContractError("no_plan", "no plan is available for approval")
    canonical = _synchronize_plan_steps(plan)
    approval = canonical.get("approval")
    if not isinstance(approval, Mapping):
        raise PlanContractError("no_pending_approval", "no plan approval is pending")
    if str(approval.get("request_id") or "") != str(request_id or ""):
        raise PlanContractError("approval_request_mismatch", "approval request id does not match")
    if str(approval.get("status") or "") != "pending":
        raise PlanContractError(
            "approval_already_resolved",
            "the plan approval request was already resolved",
            status=str(approval.get("status") or ""),
        )
    require_matching_plan(
        canonical,
        expected_revision=expected_revision,
        expected_digest=expected_digest,
    )
    timestamp = _now(now)
    normalized = str(decision or "").strip().lower()
    approved = normalized in {"approve", "allow", "yes"}
    rejected = normalized in {"reject", "deny", "no", "revise"}
    if not approved and not rejected:
        raise PlanContractError("invalid_approval_decision", "approval decision must be Yes or No")
    expires_at = approval.get("expires_at")
    if expires_at is not None and timestamp > float(expires_at):
        approved = False
        normalized = "timeout"
    result = _copy(canonical)
    result["approval"].update(
        {
            "status": "approved" if approved else ("timed_out" if normalized == "timeout" else "rejected"),
            "decision": "approve" if approved else normalized,
            "resolution_id": str(resolution_id or f"resolution-{uuid.uuid4().hex[:16]}"),
            "resolved_by": str(actor or ""),
            "resolved_at": timestamp,
        }
    )
    result.update(
        {
            "status": "approved" if approved else ("approval_timed_out" if normalized == "timeout" else "approval_rejected"),
            "mode": "build" if approved else "plan",
            "can_approve": not approved,
            "can_execute": approved,
            "updated_at": timestamp,
        }
    )
    logger.info(
        "plan_approval_resolved session=%s request=%s decision=%s actor=%s revision=%s",
        str(approval.get("session_id") or ""),
        str(request_id or ""),
        "approve" if approved else normalized,
        str(actor or ""),
        int(result.get("revision") or 0),
    )
    return result


def validate_plan_execution(
    plan: Mapping[str, Any],
    *,
    plan_id: str,
    expected_revision: int,
    expected_digest: str,
    approval_request_id: str,
) -> dict[str, Any]:
    """Authorize Build only for the exact plan the user reviewed."""

    canonical = _synchronize_plan_steps(plan)
    if not all(
        (
            str(plan_id or "").strip(),
            str(expected_digest or "").strip(),
            str(approval_request_id or "").strip(),
        )
    ):
        raise PlanContractError(
            "execution_precondition_required",
            "plan id, revision, digest, and approval request id are required",
        )
    if str(plan_id) != str(canonical.get("plan_id") or ""):
        raise PlanContractError("plan_id_mismatch", "plan id does not match")
    require_matching_plan(
        canonical,
        expected_revision=int(expected_revision),
        expected_digest=str(expected_digest),
    )
    approval = canonical.get("approval")
    if not isinstance(approval, Mapping) or str(approval.get("status") or "") != "approved":
        raise PlanContractError("plan_not_approved", "the current plan has not been approved")
    if str(approval_request_id) != str(approval.get("request_id") or ""):
        raise PlanContractError(
            "approval_request_mismatch",
            "approval request id does not match the approved plan",
        )
    build = canonical.get("build")
    if isinstance(build, Mapping) and str(build.get("status") or "") == "running":
        raise PlanContractError(
            "plan_execution_active",
            "an execution of this plan is already active",
            execution_id=str(build.get("execution_id") or ""),
        )
    if canonical.get("can_execute") is not True:
        raise PlanContractError(
            "plan_not_executable",
            "the current plan is not available for execution",
            status=str(canonical.get("status") or ""),
        )
    return _copy(canonical)


def _required_execution_id(execution_id: str) -> str:
    value = str(execution_id or "").strip()
    if not value:
        raise PlanContractError(
            "missing_execution_id",
            "plan execution id is required",
        )
    return value


def _execution_id_mismatch(*, expected: str, received: str) -> PlanContractError:
    return PlanContractError(
        "execution_id_mismatch",
        "plan execution id does not match",
        expected_execution_id=str(expected or ""),
        received_execution_id=str(received or ""),
    )


def begin_build(
    plan: Mapping[str, Any],
    *,
    execution_id: str,
    selected_step_ids: Sequence[str] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Claim the exact approved steps for one exclusive Build execution."""

    if not isinstance(plan, Mapping) or not plan:
        raise PlanContractError("no_plan", "no plan is available for execution")
    canonical = _synchronize_plan_steps(plan)
    requested_id = _required_execution_id(execution_id)
    build = canonical.get("build")
    if isinstance(build, Mapping):
        current_id = str(build.get("execution_id") or "")
        current_status = str(build.get("status") or "")
        if current_status == "running":
            requested_selection = tuple(
                dict.fromkeys(str(value or "").strip() for value in (selected_step_ids or ()))
            )
            current_selection = tuple(build.get("selected_step_ids") or ())
            if current_id == requested_id and (
                not requested_selection or requested_selection == current_selection
            ):
                return _copy(canonical)
            raise PlanContractError(
                "plan_execution_active",
                "an execution of this plan is already active",
                execution_id=current_id,
            )
        if current_status in {"finished", "failed", "cancelled", "deferred"} and current_id == requested_id:
            raise PlanContractError(
                "plan_execution_finished",
                "this execution id already has a terminal result",
                execution_id=current_id,
            )
    approval = canonical.get("approval")
    if not isinstance(approval, Mapping) or str(approval.get("status") or "") != "approved":
        raise PlanContractError("plan_not_approved", "the current plan has not been approved")
    if canonical.get("can_execute") is not True:
        raise PlanContractError(
            "plan_not_executable",
            "the current plan is not available for execution",
            status=str(canonical.get("status") or ""),
        )

    rows = [dict(step) for step in canonical.get("steps") or [] if isinstance(step, Mapping)]
    known = {str(step.get("id") or ""): step for step in rows}
    executable = {
        step_id
        for step_id, step in known.items()
        if _normalize_step_status(step.get("status")) in PLAN_EXECUTABLE_STEP_STATUSES
    }
    if selected_step_ids is None:
        selected = [str(step.get("id") or "") for step in rows if str(step.get("id") or "") in executable]
    else:
        if isinstance(selected_step_ids, (str, bytes)):
            raise PlanContractError("invalid_plan_step_selection", "selected step ids must be a sequence")
        selected = list(
            dict.fromkeys(str(value or "").strip() for value in selected_step_ids if str(value or "").strip())
        )
        unknown = [step_id for step_id in selected if step_id not in known]
        terminal = [step_id for step_id in selected if step_id in known and step_id not in executable]
        if unknown:
            raise PlanContractError(
                "unknown_plan_step",
                "selected plan step does not exist",
                unknown_step_ids=unknown,
            )
        if terminal:
            raise PlanContractError(
                "plan_step_not_executable",
                "selected plan step is already complete or skipped",
                step_ids=terminal,
            )
    if not selected:
        raise PlanContractError(
            "no_executable_plan_steps",
            "the approved plan has no unfinished steps to execute",
        )

    timestamp = _now(now)
    result = _copy(canonical)
    history = list(result.get("build_history") or [])
    if isinstance(build, Mapping) and str(build.get("status") or "") in {
        "finished",
        "failed",
        "cancelled",
        "deferred",
    }:
        history.append(_copy(dict(build)))
    if history:
        result["build_history"] = history[-50:]
    prior_statuses: dict[str, str] = {}
    selected_set = set(selected)
    for step in result.get("steps") or []:
        step_id = str(step.get("id") or "")
        if step_id in selected_set:
            prior_statuses[step_id] = _normalize_step_status(step.get("status"))
            step["status"] = "in_progress"
            step.pop("error", None)
    result["build"] = {
        "execution_id": requested_id,
        "status": "running",
        "selected_step_ids": selected,
        "prior_step_statuses": prior_statuses,
        "started_at": timestamp,
    }
    result.update(
        {
            "status": "building",
            "mode": "build",
            "can_approve": False,
            "can_execute": False,
            "updated_at": timestamp,
        }
    )
    return _synchronize_plan_steps(result)


def release_build_claim(
    plan: Mapping[str, Any],
    *,
    execution_id: str,
    reason: str,
    now: float | None = None,
) -> dict[str, Any]:
    """Return an approved-but-undispatched execution to executable state.

    This is reserved for the approval handoff boundary (for example, the user
    cancels context compaction before the synthetic execution turn is queued).
    It is not a general rollback for work that has already reached the model.
    """

    if not isinstance(plan, Mapping) or not plan:
        raise PlanContractError("no_plan", "no plan is available for execution")
    canonical = _synchronize_plan_steps(plan)
    requested_id = _required_execution_id(execution_id)
    build = canonical.get("build")
    build_state = dict(build) if isinstance(build, Mapping) else {}
    current_id = str(build_state.get("execution_id") or "")
    current_status = str(build_state.get("status") or "")
    if current_status == "deferred" and current_id == requested_id:
        return _copy(canonical)
    if current_status != "running":
        raise PlanContractError(
            "plan_execution_not_active",
            "plan execution was not claimed before handoff release",
            execution_id=requested_id,
        )
    if current_id != requested_id:
        raise _execution_id_mismatch(expected=current_id, received=requested_id)
    approval = canonical.get("approval")
    if not isinstance(approval, Mapping) or str(approval.get("status") or "") != "approved":
        raise PlanContractError("plan_not_approved", "the current plan has not been approved")

    timestamp = _now(now)
    result = _copy(canonical)
    prior_statuses = build_state.get("prior_step_statuses")
    if not isinstance(prior_statuses, Mapping):
        prior_statuses = {}
    for step in result.get("steps") or []:
        step_id = str(step.get("id") or "")
        if step_id in prior_statuses:
            step["status"] = _normalize_step_status(prior_statuses[step_id])
    result["build"] = {
        **build_state,
        "status": "deferred",
        "reason": str(reason or "execution_not_dispatched"),
        "finished_at": timestamp,
    }
    result.update(
        {
            "status": "approved",
            "mode": "build",
            "can_approve": False,
            "can_execute": True,
            "updated_at": timestamp,
        }
    )
    return _synchronize_plan_steps(result)


def record_build_outcome(
    plan: Mapping[str, Any],
    *,
    execution_id: str,
    success: bool,
    step_outcomes: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    output: Any = None,
    error: str = "",
    now: float | None = None,
) -> dict[str, Any]:
    """Settle one Build claim from exact Todo outcomes, then expose resume.

    The Agent loop decides whether to continue.  This reducer only consumes the
    protected Todo facts returned by that loop; it never infers per-step success
    from final-response prose.
    """

    if not isinstance(plan, Mapping) or not plan:
        raise PlanContractError("no_plan", "no plan is available for execution")
    canonical = _synchronize_plan_steps(plan)
    requested_id = _required_execution_id(execution_id)
    plan_status = str(canonical.get("status") or "")
    build = canonical.get("build")
    build_state = dict(build) if isinstance(build, Mapping) else {}
    current_id = str(build_state.get("execution_id") or "")
    current_status = str(build_state.get("status") or "")
    if plan_status == "cancelled":
        cancelled = canonical.get("cancelled")
        cancelled_id = (
            str(cancelled.get("execution_id") or "")
            if isinstance(cancelled, Mapping)
            else current_id
        )
        if cancelled_id and cancelled_id != requested_id:
            raise _execution_id_mismatch(expected=cancelled_id, received=requested_id)
        return _copy(canonical)
    if current_status in {"finished", "failed", "cancelled"}:
        if current_id and current_id != requested_id:
            raise _execution_id_mismatch(expected=current_id, received=requested_id)
        return _copy(canonical)
    if current_status != "running":
        raise PlanContractError(
            "plan_execution_not_active",
            "plan execution was not claimed before completion",
            execution_id=requested_id,
        )
    if current_id != requested_id:
        raise _execution_id_mismatch(expected=current_id, received=requested_id)

    result = _copy(canonical)
    timestamp = _now(now)

    selected = [str(value or "") for value in build_state.get("selected_step_ids") or []]
    if not selected:
        selected = [
            str(step.get("id") or "")
            for step in result.get("steps") or []
            if _normalize_step_status(step.get("status")) == "in_progress"
        ]
    selected_set = set(selected)
    normalized_outcomes: dict[str, str] = {}
    if isinstance(step_outcomes, Mapping):
        for raw_id, raw in step_outcomes.items():
            step_id = str(raw_id or "").strip()
            if not step_id:
                raise PlanContractError(
                    "invalid_plan_step_outcome",
                    "plan step outcome requires a non-empty id",
                )
            value = raw.get("status") if isinstance(raw, Mapping) else raw
            normalized_outcomes[step_id] = _normalize_step_status(value)
    elif isinstance(step_outcomes, Sequence) and not isinstance(step_outcomes, (str, bytes)):
        for raw in step_outcomes:
            if not isinstance(raw, Mapping):
                raise PlanContractError(
                    "invalid_plan_step_outcome",
                    "plan step outcome must be an object",
                )
            step_id = str(raw.get("id") or "").strip()
            if not step_id:
                raise PlanContractError(
                    "invalid_plan_step_outcome",
                    "plan step outcome requires a non-empty id",
                )
            if step_id in normalized_outcomes:
                raise PlanContractError(
                    "duplicate_plan_step_outcome",
                    "plan step outcome id is duplicated",
                    step_id=step_id,
                )
            normalized_outcomes[step_id] = _normalize_step_status(raw.get("status"))

    unexpected = sorted(set(normalized_outcomes) - selected_set)
    if unexpected:
        raise PlanContractError(
            "unexpected_plan_step_outcome",
            "plan step outcome was not part of this execution claim",
            step_ids=unexpected,
        )

    settled: dict[str, str] = {}
    prior_statuses = build_state.get("prior_step_statuses")
    if not isinstance(prior_statuses, Mapping):
        prior_statuses = {}
    for step in result.get("steps") or []:
        step_id = str(step.get("id") or "")
        if step_id not in selected_set:
            continue
        if step_id in normalized_outcomes:
            status = normalized_outcomes[step_id]
        else:
            # Missing evidence is not completion evidence. Restore the exact
            # pre-claim status so this step remains resumable instead of
            # fabricating success from the assistant's prose-level outcome.
            status = _normalize_step_status(prior_statuses.get(step_id))
        step["status"] = status
        if status == "failed":
            step["error"] = str(error or "Build execution failed")
        elif status == "cancelled":
            step["error"] = str(error or "Build execution was cancelled")
        else:
            step.pop("error", None)
        settled[step_id] = status

    selected_complete = bool(selected) and all(
        settled.get(step_id) in PLAN_SUCCESS_STEP_STATUSES for step_id in selected
    )
    effective_success = bool(success) and selected_complete
    effective_error = str(error or "")
    if success and not selected_complete:
        missing = [step_id for step_id in selected if step_id not in normalized_outcomes]
        if missing:
            effective_error = effective_error or (
                "selected plan step outcomes are missing: " + ", ".join(missing)
            )
        else:
            effective_error = effective_error or "selected plan steps remain unfinished"
    all_complete = bool(result.get("steps")) and not _plan_has_unfinished_steps(result)
    result["build"] = {
        "execution_id": requested_id,
        "status": "finished" if effective_success else "failed",
        "selected_step_ids": selected,
        "step_outcomes": settled,
        "output": _copy(output),
        "error": effective_error,
        "started_at": build_state.get("started_at"),
        "finished_at": timestamp,
    }
    result.update(
        {
            "status": "built" if all_complete else ("approved" if effective_success else "build_failed"),
            "mode": "build",
            "can_approve": False,
            "can_execute": not all_complete,
            "updated_at": timestamp,
        }
    )
    logger.info(
        "plan_build_finished execution=%s success=%s plan=%s revision=%s",
        requested_id,
        bool(effective_success),
        str(result.get("plan_file") or ""),
        int(result.get("revision") or 0),
    )
    return _synchronize_plan_steps(result)


def cancel_execution(
    plan: Mapping[str, Any],
    *,
    execution_id: str = "",
    reason: str = "user_cancelled",
    now: float | None = None,
) -> dict[str, Any]:
    if not isinstance(plan, Mapping) or not plan:
        raise PlanContractError("no_plan", "no plan is available for cancellation")
    canonical = _synchronize_plan_steps(plan)
    requested_id = str(execution_id or "").strip()
    build = canonical.get("build")
    build_state = dict(build) if isinstance(build, Mapping) else {}
    current_id = str(build_state.get("execution_id") or "")
    current_status = str(build_state.get("status") or "")
    plan_status = str(canonical.get("status") or "")

    if plan_status == "cancelled":
        cancelled = canonical.get("cancelled")
        cancelled_id = (
            str(cancelled.get("execution_id") or "")
            if isinstance(cancelled, Mapping)
            else current_id
        )
        if requested_id and cancelled_id and requested_id != cancelled_id:
            raise _execution_id_mismatch(expected=cancelled_id, received=requested_id)
        return _copy(canonical)
    if current_status in {"finished", "cancelled"} or (
        current_status == "failed" and requested_id and requested_id == current_id
    ):
        if requested_id and current_id and requested_id != current_id:
            raise _execution_id_mismatch(expected=current_id, received=requested_id)
        return _copy(canonical)
    if requested_id and current_status != "running":
        if current_id and current_id != requested_id:
            raise _execution_id_mismatch(expected=current_id, received=requested_id)
        raise PlanContractError(
            "plan_execution_not_active",
            "plan execution was not claimed before cancellation",
            execution_id=requested_id,
        )
    if current_status == "running":
        if requested_id and current_id != requested_id:
            raise _execution_id_mismatch(expected=current_id, received=requested_id)
        requested_id = requested_id or current_id

    result = _copy(canonical)
    timestamp = _now(now)
    if current_status == "running":
        selected_ids = set(str(value or "") for value in build_state.get("selected_step_ids") or [])
        for step in result.get("steps") or []:
            if (
                str(step.get("id") or "") in selected_ids
                and _normalize_step_status(step.get("status")) == "in_progress"
            ):
                step["status"] = "cancelled"
                step["error"] = str(reason or "user_cancelled")
        result["build"] = {
            **build_state,
            "status": "cancelled",
            "error": str(reason or "user_cancelled"),
            "finished_at": timestamp,
        }
    result.update(
        {
            "status": "cancelled",
            "mode": "build",
            "can_approve": False,
            "can_execute": _plan_has_unfinished_steps(result),
            "cancelled": {
                "execution_id": requested_id,
                "reason": str(reason or "user_cancelled"),
                "at": timestamp,
            },
            "updated_at": timestamp,
        }
    )
    return _synchronize_plan_steps(result)


def project_plan_event(plan: Mapping[str, Any], *, surface: str) -> dict[str, Any]:
    """Project transport-neutral plan state without inventing execution facts."""

    result = _synchronize_plan_steps(plan or {})
    result["surface"] = str(surface or "")
    steps = _copy(result.get("steps") or [])
    completed = sum(
        1
        for step in steps
        if _normalize_step_status(step.get("status")) in PLAN_SUCCESS_STEP_STATUSES
    )
    failed = sum(
        1 for step in steps if _normalize_step_status(step.get("status")) == "failed"
    )
    active = sum(
        1 for step in steps if _normalize_step_status(step.get("status")) == "in_progress"
    )
    result["artifact"] = {
        "content": str(result.get("content") or ""),
        "title": str(result.get("title") or ""),
        "path": str(result.get("plan_file") or ""),
        "steps": steps,
        "todos": _copy(steps),
        "plan_id": str(result.get("plan_id") or ""),
        "revision": int(result.get("revision") or 0),
        "digest": str(result.get("digest") or ""),
        "plan_file": str(result.get("plan_file") or ""),
    }
    result["progress"] = {
        "total": len(steps),
        "completed": completed,
        "failed": failed,
        "active": active,
        "remaining": max(0, len(steps) - completed),
    }
    return result


def replay_plan_state(events: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Return the latest valid snapshot from the shared append-only ledger."""

    latest: dict[str, Any] | None = None
    for event in events or []:
        try:
            payload = json.loads(str(event.get("payload_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        raw = payload.get("plan") if isinstance(payload, Mapping) else None
        latest = _copy(raw) if isinstance(raw, Mapping) else None
    return _synchronize_plan_steps(latest) if latest is not None else None


def _resolved(path: Any) -> Path | None:
    text = str(path or "").strip()
    if not text:
        return None
    try:
        return Path(text).expanduser().resolve(strict=False)
    except OSError:
        return None


def bind_plan_tool_context(mode: Any, plan_file: str | Path | None) -> None:
    """Bind the current agent mode and exact Plan artifact for tool dispatch."""

    normalized = normalize_agent_mode(mode)
    expected = _resolved(plan_file) if normalized == "plan" else None
    if expected is not None:
        # The artifact transport is allowed to bypass the terminal sandbox only
        # below the active profile.  Resolve both sides so a symlinked plans
        # directory cannot redirect the single-file grant into a workspace.
        try:
            from hairball_constants import get_hairball_home

            expected.relative_to(get_hairball_home().resolve(strict=False))
        except (ImportError, OSError, ValueError):
            logger.warning("unsafe Plan artifact path rejected: %s", expected)
            expected = None
    _PLAN_TOOL_CONTEXT.set((normalized, str(expected or "")))


def plan_tool_context() -> tuple[str, Path | None]:
    """Return the execution context propagated to the current tool worker."""

    mode, raw_path = _PLAN_TOOL_CONTEXT.get()
    return mode, _resolved(raw_path)


def is_plan_artifact_target(path: str | Path | None) -> bool:
    """Whether *path* is the one session artifact granted to this Plan turn."""

    mode, expected = plan_tool_context()
    return mode == "plan" and expected is not None and _resolved(path) == expected


def plan_terminal_read_only() -> bool:
    """Return True when terminal execution must use the Plan read-only sandbox."""

    mode, _ = plan_tool_context()
    return mode == "plan"


def write_plan_artifact(path: str | Path, content: str) -> dict[str, Any]:
    """Atomically write the exact session Plan artifact outside shell tooling.

    This is the local-artifact seam used by mature Plan implementations: the
    working tree stays read-only while the canonical plan remains writable.
    """

    target = _resolved(path)
    if target is None or not is_plan_artifact_target(target):
        raise PlanContractError(
            "plan_artifact_denied",
            "Plan artifact write is not authorized for this path",
        )
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    existed = target.exists()
    old_mode: int | None = None
    if existed:
        try:
            old_mode = target.stat().st_mode & 0o777
        except OSError:
            old_mode = None
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=parent,
            prefix=".hairball-plan-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            handle.write(str(content))
            handle.flush()
            os.fsync(handle.fileno())
        if old_mode is not None:
            os.chmod(temp_path, old_mode)
        os.replace(temp_path, target)
        temp_path = None
        try:
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    encoded = str(content).encode("utf-8")
    return {
        "success": True,
        "path": str(target),
        "resolved_path": str(target),
        "files_modified": [str(target)],
        "bytes_written": len(encoded),
        "created": not existed,
        "artifact": "plan",
    }


def _is_plan_file_target(tool: str, args: Mapping[str, Any], plan_file: str | Path | None) -> bool:
    expected = _resolved(plan_file)
    if expected is None:
        return False
    name = str(tool or "")
    if name == "write_file":
        target = _resolved(args.get("path"))
        return target == expected
    if name == "patch":
        mode = str(args.get("mode") or "replace").strip().lower()
        if mode not in {"replace", "string", "str_replace"}:
            return False
        target = _resolved(args.get("path"))
        return target == expected
    return False


def _is_explore_delegation(args: Mapping[str, Any]) -> bool:
    tasks = args.get("tasks")
    if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes)):
        rows = [row for row in tasks if isinstance(row, Mapping)]
        return bool(rows) and all(str(row.get("profile") or "").strip() == "explore" for row in rows)
    return str(args.get("profile") or "").strip() == "explore"


def gate_side_effect(
    mode: str,
    tool: str,
    args: Mapping[str, Any] | None,
    plan_file: str | Path | None,
) -> dict[str, str]:
    """Evaluate the source-faithful Plan permission subset.

    Build retains the ordinary tool surface.  Plan blocks edit tools except
    the one plan file, blocks general/collaborative delegation, and permits the
    read-only explore profile.  Other tools stay available; their existing
    Hairball approval/sandbox policy remains authoritative.
    """

    normalized_mode = normalize_agent_mode(mode)
    name = str(tool or "").strip()
    payload = dict(args or {})
    if name == "plan_exit":
        if normalized_mode == "plan":
            return {"decision": "allow", "code": "allowed", "reason": "", "permission": "plan_exit"}
        return {
            "decision": "deny",
            "code": "plan_exit_outside_plan",
            "reason": "plan_exit is available only while Plan mode is active",
            "permission": "plan_exit",
        }
    if normalized_mode != "plan":
        return {"decision": "allow", "code": "allowed", "reason": "", "permission": "*"}
    if name == "terminal":
        if bool(payload.get("background")) or bool(payload.get("pty")):
            return {
                "decision": "deny",
                "code": "plan_persistent_process_denied",
                "reason": "Plan mode terminal commands must be foreground and non-interactive",
                "permission": "read_only_exec",
            }
        return {
            "decision": "allow",
            "code": "allowed",
            "reason": "",
            "permission": "read_only_exec",
        }
    if name == "execute_code":
        return {
            "decision": "deny",
            "code": "plan_code_execution_denied",
            "reason": "Plan mode uses the read-only terminal sandbox instead of execute_code",
            "permission": "exec",
        }
    if name == "process":
        action = str(payload.get("action") or "").strip().lower()
        if action in {"list", "poll", "log", "wait"}:
            return {"decision": "allow", "code": "allowed", "reason": "", "permission": "read"}
        return {
            "decision": "deny",
            "code": "plan_process_mutation_denied",
            "reason": "Plan mode may inspect background processes but cannot control them",
            "permission": "process_control",
        }
    if name in {"write_file", "patch"}:
        if _is_plan_file_target(name, payload, plan_file):
            return {"decision": "allow", "code": "allowed", "reason": "", "permission": "edit"}
        return {
            "decision": "deny",
            "code": "plan_edit_denied",
            "reason": "Plan mode may edit only its designated plan file",
            "permission": "edit",
        }
    if name in {"ast_edit"}:
        return {
            "decision": "deny",
            "code": "plan_edit_denied",
            "reason": "Plan mode may edit only its designated plan file",
            "permission": "edit",
        }
    if name == "lsp":
        action = str(payload.get("action") or "").strip().lower()
        if action in {"rename", "rename_file"} and payload.get("apply") is not False:
            return {
                "decision": "deny",
                "code": "plan_edit_denied",
                "reason": "Plan mode cannot apply LSP edits",
                "permission": "edit",
            }
        if action == "code_actions" and payload.get("apply") is True:
            return {
                "decision": "deny",
                "code": "plan_edit_denied",
                "reason": "Plan mode cannot apply LSP edits",
                "permission": "edit",
            }
    if name == "delegate_task":
        if _is_explore_delegation(payload):
            return {"decision": "allow", "code": "allowed", "reason": "", "permission": "task"}
        return {
            "decision": "deny",
            "code": "plan_general_agent_denied",
            "reason": "Plan mode delegation must use profile='explore'",
            "permission": "task",
        }
    if name.startswith("collab_"):
        return {
            "decision": "deny",
            "code": "plan_general_agent_denied",
            "reason": "Plan mode uses delegate_task with profile='explore' for research",
            "permission": "task",
        }
    return {"decision": "allow", "code": "allowed", "reason": "", "permission": "*"}


def plan_exit_succeeded(result: Mapping[str, Any] | None) -> bool:
    """Return True only when a paired ``plan_exit`` tool result succeeded."""

    outcome = dict(result or {})
    calls: dict[str, str] = {}
    for message in outcome.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, Mapping):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), Mapping) else {}
                calls[str(call.get("id") or "")] = str(fn.get("name") or "")
        if message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "")
        name = str(message.get("name") or calls.get(call_id) or "")
        if name != "plan_exit":
            continue
        content = message.get("content")
        try:
            decoded = json.loads(content) if isinstance(content, str) else content
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, Mapping) and decoded.get("success") is True:
            return True
    return False


def ask_structured_questions(
    questions: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
    surface: str,
    request_id: str | None = None,
    timeout: float = 300.0,
    now: float | None = None,
) -> dict[str, Any]:
    """Create the multi-question shape used by Hairball's ``clarify`` adapter."""

    if isinstance(questions, (str, bytes)) or not isinstance(questions, Sequence):
        raise PlanContractError("invalid_questions", "structured questions must be a sequence")
    normalized: list[dict[str, Any]] = []
    used: set[str] = set()
    for index, raw in enumerate(questions, start=1):
        if not isinstance(raw, Mapping):
            raise PlanContractError("invalid_question", f"question {index} must be an object")
        text = str(raw.get("question") or raw.get("text") or "").strip()
        if not text:
            raise PlanContractError("invalid_question", f"question {index} has no text")
        question_id = _slug(raw.get("id") or text, fallback=f"question-{index}")
        if question_id in used:
            raise PlanContractError("duplicate_question_id", f"duplicate question id: {question_id}")
        used.add(question_id)
        normalized.append(
            {
                "id": question_id,
                "question": text,
                "choices": [str(item) for item in raw.get("choices") or []],
                "required": bool(raw.get("required", True)),
                "allow_other": bool(raw.get("allow_other", True)),
            }
        )
    if not normalized:
        raise PlanContractError("empty_questions", "at least one structured question is required")
    ttl = float(timeout)
    if ttl <= 0:
        raise PlanContractError("invalid_question_timeout", "question timeout must be positive")
    timestamp = _now(now)
    return {
        "request_id": str(request_id or f"clarify-{uuid.uuid4().hex[:16]}"),
        "session_id": str(session_id or ""),
        "surface": str(surface or ""),
        "status": "pending",
        "questions": normalized,
        "answers": {},
        "unresolved_question_ids": [row["id"] for row in normalized if row["required"]],
        "requested_at": timestamp,
        "expires_at": timestamp + ttl,
    }


def resolve_structured_answers(
    request: Mapping[str, Any],
    answers: Mapping[str, Any],
    *,
    actor: str,
    now: float | None = None,
) -> dict[str, Any]:
    if not isinstance(request, Mapping) or not request:
        raise PlanContractError("no_clarify_request", "no structured question request exists")
    if not isinstance(answers, Mapping):
        raise PlanContractError("invalid_answers", "structured answers must be an object")
    result = _copy(dict(request))
    known = {
        str(row.get("id") or ""): row
        for row in result.get("questions") or []
        if isinstance(row, Mapping)
    }
    unknown = sorted(str(key) for key in answers if str(key) not in known)
    if unknown:
        raise PlanContractError(
            "unknown_question_id",
            "answer references an unknown question",
            unknown_question_ids=unknown,
        )
    merged = dict(result.get("answers") or {})
    merged.update({str(key): _copy(value) for key, value in answers.items()})
    unresolved = [
        qid
        for qid, question in known.items()
        if bool(question.get("required", True)) and not str(merged.get(qid) or "").strip()
    ]
    result.update(
        {
            "answers": merged,
            "unresolved_question_ids": unresolved,
            "status": "resolved" if not unresolved else "pending",
            "resolved_by": str(actor or "") if not unresolved else "",
            "resolved_at": _now(now) if not unresolved else None,
        }
    )
    return result


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "PlanContractError",
    "ask_structured_questions",
    "begin_build",
    "build_plan",
    "cancel_execution",
    "bind_plan_tool_context",
    "gate_side_effect",
    "is_plan_artifact_target",
    "normalize_agent_mode",
    "plan_exit_succeeded",
    "plan_terminal_read_only",
    "plan_tool_context",
    "project_plan_event",
    "record_build_outcome",
    "release_build_claim",
    "replay_plan_state",
    "require_matching_plan",
    "request_plan_approval",
    "resolve_plan_approval",
    "resolve_plan_file",
    "resolve_plan_title",
    "resolve_structured_answers",
    "revise_plan",
    "transition_mode",
    "validate_plan_execution",
    "write_plan_artifact",
    "normalize_plan_title",
]
