"""Approved-Plan context handoff shared by every Hairball surface.

The authoritative Plan file survives every handoff.  Product surfaces choose
whether the planning transcript is discarded, distilled, or preserved; they do
not implement their own versions of those semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agent.plan_execution import PlanContractError


PLAN_CONTEXT_MODES = frozenset({"fresh", "compact", "keep"})


@dataclass(frozen=True)
class PlanHandoffResult:
    """One decision-complete handoff result for an approved execution turn."""

    history: list[dict[str, Any]]
    context_mode: str
    compaction_outcome: str
    dispatch: bool
    warning: str = ""


def normalize_plan_context_mode(value: Any, *, default: str | None = None) -> str:
    raw = str(value or "").strip().lower()
    if not raw and default is not None:
        raw = str(default).strip().lower()
    if raw not in PLAN_CONTEXT_MODES:
        raise PlanContractError(
            "invalid_plan_context_mode",
            "plan execution context must be fresh, compact, or keep",
        )
    return raw


def _interrupted(agent: Any) -> bool:
    value = getattr(agent, "_interrupt_requested", False)
    is_set = getattr(value, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(value)


def _estimate_tokens(agent: Any, history: Sequence[Mapping[str, Any]]) -> int | None:
    try:
        from agent.model_metadata import estimate_request_tokens_rough

        return int(
            estimate_request_tokens_rough(
                list(history),
                system_prompt=str(getattr(agent, "_cached_system_prompt", "") or ""),
                tools=getattr(agent, "tools", None) or None,
            )
        )
    except Exception:
        return None


def prepare_plan_handoff(
    history: Sequence[Mapping[str, Any]],
    *,
    context_mode: Any,
    plan_file: str,
    task_id: str,
    agent: Any | None = None,
) -> PlanHandoffResult:
    """Prepare history for the synthetic approved-Plan execution turn.

    ``fresh`` drops only model replay context; the visible/durable transcript is
    retained by its product surface.  ``compact`` reuses Hairball's canonical
    compressor in its supported in-place mode so the Plan ledger and run owner
    remain bound to the same session.  A compaction failure is best-effort and
    preserves context; an explicit interrupt cancels dispatch, matching the
    approval workflow's operator-cancellation contract.
    """

    mode = normalize_plan_context_mode(context_mode)
    copied = [dict(item) for item in history]
    if mode == "fresh":
        return PlanHandoffResult([], mode, "not_requested", True)
    if mode == "keep":
        return PlanHandoffResult(copied, mode, "not_requested", True)

    if agent is None:
        raise PlanContractError(
            "plan_compactor_required",
            "compact plan handoff requires the active Hairball agent",
        )
    if _interrupted(agent):
        return PlanHandoffResult(copied, mode, "cancelled", False)
    if len(copied) < 4:
        return PlanHandoffResult(copied, mode, "skipped_short", True)

    approved_path = str(plan_file or "").strip()
    focus = (
        "Distill the planning discussion for approved execution. Preserve "
        "rationale, rejected alternatives, user preferences, discovered paths "
        "and symbols. Drop tool-call noise and superseded drafts. The "
        f"authoritative approved plan is {approved_path}; preserve this exact "
        "path and the requirement to read it before execution."
    )
    previous_in_place = bool(getattr(agent, "compression_in_place", False))
    try:
        # Hairball already owns a guarded, persisted in-place compression path.
        # Using it here prevents a session rotation from severing the approved
        # Plan's exact execution owner.
        agent.compression_in_place = True
        compressed, _ = agent._compress_context(
            copied,
            None,
            approx_tokens=_estimate_tokens(agent, copied),
            task_id=str(task_id or "default"),
            focus_topic=focus,
            force=True,
        )
    except Exception as exc:
        if _interrupted(agent):
            return PlanHandoffResult(copied, mode, "cancelled", False, str(exc))
        return PlanHandoffResult(copied, mode, "failed", True, str(exc))
    finally:
        agent.compression_in_place = previous_in_place

    compacted = [dict(item) for item in (compressed or [])]
    if _interrupted(agent):
        return PlanHandoffResult(compacted or copied, mode, "cancelled", False)
    if compacted == copied or not compacted:
        return PlanHandoffResult(
            copied,
            mode,
            "failed",
            True,
            "plan context compaction made no progress; preserving full context",
        )
    return PlanHandoffResult(compacted, mode, "completed", True)


__all__ = [
    "PLAN_CONTEXT_MODES",
    "PlanHandoffResult",
    "normalize_plan_context_mode",
    "prepare_plan_handoff",
]
