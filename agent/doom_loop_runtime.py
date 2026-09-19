"""Doom-loop soft guard for the live v1 tool executor (OC-2).

Kernel v2 already checks via ``KernelEngine``; this module attaches the same
``DoomLoopDetector`` to a long-lived ``AIAgent`` so CLI / UI / gateway
``conversation_loop`` paths get the soft strategy hint without requiring the
streaming engine.

**Scope (OpenCode ``processor.ts``):** window = trailing tool calls on the
**current assistant / API step** only (OpenCode: parts of
``ctx.assistantMessage``; Hairball: one ``api_call_count`` tool batch).
``begin_doom_loop_assistant_step`` resets when the API step changes so
1×/step for 3 steps does **not** trip; 3 identical tools in one assistant
batch does. User-turn reset is belt-and-suspenders only.

**Pre-dedup proposals:** Hairball collapses identical ``(name, arguments)``
before execute. Doom observes the **model-proposed** list (stashed before
dedup), matching OpenCode counting streamed tool parts — not only the
unique calls that actually run. Hint attaches to the executed call whose
signature matches the trip.

**Phase 1 substitute:** OpenCode uses ``permission.ask`` (can stop). We only
append a soft tool-result hint. Ask UI / hard-stop are deferred — do not
claim full OpenCode parity until those exist.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal, Optional, Sequence

logger = logging.getLogger(__name__)

DoomLoopMode = Literal["soft", "off"]

#: Soft nudge when the sliding window trips and no host approval callback is
#: wired. Brand-neutral; shared with KernelEngine so v1/v2 wording matches.
#:
#: Keep this module free of ``agent.kernel`` imports at load time — ``engine``
#: imports the hint constant, and ``agent.kernel`` package init imports engine.
DOOM_LOOP_SOFT_HINT = (
    "\n\n[Tool loop warning: doom_loop; identical tool call repeated. "
    "Do not switch to text-only replies. Keep using tools, but change strategy: "
    "broaden search (parent dirs, repo root, absolute paths), try a different "
    "tool, re-check assumptions from evidence rather than docs alone, or "
    "inspect the environment with a small diagnostic before retrying.]"
)

_DEFAULT_THRESHOLD = 3


def _doom_loop_detector_cls():
    from agent.kernel.turn_loop import DoomLoopDetector

    return DoomLoopDetector


def resolve_doom_loop_mode() -> DoomLoopMode:
    """Resolve ``agent.doom_loop_mode`` (default: soft)."""
    raw = "soft"
    try:
        from hairball_cli.config import load_config

        cfg = load_config() or {}
        agent = cfg.get("agent") if isinstance(cfg, dict) else None
        if isinstance(agent, dict) and agent.get("doom_loop_mode") is not None:
            raw = agent.get("doom_loop_mode")
    except Exception:
        pass
    env = os.environ.get("HAIRBALL_DOOM_LOOP_MODE")
    if env:
        raw = env
    mode = str(raw or "soft").strip().lower()
    if mode in ("off", "soft"):
        return mode  # type: ignore[return-value]
    return "soft"


def resolve_doom_loop_threshold() -> int:
    """Resolve ``agent.doom_loop_threshold`` (default: 3)."""
    raw: Any = _DEFAULT_THRESHOLD
    try:
        from hairball_cli.config import load_config

        cfg = load_config() or {}
        agent = cfg.get("agent") if isinstance(cfg, dict) else None
        if isinstance(agent, dict) and agent.get("doom_loop_threshold") is not None:
            raw = agent.get("doom_loop_threshold")
    except Exception:
        pass
    env = os.environ.get("HAIRBALL_DOOM_LOOP_THRESHOLD")
    if env:
        raw = env
    try:
        return max(2, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_THRESHOLD


def prior_tool_calls_from_messages(messages: Optional[Sequence[Any]]) -> list[tuple[str, Any]]:
    """Extract assistant-proposed tool calls in order (for resume seed).

    The trailing assistant message that still has open tool_calls is skipped:
    the live executor is about to observe those calls, so seeding them would
    double-count and trip the window one call early.
    """
    calls: list[tuple[str, Any]] = []
    if not messages:
        return calls
    skip_trailing = _trailing_assistant_has_tool_calls(messages)
    last_index = len(messages) - 1
    for index, msg in enumerate(messages):
        if skip_trailing and index == last_index:
            break
        if isinstance(msg, dict):
            role = msg.get("role")
            tool_calls = msg.get("tool_calls") or []
        else:
            role = getattr(msg, "role", None)
            tool_calls = getattr(msg, "tool_calls", None) or []
        if role != "assistant" or not tool_calls:
            continue
        for tc in tool_calls:
            name, args = _tool_call_name_args(tc)
            if name:
                calls.append((name, args))
    return calls


def _trailing_assistant_has_tool_calls(messages: Sequence[Any]) -> bool:
    if not messages:
        return False
    msg = messages[-1]
    if isinstance(msg, dict):
        return msg.get("role") == "assistant" and bool(msg.get("tool_calls"))
    return (
        getattr(msg, "role", None) == "assistant"
        and bool(getattr(msg, "tool_calls", None))
    )


def _tool_call_name_args(tc: Any) -> tuple[str, Any]:
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        if isinstance(fn, dict):
            name = str(fn.get("name") or "").strip()
            raw_args = fn.get("arguments", {})
        else:
            name = str(tc.get("name") or "").strip()
            raw_args = tc.get("arguments", {})
    else:
        fn = getattr(tc, "function", None)
        if fn is not None:
            name = str(getattr(fn, "name", "") or "").strip()
            raw_args = getattr(fn, "arguments", {})
        else:
            name = str(getattr(tc, "name", "") or "").strip()
            raw_args = getattr(tc, "arguments", {})
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            args = {"_raw": raw_args}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {}
    return name, args


def reset_doom_loop_window(agent: Any) -> None:
    """Clear the sliding window; keep ``allow_always`` tool names."""
    detector = getattr(agent, "_doom_loop_detector", None)
    if detector is None:
        return
    reset = getattr(detector, "reset", None)
    if callable(reset):
        reset()


def _clear_doom_loop_step_state(agent: Any) -> None:
    """Drop per-step proposal / trip bookkeeping (window cleared separately)."""
    for attr in (
        "_doom_loop_proposed_tool_calls",
        "_doom_loop_proposals_observed_step",
        "_doom_loop_trip_signature",
    ):
        try:
            setattr(agent, attr, None)
        except Exception:
            pass


def reset_doom_loop_for_turn(agent: Any) -> None:
    """User-turn prologue: clear window and forget the last API step id."""
    try:
        agent._doom_loop_api_step = None
    except Exception:
        pass
    _clear_doom_loop_step_state(agent)
    reset_doom_loop_window(agent)


def begin_doom_loop_assistant_step(agent: Any, api_call_count: int = 0) -> None:
    """Reset the window when a new assistant/API step starts (OpenCode scope).

    Segmented concurrent/sequential dispatch for the **same** assistant
    message shares ``api_call_count`` and must **not** reset between segments.
    """
    try:
        step = int(api_call_count)
    except (TypeError, ValueError):
        step = 0
    last = getattr(agent, "_doom_loop_api_step", None)
    if last == step:
        return
    try:
        agent._doom_loop_api_step = step
    except Exception:
        pass
    _clear_doom_loop_step_state(agent)
    reset_doom_loop_window(agent)


def stash_doom_loop_proposed_tool_calls(agent: Any, tool_calls: Any) -> None:
    """Remember model-proposed tool_calls before ``_deduplicate_tool_calls``."""
    try:
        agent._doom_loop_proposed_tool_calls = list(tool_calls or [])
    except Exception:
        pass


def begin_and_observe_doom_proposals(
    agent: Any,
    tool_calls: Any,
    api_call_count: int = 0,
    *,
    messages: Optional[Sequence[Any]] = None,
) -> None:
    """Begin the API-step window and observe proposals once for this step.

    Prefers the pre-dedup stash when present; otherwise observes *tool_calls*
    (executor live list). Segmented dispatch for the same ``api_call_count``
    must not re-observe.

    The stash is taken **before** ``begin_doom_loop_assistant_step`` because a
    new step clears per-step bookkeeping (including the stash).
    """
    proposed = getattr(agent, "_doom_loop_proposed_tool_calls", None)
    try:
        agent._doom_loop_proposed_tool_calls = None
    except Exception:
        pass

    begin_doom_loop_assistant_step(agent, api_call_count)
    try:
        step = int(api_call_count)
    except (TypeError, ValueError):
        step = 0
    if getattr(agent, "_doom_loop_proposals_observed_step", None) == step:
        return

    to_observe = proposed if proposed is not None else list(tool_calls or [])

    trip_signature: Optional[tuple[str, str]] = None
    for tc in to_observe:
        name, args = _tool_call_name_args(tc)
        if not name:
            continue
        if observe_doom_loop(agent, name, args, messages=messages):
            try:
                from agent.kernel.turn_loop import tool_call_signature

                trip_signature = (name, tool_call_signature(name, args))
            except Exception:
                trip_signature = (name, "")
    try:
        agent._doom_loop_proposals_observed_step = step
        agent._doom_loop_trip_signature = trip_signature
    except Exception:
        pass


def doom_loop_should_hint(
    agent: Any,
    tool_name: str,
    arguments: Any,
) -> bool:
    """True when this executed call matches the proposal-window trip signature."""
    trip = getattr(agent, "_doom_loop_trip_signature", None)
    if not trip or not tool_name:
        return False
    try:
        from agent.kernel.turn_loop import tool_call_signature

        return trip == (tool_name, tool_call_signature(tool_name, arguments))
    except Exception:
        return False


def get_agent_doom_detector(
    agent: Any,
    *,
    messages: Optional[Sequence[Any]] = None,
) -> Optional[Any]:
    """Return (and lazily create) the per-agent doom detector, or None if off.

    Live path does not auto-seed from conversation history (OpenCode derives
    the window from the current assistant message's parts only).
    """
    del messages  # reserved for optional mid-turn resume seed
    if resolve_doom_loop_mode() == "off":
        return None
    Detector = _doom_loop_detector_cls()
    detector = getattr(agent, "_doom_loop_detector", None)
    if isinstance(detector, Detector):
        return detector
    detector = Detector(resolve_doom_loop_threshold())
    try:
        agent._doom_loop_detector = detector
    except Exception:
        pass
    return detector


def observe_doom_loop(
    agent: Any,
    tool_name: str,
    arguments: Any,
    *,
    messages: Optional[Sequence[Any]] = None,
) -> bool:
    """Record a proposed call; True when the live soft guard should hint."""
    if resolve_doom_loop_mode() == "off":
        return False
    detector = get_agent_doom_detector(agent, messages=messages)
    if detector is None:
        return False
    try:
        tripped = bool(detector.observe(tool_name, arguments))
    except Exception:
        logger.debug("doom-loop observe failed", exc_info=True)
        return False
    if tripped:
        logger.info(
            "doom_loop soft trip tool=%s streak=%s threshold=%s",
            tool_name,
            detector.streak,
            detector.threshold,
        )
        try:
            hits = getattr(agent, "_doom_loop_hits", None)
            if not isinstance(hits, list):
                hits = []
                agent._doom_loop_hits = hits
            hits.append({"tool": tool_name, "arguments": arguments})
        except Exception:
            pass
    return tripped


def append_doom_loop_hint(result: Any, tripped: bool) -> Any:
    """Append the soft strategy hint when *tripped* (string or multimodal)."""
    if not tripped or resolve_doom_loop_mode() != "soft":
        return result
    hint = DOOM_LOOP_SOFT_HINT
    try:
        from agent.tool_dispatch_helpers import (
            _append_subdir_hint_to_multimodal,
            _is_multimodal_tool_result,
        )

        if _is_multimodal_tool_result(result):
            _append_subdir_hint_to_multimodal(result, hint)
            return result
    except Exception:
        pass
    if isinstance(result, str):
        return result + hint
    return result


def allow_doom_loop_tool(agent: Any, tool_name: str) -> None:
    """Session-scoped allowlist for a tool name (matches OpenCode always-allow)."""
    Detector = _doom_loop_detector_cls()
    detector = getattr(agent, "_doom_loop_detector", None)
    if isinstance(detector, Detector):
        detector.allow_always(tool_name)
