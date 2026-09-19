"""Autonomy-preserving tool selection for continuation retries.

Hairball owns the decision that a candidate response is too early to end a
turn, but it must not take the model's planning authority away in order to
continue.  This module therefore exposes only two strategies:

``EXPLICIT_AUTO``
    Ask the provider for its native automatic tool selection mode.

``IMPLICIT_AUTO``
    Omit the control field and use the provider's default automatic mode.

Ordinary completion retries deliberately have no named/required strategy: a
completion judge may veto a stop, but it may not choose the model's action.
The one exception is Plan-mode convergence. After a Plan turn ends without
its required decision tool, the lifecycle may require *some* tool call on the
next sample while leaving the model free to choose which tool. Providers that
reject the control field fall back to implicit automatic selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AutonomousToolChoiceStrategy(str, Enum):
    EXPLICIT_AUTO = "explicit_auto"
    IMPLICIT_AUTO = "implicit_auto"


@dataclass(frozen=True)
class ToolControlCapabilities:
    """Resolved capability of one provider/model/transport combination."""

    api_mode: str
    supports_explicit_auto_tool_choice: bool
    source: str = "transport_default"
    reason: str = ""


@dataclass(frozen=True)
class AutonomousToolChoicePlan:
    strategy: AutonomousToolChoiceStrategy
    source: str
    reason: str = ""


def transport_default_capabilities(api_mode: str | None) -> ToolControlCapabilities:
    """Return conservative protocol defaults.

    Bedrock's Converse API defaults to automatic tool selection when
    ``toolChoice`` is absent, and omitting the field avoids model-specific
    validation differences.  Other first-class Hairball transports accept an
    explicit native auto shape.  Unknown modes use implicit auto because an
    unknown control parameter is less compatible than no control parameter.
    """

    mode = str(api_mode or "chat_completions")
    explicit = mode in {
        "chat_completions",
        "codex_responses",
        "anthropic_messages",
    }
    return ToolControlCapabilities(
        api_mode=mode,
        supports_explicit_auto_tool_choice=explicit,
        source="transport_default",
        reason=(
            "provider supports a native automatic tool-choice value"
            if explicit
            else "provider default is used to avoid an unsupported control field"
        ),
    )


def resolve_autonomous_tool_choice_plan(
    capabilities: ToolControlCapabilities,
    *,
    explicit_auto_disabled_for_session: bool = False,
) -> AutonomousToolChoicePlan:
    """Resolve a continuation plan that never reduces model autonomy."""

    explicit = (
        capabilities.supports_explicit_auto_tool_choice
        and not explicit_auto_disabled_for_session
    )
    return AutonomousToolChoicePlan(
        strategy=(
            AutonomousToolChoiceStrategy.EXPLICIT_AUTO
            if explicit
            else AutonomousToolChoiceStrategy.IMPLICIT_AUTO
        ),
        source=capabilities.source,
        reason=capabilities.reason,
    )


def apply_autonomous_tool_choice(
    api_kwargs: dict[str, Any],
    *,
    api_mode: str,
    plan: AutonomousToolChoicePlan,
) -> None:
    """Apply only auto/implicit-auto, preserving tools and reasoning byte-for-byte."""

    mode = str(api_mode or "chat_completions")
    if mode == "bedrock_converse":
        tool_config = api_kwargs.get("toolConfig")
        if isinstance(tool_config, dict):
            if plan.strategy is AutonomousToolChoiceStrategy.EXPLICIT_AUTO:
                tool_config["toolChoice"] = {"auto": {}}
            else:
                tool_config.pop("toolChoice", None)
        return

    if plan.strategy is AutonomousToolChoiceStrategy.IMPLICIT_AUTO:
        api_kwargs.pop("tool_choice", None)
        return

    if mode == "anthropic_messages":
        api_kwargs["tool_choice"] = {"type": "auto"}
    else:
        # OpenAI Chat, Responses, and Gemini's OpenAI-shaped client all map
        # this to their native automatic tool-selection mode.
        api_kwargs["tool_choice"] = "auto"


def apply_required_tool_choice(api_kwargs: dict[str, Any], *, api_mode: str) -> bool:
    """Require an unnamed tool call for a Plan decision continuation.

    Return ``False`` for transports without a known native representation so
    callers keep implicit automatic selection instead of inventing a wire
    shape.
    """

    mode = str(api_mode or "chat_completions")
    if mode == "bedrock_converse":
        tool_config = api_kwargs.get("toolConfig")
        if not isinstance(tool_config, dict):
            return False
        tool_config["toolChoice"] = {"any": {}}
        return True
    if mode == "anthropic_messages":
        api_kwargs["tool_choice"] = {"type": "any"}
        return True
    if mode in {"chat_completions", "codex_responses"}:
        api_kwargs["tool_choice"] = "required"
        return True
    return False


def apply_named_tool_choice(
    api_kwargs: dict[str, Any], *, api_mode: str, tool_name: str
) -> bool:
    """Force one named tool for an explicitly configured lifecycle prelude.

    This is intentionally separate from ordinary continuation policy.  It is
    used only when the user selected an ``always`` mode (for example
    ``agent.todo.eager: always``); normal autonomous turns and soft reminders
    never select a tool for the model.
    """

    name = str(tool_name or "").strip()
    if not name:
        return False
    mode = str(api_mode or "chat_completions")
    if mode == "bedrock_converse":
        tool_config = api_kwargs.get("toolConfig")
        if not isinstance(tool_config, dict):
            return False
        tool_config["toolChoice"] = {"tool": {"name": name}}
        return True
    if mode == "anthropic_messages":
        api_kwargs["tool_choice"] = {"type": "tool", "name": name}
        return True
    if mode == "chat_completions":
        api_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": name},
        }
        return True
    if mode == "codex_responses":
        api_kwargs["tool_choice"] = {"type": "function", "name": name}
        return True
    return False


def looks_like_tool_choice_capability_rejection(error: Any) -> bool:
    """Return True only for a definite 4xx tool-choice capability rejection."""

    status = getattr(error, "status_code", None)
    try:
        if status is not None and not (400 <= int(status) < 500):
            return False
    except (TypeError, ValueError):
        return False
    if status in {401, 403, 404, 408, 409, 429}:
        return False

    parts = [str(error)]
    for attr in ("message", "body", "response"):
        value = getattr(error, attr, None)
        if value is not None:
            parts.append(str(value))
    text = " ".join(parts).casefold()
    mentions_choice = "tool_choice" in text or "tool choice" in text
    mentions_capability = any(
        phrase in text
        for phrase in (
            "does not support",
            "not supported",
            "unsupported",
            "not compatible",
            "incompatible",
            "not allowed",
            "unknown parameter",
            "extra inputs are not permitted",
        )
    )
    return mentions_choice and mentions_capability
