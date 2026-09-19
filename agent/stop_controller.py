"""Typed turn-stop orchestration for Hairball agent loops.

Configured Stop hooks and the canonical Todo lifecycle share one typed
terminal boundary. Todo reminders operate only on explicit model-authored
state, remain bounded, and preserve automatic tool choice. They never infer
requirements from prose or add a second completion ledger.

Hairball's background-process delivery rail is the only lifecycle adapter in
this module.  It defers finalization while a process-owned completion wake is
pending, without adding model input or changing tool choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from html import escape
import json
import logging
import os
import re
from typing import Any, Mapping, Sequence
import uuid
import xml.etree.ElementTree as ET


logger = logging.getLogger(__name__)

HOOK_PROMPT_TAG = "hook_prompt"
HOOK_PROMPT_FINISH_REASON = "hook_prompt"
MID_RUN_TODO_MUTATION_THRESHOLD = 12
MID_RUN_TODO_NUDGE_MAX = 2
MUTATING_TOOLS = frozenset(
    {"ast_edit", "execute_code", "patch", "terminal", "write_file"}
)
_MARKDOWN_PROMPT_PREFIX_RE = re.compile(r"^(?:>\s*)?(?:(?:[-*+]|\d+[.)])\s+)*")
_PROMPT_LABEL_RE = re.compile(r"^(?:q(?:uestion)?|ask)\s*\d*\s*[:.)-]\s*", re.I)
_QUESTION_PROMPT_RE = re.compile(
    r"^(?:what|which|when|where|why|how|who|whom|whose|do|does|did|can|could|"
    r"would|will|should|is|are|am|may|shall)\b",
    re.I,
)
_USER_DIRECTED_PROMPT_RE = re.compile(r"\b(?:you|your|we|our)\b", re.I)
_USER_RESPONSE_CUE_RE = re.compile(
    r"^(?:please\s+)?(?:confirm|reply|choose|pick|decide|advise)\b|"
    r"^(?:please\s+)?answer\b|"
    r"^(?:please\s+)?(?:let\s+me\s+know|tell\s+me)\b|"
    r"^(?:请|麻烦)?(?:确认|回复|选择|决定|告诉我|告知我)",
    re.I,
)
_BACKGROUND_COMPLETION_RE = re.compile(
    r"(?:Background process|后台进程)\s+([A-Za-z0-9_.:-]+).*(?:completed|完成|结束)",
    re.IGNORECASE,
)
TODO_EAGER_PREFERRED = """<system-reminder>
Consider using `todo` before substantive work to keep a complex request visible
from investigation through implementation and verification. If you create the
list, continue the request in this same turn and update it only when state
materially changes.
</system-reminder>"""
TODO_EAGER_ALWAYS = """<system-reminder>
Before substantive work, call `todo` once to create a complete task list that
covers investigation, implementation, and verification. Continue the request
in this same turn after the call and update the list only when state materially
changes.
</system-reminder>"""


@dataclass(frozen=True)
class HookPromptFragment:
    """One typed Stop-hook continuation fragment."""

    text: str
    hook_run_id: str


@dataclass(frozen=True)
class StopOutcome:
    """Aggregate result of all Stop handlers for one terminal candidate."""

    should_stop: bool = False
    stop_reason: str | None = None
    should_block: bool = False
    block_reason: str | None = None
    continuation_fragments: tuple[HookPromptFragment, ...] = ()


class StopAction(str, Enum):
    ALLOW = "allow"
    RETRY_WITH_FULL_AUTONOMY = "retry_with_full_autonomy"
    DEFER_TO_ASYNC = "defer_to_async"


@dataclass(frozen=True)
class StopDecision:
    action: StopAction
    outcome: StopOutcome = StopOutcome()
    reason: str | None = None

    @property
    def should_continue(self) -> bool:
        return self.action is StopAction.RETRY_WITH_FULL_AUTONOMY

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.reason:
            reasons.append(self.reason)
        for fragment in self.outcome.continuation_fragments:
            run_id = fragment.hook_run_id
            if run_id.startswith("hairball-"):
                source, separator, _nonce = run_id[len("hairball-") :].rpartition("-")
                reason = source if separator else "stop_hook"
            else:
                reason = "stop_hook"
            if reason not in reasons:
                reasons.append(reason)
        return tuple(reasons)


def _serialize_hook_prompt_fragment(fragment: HookPromptFragment) -> str | None:
    run_id = str(fragment.hook_run_id or "").strip()
    if not run_id:
        return None
    return (
        f'<{HOOK_PROMPT_TAG} hook_run_id="{escape(run_id, quote=True)}">'
        f"{escape(str(fragment.text or ''))}"
        f"</{HOOK_PROMPT_TAG}>"
    )


def build_hook_prompt_message(
    fragments: Sequence[HookPromptFragment],
) -> dict[str, Any] | None:
    """Build the common provider-message form of typed hook fragments."""

    content = [
        serialized
        for fragment in fragments
        if (serialized := _serialize_hook_prompt_fragment(fragment)) is not None
    ]
    if not content:
        return None
    return {
        "role": "user",
        "content": "\n".join(content),
        "finish_reason": HOOK_PROMPT_FINISH_REASON,
    }


def parse_hook_prompt_fragments(message: Any) -> tuple[HookPromptFragment, ...]:
    """Parse a message only when every content item is a valid hook prompt."""

    if not isinstance(message, Mapping) or message.get("role") != "user":
        return ()
    content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, Mapping) or item.get("type") not in {
                "text",
                "input_text",
            }:
                return ()
            text = item.get("text")
            if not isinstance(text, str):
                return ()
            parts.append(text)
    elif isinstance(content, str):
        parts = [content]
    else:
        return ()

    fragments: list[HookPromptFragment] = []
    for part in parts:
        raw = part.strip()
        if not raw:
            return ()
        try:
            root = ET.fromstring(f"<root>{raw}</root>")
        except ET.ParseError:
            return ()
        if not list(root) or (root.text or "").strip():
            return ()
        for child in root:
            if child.tag != HOOK_PROMPT_TAG or child.tail and child.tail.strip():
                return ()
            run_id = str(child.attrib.get("hook_run_id") or "").strip()
            if not run_id or set(child.attrib) != {"hook_run_id"}:
                return ()
            fragments.append(
                HookPromptFragment(text="".join(child.itertext()), hook_run_id=run_id)
            )
    return tuple(fragments)


def is_hook_prompt_message(message: Any) -> bool:
    return bool(parse_hook_prompt_fragments(message))


def strip_hook_prompt_metadata(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return provider-wire copies without Hairball's local history marker."""

    cleaned: list[dict[str, Any]] = []
    for message in messages:
        row = dict(message)
        if row.get("role") == "user" and row.get("finish_reason") == HOOK_PROMPT_FINISH_REASON:
            row.pop("finish_reason", None)
        cleaned.append(row)
    return cleaned


def aggregate_stop_outcomes(outcomes: Sequence[StopOutcome]) -> StopOutcome:
    """Aggregate Stop results with explicit stop taking precedence."""

    should_stop = any(outcome.should_stop for outcome in outcomes)
    stop_reason = next(
        (outcome.stop_reason for outcome in outcomes if outcome.stop_reason),
        None,
    )
    if should_stop:
        return StopOutcome(should_stop=True, stop_reason=stop_reason)
    blocked = [outcome for outcome in outcomes if outcome.should_block]
    if not blocked:
        return StopOutcome()
    reasons = [outcome.block_reason for outcome in blocked if outcome.block_reason]
    fragments = tuple(
        fragment
        for outcome in blocked
        for fragment in outcome.continuation_fragments
    )
    return StopOutcome(
        should_block=True,
        block_reason="\n\n".join(reasons) or None,
        continuation_fragments=fragments,
    )


class StopController:
    """Session-stable dispatcher for configured Stop and SubagentStop hooks."""

    def __init__(self, agent: Any, *, enabled: bool = True) -> None:
        self.agent = agent
        self.enabled = bool(enabled)
        self._stop_hook_active = False
        self._explicit_auto_tool_choice_disabled = False
        self._pending_outcome = StopOutcome()
        self._todo_reminder_count = 0
        self._todo_reminder_awaiting_progress = False
        self._mutations_since_todo_touch = 0
        self._mid_run_todo_nudge_count = 0
        self._eager_todo_force_pending = False

    @classmethod
    def install(cls, agent: Any) -> "StopController":
        existing = getattr(agent, "_stop_controller", None)
        if isinstance(existing, cls):
            return existing
        controller = cls(agent, enabled=cls._should_enable(agent))
        agent._stop_controller = controller
        return controller

    @staticmethod
    def _should_enable(agent: Any) -> bool:
        del agent
        return True

    def start_turn(
        self,
        prompt_text: str | None = None,
        *,
        has_prior_user: bool = False,
        enabled_tools: Sequence[str] | set[str] | None = None,
    ) -> str:
        """Reset one settle cycle and return an optional eager Todo prelude.

        The prelude is folded into the current real user turn before the first
        provider call. It never mutates the cached system prompt or inserts a
        synthetic history turn.
        """

        self._stop_hook_active = False
        self._pending_outcome = StopOutcome()
        self._todo_reminder_count = 0
        self._todo_reminder_awaiting_progress = False
        self._mutations_since_todo_touch = 0
        self._mid_run_todo_nudge_count = 0
        self._eager_todo_force_pending = False

        settings = self._todo_settings()
        eager = settings["eager"]
        names = {
            str(name)
            for name in (
                enabled_tools or getattr(self.agent, "valid_tool_names", ())
            )
        }
        store = getattr(self.agent, "_todo_store", None)
        if (
            not settings["enabled"]
            or eager == "default"
            or str(getattr(self.agent, "agent_mode", "") or "").casefold()
            == "plan"
            or has_prior_user
            or "todo" not in names
            or (store is not None and bool(store.has_items()))
        ):
            return ""
        text = str(prompt_text or "").rstrip()
        if text.endswith(("?", "？", "!", "！")):
            return ""
        self._eager_todo_force_pending = eager == "always"
        return (
            TODO_EAGER_ALWAYS
            if self._eager_todo_force_pending
            else TODO_EAGER_PREFERRED
        )

    @property
    def stop_hook_active(self) -> bool:
        return self.enabled and self._stop_hook_active

    @property
    def continuation_retry_pending(self) -> bool:
        return self.stop_hook_active

    @property
    def tool_control_pending(self) -> bool:
        return self.continuation_retry_pending or self._eager_todo_force_pending

    def _todo_settings(self) -> dict[str, Any]:
        raw = getattr(self.agent, "_todo_reminder_settings", None)
        settings = raw if isinstance(raw, Mapping) else {}
        try:
            reminders_max = max(
                0, min(10, int(settings.get("reminders_max", 3)))
            )
        except (TypeError, ValueError):
            reminders_max = 3
        eager = (
            str(settings.get("eager", "default") or "default")
            .strip()
            .casefold()
        )
        if eager not in {"default", "preferred", "always"}:
            eager = "default"
        return {
            "enabled": bool(settings.get("enabled", True)),
            "reminders": bool(settings.get("reminders", True)),
            "reminders_max": reminders_max,
            "mid_run_nudges": bool(settings.get("mid_run_nudges", True)),
            "eager": eager,
        }

    @property
    def explicit_auto_tool_choice_disabled(self) -> bool:
        return self._explicit_auto_tool_choice_disabled

    def _tool_control_capabilities(self, api_mode: str):
        try:
            from providers import get_provider_profile

            profile = get_provider_profile(str(getattr(self.agent, "provider", "") or ""))
        except Exception:
            profile = None
        if profile is not None:
            return profile.tool_control_capabilities(
                model=str(getattr(self.agent, "model", "") or ""),
                api_mode=api_mode,
                base_url=str(getattr(self.agent, "base_url", "") or ""),
                reasoning_config=getattr(self.agent, "reasoning_config", None),
            )
        from providers.tool_control import transport_default_capabilities

        return transport_default_capabilities(api_mode)

    def apply_continuation_policy(
        self, api_kwargs: dict[str, Any], api_mode: str
    ) -> bool:
        """Retain the provider's normal automatic tool policy on hook retry."""

        if not self.tool_control_pending:
            return False
        from providers.tool_control import (
            apply_autonomous_tool_choice,
            apply_named_tool_choice,
            resolve_autonomous_tool_choice_plan,
        )

        if (
            self._eager_todo_force_pending
            and not self._explicit_auto_tool_choice_disabled
            and apply_named_tool_choice(
                api_kwargs, api_mode=api_mode, tool_name="todo"
            )
        ):
            return True

        plan = resolve_autonomous_tool_choice_plan(
            self._tool_control_capabilities(api_mode),
            explicit_auto_disabled_for_session=self._explicit_auto_tool_choice_disabled,
        )
        apply_autonomous_tool_choice(api_kwargs, api_mode=api_mode, plan=plan)
        return True

    def note_tool_result(
        self,
        tool_name: str,
        result: Any,
        *,
        failed: bool = False,
        blocked: bool = False,
    ) -> None:
        """Advance Todo lifecycle state after one real tool result."""

        del result
        name = str(tool_name or "")
        self._todo_reminder_awaiting_progress = False
        if name == "todo":
            self._mutations_since_todo_touch = 0
            if not failed and not blocked:
                self._eager_todo_force_pending = False
        elif not failed and not blocked and name in MUTATING_TOOLS:
            self._mutations_since_todo_touch += 1

    def take_mid_run_nudge(
        self, enabled_tools: Sequence[str] | set[str] | None = None
    ) -> dict[str, Any] | None:
        """Return the bounded hidden Todo reconciliation nudge, if due."""

        settings = self._todo_settings()
        names = {str(name) for name in (enabled_tools or ())}
        store = getattr(self.agent, "_todo_store", None)
        active = store.active_items() if store is not None else []
        if (
            self._mutations_since_todo_touch < MID_RUN_TODO_MUTATION_THRESHOLD
            or self._mid_run_todo_nudge_count >= MID_RUN_TODO_NUDGE_MAX
            or not settings["enabled"]
            or not settings["reminders"]
            or not settings["mid_run_nudges"]
            or str(getattr(self.agent, "agent_mode", "") or "").casefold()
            == "plan"
            or "todo" not in names
            or not active
        ):
            return None
        self._mutations_since_todo_touch = 0
        self._mid_run_todo_nudge_count += 1
        return build_hook_prompt_message(
            (
                HookPromptFragment(
                    text=(
                        "<system-reminder>Gentle reminder: "
                        f"{len(active)} todo item(s) are still open. If a task "
                        "finished since the last `todo` update, mark it complete; "
                        "otherwise continue working.</system-reminder>"
                    ),
                    hook_run_id=(
                        "hairball-todo_mid_run-"
                        f"{self._mid_run_todo_nudge_count}"
                    ),
                ),
            )
        )

    @staticmethod
    def _is_awaiting_user_answer(text: str) -> bool:
        body = str(text or "").strip()
        if not body:
            return False
        last = body.splitlines()[-1].strip()
        stripped = _MARKDOWN_PROMPT_PREFIX_RE.sub("", last).strip()
        without_label = _PROMPT_LABEL_RE.sub("", stripped).strip()
        had_label = without_label != stripped
        if re.search(r"[?？]\s*$", without_label):
            return bool(
                had_label
                or _QUESTION_PROMPT_RE.search(without_label)
                or _USER_DIRECTED_PROMPT_RE.search(without_label)
                or _USER_RESPONSE_CUE_RE.search(without_label)
                or re.search(
                    r"(?:吗|么|呢|哪|如何|什么|是否|能否)[?？]\s*$",
                    without_label,
                )
            )
        cue = re.sub(r"[.!?。！？]+$", "", without_label).strip()
        return bool(_USER_RESPONSE_CUE_RE.search(cue))

    @staticmethod
    def _json_object(value: Any) -> Mapping[str, Any] | None:
        if isinstance(value, Mapping):
            return value
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, Mapping) else None

    def _pending_async_wake(self, messages: Sequence[Mapping[str, Any]]) -> bool:
        """Return whether a process-owned completion wake still owns progress."""

        process_ids: list[str] = []
        delivered: set[str] = set()
        for message in messages or ():
            role = str(message.get("role") or "")
            content = message.get("content")
            if role == "user" and isinstance(content, str):
                match = _BACKGROUND_COMPLETION_RE.search(content)
                if match:
                    delivered.add(match.group(1))
                continue
            if role != "tool":
                continue
            payload = self._json_object(content)
            if payload is None:
                continue
            process_id = str(
                payload.get("session_id") or payload.get("process_id") or ""
            ).strip()
            if process_id and process_id not in process_ids:
                process_ids.append(process_id)
        if not process_ids:
            return False
        try:
            from tools.process_registry import process_registry

            inline_drain = str(getattr(self.agent, "platform", "") or "").casefold() in {
                "cli",
                "classic_cli",
            }
            for process_id in process_ids:
                if process_id in delivered:
                    continue
                session = process_registry.get(process_id)
                if session is not None:
                    watching = bool(getattr(session, "watch_patterns", ())) and not bool(
                        getattr(session, "_watch_disabled", False)
                    )
                    if watching and (
                        not bool(getattr(session, "exited", False))
                        or int(getattr(session, "_watch_hits", 0) or 0) > 0
                    ):
                        return True
                if process_registry.has_pending_completion_delivery(
                    process_id, inline_drain=inline_drain
                ):
                    return True
            return False
        except Exception:
            logger.debug("pending async wake inspection failed open", exc_info=True)
            return False

    def note_tool_choice_rejection(
        self, error: Any, api_kwargs: Mapping[str, Any] | None
    ) -> bool:
        """Use implicit auto after a definite provider capability rejection."""

        if not self.tool_control_pending or self._explicit_auto_tool_choice_disabled:
            return False
        payload = api_kwargs if isinstance(api_kwargs, Mapping) else {}
        has_choice = "tool_choice" in payload
        if not has_choice:
            tool_config = payload.get("toolConfig")
            has_choice = isinstance(tool_config, Mapping) and "toolChoice" in tool_config
        if not has_choice:
            return False
        from providers.tool_control import looks_like_tool_choice_capability_rejection

        if not looks_like_tool_choice_capability_rejection(error):
            return False
        self._explicit_auto_tool_choice_disabled = True
        logger.info(
            "Provider rejected tool_choice during Stop continuation; using "
            "implicit auto for this session (%s/%s)",
            getattr(self.agent, "provider", ""),
            getattr(self.agent, "model", ""),
        )
        return True

    def _todo_stop_outcome(
        self,
        *,
        text: str,
        messages: Sequence[Mapping[str, Any]],
        enabled_tools: Sequence[str] | set[str] | None,
    ) -> StopOutcome | None:
        settings = self._todo_settings()
        if (
            str(getattr(self.agent, "agent_mode", "") or "").casefold()
            == "plan"
        ):
            return None
        if self._todo_reminder_awaiting_progress:
            return None
        names = {str(name) for name in (enabled_tools or ())}
        if (
            not settings["enabled"]
            or not settings["reminders"]
            or "todo" not in names
        ):
            self._todo_reminder_count = 0
            return None
        store = getattr(self.agent, "_todo_store", None)
        active = store.active_items() if store is not None else []
        if not active:
            self._todo_reminder_count = 0
            self._todo_reminder_awaiting_progress = False
            return None
        if self._todo_reminder_count >= settings["reminders_max"]:
            return None
        if self._is_awaiting_user_answer(text) or self._pending_async_wake(messages):
            return None
        self._todo_reminder_count += 1
        self._todo_reminder_awaiting_progress = True
        self._mutations_since_todo_touch = 0
        todo_lines = "\n".join(
            f"- [{item['status']}] {item['content']}" for item in active
        )
        return StopOutcome(
            should_block=True,
            block_reason="todo items remain incomplete",
            continuation_fragments=(
                HookPromptFragment(
                    text=(
                        "<system-reminder>\n"
                        f"You stopped with {len(active)} incomplete todo item(s):\n"
                        f"{todo_lines}\n\n"
                        "Continue working on them or mark them complete if finished.\n"
                        f"(Reminder {self._todo_reminder_count}/"
                        f"{settings['reminders_max']})\n"
                        "</system-reminder>"
                    ),
                    hook_run_id=(
                        "hairball-todo_completion-"
                        f"{self._todo_reminder_count}"
                    ),
                ),
            ),
        )

    def evaluate_candidate(
        self,
        *,
        text: str,
        messages: Sequence[Mapping[str, Any]],
        enabled_tools: Sequence[str] | set[str] | None = None,
    ) -> StopDecision:
        """Run configured Stop handlers for a text-only terminal candidate."""

        if not self.enabled:
            return StopDecision(StopAction.ALLOW)
        if self._pending_async_wake(messages):
            self._stop_hook_active = False
            self._pending_outcome = StopOutcome()
            return StopDecision(StopAction.DEFER_TO_ASYNC, reason="pending_async_wake")
        todo_outcome = self._todo_stop_outcome(
            text=text,
            messages=messages,
            enabled_tools=enabled_tools,
        )
        if todo_outcome is not None:
            return self._decision_from_outcome(todo_outcome, handler_count=1)
        outcomes: list[StopOutcome] = []
        pre_verify = self._configured_pre_verify_outcome(text=text)
        if pre_verify is not None:
            outcomes.append(pre_verify)
        outcomes.extend(self._plugin_stop_outcomes(text=text))
        outcome = aggregate_stop_outcomes(outcomes)
        return self._decision_from_outcome(outcome, handler_count=len(outcomes))

    def _configured_pre_verify_outcome(self, *, text: str) -> StopOutcome | None:
        """Adapt an explicitly configured legacy pre-verify hook to Stop.

        This is not a built-in completion policy. It runs only when a user or
        plugin registered the hook, and maps its existing ``continue`` result
        onto the same typed outcome used by configured Stop handlers.
        """

        try:
            from hairball_cli.plugins import (
                get_pre_verify_continue_message,
                has_hook,
            )

            if not has_hook("pre_verify"):
                return None
            attempt = max(0, int(getattr(self.agent, "_pre_verify_nudges", 0) or 0))
            from agent.verify_hooks import max_verify_nudges

            if attempt >= max_verify_nudges():
                return None
            from agent.turn_workspace_tracker import turn_workspace_changed_paths

            changed_paths = turn_workspace_changed_paths(self.agent)
            guidance = get_pre_verify_continue_message(
                session_id=str(getattr(self.agent, "session_id", "") or ""),
                platform=str(getattr(self.agent, "platform", "") or ""),
                model=str(getattr(self.agent, "model", "") or ""),
                coding=bool(changed_paths),
                attempt=attempt,
                final_response=str(text or ""),
                changed_paths=list(changed_paths),
            )
        except Exception:
            logger.debug("configured pre_verify hook failed open", exc_info=True)
            return None
        if not isinstance(guidance, str) or not guidance.strip():
            return None
        self.agent._pre_verify_nudges = attempt + 1
        message = guidance.strip()
        return StopOutcome(
            should_block=True,
            block_reason=message,
            continuation_fragments=(
                HookPromptFragment(
                    text=message,
                    hook_run_id=f"hairball-pre_verify-{uuid.uuid4().hex}",
                ),
            ),
        )

    def _decision_from_outcome(
        self, outcome: StopOutcome, *, handler_count: int
    ) -> StopDecision:
        self._pending_outcome = outcome
        logger.debug(
            "stop_hook_decision handlers=%d should_stop=%s should_block=%s "
            "continuation_fragments=%d active_before=%s",
            handler_count,
            outcome.should_stop,
            outcome.should_block,
            len(outcome.continuation_fragments),
            self._stop_hook_active,
        )
        if not outcome.should_block:
            self._stop_hook_active = False
            return StopDecision(StopAction.ALLOW, outcome)
        if not outcome.continuation_fragments:
            logger.warning("Stop hook requested continuation without a prompt; ignoring block")
            self._stop_hook_active = False
            return StopDecision(StopAction.ALLOW)
        self._stop_hook_active = True
        return StopDecision(StopAction.RETRY_WITH_FULL_AUTONOMY, outcome)

    def build_continuation_message(self) -> dict[str, Any] | None:
        if not self._stop_hook_active or not self._pending_outcome.should_block:
            return None
        return build_hook_prompt_message(self._pending_outcome.continuation_fragments)

    def _plugin_stop_outcomes(self, *, text: str) -> list[StopOutcome]:
        try:
            from hairball_cli.plugins import has_hook, invoke_hook

            is_subagent = bool(getattr(self.agent, "_subagent_id", None)) or str(
                getattr(self.agent, "platform", "") or ""
            ).strip().casefold() == "subagent"
            hook_name = "subagent_stop" if is_subagent else "stop"
            if not has_hook(hook_name):
                return []
            payload: dict[str, Any] = {
                "session_id": str(
                    (
                        getattr(self.agent, "_parent_session_id", None)
                        if is_subagent
                        else getattr(self.agent, "session_id", None)
                    )
                    or getattr(self.agent, "session_id", "")
                    or ""
                ),
                "turn_id": str(
                    (
                        getattr(self.agent, "_parent_turn_id", None)
                        if is_subagent
                        else getattr(self.agent, "_current_turn_id", None)
                    )
                    or getattr(self.agent, "_current_turn_id", "")
                    or ""
                ),
                "cwd": str(getattr(self.agent, "cwd", "") or os.getcwd()),
                "transcript_path": (
                    getattr(self.agent, "transcript_path", None)
                    or getattr(self.agent, "_live_transcript_path", None)
                ),
                "model": str(getattr(self.agent, "model", "") or ""),
                "permission_mode": str(
                    getattr(self.agent, "approval_mode", "")
                    or getattr(self.agent, "permission_mode", "")
                    or ""
                ),
                "stop_hook_active": self._stop_hook_active,
                "last_assistant_message": str(text or "") or None,
            }
            if is_subagent:
                payload.update(
                    agent_id=str(
                        getattr(self.agent, "_subagent_id", "")
                        or getattr(self.agent, "session_id", "")
                        or ""
                    ),
                    agent_type=str(getattr(self.agent, "_delegate_role", "") or ""),
                    agent_transcript_path=(
                        getattr(self.agent, "_live_transcript_path", None)
                        or getattr(self.agent, "transcript_path", None)
                    ),
                )
            results = invoke_hook(hook_name, **payload)
        except Exception:
            logger.debug("Stop hook invocation failed open", exc_info=True)
            return []

        outcomes: list[StopOutcome] = []
        for result in results:
            if not isinstance(result, Mapping):
                continue
            if result.get("continue") is False:
                reason = result.get("stopReason") or result.get("stop_reason")
                outcomes.append(
                    StopOutcome(
                        should_stop=True,
                        stop_reason=str(reason).strip() if reason else None,
                    )
                )
                continue
            if str(result.get("decision") or "").strip().casefold() != "block":
                continue
            reason = result.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                logger.warning("Stop hook returned decision:block without a non-empty reason")
                continue
            run_id = str(result.get("hook_run_id") or uuid.uuid4().hex)
            guidance = reason.strip()
            outcomes.append(
                StopOutcome(
                    should_block=True,
                    block_reason=guidance,
                    continuation_fragments=(
                        HookPromptFragment(text=guidance, hook_run_id=run_id),
                    ),
                )
            )
        return outcomes


__all__ = [
    "HOOK_PROMPT_TAG",
    "HOOK_PROMPT_FINISH_REASON",
    "HookPromptFragment",
    "StopAction",
    "StopController",
    "StopDecision",
    "StopOutcome",
    "aggregate_stop_outcomes",
    "build_hook_prompt_message",
    "is_hook_prompt_message",
    "parse_hook_prompt_fragments",
    "strip_hook_prompt_metadata",
]
