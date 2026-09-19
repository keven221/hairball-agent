"""Native kernel facts emitted by Hairball's established v1 execution path.

Unlike :mod:`agent.kernel.v1_adapter`, this module does not install or wrap UI
callbacks.  ``conversation_loop`` and ``tool_executor`` call it at the exact
turn and tool execution seams, so the ledger is a side-band record of facts the
core has already decided to perform.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.kernel.contracts import EventKind, KernelEvent, new_agent_id, new_turn_id
from agent.kernel.facade import get_kernel

logger = logging.getLogger(__name__)

_SUMMARY_LIMIT = 400


def _text_of(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            str(item.get("text") or "")
            for item in value
            if isinstance(item, dict) and item.get("type") == "text"
        )
    if isinstance(value, dict):
        return str(value.get("content") or value.get("text") or "")
    return str(value or "")


def _summary(value: Any) -> str:
    text = _text_of(value)
    return text if len(text) <= _SUMMARY_LIMIT else text[:_SUMMARY_LIMIT] + "…"


def _mailbox_ledger(session_id: str) -> Any:
    """Return the opted-in ledger for a mailbox's owning parent session.

    Background completion threads no longer have the transient parent
    ``AIAgent`` object or its active turn recorder.  The mailbox already owns
    the durable routing key, though, so this is the narrow shared seam for
    recording facts that happen after the originating turn has returned.
    """
    if not session_id:
        return None
    try:
        kernel = get_kernel()
        ledger = kernel.ledger
        if ledger is None or not kernel.settings.observe_v1:
            return None
        kernel.resolve_version(session_id)
        return ledger
    except Exception:
        logger.warning("native mailbox event setup skipped", exc_info=True)
        return None


def emit_mailbox_enqueued(
    parent_session_id: Any,
    *,
    envelope_id: Any,
    envelope_kind: Any,
    agent_id: Any,
    payload: Optional[dict[str, Any]] = None,
) -> None:
    """Record a mailbox body after the real mailbox has accepted it.

    ``agent_completed`` is also the child runtime's terminal fact.  Keeping
    the completion and enqueue adjacent preserves the Single Rail: completion
    can be visible before collection, while the body remains unavailable to
    the parent model until the existing claim/complete path succeeds.
    """
    session_id = str(parent_session_id or "")
    eid = str(envelope_id or "")
    child_id = str(agent_id or "")
    kind = str(envelope_kind or "activity")
    ledger = _mailbox_ledger(session_id)
    if ledger is None or not eid:
        return
    body = dict(payload or {})
    common = {
        "envelope_id": eid,
        "agent_id": child_id,
        "envelope_kind": kind,
        "delivery_channel": "collab_mailbox",
    }
    try:
        if kind == "agent_completed":
            ledger.append(
                KernelEvent.build(
                    EventKind.AGENT_COMPLETED,
                    session_id,
                    payload={
                        **common,
                        "status": str(body.get("status") or "completed"),
                        "summary": _summary(body.get("summary")),
                        "error": _summary(body.get("error")),
                    },
                    agent_id=child_id,
                    correlation_id=child_id or eid,
                )
            )
        ledger.append(
            KernelEvent.build(
                EventKind.AGENT_MESSAGE_ENQUEUED,
                session_id,
                payload={
                    **common,
                    "has_summary": bool(body.get("summary")),
                    "has_error": bool(body.get("error")),
                },
                agent_id=child_id,
                correlation_id=eid,
            )
        )
    except Exception:
        logger.warning("native mailbox enqueue event append failed", exc_info=True)


def emit_mailbox_delivered(
    parent_session_id: Any,
    *,
    envelope_id: Any,
    envelope_kind: Any,
    agent_id: Any,
) -> None:
    """Record only a successful mailbox claim completion, never an attempt."""
    session_id = str(parent_session_id or "")
    eid = str(envelope_id or "")
    ledger = _mailbox_ledger(session_id)
    if ledger is None or not eid:
        return
    child_id = str(agent_id or "")
    try:
        ledger.append(
            KernelEvent.build(
                EventKind.AGENT_MESSAGE_DELIVERED,
                session_id,
                payload={
                    "envelope_id": eid,
                    "agent_id": child_id,
                    "envelope_kind": str(envelope_kind or "activity"),
                    "delivery_channel": "collab_mailbox",
                },
                agent_id=child_id,
                correlation_id=eid,
            )
        )
    except Exception:
        logger.warning("native mailbox delivery event append failed", exc_info=True)


@dataclass
class NativeTurnEvents:
    """Append canonical facts from the native conversation/tool path.

    The recorder is turn-scoped and lives on ``agent._native_turn_events`` only
    while a turn executes.  Every method is best-effort: observability storage
    may fail, but it must never change the model, tool, or callback behavior.
    """

    ledger: Any
    session_id: str
    turn_id: str
    agent_id: str
    provider_ref: str
    model_ref: str
    _started_tools: set[str] = field(default_factory=set)
    _terminal_tools: set[str] = field(default_factory=set)
    _last_assistant_text: str = ""
    _finished: bool = False
    _provider_attempts: dict[int, int] = field(default_factory=dict)
    _pending_provider_retries: dict[int, dict[str, Any]] = field(default_factory=dict)
    _spawned_agents: set[str] = field(default_factory=set)
    _completed_agents: set[str] = field(default_factory=set)

    @classmethod
    def open(cls, agent: Any, user_input: Any) -> Optional["NativeTurnEvents"]:
        """Open an event scope only when the existing kernel ledger is enabled."""
        if getattr(agent, "_persist_disabled", False):
            return None
        session_id = str(getattr(agent, "session_id", "") or "")
        if not session_id:
            return None
        try:
            kernel = get_kernel()
            ledger = kernel.ledger
            if ledger is None or not kernel.settings.observe_v1:
                return None
            kernel.resolve_version(session_id)
        except Exception:
            logger.warning("native kernel event setup skipped", exc_info=True)
            return None

        recorder = cls(
            ledger=ledger,
            session_id=session_id,
            turn_id=new_turn_id(),
            agent_id=new_agent_id(),
            provider_ref=str(getattr(agent, "provider", "") or ""),
            model_ref=str(getattr(agent, "model", "") or ""),
        )
        recorder._emit(
            EventKind.TURN_ACCEPTED,
            {
                "user_input_text": _text_of(user_input),
                "provider_ref": recorder.provider_ref,
                "model_ref": recorder.model_ref,
                "kernel_backend": "v1-native",
            },
        )
        recorder._emit(EventKind.TURN_STARTED, {})
        return recorder

    def _emit(
        self,
        kind: EventKind,
        payload: Optional[dict[str, Any]] = None,
        *,
        correlation_id: str = "",
    ) -> None:
        try:
            self.ledger.append(
                KernelEvent.build(
                    kind,
                    self.session_id,
                    payload=payload,
                    turn_id=self.turn_id,
                    agent_id=self.agent_id,
                    correlation_id=correlation_id or self.turn_id,
                )
            )
        except Exception:
            logger.warning("native kernel event append failed for %s", kind, exc_info=True)

    def assistant_message(self, text: Any, *, source: str) -> None:
        visible = _text_of(text)
        if not visible or visible == self._last_assistant_text:
            return
        self._last_assistant_text = visible
        self._emit(
            EventKind.ASSISTANT_MESSAGE_COMPLETED,
            {"text": visible, "source": source},
        )

    def tool_started(self, tool_call_id: Any, tool_name: Any, arguments: Any) -> None:
        call_id = str(tool_call_id or "")
        if not call_id or call_id in self._started_tools:
            return
        self._started_tools.add(call_id)
        self._emit(
            EventKind.TOOL_CALL_STARTED,
            {
                "tool_call_id": call_id,
                "tool_name": str(tool_name or ""),
                "arguments": arguments if isinstance(arguments, dict) else {},
            },
            correlation_id=call_id,
        )

    def provider_request_started(
        self,
        step: int,
        *,
        provider_ref: Any = "",
        model_ref: Any = "",
    ) -> None:
        """Record an actual provider request and settle any prior retry intent.

        ``conversation_loop`` has many recovery branches (credential refresh,
        context repair, fallback, and ordinary backoff).  Rather than duplicate
        that decision tree in observability code, a retry becomes durable only
        when the next real provider request actually starts.  This preserves
        the source-of-truth semantics: a failed attempt that ends the turn is
        not falsely presented as a retry.
        """
        step = int(step)
        attempt = self._provider_attempts.get(step, 0) + 1
        self._provider_attempts[step] = attempt
        current_provider = str(provider_ref or self.provider_ref)
        current_model = str(model_ref or self.model_ref)
        pending = self._pending_provider_retries.pop(step, None)
        if pending is not None:
            changed_route = (
                pending["provider_ref"] != current_provider
                or pending["model_ref"] != current_model
            )
            self._emit(
                EventKind.TURN_RETRY_SCHEDULED,
                {
                    "attempt": attempt,
                    "reason": pending["reason"],
                    "delay_seconds": max(0.0, time.monotonic() - pending["at"]),
                    "action": "fallback" if changed_route else "retry",
                    "message": pending["message"],
                    "next_model": current_model,
                },
            )
        self._emit(
            EventKind.PROVIDER_REQUEST_STARTED,
            {
                "step": step,
                "attempt": attempt,
                "provider_ref": current_provider,
                "model_ref": current_model,
            },
        )

    def provider_request_completed(self, step: int, duration_seconds: float, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self._emit(
            EventKind.PROVIDER_REQUEST_COMPLETED,
            {
                "step": int(step),
                "duration_ms": int(float(duration_seconds) * 1000),
                "has_usage": bool(usage),
            },
        )

    def provider_request_failed(
        self,
        step: int,
        duration_seconds: float,
        error: BaseException,
        *,
        provider_ref: Any = "",
        model_ref: Any = "",
        reason: str = "request_error",
    ) -> None:
        """Record a failed request without deciding whether core will recover."""
        self._note_provider_retry(
            step,
            reason=reason,
            message=f"{type(error).__name__}: {error}",
            provider_ref=provider_ref,
            model_ref=model_ref,
        )
        self._emit(
            EventKind.PROVIDER_WARNING,
            {
                "step": int(step),
                "duration_ms": int(float(duration_seconds) * 1000),
                "error_type": type(error).__name__,
                "error": _summary(str(error)),
                "fatal": False,
                "stage": "request",
            },
        )

    def provider_response_rejected(
        self,
        step: int,
        duration_seconds: float,
        message: str,
        *,
        provider_ref: Any = "",
        model_ref: Any = "",
    ) -> None:
        """Record a transport-successful but unusable provider response."""
        self._note_provider_retry(
            step,
            reason="invalid_response",
            message=message,
            provider_ref=provider_ref,
            model_ref=model_ref,
        )
        self._emit(
            EventKind.PROVIDER_WARNING,
            {
                "step": int(step),
                "duration_ms": int(float(duration_seconds) * 1000),
                "error_type": "InvalidAPIResponse",
                "error": _summary(message),
                "fatal": False,
                "stage": "response_validation",
            },
        )

    def provider_request_cancelled(self, step: int, duration_seconds: float) -> None:
        """Record a request cancelled by the existing interrupt machinery."""
        self._emit(
            EventKind.PROVIDER_WARNING,
            {
                "step": int(step),
                "duration_ms": int(float(duration_seconds) * 1000),
                "error_type": "InterruptedError",
                "error": "provider request interrupted",
                "fatal": False,
                "cancelled": True,
                "stage": "request",
            },
        )

    def subagent_spawned(
        self,
        subagent_id: Any,
        *,
        objective: Any,
        child_session_id: Any = "",
        depth: Any = None,
        model: Any = "",
        provider: Any = "",
    ) -> None:
        """Record a child only after the delegate runtime has built it."""
        child_id = str(subagent_id or "")
        if not child_id or child_id in self._spawned_agents:
            return
        self._spawned_agents.add(child_id)
        self._emit_agent_event(
            EventKind.AGENT_SPAWNED,
            child_id,
            {
                "agent_id": child_id,
                "parent_agent_id": self.agent_id,
                "objective": _summary(objective),
                "child_session_id": str(child_session_id or ""),
                "depth": depth if isinstance(depth, int) else None,
                "model": str(model or ""),
                "provider": str(provider or ""),
            },
        )

    def subagent_completed(
        self,
        subagent_id: Any,
        *,
        status: Any,
        summary: Any = "",
        duration_seconds: Any = 0.0,
        api_calls: Any = 0,
        exit_reason: Any = "",
    ) -> None:
        """Record the child result at the delegate runtime terminal seam."""
        child_id = str(subagent_id or "")
        if not child_id or child_id in self._completed_agents:
            return
        self._completed_agents.add(child_id)
        self._emit_agent_event(
            EventKind.AGENT_COMPLETED,
            child_id,
            {
                "agent_id": child_id,
                "status": str(status or "completed"),
                "summary": _summary(summary),
                "duration_ms": int(float(duration_seconds or 0.0) * 1000),
                "api_calls": int(api_calls or 0),
                "exit_reason": str(exit_reason or ""),
            },
        )

    def _note_provider_retry(
        self,
        step: int,
        *,
        reason: str,
        message: str,
        provider_ref: Any,
        model_ref: Any,
    ) -> None:
        self._pending_provider_retries[int(step)] = {
            "reason": str(reason or "request_error"),
            "message": _summary(message),
            "provider_ref": str(provider_ref or self.provider_ref),
            "model_ref": str(model_ref or self.model_ref),
            "at": time.monotonic(),
        }

    def _emit_agent_event(
        self, kind: EventKind, child_id: str, payload: dict[str, Any]
    ) -> None:
        """Use the child as event actor while retaining the parent's turn."""
        try:
            self.ledger.append(
                KernelEvent.build(
                    kind,
                    self.session_id,
                    payload=payload,
                    turn_id=self.turn_id,
                    agent_id=child_id,
                    correlation_id=child_id,
                )
            )
        except Exception:
            logger.warning("native subagent event append failed for %s", kind, exc_info=True)

    def tool_completed(
        self,
        tool_call_id: Any,
        tool_name: Any,
        arguments: Any,
        result: Any,
        duration_seconds: Any,
        is_error: bool,
    ) -> None:
        call_id = str(tool_call_id or "")
        if not call_id or call_id in self._terminal_tools:
            return
        if call_id not in self._started_tools:
            self.tool_started(call_id, tool_name, arguments)
        self._terminal_tools.add(call_id)
        text = _text_of(result)
        payload = {
            "tool_call_id": call_id,
            "tool_name": str(tool_name or ""),
            "duration_ms": int(float(duration_seconds or 0.0) * 1000),
            "summary": _summary(result),
            "byte_count": len(text.encode("utf-8", "replace")),
        }
        if is_error:
            payload["error"] = _summary(result)
            payload["error_kind"] = "tool_error"
            kind = EventKind.TOOL_CALL_FAILED
        else:
            kind = EventKind.TOOL_CALL_COMPLETED
        self._emit(kind, payload, correlation_id=call_id)

    def finish(self, result: Any, agent: Any) -> None:
        if self._finished:
            return
        self._finished = True
        data = result if isinstance(result, dict) else {}
        self.assistant_message(data.get("final_response"), source="result")
        if data.get("interrupted"):
            self._emit(EventKind.TURN_CANCELLED, {"reason": "interrupted"})
        elif data.get("failed") or data.get("error"):
            self._emit(
                EventKind.TURN_FAILED,
                {"error": _summary(data.get("error") or "turn failed"), "error_kind": "v1_reported_failure"},
            )
        else:
            self._emit(EventKind.TURN_COMPLETED, {"api_calls": int(data.get("api_calls") or 0)})

    def fail(self, error: BaseException) -> None:
        if self._finished:
            return
        self._finished = True
        self._emit(
            EventKind.TURN_FAILED,
            {"error": f"{type(error).__name__}: {error}", "error_kind": type(error).__name__},
        )


@dataclass
class NativeCompactionEvents:
    """Record one real context-compaction operation without owning its state.

    Automatic compaction runs inside :class:`NativeTurnEvents`; manual CLI,
    TUI, Gateway, and ACP compaction call the same ``compress_context``
    function outside a model turn.  This small scope preserves that shared
    source seam: it reuses the active turn identity when present and otherwise
    emits session-scoped facts with no synthetic conversation turn.
    """

    ledger: Any
    session_id: str
    turn_id: str = ""
    agent_id: str = ""
    correlation_id: str = ""

    @classmethod
    def open(cls, agent: Any) -> Optional["NativeCompactionEvents"]:
        active_turn = getattr(agent, "_native_turn_events", None)
        if active_turn is not None:
            return cls(
                ledger=active_turn.ledger,
                session_id=active_turn.session_id,
                turn_id=active_turn.turn_id,
                agent_id=active_turn.agent_id,
                correlation_id=f"compaction:{active_turn.turn_id}",
            )
        if getattr(agent, "_persist_disabled", False):
            return None
        session_id = str(getattr(agent, "session_id", "") or "")
        if not session_id:
            return None
        try:
            kernel = get_kernel()
            ledger = kernel.ledger
            if ledger is None or not kernel.settings.observe_v1:
                return None
            kernel.resolve_version(session_id)
        except Exception:
            logger.warning("native compaction event setup skipped", exc_info=True)
            return None
        return cls(
            ledger=ledger,
            session_id=session_id,
            agent_id=new_agent_id(),
            correlation_id=f"compaction:{new_turn_id()}",
        )

    def _emit(self, kind: EventKind, payload: dict[str, Any]) -> None:
        try:
            self.ledger.append(
                KernelEvent.build(
                    kind,
                    self.session_id,
                    payload=payload,
                    turn_id=self.turn_id,
                    agent_id=self.agent_id,
                    correlation_id=self.correlation_id,
                )
            )
        except Exception:
            logger.warning("native compaction event append failed for %s", kind, exc_info=True)

    def started(
        self,
        *,
        pre_message_count: int,
        approx_tokens: Optional[int],
        force: bool,
        in_place: bool,
        runtime: str,
    ) -> None:
        self._emit(
            EventKind.CONTEXT_COMPACTION_STARTED,
            {
                "pre_message_count": int(pre_message_count),
                "approx_tokens": int(approx_tokens or 0),
                "force": bool(force),
                "in_place": bool(in_place),
                "runtime": runtime,
            },
        )

    def completed(
        self,
        *,
        pre_message_count: int,
        post_message_count: int,
        pre_approx_tokens: Optional[int],
        post_approx_tokens: Optional[int],
        in_place: bool,
        old_session_id: str,
        new_session_id: str,
        compression_count: int,
        made_progress: bool,
        used_fallback: bool,
        runtime: str,
    ) -> None:
        self._emit(
            EventKind.CONTEXT_COMPACTION_COMPLETED,
            {
                "pre_message_count": int(pre_message_count),
                "post_message_count": int(post_message_count),
                "pre_approx_tokens": int(pre_approx_tokens or 0),
                "post_approx_tokens": int(post_approx_tokens or 0),
                "in_place": bool(in_place),
                "old_session_id": str(old_session_id or ""),
                "new_session_id": str(new_session_id or ""),
                "compression_count": int(compression_count or 0),
                "made_progress": bool(made_progress),
                "used_fallback": bool(used_fallback),
                "runtime": runtime,
            },
        )

    def skipped(self, reason: str) -> None:
        """Keep an aborted operation out of compaction metrics."""
        self._emit(
            EventKind.STATUS_NOTE,
            {
                "level": "info",
                "operation": "context.compaction",
                "outcome": "skipped",
                "reason": str(reason or "unknown"),
            },
        )
