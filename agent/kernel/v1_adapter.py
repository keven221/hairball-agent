"""Emit canonical kernel events from the existing v1 agent, changing nothing.

This is the bridge that lets kernel v2 be built and validated against real
traffic before any of its own runtime exists: v1 keeps executing exactly as it
does today, and the tap converts its callback fan-out into the append-only
event stream that ledger, projectors, and (later) the v2 engine consume.

## Why some callbacks are chained but never installed

Parts of v1 gate *behavior* on whether a callback is registered, not just on
what the callback does. Installing a wrapper where the host left the slot empty
would silently change the agent:

- ``reasoning_callback`` — ``AIAgent._fire_reasoning_delta`` only accumulates
  ``_current_streamed_reasoning_text`` when the callback exists. That is
  deliberate: ``show_reasoning=false`` leaves the slot unset so hidden provider
  thinking never becomes visible transcript content. Installing a tap there
  would leak reasoning into the transcript.
- ``stream_delta_callback`` — presence flips ``_has_stream_consumers()`` and
  causes ``_record_streamed_assistant_text``, which in turn decides the
  ``already_streamed`` flag on interim messages and suppresses ``_vprint``
  during tool execution.
- ``interim_assistant_callback`` — ``_emit_interim_assistant_message`` returns
  early when unset.

So those three are **chain-only**: wrapped when the host already registered
them, left alone otherwise. The kernel loses only streaming granularity by
that, never a durable fact, because the authoritative assistant text comes from
the turn result rather than from deltas.

The remaining callbacks are guarded by a plain ``if cb:`` that does nothing else,
so the tap may install them fresh.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Optional

from agent.kernel.contracts import (
    EventKind,
    KernelEvent,
    TurnRequest,
    new_agent_id,
    new_turn_id,
    synthetic_tool_call_id,
)
from agent.kernel.event_ledger import EventLedger
from agent.kernel.tool_outcome import looks_like_error

logger = logging.getLogger(__name__)

#: Callbacks the tap may create when the host left them unset. Each is guarded
#: in v1 by a bare presence check whose only effect is invoking the callback.
INSTALLABLE_CALLBACKS = (
    "tool_progress_callback",
    "tool_start_callback",
    "tool_complete_callback",
    "status_callback",
    "notice_callback",
    "event_callback",
    "step_callback",
)

#: Callbacks the tap only ever chains onto. See the module docstring.
CHAIN_ONLY_CALLBACKS = (
    "reasoning_callback",
    "stream_delta_callback",
    "interim_assistant_callback",
)


class V1EventTap:
    """Convert one ``AIAgent``'s callback traffic into kernel events.

    Usage::

        tap = V1EventTap(agent, ledger)
        tap.install()
        try:
            with tap.turn(request):
                result = agent.run_conversation(message)
        finally:
            tap.uninstall()

    The tap is per-agent and not reentrant across threads for a single turn;
    ``turn()`` guards against nested turn scopes.
    """

    def __init__(
        self,
        agent: Any,
        ledger: EventLedger,
        *,
        session_id: Optional[str] = None,
        agent_id: str = "",
        branch_id: str = "main",
    ) -> None:
        self._agent = agent
        self._ledger = ledger
        self._session_id = session_id or str(getattr(agent, "session_id", "") or "")
        self._agent_id = agent_id or new_agent_id()
        self._branch_id = branch_id

        self._originals: dict[str, Any] = {}
        self._installed = False
        self._lock = threading.RLock()

        self._turn_id = ""
        self._turn_active = False
        self._step = 0

        # Tool bookkeeping. v1 fires a progress event and a dedicated
        # start/complete callback for the same call; the progress event carries
        # duration/is_error while only the dedicated callback reliably carries
        # the call id. So progress is buffered and the dedicated callback emits.
        self._started_calls: set[str] = set()
        self._terminated_calls: set[str] = set()
        self._pending_progress: dict[str, dict[str, Any]] = {}
        self._synthetic_ids: dict[str, str] = {}
        self._last_progress_preview: dict[str, str] = {}

        # Assistant text bookkeeping for delta → completed folding.
        self._delta_buffer = ""
        self._last_completed_text = ""
        self._turn_outcome: Any = None

        # Session token counters as of the current turn's start, so the tap can
        # report per-turn consumption from v1's cumulative counters.
        self._usage_at_start: dict[str, float] = {}

    # ── identity ──

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def session_id(self) -> str:
        return self._session_id

    # ── install / uninstall ──

    def install(self) -> None:
        """Chain the tap onto *agent*'s callbacks. Idempotent."""
        with self._lock:
            if self._installed:
                return
            for name in INSTALLABLE_CALLBACKS:
                self._wrap(name, allow_create=True)
            for name in CHAIN_ONLY_CALLBACKS:
                self._wrap(name, allow_create=False)
            self._installed = True

    def uninstall(self) -> None:
        """Restore every original callback exactly. Idempotent."""
        with self._lock:
            if not self._installed:
                return
            for name, original in self._originals.items():
                try:
                    setattr(self._agent, name, original)
                except Exception:
                    logger.debug("failed to restore %s on agent", name, exc_info=True)
            self._originals.clear()
            self._installed = False

    def _wrap(self, name: str, *, allow_create: bool) -> None:
        original = getattr(self._agent, name, None)
        if original is None and not allow_create:
            return
        handler = getattr(self, f"_on_{name}")
        self._originals[name] = original

        def chained(*args: Any, **kwargs: Any) -> Any:
            # The host's callback runs first and its return value is what v1
            # sees, so a tap failure can never change v1's control flow.
            result = None
            if original is not None:
                result = original(*args, **kwargs)
            try:
                handler(*args, **kwargs)
            except Exception:
                logger.debug("kernel tap handler %s failed", name, exc_info=True)
            return result

        setattr(self._agent, name, chained)

    def __enter__(self) -> "V1EventTap":
        self.install()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.uninstall()

    # ── emit ──

    def _emit(
        self,
        kind: EventKind,
        payload: Optional[dict[str, Any]] = None,
        **fields: Any,
    ) -> Optional[KernelEvent]:
        if not self._session_id:
            return None
        event = KernelEvent.build(
            kind,
            self._session_id,
            payload=payload,
            branch_id=self._branch_id,
            turn_id=fields.pop("turn_id", self._turn_id),
            agent_id=fields.pop("agent_id", self._agent_id),
            correlation_id=fields.pop("correlation_id", self._turn_id),
            **fields,
        )
        try:
            return self._ledger.append(event)
        except Exception:
            # A ledger failure must never abort the turn it is observing.
            logger.warning("kernel ledger append failed for %s", kind, exc_info=True)
            return None

    # ── turn scope ──

    @contextmanager
    def turn(self, request: Optional[TurnRequest] = None, **kwargs: Any) -> Iterator[TurnRequest]:
        """Wrap one ``run_conversation`` call in turn lifecycle events.

        Emits ``turn.accepted`` and ``turn.started`` on entry and exactly one of
        ``turn.completed`` / ``turn.failed`` / ``turn.cancelled`` on exit — the
        single-terminal invariant is structural here rather than a convention
        each call site has to remember.
        """
        with self._lock:
            if self._turn_active:
                raise RuntimeError("V1EventTap.turn() is not reentrant")

        if request is None:
            request = TurnRequest(
                session_id=self._session_id,
                turn_id=kwargs.pop("turn_id", None) or new_turn_id(),
                user_input=kwargs.pop("user_input", ""),
                branch_id=self._branch_id,
                agent_id=self._agent_id,
                provider_ref=str(getattr(self._agent, "provider", "") or ""),
                model_ref=str(getattr(self._agent, "model", "") or ""),
                **kwargs,
            )

        self._begin_turn(request)
        try:
            yield request
        except BaseException as exc:
            self._end_turn_failed(exc)
            raise
        else:
            self._end_turn_completed()

    def _begin_turn(self, request: TurnRequest) -> None:
        with self._lock:
            self._turn_active = True
            self._turn_id = request.turn_id
            self._step = 0
            self._delta_buffer = ""
            self._last_completed_text = ""
            self._started_calls.clear()
            self._terminated_calls.clear()
            self._pending_progress.clear()
            self._synthetic_ids.clear()
            self._last_progress_preview.clear()
            self._usage_at_start = self._usage_snapshot()

        self._emit(
            EventKind.TURN_ACCEPTED,
            {
                "user_input_text": _text_of(request.user_input),
                "provider_ref": request.provider_ref,
                "model_ref": request.model_ref,
                "prompt_epoch": request.prompt_epoch,
                "tool_schema_epoch": request.tool_schema_epoch,
                "kernel_backend": "v1",
            },
        )
        self._emit(EventKind.TURN_STARTED, {})

    def _flush_delta_buffer(self) -> None:
        text = self._delta_buffer
        self._delta_buffer = ""
        if not text or text == self._last_completed_text:
            return
        self._last_completed_text = text
        self._emit(EventKind.ASSISTANT_MESSAGE_COMPLETED, {"text": text, "source": "stream"})

    def _close_open_tool_calls(self, reason: str, kind: EventKind) -> None:
        """Give every still-running tool call a terminal event.

        A turn that ends while a tool call is open would otherwise leave the
        ledger with a started call and no result — the same shape as an
        assistant tool call with no matching tool result in provider history.
        """
        open_calls = self._started_calls - self._terminated_calls
        for call_id in sorted(open_calls):
            self._terminated_calls.add(call_id)
            self._emit(
                kind,
                {
                    "tool_call_id": call_id,
                    "error": reason,
                    "error_kind": "turn_ended_with_open_tool_call",
                },
                correlation_id=call_id,
            )

    def finish_turn_with_result(self, result: Any) -> None:
        """Record the assistant's final answer from a ``run_conversation`` result.

        Call this inside the ``turn()`` scope, before it exits. The result dict
        is the authoritative source for the final text — it exists whether or
        not any surface registered a streaming callback.
        """
        if not isinstance(result, dict):
            return
        self._flush_delta_buffer()
        text = str(result.get("final_response") or "")
        if text and text != self._last_completed_text:
            self._last_completed_text = text
            self._emit(
                EventKind.ASSISTANT_MESSAGE_COMPLETED,
                {"text": text, "source": "result"},
            )
        turn_usage, session_usage = self._usage_from_result(result)
        payload: dict[str, Any] = {
            "api_calls": int(result.get("api_calls") or 0),
            "usage": turn_usage,
            "session_usage": session_usage,
        }
        for label in ("cost_status", "cost_source"):
            value = result.get(label)
            if value:
                payload[label] = str(value)
        self._emit(EventKind.PROVIDER_REQUEST_COMPLETED, payload)
        self._turn_outcome = result

    def _usage_snapshot(self) -> dict[str, float]:
        """Read v1's cumulative counters. Read-only — never writes to the agent."""
        return {
            key: _as_number(getattr(self._agent, attr, 0))
            for key, attr in _CUMULATIVE_USAGE_FIELDS.items()
        }

    def _usage_from_result(self, result: Mapping[str, Any]) -> tuple[dict, dict]:
        """Split v1's cumulative token counters into this turn's cost and the total.

        v1's ``run_conversation`` result carries token accounting as *flat*
        session-cumulative keys (``agent/turn_finalizer.py`` reads
        ``agent.session_*``), with no nested ``usage`` dict. Reading a
        ``result["usage"]`` key therefore yielded nothing and dropped every token
        and cost fact on the floor.

        The delta matters, not just the presence: ``MetricsProjector``
        accumulates ``usage`` with ``+=`` across events, so emitting cumulative
        totals would count turn 1's tokens again in turn 2. ``usage`` is this
        turn's consumption; ``session_usage`` keeps the running total for callers
        that want it.
        """
        session_usage: dict[str, float] = {}
        for key, attr in _CUMULATIVE_USAGE_FIELDS.items():
            if key in result:
                session_usage[key] = _as_number(result.get(key))
            else:
                # A caller may hand over a trimmed result; the agent still holds
                # the authoritative counter.
                session_usage[key] = _as_number(getattr(self._agent, attr, 0))

        turn_usage: dict[str, float] = {}
        for key, total in session_usage.items():
            # Clamp at zero: a counter reset mid-turn must not report negative
            # consumption, which would silently subtract from the metrics view.
            delta = total - self._usage_at_start.get(key, 0.0)
            turn_usage[key] = round(delta, 6) if delta > 0 else 0

        # Respect an explicit nested usage dict if some caller does supply one —
        # it is more specific than a counter difference.
        explicit = result.get("usage")
        if isinstance(explicit, dict) and explicit:
            for key, value in explicit.items():
                turn_usage[str(key)] = _as_number(value)

        return turn_usage, session_usage

    def _end_turn_completed(self) -> None:
        result = getattr(self, "_turn_outcome", None)
        interrupted = bool(isinstance(result, dict) and result.get("interrupted"))
        failed = bool(isinstance(result, dict) and result.get("failed"))

        self._flush_delta_buffer()

        if interrupted:
            self._close_open_tool_calls("turn interrupted", EventKind.TOOL_CALL_CANCELLED)
            self._emit(EventKind.TURN_CANCELLED, {"reason": "interrupted"})
        elif failed:
            self._close_open_tool_calls("turn failed", EventKind.TOOL_CALL_FAILED)
            self._emit(
                EventKind.TURN_FAILED,
                {
                    "error": str((result or {}).get("error") or "turn failed"),
                    "error_kind": "v1_reported_failure",
                },
            )
        else:
            self._close_open_tool_calls(
                "turn completed with open tool call", EventKind.TOOL_CALL_FAILED
            )
            self._emit(
                EventKind.TURN_COMPLETED,
                {"api_calls": int((result or {}).get("api_calls") or 0)},
            )
        self._reset_turn()

    def _end_turn_failed(self, exc: BaseException) -> None:
        self._flush_delta_buffer()
        if isinstance(exc, KeyboardInterrupt):
            self._close_open_tool_calls("interrupted", EventKind.TOOL_CALL_CANCELLED)
            self._emit(EventKind.TURN_CANCELLED, {"reason": "KeyboardInterrupt"})
        else:
            self._close_open_tool_calls("turn raised", EventKind.TOOL_CALL_FAILED)
            self._emit(
                EventKind.TURN_FAILED,
                {"error": f"{type(exc).__name__}: {exc}", "error_kind": type(exc).__name__},
            )
        self._reset_turn()

    def _reset_turn(self) -> None:
        with self._lock:
            self._turn_active = False
        self._turn_outcome = None

    # ── callback handlers ──

    def _resolve_call_id(self, raw_id: Any, tool_name: str) -> str:
        """Return a stable call id, minting one when the provider omitted it.

        A provider-supplied id is always preserved verbatim. When it is missing,
        the synthetic id is memoized under *tool_name* so the start and complete
        callbacks for the same call agree on it, and released once that call
        terminates so a second call to the same tool gets a fresh id instead of
        colliding with the finished one.
        """
        call_id = str(raw_id or "")
        if call_id:
            return call_id
        existing = self._synthetic_ids.get(tool_name)
        if existing and existing not in self._terminated_calls:
            return existing
        minted = synthetic_tool_call_id()
        self._synthetic_ids[tool_name] = minted
        return minted

    def _on_tool_progress_callback(
        self,
        event_type: str = "",
        name: Any = None,
        preview: Any = None,
        args: Any = None,
        **kwargs: Any,
    ) -> None:
        kind = str(event_type or "")
        tool_name = str(name or "")
        call_id = str(kwargs.get("tool_call_id") or "")

        if kind == "tool.started":
            if call_id and preview:
                self._last_progress_preview[call_id] = str(preview)
            return

        if kind == "tool.completed":
            # Buffered for the matching tool_complete_callback, which carries
            # the authoritative call id.
            record = {
                "duration_ms": int(float(kwargs.get("duration") or 0.0) * 1000),
                "tool_name": tool_name,
            }
            if "is_error" in kwargs:
                record["is_error"] = bool(kwargs["is_error"])
            self._pending_progress[call_id or f"@name:{tool_name}"] = record
            return

        if kind.startswith("subagent."):
            self._on_subagent_progress(kind, tool_name, preview, kwargs)
            return

        if kind == "tool.output_risk" and call_id:
            self._emit(
                EventKind.TOOL_CALL_PROGRESS,
                {
                    "tool_call_id": call_id,
                    "tool_name": tool_name,
                    "preview": "output risk flagged",
                    "risk_metadata": kwargs.get("risk_metadata") or {},
                },
                correlation_id=call_id,
            )

    def _on_subagent_progress(
        self, kind: str, tool_name: str, preview: Any, kwargs: dict[str, Any]
    ) -> None:
        child_id = str(
            kwargs.get("subagent_id")
            or kwargs.get("delegation_id")
            or kwargs.get("child_session_id")
            or ""
        )
        if not child_id:
            return
        suffix = kind.split(".", 1)[1]
        if suffix in {"spawn_requested", "start"}:
            self._emit(
                EventKind.AGENT_SPAWNED,
                {
                    "agent_id": child_id,
                    "parent_agent_id": self._agent_id,
                    "objective": str(preview or tool_name or ""),
                    "workspace_ref": str(kwargs.get("workspace_ref") or ""),
                },
            )
        elif suffix == "complete":
            self._emit(
                EventKind.AGENT_COMPLETED,
                {
                    "agent_id": child_id,
                    "status": str(kwargs.get("status") or "completed"),
                    "summary": str(preview or ""),
                },
            )

    def _on_tool_start_callback(
        self, tool_call_id: Any = None, function_name: Any = None, display_args: Any = None
    ) -> None:
        tool_name = str(function_name or "")
        call_id = self._resolve_call_id(tool_call_id, tool_name)
        if call_id in self._started_calls:
            return
        self._started_calls.add(call_id)
        payload: dict[str, Any] = {
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "arguments": display_args if isinstance(display_args, dict) else {},
        }
        if not tool_call_id:
            payload["synthetic_tool_call_id"] = True
        preview = self._last_progress_preview.pop(call_id, "")
        if preview:
            payload["preview"] = preview
        self._emit(EventKind.TOOL_CALL_STARTED, payload, correlation_id=call_id)

    def _on_tool_complete_callback(
        self,
        tool_call_id: Any = None,
        function_name: Any = None,
        display_args: Any = None,
        result: Any = None,
    ) -> None:
        tool_name = str(function_name or "")
        call_id = self._resolve_call_id(tool_call_id, tool_name)
        if call_id in self._terminated_calls:
            return
        if call_id not in self._started_calls:
            # v1 can complete a call whose start was suppressed (blocked tools).
            # Emit the start so the ledger never shows a terminal without one.
            self._started_calls.add(call_id)
            self._emit(
                EventKind.TOOL_CALL_STARTED,
                {
                    "tool_call_id": call_id,
                    "tool_name": tool_name,
                    "arguments": display_args if isinstance(display_args, dict) else {},
                    "inferred_from_completion": True,
                },
                correlation_id=call_id,
            )
        self._terminated_calls.add(call_id)

        progress = self._pending_progress.pop(call_id, None) or self._pending_progress.pop(
            f"@name:{tool_name}", None
        ) or {}
        text = result if isinstance(result, str) else ""
        if "is_error" in progress:
            is_error = bool(progress["is_error"])
        else:
            is_error = _looks_like_error(result)
        payload = {
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "duration_ms": int(progress.get("duration_ms") or 0),
            "summary": _summarize(text),
            "byte_count": len(text.encode("utf-8", "replace")) if text else 0,
        }
        if is_error:
            payload["error"] = _summarize(text)
            payload["error_kind"] = "tool_error"
            self._emit(EventKind.TOOL_CALL_FAILED, payload, correlation_id=call_id)
        else:
            self._emit(EventKind.TOOL_CALL_COMPLETED, payload, correlation_id=call_id)

    def _on_step_callback(self, api_call_count: Any = None, prev_tools: Any = None) -> None:
        step = int(api_call_count or 0)
        self._step = max(self._step, step)
        self._emit(EventKind.TURN_STEP, {"step": step})

    def _on_status_callback(self, kind: Any = None, message: Any = None) -> None:
        self._emit(
            EventKind.STATUS_NOTE,
            {"level": str(kind or "info"), "message": str(message or "")},
        )

    def _on_notice_callback(self, notice: Any = None) -> None:
        self._emit(
            EventKind.STATUS_NOTE,
            {
                "level": "notice",
                "message": str(getattr(notice, "message", "") or notice or ""),
                "key": str(getattr(notice, "key", "") or ""),
            },
        )

    def _on_event_callback(self, event_type: Any = None, context: Any = None) -> None:
        if str(event_type or "") != "session:compress":
            return
        ctx = context if isinstance(context, dict) else {}
        self._emit(
            EventKind.CONTEXT_COMPACTION_COMPLETED,
            {
                "in_place": bool(ctx.get("in_place")),
                "compression_count": int(ctx.get("compression_count") or 0),
                "old_session_id": str(ctx.get("old_session_id") or ""),
            },
        )

    def _on_stream_delta_callback(self, delta: Any = None) -> None:
        if delta is None:
            self._flush_delta_buffer()
            return
        text = str(delta)
        if not text:
            return
        self._delta_buffer += text
        self._emit(EventKind.ASSISTANT_MESSAGE_DELTA, {"text": text})

    def _on_reasoning_callback(self, text: Any = None) -> None:
        chunk = str(text or "")
        if chunk:
            self._emit(EventKind.ASSISTANT_REASONING_DELTA, {"text": chunk})

    def _on_interim_assistant_callback(
        self, text: Any = None, *, already_streamed: bool = False, **_: Any
    ) -> None:
        visible = str(text or "")
        if not visible or visible == self._last_completed_text:
            return
        self._delta_buffer = ""
        self._last_completed_text = visible
        self._emit(
            EventKind.ASSISTANT_MESSAGE_COMPLETED,
            {"text": visible, "source": "interim", "already_streamed": already_streamed},
        )


# ── helpers ───────────────────────────────────────────────────────────────────

_SUMMARY_LIMIT = 400

#: v1 result key → the ``AIAgent`` attribute it is copied from. Every one of
#: these is *session cumulative*, which is why the tap reports differences
#: rather than the raw values (see ``_usage_from_result``). Source:
#: ``agent/turn_finalizer.py`` builds the result from ``agent.session_*``.
_CUMULATIVE_USAGE_FIELDS: dict[str, str] = {
    "input_tokens": "session_input_tokens",
    "output_tokens": "session_output_tokens",
    "cache_read_tokens": "session_cache_read_tokens",
    "cache_write_tokens": "session_cache_write_tokens",
    "reasoning_tokens": "session_reasoning_tokens",
    "prompt_tokens": "session_prompt_tokens",
    "completion_tokens": "session_completion_tokens",
    "total_tokens": "session_total_tokens",
    "estimated_cost_usd": "session_estimated_cost_usd",
}


def _as_number(value: Any) -> float | int:
    """Coerce a counter to a number without letting a bad value break a turn."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return value
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0


def _text_of(value: Any) -> str:
    """Best-effort readable text for a user input that may be multimodal."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        return str(value.get("content") or value.get("text") or "")
    return ""


def _summarize(text: str) -> str:
    if not text:
        return ""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SUMMARY_LIMIT:
        return collapsed
    return collapsed[:_SUMMARY_LIMIT] + f"… (+{len(collapsed) - _SUMMARY_LIMIT} chars)"


def _looks_like_error(text: Any) -> bool:
    """Fallback error detection for v1 tool results that lack a flag.

    Only consulted when the progress callback did not supply ``is_error`` — the
    structured flag always wins. Shares its decision with the v2 engine so a
    given tool output cannot be a failure on one path and a success on the other.
    """
    return looks_like_error(text)


def instrument_agent(
    agent: Any,
    ledger: EventLedger,
    **kwargs: Any,
) -> V1EventTap:
    """Create and install a tap on *agent*. Caller owns ``uninstall()``."""
    tap = V1EventTap(agent, ledger, **kwargs)
    tap.install()
    return tap
