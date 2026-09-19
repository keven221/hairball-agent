"""Kernel v2's provider seam.

Phase 3 of the roadmap asks for three things: normalize provider streams, move
provider special-cases out of the loop, and build a capability matrix with
contract tests.  This module is the *destination* for the second one and the
kernel-side consumer of the other two.

What it owns:

- picking the transport for an ``api_mode`` and refusing modes that have none,
- turning a request into provider kwargs through the transport,
- folding streamed or non-streamed responses into one ``NormalizedResponse``,
- emitting canonical kernel events for the provider leg of a turn.

What it deliberately does not own — and why:

Retry policy, credential rotation/pooling, client construction, prompt-cache
control, and interrupt handling all already exist in v1 and are load-bearing
(``agent/retry_utils.py``, ``agent/credential_pool.py``,
``agent/turn_retry_state.py``, and the stale-stream circuit breaker inside
``chat_completion_helpers``).  Reimplementing them here would fork behavior that
took real incidents to get right, and the roadmap's Phase 3 gate is explicitly
"retry, credential pool, fallback 不降级".  So they arrive as *injected
callables*: the caller hands in something that performs the call, and this
runtime stays responsible only for shape.

That injection boundary is also why this module needs no changes to v1.  It can
drive a transport directly, or wrap v1's existing streaming helper, without
either path knowing about the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from agent.kernel.contracts import (
    EventKind,
    KernelError,
    KernelEvent,
    TurnRequest,
)
from agent.transports.capabilities import (
    TransportCapabilities,
    get_capabilities,
    is_known_api_mode,
    resolve_transport,
)
from agent.transports.stream import (
    CallbackStreamSource,
    StreamAccumulator,
    StreamEvent,
    StreamEventKind,
    stream_from_normalized,
)
from agent.transports.types import NormalizedResponse


class ProviderRuntimeError(KernelError):
    """Base class for provider-seam failures."""


class UnsupportedApiModeError(ProviderRuntimeError):
    """Raised when an api_mode has no ``ProviderTransport`` to drive.

    Explicit refusal beats a ``None`` transport propagating into a turn: the
    caller learns immediately that this mode needs its own runtime (today that
    is ``codex_app_server``) instead of failing later with an ``AttributeError``
    that says nothing about the real cause.
    """


@dataclass(frozen=True)
class ProviderRequest:
    """One provider call, described independently of any provider's SDK."""

    api_mode: str
    model: str
    messages: Sequence[Mapping[str, Any]]
    tools: Optional[Sequence[Mapping[str, Any]]] = None
    params: Mapping[str, Any] = field(default_factory=dict)
    stream: bool = False

    def with_params(self, **extra: Any) -> "ProviderRequest":
        merged = dict(self.params)
        merged.update(extra)
        return ProviderRequest(
            api_mode=self.api_mode,
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            params=merged,
            stream=self.stream,
        )


@dataclass
class ProviderResult:
    """A completed provider leg: the response plus how it was obtained."""

    response: NormalizedResponse
    api_mode: str
    streamed: bool
    stream_events: tuple[StreamEvent, ...] = ()
    capabilities: Optional[TransportCapabilities] = None

    @property
    def finish_reason(self) -> str:
        return self.response.finish_reason or "stop"

    @property
    def tool_calls(self):
        return self.response.tool_calls or []


class ProviderRuntime:
    """Drive one provider call and record it as kernel events.

    ``ledger`` is optional so the runtime is usable in pure unit tests and in
    kernel-disabled contexts; when it is ``None`` nothing is recorded and the
    normalization behavior is unchanged.  This mirrors the facade's rule that a
    disabled kernel is inert rather than half-recording.
    """

    def __init__(self, ledger: Any = None) -> None:
        self.ledger = ledger

    # ── Capability gating ────────────────────────────────────────────────────

    def capabilities(self, api_mode: str) -> TransportCapabilities:
        return get_capabilities(api_mode)

    def resolve(self, api_mode: str):
        """Return the transport for *api_mode*, raising if it cannot be driven.

        Two distinct failures, kept distinct because they need different fixes:
        a mode the matrix does not know at all (likely a typo or a new provider
        that needs a matrix row), and a known mode that intentionally has no
        transport (needs its own runtime).
        """
        if not is_known_api_mode(api_mode):
            raise UnsupportedApiModeError(
                f"unknown api_mode {api_mode!r}; add it to "
                "agent/transports/capabilities.py before driving it"
            )
        cap = get_capabilities(api_mode)
        if not cap.has_transport:
            raise UnsupportedApiModeError(
                f"api_mode {api_mode!r} has no ProviderTransport: {cap.notes}"
            )
        transport = resolve_transport(api_mode)
        if transport is None:
            raise UnsupportedApiModeError(
                f"api_mode {api_mode!r} is declared to have a transport but the "
                "registry returned none — check register_transport() import side effects"
            )
        return transport

    # ── Request building ─────────────────────────────────────────────────────

    def build_kwargs(self, request: ProviderRequest) -> dict[str, Any]:
        """Build provider-native call kwargs through the mode's transport."""
        transport = self.resolve(request.api_mode)
        return transport.build_kwargs(
            request.model,
            list(request.messages),
            list(request.tools) if request.tools else None,
            **dict(request.params),
        )

    # ── Non-streaming ────────────────────────────────────────────────────────

    def complete(
        self,
        request: ProviderRequest,
        *,
        invoke: Callable[[dict[str, Any]], Any],
        turn: Optional[TurnRequest] = None,
    ) -> ProviderResult:
        """Run a non-streamed provider call.

        *invoke* receives the built kwargs and returns the raw provider
        response.  It is where the caller plugs in client construction,
        credentials, and retry — this method never retries on its own, so a
        caller's retry policy stays the only one.
        """
        transport = self.resolve(request.api_mode)
        cap = get_capabilities(request.api_mode)
        kwargs = transport.build_kwargs(
            request.model,
            list(request.messages),
            list(request.tools) if request.tools else None,
            **dict(request.params),
        )

        self._emit_started(request, turn, streamed=False)
        try:
            raw = invoke(kwargs)
        except Exception as exc:
            self._emit_failed(request, turn, exc)
            raise

        if not transport.validate_response(raw):
            self._emit_warning(
                request,
                turn,
                "transport reported an invalid response",
            )

        response = transport.normalize_response(raw)
        result = ProviderResult(
            response=response,
            api_mode=request.api_mode,
            streamed=False,
            stream_events=stream_from_normalized(response),
            capabilities=cap,
        )
        self._emit_completed(request, turn, result)
        return result

    # ── Streaming ────────────────────────────────────────────────────────────

    def stream(
        self,
        request: ProviderRequest,
        *,
        invoke: Callable[[dict[str, Any], CallbackStreamSource], Any],
        turn: Optional[TurnRequest] = None,
        record_deltas: bool = False,
    ) -> ProviderResult:
        """Run a streamed provider call, normalizing its deltas.

        *invoke* receives the built kwargs and a :class:`CallbackStreamSource`
        to push deltas into; whatever it returns, if anything, is normalized
        through the transport and merged with the folded stream.  This shape
        lets a caller wrap v1's existing streaming helper — hand its
        ``_fire_*`` callbacks to the source — without v1 changing.

        ``record_deltas`` is off by default on purpose: emitting every token as
        an event costs work for output the surfaces already render live.  Note
        that emission is not persistence — delta events are ephemeral, so the
        ledger still decides whether they reach SQLite (``persist_ephemeral``).
        Live subscribers see them either way.  Keeping that policy in one place
        stops the runtime and the ledger from disagreeing about what is durable.
        """
        transport = self.resolve(request.api_mode)
        cap = get_capabilities(request.api_mode)
        kwargs = transport.build_kwargs(
            request.model,
            list(request.messages),
            list(request.tools) if request.tools else None,
            **dict(request.params),
        )

        recorder = self._delta_recorder(request, turn) if record_deltas else None
        source = CallbackStreamSource(sink=recorder)

        self._emit_started(request, turn, streamed=True)
        try:
            raw = invoke(kwargs, source)
        except Exception as exc:
            self._emit_failed(request, turn, exc)
            raise

        response = self._merge_stream(transport, source, raw)
        result = ProviderResult(
            response=response,
            api_mode=request.api_mode,
            streamed=True,
            stream_events=source.events(),
            capabilities=cap,
        )

        self._warn_on_unsupported_channels(request, turn, cap, source)
        self._emit_completed(request, turn, result)
        return result

    def _merge_stream(
        self,
        transport: Any,
        source: CallbackStreamSource,
        raw: Any,
    ) -> NormalizedResponse:
        """Reconcile folded deltas with the provider's final object.

        The final object is authoritative for structure — tool calls, usage,
        finish reason, and any provider_data that must be replayed verbatim
        (signed thinking blocks, reasoning item ids).  The deltas are
        authoritative for the text the user actually saw.  Preferring the final
        object for structure and the stream for text avoids both classic
        failures: dropping tool calls that only appear in the final object, and
        showing text the user never saw mid-stream.
        """
        acc = StreamAccumulator(strict=False)
        acc.feed_all(source.events())
        folded = acc.result()

        if raw is None:
            return folded

        final = raw if isinstance(raw, NormalizedResponse) else transport.normalize_response(raw)

        content = final.content if final.content else folded.content
        reasoning = final.reasoning if final.reasoning else folded.reasoning
        return NormalizedResponse(
            content=content,
            tool_calls=final.tool_calls if final.tool_calls else folded.tool_calls,
            finish_reason=final.finish_reason or folded.finish_reason,
            reasoning=reasoning,
            usage=final.usage or folded.usage,
            provider_data=final.provider_data or folded.provider_data,
        )

    def _warn_on_unsupported_channels(
        self,
        request: ProviderRequest,
        turn: Optional[TurnRequest],
        cap: TransportCapabilities,
        source: CallbackStreamSource,
    ) -> None:
        """Record when a stream delivered a channel the matrix says it cannot.

        A drifted matrix row is a silent correctness bug for anything that gates
        UI on declared capability, so surface it as a warning event rather than
        letting the two versions of the truth diverge unobserved.
        """
        seen = {
            StreamEventKind.TEXT_DELTA: "text",
            StreamEventKind.REASONING_DELTA: "reasoning",
            StreamEventKind.TOOL_CALL_STARTED: "tool_start",
        }
        observed = {seen[e.kind] for e in source.events() if e.kind in seen}
        undeclared = sorted(
            ch for ch in observed if not cap.supports_stream_channel(ch)
        )
        if undeclared:
            self._emit_warning(
                request,
                turn,
                "stream delivered undeclared channels: " + ", ".join(undeclared),
            )

    # ── Event emission ───────────────────────────────────────────────────────

    def _append(self, event: KernelEvent, turn: Optional[TurnRequest]) -> None:
        if self.ledger is None:
            return
        if turn is not None:
            event = event.for_turn(turn)
        self.ledger.append(event)

    def _session_id(self, turn: Optional[TurnRequest]) -> str:
        return turn.session_id if turn is not None else ""

    def _emit_started(
        self,
        request: ProviderRequest,
        turn: Optional[TurnRequest],
        *,
        streamed: bool,
    ) -> None:
        self._append(
            KernelEvent.build(
                EventKind.PROVIDER_REQUEST_STARTED,
                self._session_id(turn),
                payload={
                    "api_mode": request.api_mode,
                    "model": request.model,
                    "streamed": streamed,
                    "tool_count": len(request.tools or ()),
                    "message_count": len(request.messages),
                },
            ),
            turn,
        )

    def _emit_completed(
        self,
        request: ProviderRequest,
        turn: Optional[TurnRequest],
        result: ProviderResult,
    ) -> None:
        usage = result.response.usage
        self._append(
            KernelEvent.build(
                EventKind.PROVIDER_REQUEST_COMPLETED,
                self._session_id(turn),
                payload={
                    "api_mode": request.api_mode,
                    "model": request.model,
                    "streamed": result.streamed,
                    "finish_reason": result.finish_reason,
                    "tool_call_count": len(result.tool_calls),
                    "has_reasoning": bool(result.response.reasoning),
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "cached_tokens": usage.cached_tokens if usage else 0,
                },
            ),
            turn,
        )

    def _emit_failed(
        self,
        request: ProviderRequest,
        turn: Optional[TurnRequest],
        exc: BaseException,
    ) -> None:
        self._append(
            KernelEvent.build(
                EventKind.PROVIDER_WARNING,
                self._session_id(turn),
                payload={
                    "api_mode": request.api_mode,
                    "model": request.model,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                    "fatal": True,
                },
            ),
            turn,
        )

    def _emit_warning(
        self,
        request: ProviderRequest,
        turn: Optional[TurnRequest],
        message: str,
    ) -> None:
        self._append(
            KernelEvent.build(
                EventKind.PROVIDER_WARNING,
                self._session_id(turn),
                payload={
                    "api_mode": request.api_mode,
                    "model": request.model,
                    "warning": message,
                    "fatal": False,
                },
            ),
            turn,
        )

    def _delta_recorder(
        self, request: ProviderRequest, turn: Optional[TurnRequest]
    ) -> Callable[[StreamEvent], None]:
        """Return a sink that persists text/reasoning deltas as kernel events."""

        kinds = {
            StreamEventKind.TEXT_DELTA: EventKind.ASSISTANT_MESSAGE_DELTA,
            StreamEventKind.REASONING_DELTA: EventKind.ASSISTANT_REASONING_DELTA,
        }

        def record(event: StreamEvent) -> None:
            kind = kinds.get(event.kind)
            if kind is None or not event.text:
                return
            self._append(
                KernelEvent.build(
                    kind,
                    self._session_id(turn),
                    payload={"text": event.text, "api_mode": request.api_mode},
                ),
                turn,
            )

        return record
