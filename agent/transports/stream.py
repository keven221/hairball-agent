"""Canonical streaming vocabulary for provider transports.

``agent/transports/base.py`` deliberately leaves streaming out of the
``ProviderTransport`` contract, so today each api_mode grows its own branch
inside ``chat_completion_helpers.interruptible_streaming_api_call`` and fires
``AIAgent._fire_*`` callbacks directly.  That works, but it means "what does a
provider stream look like?" has four answers and no type.

This module gives it one answer: a small event union (:class:`StreamEvent`)
plus a fold (:class:`StreamAccumulator`) that turns any sequence of those
events into the *existing* :class:`~agent.transports.types.NormalizedResponse`.
Nothing here replaces the v1 streaming path — :class:`CallbackStreamSource`
adapts that path's callbacks into this vocabulary so a consumer can normalize a
live v1 stream without v1 changing at all.

The fold enforces the same terminality rule the kernel ledger enforces on
turns: a stream ends exactly once, and nothing follows the end.  Streams that
break that rule are the ones that produce duplicated or interleaved output
(see the single-writer guard in ``AIAgent._fire_stream_delta``), so it is worth
catching structurally rather than by inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Iterable, Iterator

from agent.transports.types import NormalizedResponse, ToolCall, Usage


class StreamEventKind(StrEnum):
    """The channels a provider stream can carry."""

    TEXT_DELTA = "text.delta"
    REASONING_DELTA = "reasoning.delta"
    TOOL_CALL_STARTED = "tool.call.started"
    USAGE = "usage"
    DONE = "done"
    ERROR = "error"


TERMINAL_STREAM_KINDS = frozenset({StreamEventKind.DONE, StreamEventKind.ERROR})


class StreamProtocolError(RuntimeError):
    """Raised when a stream violates the terminality contract."""


@dataclass(frozen=True)
class StreamEvent:
    """One observation from a provider stream.

    Frozen for the same reason kernel events are: a delta that can be mutated
    after the fact makes "what did the user actually see?" unanswerable.
    """

    kind: StreamEventKind
    text: str = ""
    tool_name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage | None = None
    finish_reason: str | None = None
    error: str | None = None
    provider_data: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def is_terminal(self) -> bool:
        return self.kind in TERMINAL_STREAM_KINDS

    # ── Constructors ─────────────────────────────────────────────────────────
    # Named constructors keep call sites readable and stop callers from minting
    # a TEXT_DELTA that carries a finish_reason, or a DONE that carries text.

    @classmethod
    def text_delta(cls, text: str) -> "StreamEvent":
        return cls(kind=StreamEventKind.TEXT_DELTA, text=text)

    @classmethod
    def reasoning_delta(cls, text: str) -> "StreamEvent":
        return cls(kind=StreamEventKind.REASONING_DELTA, text=text)

    @classmethod
    def tool_call_started(cls, tool_name: str) -> "StreamEvent":
        return cls(kind=StreamEventKind.TOOL_CALL_STARTED, tool_name=tool_name)

    @classmethod
    def usage_report(cls, usage: Usage) -> "StreamEvent":
        return cls(kind=StreamEventKind.USAGE, usage=usage)

    @classmethod
    def done(
        cls,
        *,
        finish_reason: str = "stop",
        tool_calls: Iterable[ToolCall] = (),
        usage: Usage | None = None,
        provider_data: dict[str, Any] | None = None,
    ) -> "StreamEvent":
        return cls(
            kind=StreamEventKind.DONE,
            finish_reason=finish_reason,
            tool_calls=tuple(tool_calls),
            usage=usage,
            provider_data=provider_data,
        )

    @classmethod
    def failure(cls, error: str) -> "StreamEvent":
        return cls(kind=StreamEventKind.ERROR, error=error)


class StreamAccumulator:
    """Fold :class:`StreamEvent` observations into a ``NormalizedResponse``.

    Reuses the existing normalized type rather than introducing a parallel one,
    so a streamed turn and a non-streamed turn hand downstream code the same
    object and no caller has to branch on how the response arrived.
    """

    def __init__(self, *, strict: bool = True) -> None:
        #: When strict, post-terminal events raise instead of being dropped.
        #: Tests and the kernel want the raise; a best-effort display consumer
        #: may prefer to swallow it.
        self._strict = strict
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._tool_names: list[str] = []
        self._tool_calls: tuple[ToolCall, ...] = ()
        self._usage: Usage | None = None
        self._finish_reason: str | None = None
        self._error: str | None = None
        self._terminal: StreamEventKind | None = None
        self._provider_data: dict[str, Any] | None = None

    # ── Accumulation ─────────────────────────────────────────────────────────

    def feed(self, event: StreamEvent) -> None:
        if self._terminal is not None:
            if self._strict:
                raise StreamProtocolError(
                    f"stream already ended with {self._terminal.value}; "
                    f"refusing trailing {event.kind.value}"
                )
            return

        if event.kind is StreamEventKind.TEXT_DELTA:
            if event.text:
                self._text_parts.append(event.text)
        elif event.kind is StreamEventKind.REASONING_DELTA:
            if event.text:
                self._reasoning_parts.append(event.text)
        elif event.kind is StreamEventKind.TOOL_CALL_STARTED:
            # v1 fires this once per tool name (see _fire_tool_gen_started);
            # keep that shape so a replayed stream matches a live one.
            name = event.tool_name or ""
            if name and name not in self._tool_names:
                self._tool_names.append(name)
        elif event.kind is StreamEventKind.USAGE:
            if event.usage is not None:
                self._usage = event.usage
        elif event.kind is StreamEventKind.DONE:
            # Left as-is (possibly None) so result() can infer a tool turn from
            # the calls present. Coercing to "stop" here would report a plain
            # stop for a provider that ended a tool turn without saying so.
            self._finish_reason = event.finish_reason
            if event.tool_calls:
                self._tool_calls = event.tool_calls
            if event.usage is not None:
                self._usage = event.usage
            self._terminal = StreamEventKind.DONE
            self._provider_data = event.provider_data
        elif event.kind is StreamEventKind.ERROR:
            self._error = event.error or "stream failed"
            self._terminal = StreamEventKind.ERROR

    def feed_all(self, events: Iterable[StreamEvent]) -> None:
        for event in events:
            self.feed(event)

    # ── Inspection ───────────────────────────────────────────────────────────

    @property
    def text(self) -> str:
        return "".join(self._text_parts)

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning_parts)

    @property
    def tool_names_seen(self) -> tuple[str, ...]:
        return tuple(self._tool_names)

    @property
    def ended(self) -> bool:
        return self._terminal is not None

    @property
    def failed(self) -> bool:
        return self._terminal is StreamEventKind.ERROR

    @property
    def error(self) -> str | None:
        return self._error

    # ── Result ───────────────────────────────────────────────────────────────

    def result(self) -> NormalizedResponse:
        """Return the folded response.

        Callable before the stream ends — an interrupted turn still needs the
        partial text it already showed the user.  ``finish_reason`` is only
        authoritative once :attr:`ended` is true.
        """
        if self.failed:
            raise StreamProtocolError(self._error or "stream failed")

        text = self.text
        reasoning = self.reasoning
        return NormalizedResponse(
            content=text or None,
            tool_calls=list(self._tool_calls) or None,
            finish_reason=self._finish_reason
            or ("tool_calls" if self._tool_calls else "stop"),
            reasoning=reasoning or None,
            usage=self._usage,
            provider_data=self._provider_data,
        )


def fold_stream(
    events: Iterable[StreamEvent], *, strict: bool = True
) -> NormalizedResponse:
    """Fold a complete stream into a ``NormalizedResponse``."""
    acc = StreamAccumulator(strict=strict)
    acc.feed_all(events)
    return acc.result()


class CallbackStreamSource:
    """Adapt v1's ``_fire_*`` callback style into :class:`StreamEvent`s.

    The v1 streaming helper pushes text, reasoning, and tool-start
    notifications out through three separate agent callbacks.  A consumer that
    wants the normalized vocabulary has two options: rewrite that helper (which
    touches the hottest path in the product), or observe it.  This observes it.

    Wire the ``on_*`` methods as additional callbacks — or let
    ``agent.kernel.v1_adapter`` do it — then read :meth:`events`.
    """

    def __init__(self, *, sink: Callable[[StreamEvent], None] | None = None) -> None:
        self._events: list[StreamEvent] = []
        self._sink = sink
        self._ended = False

    def _emit(self, event: StreamEvent) -> None:
        if self._ended:
            # Post-terminal callbacks are exactly what the v1 single-writer
            # guard fences out; record nothing rather than corrupt the fold.
            return
        if event.is_terminal:
            self._ended = True
        self._events.append(event)
        if self._sink is not None:
            self._sink(event)

    # ── v1 callback shapes ───────────────────────────────────────────────────

    def on_text(self, text: str) -> None:
        if text:
            self._emit(StreamEvent.text_delta(text))

    def on_reasoning(self, text: str) -> None:
        if text:
            self._emit(StreamEvent.reasoning_delta(text))

    def on_tool_start(self, tool_name: str) -> None:
        if tool_name:
            self._emit(StreamEvent.tool_call_started(tool_name))

    def finish(
        self,
        *,
        finish_reason: str = "stop",
        tool_calls: Iterable[ToolCall] = (),
        usage: Usage | None = None,
    ) -> None:
        self._emit(
            StreamEvent.done(
                finish_reason=finish_reason, tool_calls=tool_calls, usage=usage
            )
        )

    def fail(self, error: str) -> None:
        self._emit(StreamEvent.failure(error))

    # ── Reading ──────────────────────────────────────────────────────────────

    def events(self) -> tuple[StreamEvent, ...]:
        return tuple(self._events)

    def __iter__(self) -> Iterator[StreamEvent]:
        return iter(tuple(self._events))

    @property
    def ended(self) -> bool:
        return self._ended

    def fold(self, *, strict: bool = True) -> NormalizedResponse:
        return fold_stream(self._events, strict=strict)


def stream_from_normalized(response: NormalizedResponse) -> tuple[StreamEvent, ...]:
    """Render a non-streamed response as an equivalent stream.

    Lets a caller hold one code path for both modes: a provider that cannot
    stream (or a mode downgraded to non-streaming, as Bedrock does when IAM
    withholds the streaming action) still yields the same event vocabulary.
    """
    events: list[StreamEvent] = []
    if response.reasoning:
        events.append(StreamEvent.reasoning_delta(response.reasoning))
    for call in response.tool_calls or ():
        events.append(StreamEvent.tool_call_started(call.name))
    if response.content:
        events.append(StreamEvent.text_delta(response.content))
    events.append(
        StreamEvent.done(
            finish_reason=response.finish_reason or "stop",
            tool_calls=tuple(response.tool_calls or ()),
            usage=response.usage,
            provider_data=response.provider_data,
        )
    )
    return tuple(events)
