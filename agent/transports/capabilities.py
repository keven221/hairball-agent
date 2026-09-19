"""Per-``api_mode`` capability matrix for the transport layer.

Every provider path in Hairball differs in what it can actually deliver: some
report cache stats, some stream reasoning, one streams tool-call starts and
another silently does not.  Today those differences are only discoverable by
reading ``chat_completion_helpers.interruptible_streaming_api_call`` and each
adapter, which means callers either over-assume (and lose events) or
under-assume (and hide capability the provider has).

This module states the differences once, as data, with a source citation per
row.  ``evidence`` is part of the contract: a field without a citation is a
guess, and a guess in a capability matrix is worse than no matrix at all.

Deliberately *not* here: retry policy, credential rotation, prompt-cache
control, and interrupt handling.  Those are owned by ``AIAgent`` (see
``agent/transports/base.py`` — same boundary), so declaring them here would
invite a second, diverging source of truth.

Import discipline: ``agent.transports`` is imported lazily inside functions,
never at module scope.  The registry populates itself through
``_discover_transports()`` on first miss, and importing it eagerly from here
would perturb that discovery order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Known api_modes ───────────────────────────────────────────────────────────

API_MODE_CHAT_COMPLETIONS = "chat_completions"
API_MODE_ANTHROPIC_MESSAGES = "anthropic_messages"
API_MODE_CODEX_RESPONSES = "codex_responses"
API_MODE_BEDROCK_CONVERSE = "bedrock_converse"
API_MODE_CODEX_APP_SERVER = "codex_app_server"


@dataclass(frozen=True)
class TransportCapabilities:
    """What one ``api_mode`` can deliver, and the source that proves it.

    ``has_transport`` is the gate every other field hangs off: a mode without a
    registered :class:`~agent.transports.base.ProviderTransport` has no
    ``build_kwargs`` / ``normalize_response`` seam, so a caller that wants to
    drive it must go through that mode's bespoke runtime instead.
    """

    api_mode: str
    has_transport: bool

    # Streaming — which of the three delta channels the mode actually fires.
    # A False here means the surface will never see that channel for this
    # provider, not that the provider protocol lacks it.
    streams_text: bool = False
    streams_reasoning: bool = False
    streams_tool_start: bool = False

    # Response normalization — which optional ProviderTransport hooks the
    # transport overrides.  Unoverridden hooks fall back to the ABC defaults
    # (no cache stats, passthrough finish reason, always-valid).
    reports_cache_stats: bool = False
    maps_finish_reason: bool = False
    validates_response: bool = False

    # Reasoning that must be replayed verbatim on the next request or the
    # provider rejects it.  Drives whether a context rewrite may drop or
    # reorder assistant reasoning.
    carries_signed_reasoning: bool = False

    evidence: tuple[str, ...] = ()
    notes: str = ""

    def supports_stream_channel(self, channel: str) -> bool:
        """Return whether *channel* (``text`` / ``reasoning`` / ``tool_start``) streams."""
        return {
            "text": self.streams_text,
            "reasoning": self.streams_reasoning,
            "tool_start": self.streams_tool_start,
        }.get(channel, False)

    @property
    def streams_anything(self) -> bool:
        return self.streams_text or self.streams_reasoning or self.streams_tool_start

    def to_json(self) -> dict[str, Any]:
        return {
            "api_mode": self.api_mode,
            "has_transport": self.has_transport,
            "streams_text": self.streams_text,
            "streams_reasoning": self.streams_reasoning,
            "streams_tool_start": self.streams_tool_start,
            "reports_cache_stats": self.reports_cache_stats,
            "maps_finish_reason": self.maps_finish_reason,
            "validates_response": self.validates_response,
            "carries_signed_reasoning": self.carries_signed_reasoning,
            "notes": self.notes,
        }


# ── The matrix ────────────────────────────────────────────────────────────────
#
# Line numbers in ``evidence`` are load-bearing documentation, not decoration:
# they are how the next reader re-verifies a row instead of trusting it.  They
# drift as files change; the contract tests re-derive the *behavioral* claims
# (registry membership, which optional hooks are overridden) from live objects,
# so a stale citation cannot silently become a wrong capability.

CAPABILITIES: dict[str, TransportCapabilities] = {
    API_MODE_CHAT_COMPLETIONS: TransportCapabilities(
        api_mode=API_MODE_CHAT_COMPLETIONS,
        has_transport=True,
        streams_text=True,
        streams_reasoning=True,
        streams_tool_start=True,
        reports_cache_stats=True,
        maps_finish_reason=False,
        validates_response=True,
        carries_signed_reasoning=True,
        evidence=(
            "agent/transports/chat_completions.py:796 register_transport",
            "agent/transports/chat_completions.py:778 extract_cache_stats override",
            "agent/transports/chat_completions.py:768 validate_response override",
            "agent/chat_completion_helpers.py:2324,2409 _fire_reasoning_delta",
            "agent/chat_completion_helpers.py:2500 _fire_tool_gen_started",
            "agent/transports/types.py:64-76 Gemini thought_signature replay",
        ),
        notes=(
            "Native OpenAI finish-reason vocabulary, so no mapping is needed. "
            "Gemini 3 thinking models attach a thought_signature per tool call "
            "that must be replayed or the API returns HTTP 400."
        ),
    ),
    API_MODE_ANTHROPIC_MESSAGES: TransportCapabilities(
        api_mode=API_MODE_ANTHROPIC_MESSAGES,
        has_transport=True,
        streams_text=True,
        streams_reasoning=True,
        streams_tool_start=True,
        reports_cache_stats=True,
        maps_finish_reason=True,
        validates_response=True,
        carries_signed_reasoning=True,
        evidence=(
            "agent/transports/anthropic.py:251 register_transport",
            "agent/transports/anthropic.py:222 extract_cache_stats override",
            "agent/transports/anthropic.py:243 map_finish_reason override",
            "agent/transports/anthropic.py:194 validate_response override",
            "agent/chat_completion_helpers.py:2738 _fire_tool_gen_started",
            "agent/chat_completion_helpers.py:2754 _fire_reasoning_delta",
            "agent/transports/types.py:124-134 anthropic_content_blocks",
        ),
        notes=(
            "Interleaved signed thinking + tool_use must be replayed as verbatim "
            "ordered content blocks; reconstructing from parallel "
            "reasoning_details/tool_calls lists invalidates the signatures."
        ),
    ),
    API_MODE_CODEX_RESPONSES: TransportCapabilities(
        api_mode=API_MODE_CODEX_RESPONSES,
        has_transport=True,
        streams_text=True,
        streams_reasoning=True,
        streams_tool_start=False,
        reports_cache_stats=False,
        maps_finish_reason=True,
        validates_response=True,
        carries_signed_reasoning=True,
        evidence=(
            "agent/transports/codex.py:487 register_transport",
            "agent/transports/codex.py:467 map_finish_reason override",
            "agent/transports/codex.py:435 validate_response override",
            "agent/codex_runtime.py:901 _run_codex_stream",
            "agent/codex_runtime.py:919 _fire_reasoning_delta",
            "agent/transports/types.py:136-144 codex_reasoning_items",
        ),
        notes=(
            "Streams internally through _run_codex_stream rather than the shared "
            "streaming helper, and never fires tool-generation-started — a "
            "surface waiting on that channel will show no tool spinner here. "
            "Reasoning items are replayed by id, so they must survive context "
            "rewrites intact."
        ),
    ),
    API_MODE_BEDROCK_CONVERSE: TransportCapabilities(
        api_mode=API_MODE_BEDROCK_CONVERSE,
        has_transport=True,
        streams_text=True,
        streams_reasoning=True,
        streams_tool_start=True,
        reports_cache_stats=False,
        maps_finish_reason=True,
        validates_response=True,
        carries_signed_reasoning=False,
        evidence=(
            "agent/transports/bedrock.py:154 register_transport",
            "agent/transports/bedrock.py:134 map_finish_reason override",
            "agent/transports/bedrock.py:118 validate_response override",
            "agent/chat_completion_helpers.py:2050 client.converse_stream",
            "agent/chat_completion_helpers.py:2092 _fire_tool_gen_started",
            "agent/chat_completion_helpers.py:2096 _fire_reasoning_delta",
        ),
        notes=(
            "Streaming needs bedrock:InvokeModelWithResponseStream; when IAM "
            "grants only InvokeModel the helper permanently downgrades this "
            "session to non-streaming converse(), so streams_* describes the "
            "granted-permission case."
        ),
    ),
    API_MODE_CODEX_APP_SERVER: TransportCapabilities(
        api_mode=API_MODE_CODEX_APP_SERVER,
        has_transport=False,
        evidence=(
            "agent/transports/codex_app_server.py:54 CodexAppServerClient",
            "agent/transports/codex_event_projector.py projects app-server events",
        ),
        notes=(
            "Not a ProviderTransport: this mode speaks an app-server session "
            "protocol and turns its own event stream into agent state through "
            "codex_event_projector, so there is no build_kwargs/normalize_response "
            "seam to drive. Callers must route it to its own runtime."
        ),
    ),
}


def known_api_modes() -> tuple[str, ...]:
    """Return every api_mode the matrix describes, in declaration order."""
    return tuple(CAPABILITIES)


def get_capabilities(api_mode: str | None) -> TransportCapabilities:
    """Return capabilities for *api_mode*, or an all-False record when unknown.

    Unknown modes degrade instead of raising: a caller asking "can this stream
    reasoning?" about a mode we've never heard of should get "no", not a
    ``KeyError`` from deep inside a turn.  Use :func:`is_known_api_mode` when
    the distinction between "unknown" and "known but unsupported" matters.
    """
    if api_mode and api_mode in CAPABILITIES:
        return CAPABILITIES[api_mode]
    return TransportCapabilities(
        api_mode=api_mode or "",
        has_transport=False,
        notes="Unknown api_mode — no capabilities are assumed.",
    )


def is_known_api_mode(api_mode: str | None) -> bool:
    return bool(api_mode) and api_mode in CAPABILITIES


def modes_with_transport() -> tuple[str, ...]:
    """Return the api_modes the matrix claims have a registered transport."""
    return tuple(mode for mode, cap in CAPABILITIES.items() if cap.has_transport)


def resolve_transport(api_mode: str | None):
    """Return the live transport for *api_mode*, or ``None``.

    Thin wrapper over ``agent.transports.get_transport`` that keeps the import
    lazy so importing this module never triggers transport discovery.
    """
    if not api_mode:
        return None
    from agent.transports import get_transport

    return get_transport(api_mode)


def stream_channels(api_mode: str | None) -> tuple[str, ...]:
    """Return the delta channels *api_mode* actually fires."""
    cap = get_capabilities(api_mode)
    return tuple(
        channel
        for channel in ("text", "reasoning", "tool_start")
        if cap.supports_stream_channel(channel)
    )


def describe_matrix() -> list[dict[str, Any]]:
    """Return the matrix as plain dicts, for diagnostics and docs generation."""
    return [cap.to_json() for cap in CAPABILITIES.values()]
