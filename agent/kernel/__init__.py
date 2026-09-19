"""Hairball Kernel v2 — the owned agent core.

This package is the second core path described in
``docs/design/2026-07-29-hairball-kernel-v2-core-architecture-and-migration-roadmap.md``.
It is built beside the existing v1 conversation loop, not on top of it: v1 keeps
running unchanged while the kernel records what it does as a canonical event
stream, so v1 and v2 can be compared on real traffic before anything switches
over.

Current state:

- ``contracts``        — stable IDs, canonical events, turn requests, capabilities
- ``event_ledger``     — append-only durable log on the existing ``state.db``
- ``projectors``       — chat / workspace / sidebar / agent-tree / metrics read models
- ``v1_adapter``       — emits canonical events from v1's callbacks, behavior-neutral
- ``facade``           — one entry point, session-pinned kernel version routing
- ``provider_runtime`` — transport-driven provider seam, capability-gated
- ``turn_loop``        — three-state step machine with guaranteed tool settlement
- ``engine``           — the v2 execution path: drives the loop against real
                         providers and real tool dispatch

The provider seam borrows rather than forks: retry, credential pooling, client
construction, and interrupt handling stay in v1 and arrive as injected
callables, because those are the parts that took real incidents to get right.
v1's own streaming helper is untouched — the normalized stream vocabulary in
``agent/transports/stream.py`` observes it instead of replacing it.

The turn loop follows the same rule in the other direction: compaction policy,
tool guardrails, and the choice of tool-concurrency strategy stay with
Hairball's existing implementations because current product evidence shows
those seams are load-bearing. See
``docs/design/2026-08-04-hairball-independent-core-knowledge.md`` section 3.
The loop only owns loop control.

``engine`` deliberately borrows v1's system prompt and tool schemas rather than
building its own, so a v1-vs-v2 comparison varies only loop control. Replacing
the legacy-derived prompt builder is a separate, later experiment; changing both
at once would make any measured difference unattributable. See
``tests/kernel/test_dual_path_equivalence.py`` and ``scripts/kernel_ab.py``.

Not built yet, deliberately (speculative infrastructure with no consumer is
worse than none): the tool runtime / capability catalog and the context runtime.

``facade`` routes ``default_version: v2`` to ``KernelEngine`` today, but product
surfaces must keep ``default_version: v1`` until the engine streams
(``ProviderRuntime.stream`` + ``ASSISTANT_MESSAGE_DELTA``). hairball-ui
``chat_bridge`` already calls ``get_kernel().run_turn`` (UPG-A2 Phase1): with
``kernel.enabled: false`` that is a passthrough; with enabled + v1 it observes
via ``V1EventTap`` without breaking live deltas.

Everything here is inert for ledger/observe until ``kernel.enabled`` is true
in ``config.yaml``.
"""

from agent.kernel.contracts import (
    EVENT_SCHEMA_VERSION,
    KERNEL_PROTOCOL_VERSION,
    KERNEL_V1,
    KERNEL_V2,
    TERMINAL_TOOL_KINDS,
    TERMINAL_TURN_KINDS,
    ArtifactRef,
    Budget,
    CapabilityToken,
    ControlCommand,
    ControlKind,
    EventKind,
    InvariantViolation,
    KernelError,
    KernelEvent,
    LedgerUnavailable,
    ToolCallState,
    ToolOutcome,
    TurnRequest,
    TurnState,
    new_agent_id,
    new_artifact_id,
    new_attempt_id,
    new_branch_id,
    new_event_id,
    new_turn_id,
)
from agent.kernel.engine import DEFAULT_API_MODE, EngineResult, KernelEngine
from agent.kernel.event_ledger import EventLedger, InMemoryLedger, row_to_event
from agent.kernel.facade import Kernel, KernelSettings, TurnHandle, get_kernel
from agent.kernel.provider_runtime import (
    ProviderRequest,
    ProviderResult,
    ProviderRuntime,
    ProviderRuntimeError,
    UnsupportedApiModeError,
)
from agent.kernel.projectors import (
    AgentTreeProjector,
    ChatProjector,
    InvariantReport,
    MetricsProjector,
    Projector,
    SidebarProjector,
    WorkspaceProjector,
    check_invariants,
    replay,
)
from agent.kernel.turn_loop import (
    ABORTED_TOOL_TEXT,
    DOOM_LOOP_THRESHOLD,
    CompactionCheck,
    DoomLoopDetector,
    PendingInput,
    PendingInputQueue,
    RetryReport,
    SnapshotSource,
    StepReport,
    StopReason,
    TurnDecision,
    TurnLoop,
    TurnUsage,
    orphan_tool_results,
    tool_call_signature,
)
from agent.kernel.v1_adapter import V1EventTap, instrument_agent

__all__ = [
    "ABORTED_TOOL_TEXT",
    "DOOM_LOOP_THRESHOLD",
    "EVENT_SCHEMA_VERSION",
    "KERNEL_PROTOCOL_VERSION",
    "KERNEL_V1",
    "KERNEL_V2",
    "TERMINAL_TOOL_KINDS",
    "TERMINAL_TURN_KINDS",
    "AgentTreeProjector",
    "ArtifactRef",
    "Budget",
    "CapabilityToken",
    "ChatProjector",
    "CompactionCheck",
    "ControlCommand",
    "ControlKind",
    "DoomLoopDetector",
    "EventKind",
    "EventLedger",
    "InMemoryLedger",
    "InvariantReport",
    "InvariantViolation",
    "Kernel",
    "KernelError",
    "KernelEvent",
    "KernelSettings",
    "LedgerUnavailable",
    "MetricsProjector",
    "PendingInput",
    "PendingInputQueue",
    "Projector",
    "ProviderRequest",
    "ProviderResult",
    "ProviderRuntime",
    "ProviderRuntimeError",
    "RetryReport",
    "SidebarProjector",
    "SnapshotSource",
    "StepReport",
    "StopReason",
    "ToolCallState",
    "ToolOutcome",
    "TurnDecision",
    "TurnHandle",
    "TurnLoop",
    "TurnRequest",
    "TurnState",
    "TurnUsage",
    "UnsupportedApiModeError",
    "V1EventTap",
    # engine
    "KernelEngine",
    "EngineResult",
    "DEFAULT_API_MODE",
    "WorkspaceProjector",
    "check_invariants",
    "get_kernel",
    "instrument_agent",
    "new_agent_id",
    "new_artifact_id",
    "new_attempt_id",
    "new_branch_id",
    "new_event_id",
    "new_turn_id",
    "orphan_tool_results",
    "replay",
    "row_to_event",
    "tool_call_signature",
]
