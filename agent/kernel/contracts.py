"""Kernel v2 wire contracts: stable IDs, canonical events, turn requests.

This module is the *narrow waist* of the Hairball kernel. Everything that
crosses a kernel boundary — product surfaces, the event ledger, projectors,
provider/tool runtimes — speaks the types defined here and nothing else.

Design constraints that shaped this file:

- **Serializable and versioned.** Every event carries ``schema_version`` so a
  ledger written by a newer Hairball stays readable by an older projector.
- **No runtime imports.** Contracts must be importable without pulling in
  SQLite, providers, tools, or the conversation loop, so that pure state-machine
  and projector tests stay fast and hermetic.
- **Frozen.** Events and requests are immutable once minted; mutable execution
  state belongs in the runtime layer, not on the wire.

Source adoption: the event-union shape (one flat discriminated set of kinds,
tool events always carrying a stable call id) is ADAPTed from Pi's
``AgentEvent`` (``packages/agent/src/types.ts``, MIT, commit ``dd6bea41``) and
the turn/session id layering from ChatGPT ``codex-rs/core/src/session``
(Apache-2.0). Naming, payload shapes, and the kind taxonomy are Hairball's.
See ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any, Mapping, Optional

# ── Versioning ────────────────────────────────────────────────────────────────

# Bump when the *serialized* shape of KernelEvent changes incompatibly.
# Additive payload keys do NOT require a bump — projectors must tolerate
# unknown payload keys by construction.
EVENT_SCHEMA_VERSION = 1

# Bump when the kernel's own execution semantics change such that a session
# started under an older kernel must not be resumed by a newer one.
KERNEL_PROTOCOL_VERSION = 1

KERNEL_V1 = "v1"
KERNEL_V2 = "v2"


# ── Stable IDs ────────────────────────────────────────────────────────────────

_ID_LOCK = threading.Lock()
_LAST_MS = 0
_SEQ_IN_MS = 0

# Time-ordered id prefixes keep ledger rows sortable by id even when several
# events land inside the same millisecond, and make raw SQL dumps readable.
_ID_PREFIXES = {
    "event": "ev",
    "turn": "tn",
    "agent": "ag",
    "branch": "br",
    "attempt": "at",
    "artifact": "af",
    "workspace": "ws",
    "tool_call": "tc",
    "api_request": "rq",
    "session": "se",
}


def _time_ordered_suffix() -> str:
    """Return a lexicographically sortable ``<ms>-<seq>-<rand>`` suffix.

    Adapted from the uuidv7 property Pi relies on (timestamp-derived prefix,
    random tail) but rendered as base-36 so ids stay short in log lines and
    SQLite indexes. The in-millisecond counter guarantees strict ordering for
    ids minted back-to-back on one process, which matters because callers use
    id order as a tiebreaker when two events share a ledger timestamp.
    """
    global _LAST_MS, _SEQ_IN_MS
    with _ID_LOCK:
        now_ms = int(time.time() * 1000)
        if now_ms == _LAST_MS:
            _SEQ_IN_MS += 1
        elif now_ms > _LAST_MS:
            _LAST_MS = now_ms
            _SEQ_IN_MS = 0
        else:
            # Clock moved backwards (NTP step, suspend/resume). Keep minting
            # forward-ordered ids rather than emitting a duplicate-sorting id.
            _SEQ_IN_MS += 1
            now_ms = _LAST_MS
        seq = _SEQ_IN_MS
    rand = secrets.token_hex(4)
    return f"{now_ms:09x}{seq:03x}{rand}"


def new_id(kind: str) -> str:
    """Mint a new time-ordered id for *kind* (``"event"``, ``"turn"``, ...)."""
    prefix = _ID_PREFIXES.get(kind)
    if prefix is None:
        raise ValueError(f"unknown kernel id kind: {kind!r}")
    return f"{prefix}_{_time_ordered_suffix()}"


def new_event_id() -> str:
    return new_id("event")


def new_turn_id() -> str:
    return new_id("turn")


def new_agent_id() -> str:
    return new_id("agent")


def new_branch_id() -> str:
    return new_id("branch")


def new_attempt_id() -> str:
    return new_id("attempt")


def new_artifact_id() -> str:
    return new_id("artifact")


def synthetic_tool_call_id() -> str:
    """Mint a kernel-owned id for a provider that omitted ``tool_call_id``.

    The kernel never rewrites a provider-supplied call id — it records a
    mapping instead — so the synthetic prefix stays visually distinct.
    """
    return new_id("tool_call")


# ── Event kinds ───────────────────────────────────────────────────────────────


class EventKind(StrEnum):
    """The canonical kernel event vocabulary.

    Dotted names are intentional: projectors and surfaces filter on prefixes
    (``tool.call.``, ``agent.``) rather than maintaining their own mapping
    tables.
    """

    SESSION_CREATED = "session.created"

    TURN_ACCEPTED = "turn.accepted"
    TURN_STARTED = "turn.started"
    TURN_STEP = "turn.step"
    TURN_STEP_COMPLETED = "turn.step.completed"
    TURN_RETRY_SCHEDULED = "turn.retry_scheduled"
    TURN_COMPLETED = "turn.completed"
    TURN_FAILED = "turn.failed"
    TURN_CANCELLED = "turn.cancelled"

    PROVIDER_REQUEST_STARTED = "provider.request.started"
    PROVIDER_REQUEST_COMPLETED = "provider.request.completed"
    PROVIDER_WARNING = "provider.warning"

    ASSISTANT_REASONING_STARTED = "assistant.reasoning.started"
    ASSISTANT_REASONING_DELTA = "assistant.reasoning.delta"
    ASSISTANT_REASONING_COMPLETED = "assistant.reasoning.completed"
    ASSISTANT_MESSAGE_STARTED = "assistant.message.started"
    ASSISTANT_MESSAGE_DELTA = "assistant.message.delta"
    ASSISTANT_MESSAGE_COMPLETED = "assistant.message.completed"

    # The input phase is separate from TOOL_CALL_PROPOSED because a streaming
    # provider reveals the tool name before its arguments finish arriving.
    # Surfaces that want "running `read_file`…" while the path is still
    # streaming need that earlier boundary; PROPOSED stays the point at which
    # the arguments are complete and the call is executable.
    TOOL_CALL_INPUT_STARTED = "tool.call.input.started"
    TOOL_CALL_INPUT_DELTA = "tool.call.input.delta"
    TOOL_CALL_PROPOSED = "tool.call.proposed"
    TOOL_CALL_APPROVAL_REQUIRED = "tool.call.approval_required"
    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_PROGRESS = "tool.call.progress"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"
    TOOL_CALL_CANCELLED = "tool.call.cancelled"

    ARTIFACT_CREATED = "artifact.created"
    WORKSPACE_PATCH = "workspace.patch"

    # Plan is a durable artifact whose lifecycle is projected across CLI, TUI,
    # gateway, and desktop processes.  The snapshot remains a projection; the
    # Markdown artifact is the execution source of truth.
    PLAN_SNAPSHOT = "plan.snapshot.v1"

    CONTEXT_COMPACTION_STARTED = "context.compaction.started"
    CONTEXT_COMPACTION_COMPLETED = "context.compaction.completed"

    AGENT_SPAWNED = "agent.spawned"
    AGENT_INBOX_SPLICED = "agent.inbox.spliced"
    AGENT_MESSAGE_ENQUEUED = "agent.message.enqueued"
    AGENT_MESSAGE_DELIVERED = "agent.message.delivered"
    AGENT_COMPLETED = "agent.completed"

    STATUS_NOTE = "status.note"


#: A turn ends in exactly one of these. Enforced by ledger invariant checks.
TERMINAL_TURN_KINDS: frozenset[EventKind] = frozenset(
    {
        EventKind.TURN_COMPLETED,
        EventKind.TURN_FAILED,
        EventKind.TURN_CANCELLED,
    }
)

#: A started tool call ends in exactly one of these.
TERMINAL_TOOL_KINDS: frozenset[EventKind] = frozenset(
    {
        EventKind.TOOL_CALL_COMPLETED,
        EventKind.TOOL_CALL_FAILED,
        EventKind.TOOL_CALL_CANCELLED,
    }
)

#: Kinds whose payload must carry a ``tool_call_id``.
TOOL_SCOPED_KINDS: frozenset[EventKind] = frozenset(
    {
        EventKind.TOOL_CALL_INPUT_STARTED,
        EventKind.TOOL_CALL_INPUT_DELTA,
        EventKind.TOOL_CALL_PROPOSED,
        EventKind.TOOL_CALL_APPROVAL_REQUIRED,
        EventKind.TOOL_CALL_STARTED,
        EventKind.TOOL_CALL_PROGRESS,
        EventKind.TOOL_CALL_COMPLETED,
        EventKind.TOOL_CALL_FAILED,
        EventKind.TOOL_CALL_CANCELLED,
    }
)

#: High-frequency streaming kinds. Durable ledgers may drop these (they are
#: notification, not truth — the completed message carries the full text), so
#: no projector may treat them as the sole source of a durable fact.
EPHEMERAL_KINDS: frozenset[EventKind] = frozenset(
    {
        EventKind.ASSISTANT_MESSAGE_DELTA,
        EventKind.ASSISTANT_REASONING_DELTA,
        EventKind.TOOL_CALL_INPUT_DELTA,
        EventKind.TOOL_CALL_PROGRESS,
    }
)


class ToolCallState(StrEnum):
    """Lifecycle position of a single tool call, as projected from events."""

    # The tool name is known but its arguments are still streaming, so the call
    # is not yet executable. Surfaces use this to show "running `read_file`…"
    # before the path has finished arriving.
    PENDING = "pending"
    PROPOSED = "proposed"
    APPROVAL_REQUIRED = "approval_required"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TurnState(StrEnum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TOOL_KIND_TO_STATE: dict[EventKind, ToolCallState] = {
    EventKind.TOOL_CALL_INPUT_STARTED: ToolCallState.PENDING,
    EventKind.TOOL_CALL_PROPOSED: ToolCallState.PROPOSED,
    EventKind.TOOL_CALL_APPROVAL_REQUIRED: ToolCallState.APPROVAL_REQUIRED,
    EventKind.TOOL_CALL_STARTED: ToolCallState.RUNNING,
    EventKind.TOOL_CALL_COMPLETED: ToolCallState.COMPLETED,
    EventKind.TOOL_CALL_FAILED: ToolCallState.FAILED,
    EventKind.TOOL_CALL_CANCELLED: ToolCallState.CANCELLED,
}


def tool_state_for_kind(kind: EventKind | str) -> Optional[ToolCallState]:
    """Map a tool event kind to the state it puts the call in.

    Returns ``None`` for the streaming kinds — progress and input deltas report
    movement inside a state rather than a transition between states — and for
    non-tool kinds.
    """
    try:
        return _TOOL_KIND_TO_STATE.get(EventKind(kind))
    except ValueError:
        return None


# ── Capability tokens ─────────────────────────────────────────────────────────

#: The capability vocabulary. Subagents receive a token whose grants are a
#: subset of what the parent is allowed to delegate — never a superset.
CAPABILITY_WORKSPACE_READ = "workspace.read"
CAPABILITY_WORKSPACE_WRITE = "workspace.write"
CAPABILITY_PROCESS_SPAWN = "process.spawn"
CAPABILITY_PROCESS_SIGNAL = "process.signal"
CAPABILITY_NETWORK_CLIENT = "network.client"
CAPABILITY_BROWSER_CONTROL = "browser.control"
CAPABILITY_SESSION_SPAWN = "session.spawn"
CAPABILITY_SESSION_MESSAGE = "session.message"
CAPABILITY_SESSION_INTERRUPT = "session.interrupt"
CAPABILITY_MODEL_SMALL = "model.small"
CAPABILITY_MODEL_MAIN = "model.main"

#: Scoped capability prefixes: ``secret.read:<scope>``, ``external.mutate:<svc>``.
CAPABILITY_SECRET_READ_PREFIX = "secret.read:"
CAPABILITY_EXTERNAL_MUTATE_PREFIX = "external.mutate:"


@dataclass(frozen=True)
class CapabilityToken:
    """An immutable grant set carried by a turn or an agent run.

    ``delegable`` is the subset a holder may hand to a child. It defaults to
    ``grants`` minus the session-control capabilities, so a leaf worker cannot
    silently gain the ability to spawn its own fleet.
    """

    grants: frozenset[str] = frozenset()
    delegable: frozenset[str] = frozenset()
    issued_to: str = ""

    @classmethod
    def of(
        cls,
        *grants: str,
        delegable: Optional[frozenset[str] | set[str]] = None,
        issued_to: str = "",
    ) -> "CapabilityToken":
        grant_set = frozenset(grants)
        if delegable is None:
            delegable_set = grant_set - {
                CAPABILITY_SESSION_SPAWN,
                CAPABILITY_SESSION_INTERRUPT,
            }
        else:
            delegable_set = frozenset(delegable)
        return cls(grants=grant_set, delegable=delegable_set, issued_to=issued_to)

    def allows(self, capability: str) -> bool:
        return capability in self.grants

    def is_subset_of(self, other: "CapabilityToken") -> bool:
        """True when this token grants nothing *other* cannot delegate."""
        return self.grants <= other.delegable

    def narrow(
        self,
        *,
        keep: Optional[frozenset[str] | set[str]] = None,
        issued_to: str = "",
    ) -> "CapabilityToken":
        """Derive a child token that can only ever be narrower than this one.

        The child's grants are intersected with this token's ``delegable`` set,
        so a caller passing an over-broad ``keep`` gets a clamped token instead
        of a privilege escalation.
        """
        requested = self.delegable if keep is None else frozenset(keep)
        granted = requested & self.delegable
        return CapabilityToken(
            grants=granted,
            delegable=granted
            - {CAPABILITY_SESSION_SPAWN, CAPABILITY_SESSION_INTERRUPT},
            issued_to=issued_to or self.issued_to,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "grants": sorted(self.grants),
            "delegable": sorted(self.delegable),
            "issued_to": self.issued_to,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any] | None) -> "CapabilityToken":
        if not raw:
            return cls()
        return cls(
            grants=frozenset(raw.get("grants") or ()),
            delegable=frozenset(raw.get("delegable") or ()),
            issued_to=str(raw.get("issued_to") or ""),
        )


# ── Budget ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Budget:
    """Per-turn ceilings. ``0``/``None`` means "inherit the runtime default"."""

    max_iterations: int = 0
    max_input_tokens: int = 0
    max_output_tokens: int = 0
    max_wall_seconds: float = 0.0
    max_cost_usd: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: Mapping[str, Any] | None) -> "Budget":
        if not raw:
            return cls()
        return cls(
            max_iterations=int(raw.get("max_iterations") or 0),
            max_input_tokens=int(raw.get("max_input_tokens") or 0),
            max_output_tokens=int(raw.get("max_output_tokens") or 0),
            max_wall_seconds=float(raw.get("max_wall_seconds") or 0.0),
            max_cost_usd=float(raw.get("max_cost_usd") or 0.0),
        )


# ── Turn request ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TurnRequest:
    """Everything the kernel needs to run one turn, fixed at accept time.

    ``prompt_epoch`` and ``tool_schema_epoch`` are the prompt-cache contract:
    while they are unchanged, the kernel guarantees a byte-stable system prefix
    and a stable core tool schema set. Changing either requires a new branch,
    never an in-place mutation of a live session.
    """

    session_id: str
    turn_id: str
    user_input: Any
    branch_id: str = ""
    agent_id: str = ""
    provider_ref: str = ""
    model_ref: str = ""
    prompt_epoch: str = ""
    tool_schema_epoch: str = ""
    capability_token: CapabilityToken = field(default_factory=CapabilityToken)
    workspace_ref: Optional[str] = None
    budget: Budget = field(default_factory=Budget)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        session_id: str,
        user_input: Any,
        **kwargs: Any,
    ) -> "TurnRequest":
        kwargs.setdefault("turn_id", new_turn_id())
        kwargs.setdefault("branch_id", "main")
        return cls(session_id=session_id, user_input=user_input, **kwargs)

    def with_metadata(self, **extra: Any) -> "TurnRequest":
        merged = dict(self.metadata)
        merged.update(extra)
        return replace(self, metadata=merged)


# ── Control commands ──────────────────────────────────────────────────────────


class ControlKind(StrEnum):
    """Out-of-band commands a surface can send into a running turn.

    ``STEER`` is supplementary input for the turn already executing; ``FOLLOW_UP``
    is the next input, delivered only after the current turn settles. Collapsing
    the two is the classic way to end up faking a user message mid tool loop, so
    the kernel keeps them as distinct commands all the way down.
    """

    INTERRUPT = "interrupt"
    CANCEL = "cancel"
    STEER = "steer"
    FOLLOW_UP = "follow_up"
    APPROVE = "approve"
    DENY = "deny"


@dataclass(frozen=True)
class ControlCommand:
    kind: ControlKind
    session_id: str
    turn_id: str = ""
    tool_call_id: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


# ── The event ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KernelEvent:
    """One durable fact about a session.

    ``seq`` is assigned by the ledger on append (monotonic per session, never
    reused, never rewritten). An event built in memory before append carries
    ``seq=0``; treat any event with ``seq == 0`` as not-yet-durable.

    ``causation_id`` points at the event that directly caused this one;
    ``correlation_id`` groups a whole logical operation (typically the
    ``turn_id`` or a ``tool_call_id``). Both are optional but make ledger
    forensics tractable when a batch of parallel tools interleaves.
    """

    kind: EventKind
    session_id: str
    event_id: str = field(default_factory=new_event_id)
    seq: int = 0
    timestamp: float = field(default_factory=time.time)
    branch_id: str = ""
    turn_id: str = ""
    agent_id: str = ""
    schema_version: int = EVENT_SCHEMA_VERSION
    causation_id: str = ""
    correlation_id: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    # ── construction ──

    @classmethod
    def build(
        cls,
        kind: EventKind | str,
        session_id: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "KernelEvent":
        return cls(
            kind=EventKind(kind),
            session_id=session_id,
            payload=dict(payload or {}),
            **kwargs,
        )

    def for_turn(self, request: TurnRequest) -> "KernelEvent":
        """Return a copy stamped with *request*'s identity fields."""
        return replace(
            self,
            session_id=request.session_id,
            branch_id=self.branch_id or request.branch_id,
            turn_id=self.turn_id or request.turn_id,
            agent_id=self.agent_id or request.agent_id,
            correlation_id=self.correlation_id or request.turn_id,
        )

    # ── properties ──

    @property
    def tool_call_id(self) -> str:
        return str(self.payload.get("tool_call_id") or "")

    @property
    def is_terminal_turn(self) -> bool:
        return self.kind in TERMINAL_TURN_KINDS

    @property
    def is_terminal_tool(self) -> bool:
        return self.kind in TERMINAL_TOOL_KINDS

    @property
    def is_ephemeral(self) -> bool:
        return self.kind in EPHEMERAL_KINDS

    # ── serialization ──

    def to_json(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "branch_id": self.branch_id,
            "turn_id": self.turn_id,
            "agent_id": self.agent_id,
            "kind": str(self.kind),
            "schema_version": self.schema_version,
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "KernelEvent":
        """Rebuild an event from a ledger row or a JSONL export.

        Unknown ``kind`` values raise, because a projector silently skipping an
        unrecognized durable fact is how "the UI shows one thing and the ledger
        says another" bugs start. Callers replaying a ledger from a newer
        Hairball should gate on ``schema_version`` instead.
        """
        return cls(
            kind=EventKind(raw["kind"]),
            session_id=str(raw.get("session_id") or ""),
            event_id=str(raw.get("event_id") or new_event_id()),
            seq=int(raw.get("seq") or 0),
            timestamp=float(raw.get("timestamp") or 0.0),
            branch_id=str(raw.get("branch_id") or ""),
            turn_id=str(raw.get("turn_id") or ""),
            agent_id=str(raw.get("agent_id") or ""),
            schema_version=int(raw.get("schema_version") or EVENT_SCHEMA_VERSION),
            causation_id=str(raw.get("causation_id") or ""),
            correlation_id=str(raw.get("correlation_id") or ""),
            payload=dict(raw.get("payload") or {}),
        )

    def payload_json(self) -> str:
        """Serialize the payload for durable storage.

        Falls back to ``default=str`` rather than raising: a tool that returned
        an exotic object in its progress payload must not be able to abort the
        turn it is reporting on.
        """
        return json.dumps(self.payload, ensure_ascii=False, default=str)


# ── Artifacts ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArtifactRef:
    """A pointer to large output kept out of the model's context.

    The model sees ``summary`` (+ optional ``preview``); the workspace surface
    and the user can open the full bytes. ``byte_count``/``digest`` make it
    possible to prove the artifact on disk is the one the turn produced.
    """

    artifact_id: str
    session_id: str
    kind: str
    byte_count: int = 0
    digest: str = ""
    path: str = ""
    media_type: str = "text/plain"
    summary: str = ""
    truncated: bool = False
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            artifact_id=str(raw.get("artifact_id") or ""),
            session_id=str(raw.get("session_id") or ""),
            kind=str(raw.get("kind") or ""),
            byte_count=int(raw.get("byte_count") or 0),
            digest=str(raw.get("digest") or ""),
            path=str(raw.get("path") or ""),
            media_type=str(raw.get("media_type") or "text/plain"),
            summary=str(raw.get("summary") or ""),
            truncated=bool(raw.get("truncated")),
            created_at=float(raw.get("created_at") or 0.0),
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ToolOutcome:
    """What a tool returned, in the shape the kernel records and replays.

    ``summary`` is what enters the model's context when ``artifact_ref`` is set;
    ``inline`` is the full text when the output was small enough to keep.
    """

    tool_call_id: str
    tool_name: str
    ok: bool
    inline: str = ""
    summary: str = ""
    preview: str = ""
    artifact_ref: str = ""
    byte_count: int = 0
    truncated: bool = False
    duration_ms: int = 0
    error_kind: str = ""

    def model_text(self) -> str:
        """The text that should be handed back to the model."""
        if self.artifact_ref:
            parts = [self.summary or "(output stored as artifact)"]
            if self.preview:
                parts.append(self.preview)
            parts.append(
                f"[full output: {self.artifact_ref} · {self.byte_count} bytes]"
            )
            return "\n".join(p for p in parts if p)
        return self.inline


# ── Errors ────────────────────────────────────────────────────────────────────


class KernelError(Exception):
    """Base class for kernel contract violations."""


class InvariantViolation(KernelError):
    """A durable invariant from the design doc was broken.

    Raised (not logged-and-continued) so that a violation surfaces in tests
    instead of silently corrupting a projection.
    """


class LedgerUnavailable(KernelError):
    """The durable ledger could not be opened or migrated."""


# ── Environment helper ────────────────────────────────────────────────────────


def kernel_debug_enabled() -> bool:
    """True when kernel internals should log verbosely.

    Debug verbosity is a developer switch, not user configuration, which is why
    it reads an env var while every behavioral kernel setting lives in
    ``config.yaml`` under ``kernel:``.
    """
    return os.getenv("HAIRBALL_KERNEL_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
