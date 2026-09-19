"""Read models projected from the kernel event ledger.

Each product surface gets its own projector over the *same* event stream
instead of maintaining its own guess at lifecycle state. That separation is the
point: the chat projector may drop everything except user-visible messages, and
the workspace projector still holds every tool call, argument, and artifact,
because they are independent functions of one immutable log rather than two
readers of one mutable store.

Every projector here is:

- **Replay-idempotent.** Applying the same event twice is a no-op, so a
  reconnecting surface can safely re-read from a stale cursor.
- **Order-tolerant on ephemera, order-strict on facts.** Deltas may be missing
  entirely (the ledger does not persist them); the completed message is
  authoritative.
- **Pure.** No SQLite, no I/O, no clock. Given the same events, they produce the
  same snapshot — which is what makes v1/v2 differential testing possible.

Source adoption: the "one event log, many projections" split and the per-call-id
tool lifecycle (pending → running → completed/error, updated by call id) are
ADAPTed from OpenCode's ``session/processor.ts`` + ``server/projectors.ts``
(MIT) and reference's ``session/rollout_reconstruction.rs`` (Apache-2.0). No code
was copied; the state shapes and names are Hairball's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from agent.kernel.contracts import (
    TERMINAL_TOOL_KINDS,
    TERMINAL_TURN_KINDS,
    TOOL_SCOPED_KINDS,
    EventKind,
    InvariantViolation,
    KernelEvent,
    ToolCallState,
    TurnState,
    tool_state_for_kind,
)


class Projector:
    """Base class: feed events in, read a snapshot out."""

    name: str = "base"

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self.last_seq: int = 0

    def apply(self, event: KernelEvent) -> bool:
        """Apply *event*; return ``False`` when it was a duplicate.

        Deduplication is by ``event_id`` rather than ``seq`` because ephemeral
        events legitimately carry ``seq == 0``.
        """
        if event.event_id and event.event_id in self._seen:
            return False
        if event.event_id:
            self._seen.add(event.event_id)
        self.last_seq = max(self.last_seq, event.seq)
        self._handle(event)
        return True

    def apply_all(self, events: Iterable[KernelEvent]) -> int:
        applied = 0
        for event in events:
            if self.apply(event):
                applied += 1
        return applied

    def _handle(self, event: KernelEvent) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError


# ── Chat ──────────────────────────────────────────────────────────────────────


@dataclass
class ChatMessage:
    """One bubble in the user-visible transcript."""

    role: str
    text: str = ""
    turn_id: str = ""
    agent_id: str = ""
    seq: int = 0
    streaming: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "turn_id": self.turn_id,
            "agent_id": self.agent_id,
            "seq": self.seq,
            "streaming": self.streaming,
        }


class ChatProjector(Projector):
    """The transcript a human reads.

    Deliberately narrow: no tool arguments, no reasoning bodies, no subagent
    internals. Those live in the workspace projection. Reasoning is surfaced
    only as a boolean "this turn thought" flag, so private reasoning content
    cannot leak into a chat bubble.
    """

    name = "chat"

    def __init__(self, *, include_subagents: bool = False) -> None:
        super().__init__()
        self.messages: list[ChatMessage] = []
        self.include_subagents = include_subagents
        self._streaming: dict[str, ChatMessage] = {}
        self._root_agent_id: str = ""

    def _is_subagent(self, event: KernelEvent) -> bool:
        if not event.agent_id:
            return False
        if not self._root_agent_id:
            return False
        return event.agent_id != self._root_agent_id

    def _handle(self, event: KernelEvent) -> None:
        if event.kind == EventKind.TURN_ACCEPTED:
            if not self._root_agent_id:
                self._root_agent_id = event.agent_id
            text = str(event.payload.get("user_input_text") or "")
            if text:
                self.messages.append(
                    ChatMessage(
                        role="user", text=text, turn_id=event.turn_id,
                        agent_id=event.agent_id, seq=event.seq,
                    )
                )
            return

        if not self.include_subagents and self._is_subagent(event):
            return

        if event.kind == EventKind.ASSISTANT_MESSAGE_DELTA:
            # Deltas are notification only. Maintain a live bubble for surfaces
            # that render while streaming, but the completed event replaces it,
            # so a session resumed from the ledger (which has no deltas) still
            # produces the identical final transcript.
            bubble = self._streaming.get(event.turn_id)
            if bubble is None:
                bubble = ChatMessage(
                    role="assistant", turn_id=event.turn_id,
                    agent_id=event.agent_id, seq=event.seq, streaming=True,
                )
                self._streaming[event.turn_id] = bubble
                self.messages.append(bubble)
            bubble.text += str(event.payload.get("text") or "")
            return

        if event.kind == EventKind.ASSISTANT_MESSAGE_COMPLETED:
            text = str(event.payload.get("text") or "")
            bubble = self._streaming.pop(event.turn_id, None)
            if bubble is not None:
                bubble.text = text or bubble.text
                bubble.streaming = False
                bubble.seq = event.seq or bubble.seq
                return
            if text:
                self.messages.append(
                    ChatMessage(
                        role="assistant", text=text, turn_id=event.turn_id,
                        agent_id=event.agent_id, seq=event.seq,
                    )
                )
            return

        if event.kind == EventKind.TURN_FAILED:
            error = str(event.payload.get("error") or "turn failed")
            self._streaming.pop(event.turn_id, None)
            self.messages.append(
                ChatMessage(
                    role="error", text=error, turn_id=event.turn_id,
                    agent_id=event.agent_id, seq=event.seq,
                )
            )
            return

        if event.kind in (EventKind.TURN_COMPLETED, EventKind.TURN_CANCELLED):
            self._streaming.pop(event.turn_id, None)

    def clear_live(self) -> None:
        """Drop the in-flight streaming bubbles only.

        This is the operation that used to nuke shared state. Here it can only
        forget partial bubbles — durable messages, and everything the workspace
        projection holds, are untouched because they are separate projections of
        an append-only log.
        """
        for bubble in self._streaming.values():
            if bubble in self.messages and not bubble.text:
                self.messages.remove(bubble)
        self._streaming.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "projector": self.name,
            "last_seq": self.last_seq,
            "messages": [m.to_json() for m in self.messages],
        }


# ── Workspace ─────────────────────────────────────────────────────────────────


@dataclass
class ToolCallView:
    """Full detail for one tool call — the workspace's evidence record."""

    tool_call_id: str
    tool_name: str = ""
    state: ToolCallState = ToolCallState.PROPOSED
    turn_id: str = ""
    agent_id: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    progress: list[str] = field(default_factory=list)
    result_summary: str = ""
    artifact_ref: str = ""
    error: str = ""
    duration_ms: int = 0
    started_seq: int = 0
    terminal_seq: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "state": str(self.state),
            "turn_id": self.turn_id,
            "agent_id": self.agent_id,
            "arguments": dict(self.arguments),
            "progress": list(self.progress),
            "result_summary": self.result_summary,
            "artifact_ref": self.artifact_ref,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "started_seq": self.started_seq,
            "terminal_seq": self.terminal_seq,
        }


@dataclass
class TurnView:
    turn_id: str
    state: TurnState = TurnState.ACCEPTED
    agent_id: str = ""
    branch_id: str = ""
    user_input_text: str = ""
    model_ref: str = ""
    provider_ref: str = ""
    steps: int = 0
    retries: int = 0
    tool_call_ids: list[str] = field(default_factory=list)
    compactions: int = 0
    error: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "state": str(self.state),
            "agent_id": self.agent_id,
            "branch_id": self.branch_id,
            "user_input_text": self.user_input_text,
            "model_ref": self.model_ref,
            "provider_ref": self.provider_ref,
            "steps": self.steps,
            "retries": self.retries,
            "tool_call_ids": list(self.tool_call_ids),
            "compactions": self.compactions,
            "error": self.error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


class WorkspaceProjector(Projector):
    """Complete execution evidence: turns, tool calls, artifacts, compactions.

    This projection keeps everything the chat throws away, including subagent
    activity. Nothing here is ever removed by a chat-side clear.
    """

    name = "workspace"

    def __init__(self) -> None:
        super().__init__()
        self.turns: dict[str, TurnView] = {}
        self.tool_calls: dict[str, ToolCallView] = {}
        self.artifacts: list[dict[str, Any]] = []
        self.notes: list[dict[str, Any]] = []

    def _turn(self, event: KernelEvent) -> TurnView:
        view = self.turns.get(event.turn_id)
        if view is None:
            view = TurnView(
                turn_id=event.turn_id,
                agent_id=event.agent_id,
                branch_id=event.branch_id,
            )
            self.turns[event.turn_id] = view
        return view

    def _tool(self, event: KernelEvent) -> ToolCallView:
        call_id = event.tool_call_id
        view = self.tool_calls.get(call_id)
        if view is None:
            view = ToolCallView(
                tool_call_id=call_id,
                turn_id=event.turn_id,
                agent_id=event.agent_id,
            )
            self.tool_calls[call_id] = view
            turn = self._turn(event)
            if call_id not in turn.tool_call_ids:
                turn.tool_call_ids.append(call_id)
        return view

    def _handle(self, event: KernelEvent) -> None:
        kind = event.kind

        if kind == EventKind.TURN_ACCEPTED:
            turn = self._turn(event)
            turn.user_input_text = str(event.payload.get("user_input_text") or "")
            turn.model_ref = str(event.payload.get("model_ref") or "")
            turn.provider_ref = str(event.payload.get("provider_ref") or "")
            turn.started_at = event.timestamp
            return

        if kind == EventKind.TURN_STARTED:
            turn = self._turn(event)
            turn.state = TurnState.RUNNING
            if not turn.started_at:
                turn.started_at = event.timestamp
            return

        if kind == EventKind.TURN_STEP:
            self._turn(event).steps = max(
                self._turn(event).steps,
                int(event.payload.get("step") or 0),
            )
            return

        if kind == EventKind.TURN_RETRY_SCHEDULED:
            self._turn(event).retries += 1
            return

        if kind in TERMINAL_TURN_KINDS:
            turn = self._turn(event)
            turn.state = {
                EventKind.TURN_COMPLETED: TurnState.COMPLETED,
                EventKind.TURN_FAILED: TurnState.FAILED,
                EventKind.TURN_CANCELLED: TurnState.CANCELLED,
            }[kind]
            turn.ended_at = event.timestamp
            if kind == EventKind.TURN_FAILED:
                turn.error = str(event.payload.get("error") or "")
            return

        if kind == EventKind.CONTEXT_COMPACTION_COMPLETED:
            self._turn(event).compactions += 1
            return

        if kind == EventKind.ARTIFACT_CREATED:
            self.artifacts.append(dict(event.payload))
            call_id = event.tool_call_id
            if call_id and call_id in self.tool_calls:
                self.tool_calls[call_id].artifact_ref = str(
                    event.payload.get("artifact_id") or ""
                )
            return

        if kind == EventKind.STATUS_NOTE:
            self.notes.append(
                {
                    "seq": event.seq,
                    "turn_id": event.turn_id,
                    "level": str(event.payload.get("level") or "info"),
                    "message": str(event.payload.get("message") or ""),
                }
            )
            return

        if kind == EventKind.TOOL_CALL_PROGRESS:
            preview = str(event.payload.get("preview") or "")
            if preview:
                self._tool(event).progress.append(preview)
            return

        state = tool_state_for_kind(kind)
        if state is None:
            return

        view = self._tool(event)
        view.state = state
        if event.payload.get("tool_name"):
            view.tool_name = str(event.payload["tool_name"])
        args = event.payload.get("arguments")
        if isinstance(args, dict) and args:
            view.arguments = dict(args)
        if kind == EventKind.TOOL_CALL_STARTED:
            view.started_seq = event.seq
        if kind in TERMINAL_TOOL_KINDS:
            view.terminal_seq = event.seq
            view.duration_ms = int(event.payload.get("duration_ms") or 0)
            view.result_summary = str(event.payload.get("summary") or "")
            if event.payload.get("artifact_ref"):
                view.artifact_ref = str(event.payload["artifact_ref"])
            if kind == EventKind.TOOL_CALL_FAILED:
                view.error = str(event.payload.get("error") or "")

    def snapshot(self) -> dict[str, Any]:
        return {
            "projector": self.name,
            "last_seq": self.last_seq,
            "turns": [
                self.turns[t].to_json()
                for t in sorted(self.turns, key=lambda k: self.turns[k].started_at)
            ],
            "tool_calls": [
                self.tool_calls[c].to_json()
                for c in sorted(
                    self.tool_calls,
                    key=lambda k: (
                        self.tool_calls[k].started_seq,
                        self.tool_calls[k].tool_call_id,
                    ),
                )
            ],
            "artifacts": list(self.artifacts),
            "notes": list(self.notes),
        }


# ── Agent tree ────────────────────────────────────────────────────────────────


@dataclass
class AgentNode:
    agent_id: str
    parent_agent_id: str = ""
    objective: str = ""
    status: str = "running"
    workspace_ref: str = ""
    result_summary: str = ""
    messages_enqueued: int = 0
    messages_delivered: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "objective": self.objective,
            "status": self.status,
            "workspace_ref": self.workspace_ref,
            "result_summary": self.result_summary,
            "messages_enqueued": self.messages_enqueued,
            "messages_delivered": self.messages_delivered,
        }


class AgentTreeProjector(Projector):
    """Parent/child agent status and mailbox accounting.

    ``messages_enqueued`` vs ``messages_delivered`` is the single-rail check:
    a body that was enqueued once must be delivered at most once. Divergence
    here is how double-injection into a parent context gets caught.
    """

    name = "agent_tree"

    def __init__(self) -> None:
        super().__init__()
        self.agents: dict[str, AgentNode] = {}
        self._delivered_envelopes: set[str] = set()
        self.duplicate_deliveries: list[str] = []

    def _handle(self, event: KernelEvent) -> None:
        if event.kind == EventKind.AGENT_SPAWNED:
            agent_id = str(event.payload.get("agent_id") or event.agent_id)
            self.agents[agent_id] = AgentNode(
                agent_id=agent_id,
                parent_agent_id=str(event.payload.get("parent_agent_id") or ""),
                objective=str(event.payload.get("objective") or ""),
                workspace_ref=str(event.payload.get("workspace_ref") or ""),
            )
            return

        if event.kind == EventKind.AGENT_MESSAGE_ENQUEUED:
            node = self.agents.get(str(event.payload.get("agent_id") or event.agent_id))
            if node is not None:
                node.messages_enqueued += 1
            return

        if event.kind == EventKind.AGENT_MESSAGE_DELIVERED:
            envelope_id = str(event.payload.get("envelope_id") or "")
            if envelope_id and envelope_id in self._delivered_envelopes:
                self.duplicate_deliveries.append(envelope_id)
                return
            if envelope_id:
                self._delivered_envelopes.add(envelope_id)
            node = self.agents.get(str(event.payload.get("agent_id") or event.agent_id))
            if node is not None:
                node.messages_delivered += 1
            return

        if event.kind == EventKind.AGENT_COMPLETED:
            agent_id = str(event.payload.get("agent_id") or event.agent_id)
            node = self.agents.get(agent_id)
            if node is None:
                node = AgentNode(agent_id=agent_id)
                self.agents[agent_id] = node
            node.status = str(event.payload.get("status") or "completed")
            node.result_summary = str(event.payload.get("summary") or "")

    def snapshot(self) -> dict[str, Any]:
        return {
            "projector": self.name,
            "last_seq": self.last_seq,
            "agents": [self.agents[a].to_json() for a in sorted(self.agents)],
            "duplicate_deliveries": list(self.duplicate_deliveries),
        }


# ── Sidebar ───────────────────────────────────────────────────────────────────


class SidebarProjector(Projector):
    """Session-list row: title hint, activity time, counts, unread."""

    name = "sidebar"

    def __init__(self) -> None:
        super().__init__()
        self.first_user_text: str = ""
        self.last_activity: float = 0.0
        self.turn_count: int = 0
        self.completed_turns: int = 0
        self.failed_turns: int = 0
        self.tool_call_count: int = 0
        self.unread: int = 0
        self.running: bool = False

    def _handle(self, event: KernelEvent) -> None:
        self.last_activity = max(self.last_activity, event.timestamp)
        if event.kind == EventKind.TURN_ACCEPTED:
            self.turn_count += 1
            self.running = True
            if not self.first_user_text:
                self.first_user_text = str(event.payload.get("user_input_text") or "")
        elif event.kind == EventKind.TOOL_CALL_STARTED:
            self.tool_call_count += 1
        elif event.kind == EventKind.ASSISTANT_MESSAGE_COMPLETED:
            self.unread += 1
        elif event.kind in TERMINAL_TURN_KINDS:
            self.running = False
            if event.kind == EventKind.TURN_COMPLETED:
                self.completed_turns += 1
            elif event.kind == EventKind.TURN_FAILED:
                self.failed_turns += 1

    def mark_read(self) -> None:
        self.unread = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "projector": self.name,
            "last_seq": self.last_seq,
            "title_hint": self.first_user_text[:80],
            "last_activity": self.last_activity,
            "turn_count": self.turn_count,
            "completed_turns": self.completed_turns,
            "failed_turns": self.failed_turns,
            "tool_call_count": self.tool_call_count,
            "unread": self.unread,
            "running": self.running,
        }


# ── Metrics ───────────────────────────────────────────────────────────────────


class MetricsProjector(Projector):
    """Latency, token, and error counters for the performance gates.

    Populated from durable events only, so the numbers a load test reports are
    the same numbers a post-hoc ledger replay reports.
    """

    name = "metrics"

    def __init__(self) -> None:
        super().__init__()
        self.turn_wall_ms: list[int] = []
        self.time_to_first_token_ms: list[int] = []
        self.tool_durations_ms: list[int] = []
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cache_read_tokens: int = 0
        self.cache_write_tokens: int = 0
        self.errors: dict[str, int] = {}
        self.retries: int = 0
        self.compactions: int = 0
        self._turn_start: dict[str, float] = {}
        self._first_token_seen: set[str] = set()

    def _handle(self, event: KernelEvent) -> None:
        kind = event.kind
        if kind == EventKind.TURN_STARTED:
            self._turn_start[event.turn_id] = event.timestamp
        elif kind in (
            EventKind.ASSISTANT_MESSAGE_DELTA,
            EventKind.ASSISTANT_MESSAGE_COMPLETED,
        ):
            if event.turn_id not in self._first_token_seen:
                self._first_token_seen.add(event.turn_id)
                start = self._turn_start.get(event.turn_id)
                if start:
                    self.time_to_first_token_ms.append(
                        max(0, int((event.timestamp - start) * 1000))
                    )
        elif kind == EventKind.PROVIDER_REQUEST_COMPLETED:
            usage = event.payload.get("usage") or {}
            if isinstance(usage, dict):
                self.input_tokens += int(usage.get("input_tokens") or 0)
                self.output_tokens += int(usage.get("output_tokens") or 0)
                self.cache_read_tokens += int(usage.get("cache_read_tokens") or 0)
                self.cache_write_tokens += int(usage.get("cache_write_tokens") or 0)
        elif kind == EventKind.TURN_RETRY_SCHEDULED:
            self.retries += 1
        elif kind == EventKind.CONTEXT_COMPACTION_COMPLETED:
            self.compactions += 1
        elif kind in TERMINAL_TOOL_KINDS:
            duration = int(event.payload.get("duration_ms") or 0)
            if duration:
                self.tool_durations_ms.append(duration)
            if kind == EventKind.TOOL_CALL_FAILED:
                key = str(event.payload.get("error_kind") or "tool_error")
                self.errors[key] = self.errors.get(key, 0) + 1
        elif kind in TERMINAL_TURN_KINDS:
            start = self._turn_start.pop(event.turn_id, None)
            if start:
                self.turn_wall_ms.append(max(0, int((event.timestamp - start) * 1000)))
            if kind == EventKind.TURN_FAILED:
                key = str(event.payload.get("error_kind") or "turn_error")
                self.errors[key] = self.errors.get(key, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "projector": self.name,
            "last_seq": self.last_seq,
            "turn_wall_ms": list(self.turn_wall_ms),
            "time_to_first_token_ms": list(self.time_to_first_token_ms),
            "tool_durations_ms": list(self.tool_durations_ms),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "errors": dict(self.errors),
            "retries": self.retries,
            "compactions": self.compactions,
        }


# ── Replay helpers ────────────────────────────────────────────────────────────


DEFAULT_PROJECTORS: tuple[type[Projector], ...] = (
    ChatProjector,
    WorkspaceProjector,
    AgentTreeProjector,
    SidebarProjector,
    MetricsProjector,
)


def replay(
    events: Iterable[KernelEvent],
    projectors: Optional[Iterable[Projector]] = None,
) -> dict[str, dict[str, Any]]:
    """Feed *events* through *projectors* and return snapshots by name."""
    views = list(projectors) if projectors is not None else [
        cls() for cls in DEFAULT_PROJECTORS
    ]
    event_list = list(events)
    for view in views:
        view.apply_all(event_list)
    return {view.name: view.snapshot() for view in views}


@dataclass
class InvariantReport:
    """Result of checking a ledger against the kernel's durable invariants."""

    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def raise_if_violated(self) -> None:
        if self.violations:
            raise InvariantViolation("; ".join(self.violations))


def check_invariants(events: Iterable[KernelEvent]) -> InvariantReport:
    """Verify the durable invariants the design doc treats as non-negotiable.

    Checks:

    - every accepted turn has at most one terminal event, and any started turn
      that terminated did so exactly once;
    - every started tool call has exactly one terminal event;
    - ``seq`` is strictly increasing across durable events and never reused;
    - tool-scoped events always carry a ``tool_call_id``;
    - a mailbox envelope is delivered at most once.

    Returns a report rather than raising so a caller can log every violation in
    one pass instead of stopping at the first.
    """
    report = InvariantReport()

    accepted_turns: set[str] = set()
    terminal_turns: dict[str, int] = {}
    started_tools: set[str] = set()
    terminal_tools: dict[str, int] = {}
    delivered: dict[str, int] = {}
    seen_seqs: set[int] = set()
    last_seq = 0

    for event in events:
        if event.seq:
            if event.seq in seen_seqs:
                report.violations.append(f"duplicate seq {event.seq}")
            elif event.seq < last_seq:
                report.violations.append(
                    f"seq {event.seq} out of order after {last_seq}"
                )
            seen_seqs.add(event.seq)
            last_seq = max(last_seq, event.seq)

        if event.kind in TOOL_SCOPED_KINDS and not event.tool_call_id:
            report.violations.append(f"{event.kind} at seq {event.seq} has no tool_call_id")

        if event.kind == EventKind.TURN_ACCEPTED:
            accepted_turns.add(event.turn_id)
        elif event.kind in TERMINAL_TURN_KINDS:
            terminal_turns[event.turn_id] = terminal_turns.get(event.turn_id, 0) + 1
        elif event.kind == EventKind.TOOL_CALL_STARTED:
            started_tools.add(event.tool_call_id)
        elif event.kind in TERMINAL_TOOL_KINDS:
            terminal_tools[event.tool_call_id] = (
                terminal_tools.get(event.tool_call_id, 0) + 1
            )
        elif event.kind == EventKind.AGENT_MESSAGE_DELIVERED:
            envelope_id = str(event.payload.get("envelope_id") or "")
            if envelope_id:
                delivered[envelope_id] = delivered.get(envelope_id, 0) + 1

    for turn_id, count in terminal_turns.items():
        if count > 1:
            report.violations.append(
                f"turn {turn_id} has {count} terminal events (must be exactly 1)"
            )

    for call_id in started_tools:
        count = terminal_tools.get(call_id, 0)
        if count == 0:
            report.violations.append(
                f"tool call {call_id} started but never reached a terminal event"
            )
        elif count > 1:
            report.violations.append(
                f"tool call {call_id} has {count} terminal events (must be exactly 1)"
            )

    for envelope_id, count in delivered.items():
        if count > 1:
            report.violations.append(
                f"mailbox envelope {envelope_id} delivered {count} times "
                "(single rail allows 1)"
            )

    return report
