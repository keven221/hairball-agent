"""Kernel v2 turn loop: a three-state step machine with guaranteed cleanup.

This is the spine of the second kernel route. It owns the decision "what
happens after this step" and nothing else: it does not call providers, does not
execute tools, and does not build prompts. Those arrive as injected callables so
that the loop stays a pure state machine that unit tests can drive to every
terminal state without a network, a database, or a model.

**Severance posture.** The v1 loop (``agent/conversation_loop.py``) is
legacy-derived. It is treated here as a *behavior contract* only — its inputs,
outputs, state transitions and invariants — with no code lineage. The capability
comparison that decided each design point below is summarized in
``docs/design/2026-08-04-hairball-independent-core-knowledge.md`` section 3.

What this module implements, and where the contract came from:

- **Three-state step result** (``TurnDecision``). A step ends in ``CONTINUE``,
  ``STOP`` or ``COMPACT``. Making compaction a first-class loop outcome instead
  of an exception path is ADAPTed from OpenCode's ``Result`` type
  (``packages/opencode/src/session/processor.ts:30,679-681``, MIT).
- **Doom-loop detection** (``DoomLoopDetector``). Repeating the identical tool
  call N times in a row stops being work and starts being a spin; the loop
  demands approval rather than burning the budget. Contract from OpenCode's
  ``DOOM_LOOP_THRESHOLD`` (``processor.ts:29,356-379``). No other reference
  implementation reviewed had this.
- **Guaranteed tool-call settlement** (``TurnLoop`` as a context manager). On
  any exit — normal, exception, interrupt — every tool call that started gets
  exactly one terminal event. Contract from OpenCode's
  ``Effect.ensuring(cleanup())`` (``processor.ts:576-593``), which marks orphans
  aborted/interrupted, and from Pi's ``failToolCallsFromTruncatedMessage``
  (``packages/agent/src/agent-loop.ts:381``, MIT). This is what keeps a provider
  history from ever holding an assistant tool call with no matching result.
- **Explicit pending-input queue** (``PendingInputQueue``). Steer and follow-up
  are kept as separate queues with different delivery rules; the shape is
  ADAPTed from Pi's ``pendingMessages`` (``agent-loop.ts:174,183``).
- **Retry observability** (``RetryReport``). Retries report attempt, reason,
  action and delay as durable events rather than log lines, following
  OpenCode's ``SessionRetry`` status callback (``processor.ts:660-674``).

What this module deliberately keeps from Hairball's own implementation, because
the capability comparison found it stronger than every reference:

- **Compaction policy** stays injected (``CompactionCheck``) so the kernel keeps
  using Hairball's ``should_compress`` / ``should_compress_preflight`` with its
  cooldown and anti-thrash guards. The references check a single overflow
  boolean; Hairball's policy was shaped by real thrash incidents.
- **Tool guardrails** are evaluated by the executor and reported to the loop via
  ``StepReport.guardrail_halt``. The loop does not re-implement or second-guess
  them.
- **Tool concurrency strategy** (concurrent / sequential / segmented) stays with
  the executor. The loop is handed a settled batch and never picks a strategy.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from agent.kernel.contracts import (
    EventKind,
    InvariantViolation,
    KernelEvent,
    TurnRequest,
)

#: Consecutive identical tool calls tolerated before the loop demands approval.
#: Two is a legitimate retry ("the file wasn't there, try again"); three in a
#: row with byte-identical arguments is a spin.
DOOM_LOOP_THRESHOLD = 3

#: Text recorded for a tool call the loop had to settle on the turn's behalf.
#: Kept as a constant because both the event payload and the model-visible tool
#: result must say the same thing — a surface showing "aborted" while the model
#: was told something else is how post-interrupt confusion starts.
ABORTED_TOOL_TEXT = "Tool execution aborted"


class TurnDecision(StrEnum):
    """What the runtime should do after a step."""

    CONTINUE = "continue"
    STOP = "stop"
    COMPACT = "compact"


class StopReason(StrEnum):
    """Why a turn stopped. Recorded on the terminal event.

    Distinct from ``TurnDecision`` on purpose: surfaces render "hit the
    iteration cap" very differently from "the model was done", and metrics need
    to tell an interrupt apart from a failure.
    """

    NATURAL = "natural"
    BUDGET_ITERATIONS = "budget.iterations"
    BUDGET_WALL_CLOCK = "budget.wall_clock"
    BUDGET_COST = "budget.cost"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    ERROR = "error"
    GUARDRAIL_HALT = "guardrail.halt"
    DOOM_LOOP = "doom_loop"


#: Stop reasons that mean the turn did not finish its work.
UNFINISHED_STOP_REASONS: frozenset[StopReason] = frozenset(
    {
        StopReason.BUDGET_ITERATIONS,
        StopReason.BUDGET_WALL_CLOCK,
        StopReason.BUDGET_COST,
        StopReason.INTERRUPTED,
        StopReason.CANCELLED,
        StopReason.ERROR,
        StopReason.GUARDRAIL_HALT,
        StopReason.DOOM_LOOP,
    }
)


# ── Call signatures / doom loop ───────────────────────────────────────────────


def tool_call_signature(tool_name: str, arguments: Any) -> str:
    """A stable identity for "the same tool call again".

    Arguments arrive as a dict from some providers and as a JSON string from
    others, and key order is not stable across serializations. Both are folded
    to a canonical form so that a genuine repeat is detected regardless of how
    the provider happened to encode it, and so that a mere key reshuffle is not
    mistaken for a different call.
    """
    payload = arguments
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            return f"{tool_name}:{payload.strip()}"
    try:
        encoded = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        encoded = repr(payload)
    return f"{tool_name}:{encoded}"


class DoomLoopDetector:
    """Detect a model repeating one tool call verbatim.

    Semantics are a sliding window over the most recent proposed calls: the
    detector trips when the last ``threshold`` calls are all the same tool with
    the same arguments. This matches OpenCode's check, which slices the last
    ``DOOM_LOOP_THRESHOLD`` message parts and requires every one of them to be
    the same tool with an identical JSON input (``processor.ts:356-369``).

    Two deliberate differences, both improvements on the reference:

    - **Cost.** OpenCode re-reads the assistant message's parts from its
      database on every tool-input-end, so the check costs a query per tool
      call. The window here is an in-memory ring buffer, so it costs nothing per
      call.
    - **Resume.** Because OpenCode derives the window from persisted history, a
      resumed session keeps its streak; an in-memory counter would silently
      forget it and let a spin continue past a restart or a compaction. ``seed``
      restores the window so the cheap representation keeps that property.

    Approval is keyed on the **tool name**, not the signature, matching
    OpenCode's ``permission.ask({ always: [value.name] })``: once a user says
    this tool may repeat, they are not asked again for a different argument set
    of the same tool.
    """

    def __init__(self, threshold: int = DOOM_LOOP_THRESHOLD) -> None:
        # A threshold below 2 would trip on a call's first appearance, which
        # would make every tool call require approval.
        self._threshold = max(2, int(threshold))
        self._window: deque[tuple[str, str]] = deque(maxlen=self._threshold)
        self._always_allowed: set[str] = set()

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def streak(self) -> int:
        """How many trailing window entries share the newest signature."""
        if not self._window:
            return 0
        newest = self._window[-1]
        count = 0
        for entry in reversed(self._window):
            if entry != newest:
                break
            count += 1
        return count

    def observe(self, tool_name: str, arguments: Any) -> bool:
        """Record a proposed call; return True when the window trips."""
        entry = (tool_name, tool_call_signature(tool_name, arguments))
        self._window.append(entry)
        if tool_name in self._always_allowed:
            return False
        return len(self._window) == self._threshold and all(
            other == entry for other in self._window
        )

    def allow_always(self, tool_name: str) -> None:
        """Stop asking about repeats of *tool_name* for the rest of the session.

        Clearing the window instead would ask again as soon as the streak
        rebuilt, which is the "approved once, prompted forever" failure mode.
        """
        self._always_allowed.add(tool_name)

    def is_always_allowed(self, tool_name: str) -> bool:
        return tool_name in self._always_allowed

    def seed(self, calls: Sequence[tuple[str, Any]]) -> None:
        """Rebuild the window from prior calls, for a resumed turn.

        Only the trailing ``threshold`` entries matter, so callers may hand over
        as much history as they have.
        """
        for tool_name, arguments in calls:
            self._window.append(
                (tool_name, tool_call_signature(tool_name, arguments))
            )

    def reset(self) -> None:
        self._window.clear()


# ── Pending input ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PendingInput:
    """Input that arrived while a turn was already running."""

    text: str
    kind: str = "steer"
    received_at: float = field(default_factory=time.time)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PendingInputQueue:
    """Two queues with different delivery rules.

    ``steer`` is supplementary input for the turn that is *already running*: the
    loop drains it between steps and hands it to the runtime to attach to the
    next provider call. ``follow_up`` is the next turn's input and is held until
    the current turn settles.

    Collapsing the two is the classic way to end up injecting a synthetic user
    message into the middle of a tool loop, which breaks strict role alternation
    and invalidates the prompt cache. Keeping them apart all the way down is the
    point of this class.

    Thread-safe because control commands arrive from a surface thread while the
    loop runs on the turn thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steer: list[PendingInput] = []
        self._follow_up: list[PendingInput] = []

    def push_steer(self, text: str, **metadata: Any) -> PendingInput:
        item = PendingInput(text=text, kind="steer", metadata=metadata)
        with self._lock:
            self._steer.append(item)
        return item

    def push_follow_up(self, text: str, **metadata: Any) -> PendingInput:
        item = PendingInput(text=text, kind="follow_up", metadata=metadata)
        with self._lock:
            self._follow_up.append(item)
        return item

    @property
    def has_steer(self) -> bool:
        with self._lock:
            return bool(self._steer)

    @property
    def has_follow_up(self) -> bool:
        with self._lock:
            return bool(self._follow_up)

    def drain_steer(self) -> tuple[PendingInput, ...]:
        """Take everything steered so far. Called between steps."""
        with self._lock:
            items = tuple(self._steer)
            self._steer.clear()
        return items

    def take_follow_ups(self) -> tuple[PendingInput, ...]:
        """Take held follow-ups. Called only after the turn settles."""
        with self._lock:
            items = tuple(self._follow_up)
            self._follow_up.clear()
        return items


# ── Reports ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetryReport:
    """One retry attempt, as a durable fact rather than a log line."""

    attempt: int
    reason: str
    delay_seconds: float = 0.0
    action: str = ""
    message: str = ""
    next_model: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "reason": self.reason,
            "delay_seconds": self.delay_seconds,
            "action": self.action,
            "message": self.message,
            "next_model": self.next_model,
        }


@dataclass(frozen=True)
class TurnUsage:
    """Accumulated cost of a turn so far."""

    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def plus(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> "TurnUsage":
        return TurnUsage(
            steps=self.steps + 1,
            input_tokens=self.input_tokens + max(0, input_tokens),
            output_tokens=self.output_tokens + max(0, output_tokens),
            cost_usd=self.cost_usd + max(0.0, cost_usd),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass(frozen=True)
class StepReport:
    """What the runtime observed during one step, handed back to the loop.

    ``tool_call_count`` rather than the calls themselves: the loop's decision
    only depends on whether the model asked for more work, and keeping tool
    payloads out of the state machine keeps it cheap to test.

    ``force_continue`` carries v1's intent-ack / verify-on-stop follow-ups:
    the model emitted a text-only "I'll do X next" that must not end the turn.
    """

    finish_reason: str = "stop"
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    guardrail_halt: bool = False
    error: str = ""
    force_continue: bool = False


# ── Injected collaborators ────────────────────────────────────────────────────

#: Given the turn's accumulated usage, is compaction required before continuing?
#: Wired to Hairball's own ``should_compress`` so the kernel inherits its
#: cooldown and anti-thrash behavior instead of re-deriving a threshold.
CompactionCheck = Callable[[TurnUsage], bool]


class SnapshotSource(Protocol):
    """Per-step workspace change attribution.

    ``track`` is called before a step and returns an opaque handle (or ``None``
    when snapshots are unavailable — a dirty tree, no VCS, snapshots disabled).
    ``patch`` is called after the step and returns a description of what changed,
    or ``None``/empty when nothing did.
    """

    def track(self) -> Optional[str]: ...

    def patch(self, handle: str) -> Optional[Mapping[str, Any]]: ...


# ── The loop ──────────────────────────────────────────────────────────────────


@dataclass
class _OpenToolCall:
    tool_call_id: str
    tool_name: str
    started_at: float


class TurnLoop:
    """Loop control for exactly one turn.

    Usage — the context manager form is the supported one, because the cleanup
    guarantee is what makes the tool-settlement invariant hold::

        with TurnLoop(request, ledger=ledger) as loop:
            while loop.begin_step() is not None:
                report = runtime.execute_one_step(loop)
                decision = loop.end_step(report)
                if decision is TurnDecision.STOP:
                    break
                if decision is TurnDecision.COMPACT:
                    runtime.compact()
                    loop.note_compaction_done()

    On exit the loop settles any tool call left open and emits exactly one
    terminal turn event. Both happen even when the body raised.

    ``ledger`` is optional so the state machine is testable in isolation and
    usable when the kernel is disabled; with ``None`` no events are recorded and
    every decision still behaves identically.
    """

    def __init__(
        self,
        request: TurnRequest,
        *,
        ledger: Any = None,
        compaction_check: Optional[CompactionCheck] = None,
        snapshot: Optional[SnapshotSource] = None,
        doom_loop_threshold: int = DOOM_LOOP_THRESHOLD,
        prior_tool_calls: Optional[Sequence[tuple[str, Any]]] = None,
        max_iterations: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.request = request
        self.ledger = ledger
        self._compaction_check = compaction_check
        self._snapshot = snapshot
        self._clock = clock

        # An explicit budget on the request wins; the constructor default is the
        # runtime's fallback for callers that do not build a Budget.
        self._max_iterations = request.budget.max_iterations or max_iterations
        self._max_wall_seconds = request.budget.max_wall_seconds
        self._max_cost_usd = request.budget.max_cost_usd

        self.doom_loop = DoomLoopDetector(doom_loop_threshold)
        if prior_tool_calls:
            # A resumed turn must not forget a spin that was already underway.
            self.doom_loop.seed(prior_tool_calls)
        self.pending = PendingInputQueue()

        self._lock = threading.Lock()
        self._open_tools: dict[str, _OpenToolCall] = {}
        self._usage = TurnUsage()
        self._started_at: Optional[float] = None
        self._step_index = 0
        self._step_handle: Optional[str] = None
        self._step_open = False
        self._needs_compaction = False
        self._interrupt: Optional[StopReason] = None
        self._stop_reason: Optional[StopReason] = None
        self._terminal_emitted = False
        self._final_text = ""

    # ── properties ──

    @property
    def usage(self) -> TurnUsage:
        return self._usage

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def stop_reason(self) -> Optional[StopReason]:
        return self._stop_reason

    @property
    def final_text(self) -> str:
        """The turn's answer as the loop last saw it.

        Read-only so a host reads the same string the terminal event carried,
        instead of tracking its own copy that can drift from the ledger.
        """
        return self._final_text

    @property
    def open_tool_call_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._open_tools)

    @property
    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, self._clock() - self._started_at)

    # ── lifecycle ──

    def __enter__(self) -> "TurnLoop":
        self.begin()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Never swallow the exception: the caller's error handling stays intact,
        # this only guarantees the ledger is left in a consistent state.
        if exc is not None and self._stop_reason is None:
            self._stop_reason = StopReason.ERROR
            self._final_text = self._final_text or str(exc)
        self.close()
        return False

    def begin(self) -> None:
        if self._started_at is not None:
            raise InvariantViolation("TurnLoop.begin() called twice")
        self._started_at = self._clock()
        self._emit(EventKind.TURN_STARTED, {"budget": self.request.budget.to_json()})

    def begin_step(self) -> Optional[int]:
        """Open the next step, or return ``None`` when the turn must stop.

        Budget ceilings are enforced here rather than after the fact so that a
        turn cannot exceed its cap by one extra provider call — the expensive
        thing being capped.
        """
        if self._step_open:
            raise InvariantViolation(
                f"step {self._step_index} is still open; call end_step() first"
            )
        if self._stop_reason is not None:
            return None

        blocked = self._interrupt or self._budget_stop()
        if blocked is not None:
            self._stop_reason = blocked
            return None

        self._step_index += 1
        self._step_open = True
        self._step_handle = self._snapshot.track() if self._snapshot else None
        self._emit(
            EventKind.TURN_STEP,
            {"step": self._step_index, "usage": self._usage.to_payload()},
        )
        return self._step_index

    def end_step(self, report: StepReport) -> TurnDecision:
        """Close the current step and decide what happens next."""
        if not self._step_open:
            raise InvariantViolation("end_step() called without an open step")
        self._step_open = False
        self._usage = self._usage.plus(
            input_tokens=report.input_tokens,
            output_tokens=report.output_tokens,
            cost_usd=report.cost_usd,
        )
        self._record_step_patch()

        decision, reason = self._decide(report)
        if reason is not None:
            self._stop_reason = reason
        self._emit(
            EventKind.TURN_STEP_COMPLETED,
            {
                "step": self._step_index,
                "decision": str(decision),
                "stop_reason": str(reason) if reason else "",
                "finish_reason": report.finish_reason,
                "tool_calls": report.tool_call_count,
                "usage": self._usage.to_payload(),
            },
        )
        return decision

    def _decide(self, report: StepReport) -> tuple[TurnDecision, Optional[StopReason]]:
        """Resolve one step into a decision plus an optional stop reason.

        Ordering note. OpenCode checks compaction *first*
        (``processor.ts:679-681``), so an overflowed context is repaired before
        anything else. Hairball checks cancellation, error and guardrail halts
        first instead: compacting costs a model call, and spending one on a turn
        the user just cancelled — or one that already failed — is waste the
        user pays for. The overflowed history is not lost by deferring, because
        Hairball's own preflight compression check runs at the start of the next
        turn (``agent/context_engine.py:134``), which is the capability the
        references do not have. This is the "same contract, tuned to Hairball's
        stronger compaction policy" case from the severance ledger.
        """
        if self._interrupt is not None:
            return TurnDecision.STOP, self._interrupt
        if report.error:
            self._final_text = self._final_text or report.error
            return TurnDecision.STOP, StopReason.ERROR
        if report.guardrail_halt:
            return TurnDecision.STOP, StopReason.GUARDRAIL_HALT

        if self._needs_compaction or self._compaction_needed():
            self._needs_compaction = True
            return TurnDecision.COMPACT, None

        if report.force_continue:
            budget_stop = self._budget_stop()
            if budget_stop is not None:
                return TurnDecision.STOP, budget_stop
            return TurnDecision.CONTINUE, None

        if report.tool_call_count <= 0:
            return TurnDecision.STOP, StopReason.NATURAL

        budget_stop = self._budget_stop()
        if budget_stop is not None:
            return TurnDecision.STOP, budget_stop
        return TurnDecision.CONTINUE, None

    def _compaction_needed(self) -> bool:
        if self._compaction_check is None:
            return False
        try:
            return bool(self._compaction_check(self._usage))
        except Exception:
            # A compaction policy that raises must not take the turn down with
            # it; the next turn's preflight check gets another chance.
            return False

    def _budget_stop(self) -> Optional[StopReason]:
        if self._max_iterations and self._step_index >= self._max_iterations:
            return StopReason.BUDGET_ITERATIONS
        if self._max_wall_seconds and self.elapsed_seconds >= self._max_wall_seconds:
            return StopReason.BUDGET_WALL_CLOCK
        if self._max_cost_usd and self._usage.cost_usd >= self._max_cost_usd:
            return StopReason.BUDGET_COST
        return None

    def note_compaction_done(self) -> None:
        """Clear the compaction latch after the runtime compacted."""
        self._needs_compaction = False

    def note_retry(self, report: RetryReport) -> None:
        self._emit(EventKind.TURN_RETRY_SCHEDULED, report.to_payload())

    def request_stop(self, reason: StopReason) -> None:
        """Ask the loop to stop at the next boundary.

        Called from a control thread. The loop stops between steps rather than
        mid-step so that a tool already running still gets its terminal event
        through the normal path.
        """
        self._interrupt = reason

    def set_final_text(self, text: str) -> None:
        self._final_text = text or ""

    # ── tool call bookkeeping ──

    def tool_started(
        self,
        tool_call_id: str,
        tool_name: str,
        *,
        arguments: Any = None,
    ) -> None:
        """Record a tool call as running. Pairs with ``tool_settled``."""
        if not tool_call_id:
            raise InvariantViolation("tool_started() requires a tool_call_id")
        with self._lock:
            if tool_call_id in self._open_tools:
                raise InvariantViolation(
                    f"tool call {tool_call_id} is already open"
                )
            self._open_tools[tool_call_id] = _OpenToolCall(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                started_at=self._clock(),
            )
        payload: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
        }
        if arguments is not None:
            payload["arguments"] = arguments
        self._emit(EventKind.TOOL_CALL_STARTED, payload)

    def tool_settled(
        self,
        tool_call_id: str,
        *,
        ok: bool = True,
        cancelled: bool = False,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record the single terminal event for a running tool call."""
        with self._lock:
            open_call = self._open_tools.pop(tool_call_id, None)
        if open_call is None:
            raise InvariantViolation(
                f"tool call {tool_call_id} is not open; refusing a second "
                f"terminal event"
            )
        if cancelled:
            kind = EventKind.TOOL_CALL_CANCELLED
        elif ok:
            kind = EventKind.TOOL_CALL_COMPLETED
        else:
            kind = EventKind.TOOL_CALL_FAILED
        body: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "tool_name": open_call.tool_name,
            "duration_ms": int((self._clock() - open_call.started_at) * 1000),
        }
        body.update(payload or {})
        self._emit(kind, body)

    def approve_doom_loop(self, tool_name: str) -> None:
        """Record that the user allowed *tool_name* to keep repeating."""
        self.doom_loop.allow_always(tool_name)

    def check_doom_loop(self, tool_name: str, arguments: Any) -> bool:
        """Observe a proposed call; emit an approval request if it is a spin.

        Returns True when the caller must obtain approval before executing. The
        loop does not block — approval routing belongs to the surface — it only
        states that the streak tripped and records why.
        """
        if not self.doom_loop.observe(tool_name, arguments):
            return False
        self._emit(
            EventKind.TOOL_CALL_APPROVAL_REQUIRED,
            {
                # No executable call exists yet, so this event is scoped to the
                # streak rather than to a call id the surface could act on.
                "tool_call_id": f"doomloop:{tool_name}",
                "tool_name": tool_name,
                "reason": "doom_loop",
                "streak": self.doom_loop.streak,
                "threshold": self.doom_loop.threshold,
            },
        )
        return True

    # ── workspace attribution ──

    def _record_step_patch(self) -> None:
        handle, self._step_handle = self._step_handle, None
        if not handle or self._snapshot is None:
            return
        try:
            patch = self._snapshot.patch(handle)
        except Exception:
            # Attribution is diagnostic. A snapshot backend failing (detached
            # HEAD, permissions, a huge tree) must not fail the step whose work
            # already succeeded.
            return
        if not patch:
            return
        body = {"step": self._step_index, "snapshot": handle}
        body.update(patch)
        self._emit(EventKind.WORKSPACE_PATCH, body)

    # ── termination ──

    def close(self, *, reason: Optional[StopReason] = None) -> StopReason:
        """Settle the turn: drain open tool calls, emit one terminal event.

        Idempotent, because it runs both from ``__exit__`` and from callers that
        close explicitly.
        """
        if self._terminal_emitted:
            return self._stop_reason or StopReason.NATURAL
        if reason is not None:
            self._stop_reason = reason
        if self._stop_reason is None:
            self._stop_reason = self._interrupt or StopReason.NATURAL

        self._drain_open_tools()

        settled = self._stop_reason
        if settled is StopReason.CANCELLED or settled is StopReason.INTERRUPTED:
            kind = EventKind.TURN_CANCELLED
        elif settled is StopReason.ERROR:
            kind = EventKind.TURN_FAILED
        else:
            kind = EventKind.TURN_COMPLETED
        self._terminal_emitted = True
        self._emit(
            kind,
            {
                "stop_reason": str(settled),
                "unfinished": settled in UNFINISHED_STOP_REASONS,
                "steps": self._step_index,
                "usage": self._usage.to_payload(),
                "final_text": self._final_text,
                "elapsed_seconds": round(self.elapsed_seconds, 3),
            },
        )
        return settled

    def _drain_open_tools(self) -> None:
        """Give every still-open tool call its one terminal event.

        This is the invariant that makes an interrupted turn resumable: a
        provider history holding an assistant tool call with no matching result
        is rejected by several providers outright, and confuses the rest.
        """
        with self._lock:
            orphans = list(self._open_tools.values())
            self._open_tools.clear()
        for orphan in orphans:
            self._emit(
                EventKind.TOOL_CALL_CANCELLED,
                {
                    "tool_call_id": orphan.tool_call_id,
                    "tool_name": orphan.tool_name,
                    "interrupted": True,
                    "error": ABORTED_TOOL_TEXT,
                    "duration_ms": int((self._clock() - orphan.started_at) * 1000),
                },
            )

    # ── emission ──

    def _emit(self, kind: EventKind, payload: Mapping[str, Any]) -> Optional[KernelEvent]:
        if self.ledger is None:
            return None
        event = KernelEvent.build(kind, self.request.session_id, payload=payload)
        return self.ledger.append(event.for_turn(self.request))


def orphan_tool_results(
    open_calls: Sequence[str],
    *,
    text: str = ABORTED_TOOL_TEXT,
) -> list[dict[str, Any]]:
    """Build provider-shaped tool results for calls that never completed.

    The ledger records what happened; the *model* still needs a tool message per
    outstanding call or the next request carries an unanswered tool call. This
    helper keeps both sides using the same wording.
    """
    return [
        {"role": "tool", "tool_call_id": call_id, "content": text}
        for call_id in open_calls
    ]
