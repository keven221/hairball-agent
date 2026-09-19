"""The Kernel v2 execution path: a real second engine, not an observer.

Until this module existed, ``agent/kernel/facade.py`` had to log a warning and
run v1 whenever a session was pinned to v2, because there was nothing to drive.
This is the thing that drives: it owns the turn, calls the provider, dispatches
tools, and settles the ledger itself.

**Stage 1 holds prompt and tools constant on purpose.** The system prompt comes
from v1's own ``AIAgent._build_system_prompt`` and the tool schemas from the same
``get_tool_definitions`` call v1 makes, so a v1-vs-v2 comparison varies *only*
loop control and tool orchestration. If the prompt changed at the same time, any
measured difference could not be attributed to the new loop — the prompt is the
dominant behavioral variable. Replacing the legacy-derived prompt builder is
stage 2, run as a second experiment against a stage-1 baseline.

Known deviations from v1, kept explicit rather than hidden:

- Non-streaming provider calls. v1 streams; this engine uses
  ``ProviderRuntime.complete``. Content, tool calls, and usage are equivalent
  through the same transport normalization, but a surface that needs live
  deltas is not served yet.
- No compaction unless a ``compaction_check`` is injected. Tool-loop
  guardrails reuse v1's ``ToolCallGuardrailController`` (soft recovery
  guidance by default; hard stops only when config enables them).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from agent.display import _detect_tool_failure
from agent.doom_loop_runtime import DOOM_LOOP_SOFT_HINT as _DOOM_LOOP_SOFT_HINT
from agent.kernel.contracts import (
    Budget,
    EventKind,
    KernelEvent,
    TurnRequest,
    new_turn_id,
)
from agent.kernel.provider_runtime import ProviderRequest, ProviderRuntime
from agent.kernel.tool_outcome import looks_like_error
from agent.kernel.turn_loop import (
    CompactionCheck,
    SnapshotSource,
    StepReport,
    StopReason,
    TurnDecision,
    TurnLoop,
)
from agent.tool_guardrails import (
    ToolCallGuardrailController,
    append_toolguard_guidance,
    toolguard_synthetic_result,
)

logger = logging.getLogger(__name__)

#: v1 falls back to this mode when an agent does not name one.
DEFAULT_API_MODE = "chat_completions"


@dataclass
class EngineResult:
    """A turn's outcome, shaped so it can be diffed against v1's result dict.

    v1's ``run_conversation`` returns a plain dict (``agent/turn_finalizer.py``).
    Mirroring its key names is what makes an A/B a field-by-field comparison
    instead of a translation exercise.
    """

    final_response: str = ""
    messages: list = field(default_factory=list)
    api_calls: int = 0
    completed: bool = False
    failed: bool = False
    interrupted: bool = False
    turn_exit_reason: str = ""
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    session_id: str = ""
    turn_id: str = ""
    kernel_version: str = "v2"
    tool_calls: list = field(default_factory=list)
    doom_loop_hits: list = field(default_factory=list)
    completion_assessment: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        payload = {
            "final_response": self.final_response,
            "messages": self.messages,
            "api_calls": self.api_calls,
            "completed": self.completed,
            "failed": self.failed,
            "interrupted": self.interrupted,
            "turn_exit_reason": self.turn_exit_reason,
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "kernel_version": self.kernel_version,
            "tool_calls": self.tool_calls,
            "doom_loop_hits": self.doom_loop_hits,
        }
        if self.completion_assessment is not None:
            payload["completion_assessment"] = dict(self.completion_assessment)
        return payload


class KernelEngine:
    """Runs a turn on the v2 path, borrowing v1's prompt, tools, and client.

    The agent is a *resource*, not a driver: this engine reads its system
    prompt, toolset configuration, and API client, and never calls
    ``run_conversation``. That separation is what lets the same agent object
    serve both paths in an A/B with identical inputs.
    """

    def __init__(
        self,
        agent: Any,
        *,
        ledger: Any = None,
        session_id: str = "",
        compaction_check: Optional[CompactionCheck] = None,
        snapshot: Optional[SnapshotSource] = None,
        on_doom_loop: Optional[Callable[[str, Any], bool]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.agent = agent
        self.ledger = ledger
        self.session_id = session_id or str(getattr(agent, "session_id", "") or "")
        self.runtime = ProviderRuntime(ledger=ledger)
        self._compaction_check = compaction_check
        self._snapshot = snapshot
        self._on_doom_loop = on_doom_loop
        self._clock = clock
        self._doom_hits: list = []
        # Raw provider response for the step in flight; read only for pricing,
        # which needs the cache buckets the normalized form drops.
        self._last_raw: Any = None
        # Bounded follow-ups mirroring conversation_loop intent-ack / verify-on-stop.
        self._turn_user_message: Any = None

    # ── inputs borrowed from v1 ──

    def system_prompt(self, system_message: Optional[str] = None) -> str:
        """v1's prompt, byte for byte.

        Reuses the cached prompt when v1 already built one so a resumed session
        keeps the same cached prefix; otherwise calls v1's own builder. Inventing
        prompt text here would silently break the stage-1 control.
        """
        cached = getattr(self.agent, "_cached_system_prompt", None)
        if cached:
            return str(cached)
        builder = getattr(self.agent, "_build_system_prompt", None)
        if callable(builder):
            return str(builder(system_message) or "")
        return str(system_message or "")

    def tool_schemas(self) -> list:
        """The exact session-stable schema snapshot v1 would send.

        Recomputing from toolsets drops per-agent memory/context tools and the
        stop controller schema, while a registry refresh can also add tools
        mid-session and invalidate the prompt cache.  ``agent.tools`` is the
        canonical initialized snapshot for both engines.
        """
        import copy

        return copy.deepcopy(list(getattr(self.agent, "tools", None) or ()))

    @staticmethod
    def _tool_names(schemas: Sequence[Mapping[str, Any]]) -> list:
        """Names the model was actually offered.

        Passed to ``handle_function_call`` as ``enabled_tools`` so dispatch
        authorizes exactly what was advertised — the pitfall of leaving it
        implicit is that dispatch and schema can drift apart.
        """
        names = []
        for schema in schemas or []:
            function = schema.get("function") if isinstance(schema, Mapping) else None
            name = (function or {}).get("name") if isinstance(function, Mapping) else None
            name = name or (schema.get("name") if isinstance(schema, Mapping) else None)
            if name:
                names.append(str(name))
        return names

    # ── the turn ──

    def run_turn(
        self,
        user_message: Any,
        *,
        request: Optional[TurnRequest] = None,
        conversation_history: Optional[Sequence[Mapping[str, Any]]] = None,
        system_message: Optional[str] = None,
        max_iterations: int = 0,
    ) -> dict:
        agent = self.agent
        schemas = self.tool_schemas()
        enabled_tools = self._tool_names(schemas)

        turn = request or TurnRequest(
            session_id=self.session_id,
            turn_id=new_turn_id(),
            user_input=user_message,
            provider_ref=str(getattr(agent, "provider", "") or ""),
            model_ref=str(getattr(agent, "model", "") or ""),
            budget=Budget(
                max_iterations=(
                    max_iterations or int(getattr(agent, "max_iterations", 0) or 0)
                )
            ),
        )

        messages: list = [{"role": "system", "content": self.system_prompt(system_message)}]
        messages.extend(dict(item) for item in (conversation_history or []))
        messages.append({"role": "user", "content": user_message})

        api_mode = str(getattr(agent, "api_mode", "") or DEFAULT_API_MODE)
        model = str(getattr(agent, "model", "") or "")

        result = EngineResult(
            model=model,
            provider=str(getattr(agent, "provider", "") or ""),
            session_id=self.session_id,
            turn_id=turn.turn_id,
        )

        self._emit_accepted(turn, user_message)
        self._turn_user_message = user_message
        # Same per-turn reset v1's turn_context performs: without it, soft
        # recovery hints never fire and the model stays stuck in compliance mode.
        guardrails = self._guardrails()
        if guardrails is not None:
            guardrails.reset_for_turn()
        try:
            self.agent._verification_stop_nudges = 0
            self.agent._pre_verify_nudges = 0
            if not hasattr(self.agent, "_turn_file_mutation_paths"):
                self.agent._turn_file_mutation_paths = set()
            else:
                self.agent._turn_file_mutation_paths = set()
            self.agent._turn_failed_file_mutations = {}
            from agent.turn_workspace_tracker import begin_turn_workspace_tracking

            begin_turn_workspace_tracking(self.agent)
            from agent.completion_contract import completion_contract_runtime_for_turn
            self.agent._completion_contract_runtime = completion_contract_runtime_for_turn(
                self.agent, user_message
            )
        except Exception:
            pass
        try:
            controller = getattr(self.agent, "_stop_controller", None)
            todo_prelude = (
                controller.start_turn(
                    user_message if isinstance(user_message, str) else None,
                    has_prior_user=any(
                        isinstance(item, dict) and item.get("role") == "user"
                        for item in (conversation_history or [])
                    ),
                    enabled_tools=enabled_tools,
                )
                if controller is not None
                else ""
            )
            if todo_prelude and messages and isinstance(messages[-1], dict):
                messages[-1]["content"] = f"{messages[-1]['content']}\n\n{todo_prelude}"
        except Exception:
            logger.debug("kernel settle controller turn setup skipped", exc_info=True)

        with TurnLoop(
            turn,
            ledger=self.ledger,
            compaction_check=self._compaction_check,
            snapshot=self._snapshot,
            clock=self._clock,
        ) as loop:
            while loop.begin_step() is not None:
                report, stop = self._run_step(
                    loop,
                    api_mode=api_mode,
                    model=model,
                    messages=messages,
                    schemas=schemas,
                    enabled_tools=enabled_tools,
                    turn=turn,
                    result=result,
                )
                result.api_calls += 1
                decision = loop.end_step(report)
                if stop or decision is TurnDecision.STOP:
                    break
                if decision is TurnDecision.COMPACT:
                    # Stage 1 injects no compactor. Acknowledging and continuing
                    # is better than looping on an unsatisfiable request.
                    loop.note_compaction_done()

            result.final_response = loop.final_text or result.final_response
            usage = loop.usage
            result.input_tokens = usage.input_tokens
            result.output_tokens = usage.output_tokens
            result.estimated_cost_usd = usage.cost_usd

        stop_reason = loop.stop_reason
        result.turn_exit_reason = str(stop_reason.value if stop_reason else "")
        result.completed = stop_reason in (StopReason.NATURAL, None)
        result.failed = stop_reason is StopReason.ERROR
        result.interrupted = stop_reason is StopReason.CANCELLED
        result.messages = messages
        result.prompt_tokens = result.input_tokens
        result.completion_tokens = result.output_tokens
        result.total_tokens = result.input_tokens + result.output_tokens
        result.doom_loop_hits = list(self._doom_hits)
        try:
            tracker = getattr(self.agent, "_completion_contract_runtime", None)
            if tracker is not None:
                if stop_reason in {StopReason.CANCELLED, StopReason.INTERRUPTED}:
                    completion_reason = "cancelled"
                elif stop_reason in {
                    StopReason.BUDGET_ITERATIONS,
                    StopReason.BUDGET_WALL_CLOCK,
                    StopReason.BUDGET_COST,
                }:
                    completion_reason = "budget_iterations"
                elif stop_reason is StopReason.NATURAL or stop_reason is None:
                    completion_reason = "natural"
                else:
                    completion_reason = "failed"
                result.completion_assessment = tracker.assess(
                    terminal_reason=completion_reason,
                    phase="terminal",
                ).to_dict()
        except Exception:
            logger.debug("completion contract v2 terminal assessment skipped", exc_info=True)
        return result.to_dict()

    # ── one step ──

    def _run_step(
        self,
        loop: TurnLoop,
        *,
        api_mode: str,
        model: str,
        messages: list,
        schemas: list,
        enabled_tools: list,
        turn: TurnRequest,
        result: EngineResult,
    ) -> tuple[StepReport, bool]:
        from agent.stop_controller import strip_hook_prompt_metadata

        provider_request = ProviderRequest(
            api_mode=api_mode,
            model=model,
            messages=strip_hook_prompt_metadata(messages),
            tools=schemas or None,
            params=self._provider_params(),
        )

        # Clear first: pricing must never read the previous step's response if
        # this step's invoke never lands.
        self._last_raw = None
        try:
            provider_result = self.runtime.complete(
                provider_request, invoke=self._invoke, turn=turn
            )
        except Exception as exc:  # provider failure ends the turn
            logger.warning("kernel v2 provider call failed: %s", exc)
            return StepReport(finish_reason="error", error=str(exc)), True

        response = provider_result.response
        usage = response.usage
        report_kwargs = {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "cost_usd": self._step_cost(model, api_mode, usage),
        }

        text = response.content or ""
        tool_calls = list(response.tool_calls or [])

        assistant: dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": call.id or f"call_{index}",
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for index, call in enumerate(tool_calls)
            ]
        if not tool_calls:
            controller = getattr(self.agent, "_stop_controller", None)
            decision = (
                controller.evaluate_candidate(
                    text=text,
                    messages=messages,
                    enabled_tools=enabled_tools,
                )
                if controller is not None
                else None
            )
            if decision is not None and decision.should_continue:
                hook_prompt = controller.build_continuation_message()
                if hook_prompt is not None:
                    messages.extend([assistant, hook_prompt])
                return StepReport(
                    finish_reason="incomplete",
                    force_continue=True,
                    **report_kwargs,
                ), False
            messages.append(assistant)
            if text:
                loop.set_final_text(text)
                self._emit(
                    EventKind.ASSISTANT_MESSAGE_COMPLETED, turn, {"text": text}
                )
            return StepReport(
                finish_reason=response.finish_reason or "stop", **report_kwargs
            ), False

        messages.append(assistant)
        halt = self._run_tools(
            loop,
            tool_calls=tool_calls,
            messages=messages,
            enabled_tools=enabled_tools,
            turn=turn,
            result=result,
        )
        return StepReport(
            finish_reason=response.finish_reason or "tool_calls",
            tool_call_count=len(tool_calls),
            **report_kwargs,
        ), halt

    def _guardrails(self) -> Optional[ToolCallGuardrailController]:
        """Reuse the agent's v1 controller when present; otherwise a fresh one."""
        existing = getattr(self.agent, "_tool_guardrails", None)
        if isinstance(existing, ToolCallGuardrailController):
            return existing
        if existing is not None:
            return existing  # type: ignore[return-value]
        controller = ToolCallGuardrailController()
        try:
            self.agent._tool_guardrails = controller
        except Exception:
            pass
        return controller

    def _run_tools(
        self,
        loop: TurnLoop,
        *,
        tool_calls: Sequence[Any],
        messages: list,
        enabled_tools: list,
        turn: TurnRequest,
        result: EngineResult,
    ) -> bool:
        """Dispatch each call, settling every one exactly once."""
        from model_tools import (
            SessionToolRouter,
            handle_function_call,
            is_agent_loop_tool,
        )

        halt = False
        guardrails = self._guardrails()
        session_tool_router = getattr(self.agent, "_session_tool_router", None)
        if not isinstance(session_tool_router, SessionToolRouter):
            session_tool_router = SessionToolRouter.from_names(enabled_tools)
        for index, call in enumerate(tool_calls):
            call_id = call.id or f"call_{index}"
            args = _parse_arguments(call.arguments)
            doom_tripped = False

            # Kernel v2 owns dispatch and therefore cannot rely on v1's
            # sequential/concurrent executor to enforce Plan mode.  Reuse the
            # exact same execution-time decision seam before guardrails or
            # registry dispatch; this also binds the Plan ContextVar used by
            # the single-artifact writer and read-only terminal sandbox.
            from agent.tool_executor import _plan_mode_tool_block

            plan_block = _plan_mode_tool_block(self.agent, call.name, args)
            if plan_block is not None:
                output = plan_block["result"]
                loop.tool_started(call_id, call.name, arguments=args)
                loop.tool_settled(
                    call_id,
                    ok=False,
                    payload={
                        "error": plan_block["message"],
                        "code": plan_block["error_type"],
                    },
                )
                result.tool_calls.append(
                    {"id": call_id, "name": call.name, "arguments": args}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": call.name,
                        "content": output,
                    }
                )
                try:
                    controller = getattr(self.agent, "_stop_controller", None)
                    if controller is not None:
                        controller.note_tool_result(
                            call.name,
                            output,
                            failed=True,
                            blocked=True,
                        )
                except Exception:
                    logger.debug("kernel Plan blocked-tool observation skipped", exc_info=True)
                try:
                    tracker = getattr(self.agent, "_completion_contract_runtime", None)
                    if tracker is not None:
                        tracker.record_tool_outcome(
                            tool_name=call.name,
                            tool_call_id=call_id,
                            tool_result=output,
                            failed=True,
                            details={
                                "status": "blocked",
                                "guardrail": plan_block["error_type"],
                            },
                        )
                except Exception:
                    logger.debug("completion contract v2 Plan block evidence skipped", exc_info=True)
                continue

            if loop.check_doom_loop(call.name, args):
                # A repeat spin is information, not automatically a stop: the
                # host decides, because only it knows whether a user is there to
                # approve. Recording it either way keeps the A/B able to see it.
                self._note_doom_hit(call.name, args)
                doom_tripped = True
                if self._on_doom_loop is not None and not self._on_doom_loop(
                    call.name, args
                ):
                    loop.request_stop(StopReason.GUARDRAIL_HALT)
                    halt = True

            if guardrails is not None:
                before = guardrails.before_call(call.name, args)
                if not before.allows_execution:
                    output = toolguard_synthetic_result(before)
                    loop.tool_started(call_id, call.name, arguments=args)
                    loop.tool_settled(
                        call_id,
                        ok=False,
                        payload={"error": before.message, "guardrail": before.code},
                    )
                    # V2 owns direct tool dispatch, so it must adapt this real
                    # blocked terminal fact into the same deep completion
                    # module the v1 executor uses.  A blocked call is evidence
                    # of a blocker, never a silent omission or false success.
                    try:
                        tracker = getattr(self.agent, "_completion_contract_runtime", None)
                        if tracker is not None:
                            tracker.record_tool_outcome(
                                tool_name=call.name,
                                tool_call_id=call_id,
                                tool_result=output,
                                failed=True,
                                details={
                                    "status": "blocked",
                                    "guardrail": before.code,
                                },
                            )
                    except Exception:
                        logger.debug("completion contract v2 blocked-tool evidence skipped", exc_info=True)
                    result.tool_calls.append(
                        {"id": call_id, "name": call.name, "arguments": args}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": call.name,
                            "content": output,
                        }
                    )
                    if before.should_halt:
                        loop.request_stop(StopReason.GUARDRAIL_HALT)
                        halt = True
                    if halt:
                        break
                    continue

            loop.tool_started(call_id, call.name, arguments=args)
            try:
                if is_agent_loop_tool(call.name):
                    # These tools depend on live AIAgent state. Reuse the
                    # legacy loop's existing stateful adapter rather than the
                    # registry path that intentionally rejects them.
                    output = session_tool_router.authorization_error(call.name)
                    if output is None:
                        output = self.agent._invoke_tool(
                            call.name,
                            args,
                            turn.turn_id,
                            tool_call_id=call_id,
                            messages=messages,
                        )
                else:
                    output = handle_function_call(
                        call.name,
                        args,
                        tool_call_id=call_id,
                        session_id=self.session_id,
                        turn_id=turn.turn_id,
                        enabled_tools=enabled_tools,
                        enabled_toolsets=getattr(self.agent, "enabled_toolsets", None),
                        disabled_toolsets=getattr(self.agent, "disabled_toolsets", None),
                        session_tool_router=session_tool_router,
                    )
            except Exception as exc:
                output = json.dumps({"error": str(exc)})
                failed = True
                loop.tool_settled(call_id, ok=False, payload={"error": str(exc)})
            else:
                failed, _ = _detect_tool_failure(call.name, output)
                failed = failed or looks_like_error(output)
                loop.tool_settled(
                    call_id,
                    ok=not failed,
                    payload={"summary": output[:400]},
                )
            try:
                self.agent._record_file_mutation_result(
                    call.name,
                    args,
                    output,
                    bool(failed),
                )
            except Exception:
                logger.debug("kernel workspace mutation reconciliation skipped", exc_info=True)

            try:
                controller = getattr(self.agent, "_stop_controller", None)
                if controller is not None:
                    controller.note_tool_result(
                        call.name,
                        output,
                        failed=bool(failed),
                        blocked=False,
                    )
            except Exception:
                logger.debug("kernel Plan decision observation skipped", exc_info=True)

            # This is the v2 equivalent of v1's executor terminal seam. It
            # observes the canonical result before any guardrail prose is
            # appended, so structured verification evidence keeps its shape.
            try:
                tracker = getattr(self.agent, "_completion_contract_runtime", None)
                if tracker is not None:
                    tracker.record_tool_outcome(
                        tool_name=call.name,
                        tool_call_id=call_id,
                        tool_result=output,
                        failed=bool(failed),
                    )
            except Exception:
                logger.debug("completion contract v2 tool evidence skipped", exc_info=True)

            if guardrails is not None:
                after = guardrails.after_call(
                    call.name, args, output, failed=failed
                )
                output = append_toolguard_guidance(output, after)
                if after.should_halt:
                    loop.request_stop(StopReason.GUARDRAIL_HALT)
                    halt = True

            if doom_tripped and self._on_doom_loop is None and not halt:
                output = (output or "") + _DOOM_LOOP_SOFT_HINT

            result.tool_calls.append(
                {"id": call_id, "name": call.name, "arguments": args}
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": call.name,
                    "content": output,
                }
            )
            if halt:
                break
        if not halt:
            try:
                controller = getattr(self.agent, "_stop_controller", None)
                nudge = (
                    controller.take_mid_run_nudge(enabled_tools)
                    if controller is not None
                    else None
                )
                if nudge is not None:
                    messages.append(nudge)
            except Exception:
                logger.debug("kernel mid-run Todo reconciliation skipped", exc_info=True)
        return halt

    # ── provider plumbing ──

    def _provider_params(self) -> dict:
        """Sampling params from the agent, so both paths ask for the same thing."""
        params: dict[str, Any] = {}
        for attr, key in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("max_tokens", "max_tokens"),
        ):
            value = getattr(self.agent, attr, None)
            if value is not None:
                params[key] = value
        return params

    def _step_cost(self, model: str, api_mode: str, usage: Any) -> float:
        """Price one call through v1's own usage normalization and pricing tables.

        Prices from the **raw** provider response, not the normalized one.
        ``NormalizedResponse.usage`` carries three totals and no cache buckets, so
        pricing from it bills cache hits at the full input rate — on a provider
        with a large cache discount (DeepSeek) that roughly doubles the reported
        cost even when the token counts are identical.

        ``normalize_usage`` is v1's own extractor and already handles the
        per-provider shapes (``prompt_tokens_details.cached_tokens`` plus the
        top-level fallbacks some proxies use). Reusing it is what keeps the two
        paths agreeing with the user's invoice; a second implementation is
        exactly where they would silently diverge.

        Returns ``0.0`` when pricing is unknown or the lookup fails: a missing
        price must not end an otherwise healthy turn.
        """
        raw_usage = getattr(self._last_raw, "usage", None) if self._last_raw else None
        if raw_usage is None and usage is None:
            return 0.0
        try:
            from agent.usage_pricing import (
                CanonicalUsage,
                estimate_usage_cost,
                normalize_usage,
            )

            provider = str(getattr(self.agent, "provider", "") or "") or None
            if raw_usage is not None:
                canonical = normalize_usage(
                    raw_usage, provider=provider, api_mode=api_mode
                )
            else:
                # No raw response to read (a transport that returns an already
                # normalized object). Cache buckets are unavailable here, so this
                # is an upper bound on cost rather than an exact figure.
                canonical = CanonicalUsage(
                    input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                )
            result = estimate_usage_cost(
                model,
                canonical,
                provider=provider,
                base_url=str(getattr(self.agent, "base_url", "") or "") or None,
                api_key=getattr(self.agent, "api_key", "") or None,
            )
        except Exception as exc:
            logger.debug("kernel v2 cost estimation failed: %s", exc)
            return 0.0
        amount = getattr(result, "amount_usd", None)
        return float(amount) if amount is not None else 0.0

    def _invoke(self, kwargs: dict) -> Any:
        """Reuse v1's already-constructed client — same credentials, same base URL.

        The raw response is kept for this step so cost can be priced from the
        provider's full usage object, which carries cache buckets the normalized
        form drops.
        """
        client = getattr(self.agent, "client", None)
        if client is None:
            raise RuntimeError("agent has no client; kernel v2 cannot call the provider")
        controller = getattr(self.agent, "_stop_controller", None)
        if controller is not None:
            controller.apply_continuation_policy(
                kwargs,
                str(getattr(self.agent, "api_mode", "") or DEFAULT_API_MODE),
            )
        try:
            raw = client.chat.completions.create(**kwargs)
        except Exception as exc:
            if controller is None or not controller.note_tool_choice_rejection(
                exc, kwargs
            ):
                raise
            # One bounded capability recovery: same request, same tools and
            # reasoning, but omit the rejected control field (implicit auto).
            controller.apply_continuation_policy(
                kwargs,
                str(getattr(self.agent, "api_mode", "") or DEFAULT_API_MODE),
            )
            raw = client.chat.completions.create(**kwargs)
        self._last_raw = raw
        return raw

    def _note_doom_hit(self, tool_name: str, args: Any) -> None:
        self._doom_hits.append({"tool_name": tool_name, "arguments": args})

    # ── ledger ──

    def _emit_accepted(self, turn: TurnRequest, user_message: Any) -> None:
        self._emit(
            EventKind.TURN_ACCEPTED,
            turn,
            {
                "user_input_text": _text_of(user_message),
                "provider_ref": turn.provider_ref,
                "model_ref": turn.model_ref,
                "kernel_backend": "v2",
            },
        )

    def _emit(self, kind: EventKind, turn: TurnRequest, payload: Mapping[str, Any]) -> None:
        if self.ledger is None:
            return
        event = KernelEvent.build(kind, turn.session_id, payload=dict(payload))
        self.ledger.append(event.for_turn(turn))


def _parse_arguments(raw: Any) -> dict:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _looks_like_error(output: str) -> bool:
    """Shared with the v1 tap for dual-path settlement agreement tests."""
    return looks_like_error(output)


def _text_of(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value.get("content") or value.get("text") or "")
    return str(value or "")
