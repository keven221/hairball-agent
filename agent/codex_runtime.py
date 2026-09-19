"""reference API runtime — App Server and Responses-API streaming paths.

Extracted from :class:`AIAgent` to keep the agent loop file focused.
Each function takes the parent ``AIAgent`` as its first argument
(``agent``).  AIAgent keeps thin forwarder methods for backward
compatibility.

* ``run_codex_app_server_turn`` — drives one turn through the
  ``codex_app_server`` subprocess client (used when a reference CLI install
  is the active provider).
* ``run_codex_stream`` — streams a reference Responses API call (the
  ``codex_responses`` api_mode).
* ``run_codex_create_stream_fallback`` — recovery path when the
  Responses ``stream=True`` initial create fails.
"""

from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List

from agent.stream_single_writer import claim_stream_writer, stream_writer_is_current

logger = logging.getLogger(__name__)


def _codex_note_to_tool_progress(note: dict) -> tuple[str, str, str, dict] | None:
    """Map a reference app-server item notification to Hairball tool-progress.

    Returns ``(event_type, tool_name, preview, args)`` where ``event_type`` is
    ``tool.started`` (from ``item/started``) or ``tool.completed`` (from
    ``item/completed``). Returns None for items that aren't tool-shaped.

    The reference app-server runtime processes item notifications for command
    execution, file changes, and MCP/dynamic tool calls. Bridging both start
    and complete keeps the same ``item.id`` ledger contract as executor Phase2
    (P2c) so UI/gateway rows do not stay stuck on started-only.
    """
    if not isinstance(note, dict):
        return None
    method = str(note.get("method") or "")
    if method == "item/started":
        event_type = "tool.started"
    elif method == "item/completed":
        event_type = "tool.completed"
    else:
        return None
    params = note.get("params") or {}
    item = params.get("item") or {}
    if not isinstance(item, dict):
        return None

    item_type = item.get("type") or ""
    if item_type == "commandExecution":
        command = item.get("command") or ""
        preview = command
        if event_type == "tool.completed":
            out = str(item.get("aggregatedOutput") or "").strip()
            if out:
                preview = out[:400]
        return event_type, "exec_command", preview, {"command": command, "cwd": item.get("cwd") or ""}

    if item_type == "fileChange":
        changes = item.get("changes") or []
        preview = "file changes"
        if isinstance(changes, list) and changes:
            paths = [
                str(change.get("path"))
                for change in changes
                if isinstance(change, dict) and change.get("path")
            ]
            if paths:
                preview = ", ".join(paths[:3])
                if len(paths) > 3:
                    preview += f", +{len(paths) - 3} more"
        return event_type, "apply_patch", preview, {"changes": changes}

    if item_type == "mcpToolCall":
        server = item.get("server") or "mcp"
        tool = item.get("tool") or "unknown"
        args = item.get("arguments") or {}
        if not isinstance(args, dict):
            args = {"arguments": args}
        return event_type, f"mcp.{server}.{tool}", tool, args

    if item_type == "dynamicToolCall":
        tool = item.get("tool") or "unknown"
        args = item.get("arguments") or {}
        if not isinstance(args, dict):
            args = {"arguments": args}
        return event_type, tool, tool, args

    return None


def _codex_completed_progress_extras(item: dict) -> dict[str, Any]:
    """Extra kwargs for tool.completed from a reference item/completed payload."""
    if not isinstance(item, dict):
        return {}
    extras: dict[str, Any] = {}
    item_type = item.get("type") or ""
    if item_type == "commandExecution":
        out = item.get("aggregatedOutput")
        if out is not None:
            extras["result"] = str(out)[:12000]
        exit_code = item.get("exitCode")
        status = str(item.get("status") or "").lower()
        if (isinstance(exit_code, int) and exit_code != 0) or status in {"failed", "error"}:
            extras["is_error"] = True
    elif item_type in {"mcpToolCall", "dynamicToolCall"}:
        result = item.get("result")
        if result is not None:
            extras["result"] = result if isinstance(result, str) else str(result)[:12000]
        if item.get("isError") or item.get("is_error") or str(item.get("status") or "").lower() in {"failed", "error"}:
            extras["is_error"] = True
    elif item_type == "fileChange":
        # No stdout body; still mark completed so live rows can seal.
        extras["result"] = "ok"
    return extras


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


def _record_codex_app_server_usage(agent, turn) -> dict[str, Any]:
    """Translate reference app-server token usage into Hairball accounting.

    reference app-server reports usage via thread/tokenUsage/updated as:
    inputTokens, cachedInputTokens, outputTokens, reasoningOutputTokens,
    totalTokens.

    Hairball' canonical prompt bucket includes uncached input + cached input.
    The reference app-server protocol does not currently expose cache-write tokens,
    so that bucket remains zero on this runtime.

    Even when reference omits usage for a turn, Hairball should still count that turn
    as one API call for session/status accounting.
    """
    agent.session_api_calls += 1

    usage = getattr(turn, "token_usage_last", None)
    if not isinstance(usage, dict) or not usage:
        compressor = getattr(agent, "context_compressor", None)
        if (
            compressor is not None
            and getattr(compressor, "awaiting_real_usage_after_compression", False)
        ):
            # No usage means this turn cannot adjudicate the pending compaction.
            # Consume the marker so a later unrelated reading is not charged to
            # it and preflight deferral cannot stay latched indefinitely.
            compressor.update_from_response({})
        if agent._session_db and agent.session_id:
            try:
                if not agent._session_db_created:
                    agent._ensure_db_session()
                agent._session_db.update_token_counts(
                    agent.session_id,
                    model=agent.model,
                    billing_provider=agent.provider,
                    billing_base_url=agent.base_url,
                    billing_mode="subscription_included",
                    api_call_count=1,
                )
            except Exception as exc:
                logger.debug(
                    "reference app-server api-call persistence failed (session=%s): %s",
                    agent.session_id, exc,
                )
        return {}

    from agent.usage_pricing import (
        CanonicalUsage,
        accumulate_session_cost,
        estimate_usage_cost,
        extract_provider_cost_usd,
    )

    input_tokens = _coerce_usage_int(usage.get("inputTokens"))
    cache_read_tokens = _coerce_usage_int(usage.get("cachedInputTokens"))
    output_tokens = _coerce_usage_int(usage.get("outputTokens"))
    reasoning_tokens = _coerce_usage_int(usage.get("reasoningOutputTokens"))
    reported_total = _coerce_usage_int(usage.get("totalTokens"))

    canonical_usage = CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=0,
        reasoning_tokens=reasoning_tokens,
        raw_usage=usage,
    )
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = reported_total or canonical_usage.total_tokens
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": canonical_usage.reasoning_tokens,
    }

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        try:
            compressor.update_from_response(usage_dict)
            context_window = getattr(turn, "model_context_window", None)
            if isinstance(context_window, int) and context_window > 0:
                compressor.context_length = context_window
        except Exception:
            logger.debug("codex app-server usage update failed", exc_info=True)

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens
    agent.session_reasoning_tokens += canonical_usage.reasoning_tokens

    cost_result = estimate_usage_cost(
        agent.model,
        canonical_usage,
        provider=agent.provider,
        base_url=agent.base_url,
        api_key=getattr(agent, "api_key", ""),
    )
    reported_actual_cost = extract_provider_cost_usd(
        usage,
        provider=agent.provider,
        base_url=agent.base_url,
    )
    billable_call_count = 0 if cost_result.status == "included" else 1
    cost_summary = accumulate_session_cost(
        agent,
        estimated_cost_usd=cost_result.amount_usd,
        actual_cost_usd=reported_actual_cost,
        actual_call_count=(
            1
            if reported_actual_cost is not None and billable_call_count
            else 0
        ),
        billable_call_count=billable_call_count,
        estimated_status=cost_result.status,
        estimated_source=cost_result.source,
    )

    if agent._session_db and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.update_token_counts(
                agent.session_id,
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                reasoning_tokens=canonical_usage.reasoning_tokens,
                estimated_cost_usd=float(cost_result.amount_usd)
                if cost_result.amount_usd is not None else None,
                actual_cost_usd=(
                    reported_actual_cost if billable_call_count else None
                ),
                cost_actual_call_count=(
                    1
                    if reported_actual_cost is not None and billable_call_count
                    else 0
                ),
                cost_total_call_count=billable_call_count,
                cost_status=cost_summary.status,
                cost_source=cost_summary.source,
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included"
                if cost_result.status == "included" else None,
                model=agent.model,
                api_call_count=1,
            )
        except Exception as exc:
            logger.debug(
                "reference app-server token persistence failed (session=%s, tokens=%d): %s",
                agent.session_id, total_tokens, exc,
            )

    return {
        **usage_dict,
        "last_prompt_tokens": prompt_tokens,
        "estimated_cost_usd": float(cost_result.amount_usd)
        if cost_result.amount_usd is not None else None,
        "cost_usd": cost_summary.display_cost_usd,
        "actual_cost_usd": cost_summary.actual_cost_usd,
        "cost_actual_call_count": cost_summary.actual_call_count,
        "cost_total_call_count": cost_summary.total_call_count,
        "actual_cost_coverage": cost_summary.actual_coverage,
        "actual_cost_complete": cost_summary.actual_complete,
        "cost_status": cost_summary.status,
        "cost_source": cost_summary.source,
    }


def _record_codex_app_server_compaction(
    agent,
    turn,
    *,
    approx_tokens: int | None = None,
    force: bool = False,
) -> bool:
    """Record a reference-native context compaction boundary in Hairball state.

    The app-server owns the compacted thread context, so Hairball should not
    rewrite local transcript rows here; state.db records the boundary via the
    session event/usage counters while preserving the visible transcript.
    """
    if not force and not getattr(turn, "compacted", False):
        return False

    thread_id = getattr(turn, "thread_id", None) or ""
    turn_id = getattr(turn, "turn_id", None) or ""
    logger.info(
        "codex app-server compaction observed: session=%s thread=%s turn=%s force=%s",
        getattr(agent, "session_id", None) or "none",
        thread_id,
        turn_id,
        force,
    )
    if not force:
        try:
            from agent.conversation_compression import COMPACTION_STATUS

            agent._emit_status(COMPACTION_STATUS)
        except Exception:
            pass

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        compressor.compression_count = getattr(
            compressor, "compression_count", 0
        ) + 1
        compressor.last_compression_rough_tokens = approx_tokens or 0
        # The app server has already completed a real compaction boundary. Its
        # usage update (when supplied) is therefore the same real-vs-real
        # effectiveness verdict used by the normal compression path.
        record_boundary = getattr(
            type(compressor), "record_completed_compaction", None
        )
        if callable(record_boundary):
            # reference owns this summary. A prior Hairball deterministic-fallback
            # flag must not leak into the native boundary's quality verdict.
            record_boundary(compressor, used_fallback=False)
        elif hasattr(compressor, "_verify_compaction_cleared_threshold"):
            compressor._verify_compaction_cleared_threshold = True
        if not getattr(turn, "token_usage_last", None):
            compressor.last_prompt_tokens = -1
            compressor.last_completion_tokens = 0
            compressor.awaiting_real_usage_after_compression = True

    agent._last_compaction_in_place = False
    try:
        if getattr(agent, "event_callback", None):
            agent.event_callback(
                "session:compress",
                {
                    "platform": getattr(agent, "platform", None) or "",
                    "session_id": getattr(agent, "session_id", None) or "",
                    "old_session_id": "",
                    "in_place": False,
                    "compression_count": getattr(
                        compressor, "compression_count", 0
                    )
                    if compressor is not None
                    else 0,
                    "runtime": "codex_app_server",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                },
            )
    except Exception:
        logger.debug("event_callback error on codex session:compress", exc_info=True)

    return True


def run_codex_app_server_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """reference app-server runtime path. Hands the entire turn to a `codex
    app-server` subprocess and projects its events back into Hairball'
    messages list so memory/skill review keep working.

    Called from run_conversation() when agent.api_mode == "codex_app_server".
    Returns the same dict shape as the chat_completions path.
    """
    from agent.transports.codex_app_server_session import (
        CodexAppServerSession,
        _ServerRequestRouting,
    )

    # Lazy session: one CodexAppServerSession per AIAgent instance.
    # Spawned on first turn, reused across turns, closed at AIAgent
    # shutdown (see _cleanup hook).
    if not hasattr(agent, "_codex_session") or agent._codex_session is None:
        from agent.runtime_cwd import resolve_agent_cwd

        cwd = getattr(agent, "session_cwd", None) or str(resolve_agent_cwd())
        # Approval callback: defer to Hairball' standard prompt flow if a
        # CLI thread has installed one. Gateway / cron contexts get the
        # codex-side fail-closed default.
        try:
            from tools.terminal_tool import _get_approval_callback
            approval_callback = _get_approval_callback()
        except Exception:
            approval_callback = None

        # Gateway / cron contexts have no UI to surface codex's approval
        # requests through, so codex app-server exec / apply_patch requests
        # fail closed (silently decline) by default. When the user has
        # explicitly opted out of Hairball approvals — via `approvals.mode: off`
        # in config, the /yolo session toggle, or --yolo / HAIRBALL_YOLO_MODE —
        # honor that and let codex's own sandbox permission profile
        # (~/.codex/config.toml) be the policy gate instead of double-gating
        # with a missing Hairball UI. Defaults (manual/smart/unset) preserve the
        # current fail-closed behavior — this is a no-op for those users.
        auto_approve_requests = False
        try:
            from tools.approval import is_approval_bypass_active

            auto_approve_requests = is_approval_bypass_active()
        except Exception:
            logger.debug(
                "codex app-server: approval-bypass lookup failed; "
                "keeping fail-closed default",
                exc_info=True,
            )

        def _on_codex_event(note: dict) -> None:
            # Bridge reference app-server item/started|completed to Hairball
            # tool-progress so gateways show verbose breadcrumbs and the UI
            # ledger can complete the same item.id (P2c).
            progress_callback = getattr(agent, "tool_progress_callback", None)
            if progress_callback is None:
                return
            mapped = _codex_note_to_tool_progress(note)
            if mapped is None:
                return
            event_type, tool_name, preview, args = mapped
            progress_kw: Dict[str, Any] = {}
            try:
                from agent.tool_executor import tool_progress_api_call_id_enabled

                item = (note.get("params") or {}).get("item") or {}
                if not isinstance(item, dict):
                    item = {}
                if tool_progress_api_call_id_enabled():
                    call_id = str(item.get("id") or "").strip()
                    if call_id:
                        progress_kw["tool_call_id"] = call_id
                if event_type == "tool.completed":
                    progress_kw.update(_codex_completed_progress_extras(item))
            except Exception:
                progress_kw = {}
            try:
                progress_callback(event_type, tool_name, preview, args, **progress_kw)
            except Exception:
                logger.debug("codex tool-progress callback raised", exc_info=True)

        agent._codex_session = CodexAppServerSession(
            cwd=cwd,
            approval_callback=approval_callback,
            request_routing=_ServerRequestRouting(
                auto_approve_exec=auto_approve_requests,
                auto_approve_apply_patch=auto_approve_requests,
            ),
            on_event=_on_codex_event,
        )

    # NOTE: the user message is ALREADY appended to messages by the
    # standard run_conversation() flow (line ~11823) before the early
    # return reaches us. Do NOT append again — that would duplicate.

    try:
        turn = agent._codex_session.run_turn(user_input=user_message)
    except Exception as exc:
        logger.exception("codex app-server turn failed")
        # Crash → unconditionally drop the session so the next turn
        # respawns from scratch instead of reusing a dead client.
        try:
            agent._codex_session.close()
        except Exception:
            pass
        agent._codex_session = None
        return {
            "final_response": (
                f"reference app-server turn failed: {exc}. "
                f"Fall back to default runtime with `/codex-runtime auto`."
            ),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
        }

    # If the turn signalled the underlying client is wedged (deadline
    # blown, post-tool watchdog tripped, OAuth refresh died, subprocess
    # exited), retire the session so the next turn respawns codex
    # rather than riding the broken process. Mirrors openclaw beta.8's
    # "retire timed-out app-server clients" fix.
    if getattr(turn, "should_retire", False):
        logger.warning(
            "codex app-server session retired (turn error: %s)",
            turn.error,
        )
        try:
            agent._codex_session.close()
        except Exception:
            pass
        agent._codex_session = None

    # Splice projected messages into the conversation. The projector emits
    # standard {role, content, tool_calls, tool_call_id} entries, which
    # is exactly what curator.py / sessions DB expect.
    if turn.projected_messages:
        messages.extend(turn.projected_messages)

        # Persist the newly-projected assistant/tool messages ourselves.
        # This path is an early return that bypasses conversation_loop, whose
        # normal per-step _persist_session() calls would otherwise flush them.
        # The inbound user turn was already flushed at turn start
        # (turn_context.py _persist_session), and _flush_messages_to_session_db
        # is idempotent via the intrinsic _DB_PERSISTED_MARKER — so this writes
        # ONLY the new codex projected rows and does NOT re-write the user turn.
        # Keeping the agent as the sole persister lets us return
        # agent_persisted=True below, so the gateway skips its own DB write and
        # we avoid the #860/#42039 duplicate user-message write (append_message
        # is a raw INSERT with no dedup, so a gateway re-write would duplicate
        # the already-flushed user turn). See gateway/run.py agent_persisted.
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug(
                    "codex app-server projected-message flush failed",
                    exc_info=True,
                )


    # Counter ticks for the agent-improvement loop.
    # _turns_since_memory and _user_turn_count are ALREADY incremented
    # in the run_conversation() pre-loop block (lines ~11793-11817) so we
    # do NOT touch them here — that would double-count.
    # Only _iters_since_skill needs explicit increment, since the
    # chat_completions loop bumps it per tool iteration (line ~12110)
    # and that loop is bypassed on this path.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )
    _record_codex_app_server_compaction(agent, turn)
    usage_result = _record_codex_app_server_usage(agent, turn)
    api_calls = 1

    # Now check the skill nudge AFTER iters were incremented — same
    # pattern the chat_completions path uses (line ~15432).
    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    # External memory provider sync (mirrors line ~15439). Skipped on
    # interrupt/error to avoid feeding partial transcripts to memory.
    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
                messages=messages,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    # Background review fork — same cadence + signature as the default
    # path (line ~15449). Only fires when a trigger actually tripped AND
    # we have a real final response.
    if (
        turn.final_text
        and not turn.interrupted
        and (should_review_memory or should_review_skills)
    ):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=should_review_memory,
                review_skills=should_review_skills,
            )
        except Exception:
            logger.debug("background review spawn raised", exc_info=True)

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": api_calls,
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "error": turn.error,
        # The codex app-server runtime IS an early-return path that bypasses
        # conversation_loop, but we flush the projected assistant/tool messages
        # ourselves above (see the _flush_messages_to_session_db call after
        # messages.extend). The inbound user turn was already flushed at turn
        # start (turn_context._persist_session) and the flush dedups via
        # _DB_PERSISTED_MARKER, so state.db ends up with each real message
        # exactly once and session_search / conversation-distill see the full
        # gateway conversation. Report agent_persisted=True so the gateway
        # skips its own append_to_transcript DB write — writing again there
        # would re-INSERT the already-flushed user turn (append_message has no
        # dedup), reintroducing the #860 / #42039 duplicate-write bug.
        "agent_persisted": True,
        "codex_thread_id": turn.thread_id,
        "codex_turn_id": turn.turn_id,
        **usage_result,
    }


# ---------------------------------------------------------------------------
# Event-driven Responses streaming
#
# OpenAI ships its consumer reference backend (chatgpt.com/backend-api/codex) on
# a different schedule from the openai Python SDK.  The high-level
# ``client.responses.stream(...)`` helper reconstructs a typed Response from
# the terminal ``response.completed`` event's ``response.output`` field, and
# when that field drifts to ``null`` (gpt-5.5, May 2026) the SDK raises
# ``TypeError: 'NoneType' object is not iterable`` mid-iteration.
#
# We sidestep the whole class of failure by going one level lower:
# ``client.responses.create(stream=True)`` returns the raw AsyncIterable of
# SSE events, and we assemble the final response object purely from
# ``response.output_item.done`` events as they arrive.  We never read
# ``response.completed.response.output`` for content reconstruction, so the
# backend can return ``null``, ``[]``, a string, or omit the field entirely
# and we don't care.
#
# This mirrors what the legacy TS implementation does for the same backend
# and is structurally immune to the bug class rather than patched.
# ---------------------------------------------------------------------------


_TERMINAL_EVENT_TYPES = frozenset({
    "response.completed",
    "response.incomplete",
    "response.failed",
})


def _event_field(event: Any, name: str, default: Any = None) -> Any:
    """Field access that handles both attr-style (SDK objects) and dict (raw JSON) events."""
    value = getattr(event, name, None)
    if value is None and isinstance(event, dict):
        value = event.get(name, default)
    return value if value is not None else default


def _item_field(item: Any, name: str, default: Any = None) -> Any:
    """Field access for nested Response items (attr-style SDK object or dict)."""
    value = getattr(item, name, None)
    if value is None and isinstance(item, dict):
        value = item.get(name, default)
    return value if value is not None else default


def _raise_stream_error(event: Any) -> None:
    """Raise a ``_StreamErrorEvent`` from a ``type=error`` SSE frame.

    Imported lazily so this module stays importable from places that don't
    pull in ``run_agent`` (e.g. plugin code, doc tools).
    """
    from run_agent import _StreamErrorEvent
    message = (_event_field(event, "message", "") or "stream emitted error event").strip()
    raise _StreamErrorEvent(
        message,
        code=_event_field(event, "code"),
        param=_event_field(event, "param"),
    )


def _consume_codex_event_stream(
    event_iter: Any,
    *,
    model: str,
    on_text_delta=None,
    on_reasoning_delta=None,
    on_first_delta=None,
    on_event=None,
    interrupt_check=None,
) -> SimpleNamespace:
    """Consume a reference Responses SSE event stream and return a final response.

    The returned object is a ``SimpleNamespace`` shaped like the SDK's typed
    ``Response`` for the fields downstream code actually reads:

    * ``output``: list of output items, assembled from ``response.output_item.done``.
      For tool-call turns this contains the function_call items; for plain-text
      turns it contains a synthesized ``message`` item built from streamed deltas
      if no message item was emitted directly.
    * ``output_text``: assembled text from ``response.output_text.delta`` deltas.
    * ``usage``: copied from the terminal event's ``response.usage`` (when present).
    * ``status``: ``completed`` / ``incomplete`` / ``failed`` (or ``completed`` if
      the stream ended without a terminal frame but produced content).
    * ``id``: ``response.id`` when present.
    * ``incomplete_details``: passed through for ``response.incomplete`` frames.
    * ``error``: passed through for ``response.failed`` frames.
    * ``model``: from kwargs (the wire model name is not authoritative).

    Critically, we never read ``response.output`` from the terminal event for
    content reconstruction — only ``usage``, ``status``, ``id``.  That field
    being ``null`` / ``[]`` / missing is fine.

    Callbacks:

    * ``on_text_delta(str)`` — fires per ``response.output_text.delta``, suppressed
      once a function_call event is seen (so tool-call turns don't bleed text
      into the chat).
    * ``on_reasoning_delta(str)`` — fires per ``response.reasoning.*.delta``.
    * ``on_first_delta()`` — one-shot, fires on the first text delta only.
    * ``on_event(event)`` — fires for every event before any other processing.
      Used for watchdog activity, debug logging, anything wire-shape-agnostic.
    * ``interrupt_check()`` — returns True to break the loop early.
    """
    collected_output_items: List[Any] = []
    collected_text_deltas: List[str] = []
    has_tool_calls = False
    first_delta_fired = False
    active_message_phase: str | None = None
    terminal_status: str = "completed"
    terminal_usage: Any = None
    terminal_response_id: str = None
    terminal_incomplete_details: Any = None
    terminal_error: Any = None
    saw_terminal = False

    for event in event_iter:
        if on_event is not None:
            try:
                on_event(event)
            except (TimeoutError, InterruptedError):
                # Control-flow signals from watchdog/cancellation hooks must
                # propagate, not get swallowed as "debug noise".
                raise
            except Exception:
                # Genuine bugs in third-party debug/log hooks shouldn't break
                # stream consumption.
                logger.debug("reference stream on_event hook raised", exc_info=True)
        if interrupt_check is not None and interrupt_check():
            break

        event_type = _event_field(event, "type", "")
        if not isinstance(event_type, str):
            event_type = ""

        # ``error`` SSE frames carry the provider's real failure reason
        # (subscription / quota / model-not-available / rejected-reasoning-replay)
        # but never appear in the terminal set.  Surface them as a structured
        # exception so the credential pool + error classifier see the body.
        if event_type == "error":
            _raise_stream_error(event)

        # Track the phase of the active streamed message item.  reference/Harmony
        # ``commentary``/``analysis`` text is mid-turn preamble/progress
        # narration, never the final answer.  We still collect completed output
        # items for replay, but route those deltas to the reasoning callback so
        # they display like thinking text instead of assistant content.
        if event_type == "response.output_item.added":
            item = _event_field(event, "item")
            item_type = _item_field(item, "type", "")
            if item_type == "message":
                phase = _item_field(item, "phase", None)
                active_message_phase = phase.strip().lower() if isinstance(phase, str) else None
            else:
                active_message_phase = None
            if "function_call" in str(item_type):
                has_tool_calls = True
            continue

        if "output_text.delta" in event_type or event_type == "response.output_text.delta":
            delta_text = _event_field(event, "delta", "")
            is_commentary_delta = active_message_phase in {"commentary", "analysis"}
            if delta_text and is_commentary_delta:
                # Commentary streams through the reasoning channel, not the
                # visible answer stream (and stays out of output_text).
                if on_reasoning_delta is not None:
                    try:
                        on_reasoning_delta(delta_text)
                    except Exception:
                        logger.debug("reference stream on_reasoning_delta raised", exc_info=True)
            elif delta_text:
                collected_text_deltas.append(delta_text)
                if not has_tool_calls:
                    if not first_delta_fired:
                        first_delta_fired = True
                        if on_first_delta is not None:
                            try:
                                on_first_delta()
                            except Exception:
                                logger.debug("reference stream on_first_delta raised", exc_info=True)
                    if on_text_delta is not None:
                        try:
                            on_text_delta(delta_text)
                        except Exception:
                            logger.debug("reference stream on_text_delta raised", exc_info=True)
            continue

        if "function_call" in event_type:
            has_tool_calls = True
            # fall through — function_call items still get added on output_item.done

        if "reasoning" in event_type and "delta" in event_type:
            reasoning_text = _event_field(event, "delta", "")
            if reasoning_text and on_reasoning_delta is not None:
                try:
                    on_reasoning_delta(reasoning_text)
                except Exception:
                    logger.debug("reference stream on_reasoning_delta raised", exc_info=True)
            continue

        if event_type == "response.output_item.done":
            done_item = _event_field(event, "item")
            if done_item is not None:
                collected_output_items.append(done_item)
            continue

        if event_type in _TERMINAL_EVENT_TYPES:
            saw_terminal = True
            resp_obj = _event_field(event, "response")
            if resp_obj is not None:
                terminal_usage = getattr(resp_obj, "usage", None)
                if terminal_usage is None and isinstance(resp_obj, dict):
                    terminal_usage = resp_obj.get("usage")
                rid = getattr(resp_obj, "id", None)
                if rid is None and isinstance(resp_obj, dict):
                    rid = resp_obj.get("id")
                terminal_response_id = rid
                rstatus = getattr(resp_obj, "status", None)
                if rstatus is None and isinstance(resp_obj, dict):
                    rstatus = resp_obj.get("status")
                if isinstance(rstatus, str):
                    terminal_status = rstatus
                if event_type == "response.incomplete":
                    terminal_incomplete_details = getattr(resp_obj, "incomplete_details", None)
                    if terminal_incomplete_details is None and isinstance(resp_obj, dict):
                        terminal_incomplete_details = resp_obj.get("incomplete_details")
                if event_type == "response.failed":
                    terminal_error = getattr(resp_obj, "error", None)
                    if terminal_error is None and isinstance(resp_obj, dict):
                        terminal_error = resp_obj.get("error")
            if event_type == "response.completed":
                terminal_status = terminal_status or "completed"
            elif event_type == "response.incomplete":
                terminal_status = terminal_status or "incomplete"
            elif event_type == "response.failed":
                terminal_status = terminal_status or "failed"
            # Stop on terminal event.
            break

    # Build the final output list.  Prefer items observed via output_item.done;
    # if none arrived but we streamed plain text deltas (no tool calls), synthesize
    # a single message item so downstream normalization has something to work with.
    if collected_output_items:
        output = list(collected_output_items)
    elif collected_text_deltas and not has_tool_calls:
        assembled = "".join(collected_text_deltas)
        output = [SimpleNamespace(
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text=assembled)],
        )]
    else:
        output = []

    # If the stream ended without any terminal event AND produced no usable
    # content (no items, no text deltas), surface that as a RuntimeError so
    # callers can distinguish "stream truncated mid-flight / provider rejected
    # the call" from "stream completed with empty body".  This preserves the
    # signal the SDK's high-level helper used to raise as
    # ``RuntimeError("Didn't receive a `response.completed` event.")``.
    if not saw_terminal and not output:
        raise RuntimeError(
            "reference Responses stream did not emit a terminal response"
        )

    assembled_text = "".join(collected_text_deltas)

    final = SimpleNamespace(
        output=output,
        output_text=assembled_text,
        usage=terminal_usage,
        status=terminal_status,
        id=terminal_response_id,
        model=model,
        incomplete_details=terminal_incomplete_details,
        error=terminal_error,
    )
    return final


def run_codex_stream(agent, api_kwargs: dict, client: Any = None, on_first_delta=None):
    """Execute one streaming Responses API request and return the final response.

    Uses ``responses.create(stream=True)`` (low-level raw event iteration)
    rather than the high-level ``responses.stream(...)`` helper.  This makes
    us structurally immune to backend drift in the ``response.completed``
    payload shape — we never let the SDK reconstruct a typed object from
    the terminal event's ``output`` field.
    """
    import httpx as _httpx

    active_client = client or agent._ensure_primary_openai_client(reason="codex_stream_direct")
    max_stream_retries = 1
    # Accumulate streamed text so callers / compat shims can read it.
    agent._codex_streamed_text_parts: list = []

    def _on_text_delta(text: str) -> None:
        agent._codex_streamed_text_parts.append(text)
        agent._fire_stream_delta(text)

    def _on_reasoning_delta(text: str) -> None:
        agent._fire_reasoning_delta(text)

    def _on_event(event: Any) -> None:
        # TTFB watchdog and activity touch — runs once per SSE event.
        agent._codex_stream_last_event_ts = time.time()
        agent._touch_activity("receiving stream response")

    def _interrupt_check() -> bool:
        return bool(agent._interrupt_requested)

    for attempt in range(max_stream_retries + 1):
        if agent._interrupt_requested:
            raise InterruptedError("Agent interrupted before reference stream retry")

        stream_kwargs = dict(api_kwargs)
        stream_kwargs["stream"] = True

        try:
            event_stream = active_client.responses.create(**stream_kwargs)
        except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
            if attempt < max_stream_retries:
                logger.debug(
                    "reference Responses stream connect failed (attempt %s/%s); retrying. %s error=%s",
                    attempt + 1, max_stream_retries + 1,
                    agent._client_log_context(), exc,
                )
                continue
            raise

        # Claim the delta sink for THIS attempt (#65991) — parity with the
        # chat_completions/anthropic/bedrock paths.
        _writer_token = claim_stream_writer(agent)

        def _interrupt_or_superseded(_tok=_writer_token) -> bool:
            if agent._interrupt_requested:
                return True
            if not stream_writer_is_current(agent, _tok):
                logger.warning(
                    "reference streaming attempt superseded by a newer stream; "
                    "stopping consumption to preserve the single-writer "
                    "invariant (model=%s).",
                    api_kwargs.get("model", "unknown"),
                )
                return True
            return False

        try:
            # Compatibility: some mocks/providers return a concrete response
            # instead of an iterable.  Pass it straight through.
            if hasattr(event_stream, "output") and not hasattr(event_stream, "__iter__"):
                return event_stream

            try:
                final = _consume_codex_event_stream(
                    event_stream,
                    model=api_kwargs.get("model"),
                    on_text_delta=_on_text_delta,
                    on_reasoning_delta=_on_reasoning_delta,
                    on_first_delta=on_first_delta,
                    on_event=_on_event,
                    interrupt_check=_interrupt_or_superseded,
                )
            except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
                if attempt < max_stream_retries:
                    logger.debug(
                        "reference Responses stream transport failed mid-iteration "
                        "(attempt %s/%s); retrying. %s error=%s",
                        attempt + 1, max_stream_retries + 1,
                        agent._client_log_context(), exc,
                    )
                    continue
                raise

            if final.status in {"incomplete", "failed"}:
                logger.warning(
                    "reference Responses stream terminal status=%s "
                    "(incomplete_details=%s, error=%s, streamed_chars=%d). %s",
                    final.status, final.incomplete_details, final.error,
                    sum(len(p) for p in agent._codex_streamed_text_parts),
                    agent._client_log_context(),
                )

            return final
        finally:
            close_fn = getattr(event_stream, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass


def run_codex_create_stream_fallback(agent, api_kwargs: dict, client: Any = None):
    """Backward-compatible alias for the unified event-driven path.

    Historically this was the fallback when the SDK's high-level
    ``responses.stream(...)`` helper raised on shape drift.  The primary
    path now does exactly what the fallback did, so this just forwards.
    Kept as a public symbol because tests and a small number of call sites
    still reference it by name.
    """
    return run_codex_stream(agent, api_kwargs, client=client)


__all__ = [
    "run_codex_app_server_turn",
    "run_codex_stream",
    "run_codex_create_stream_fallback",
    "_consume_codex_event_stream",
]
