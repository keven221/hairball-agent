"""Collab V2 control plane — thin wrap over async_delegation + mailbox.

Feature-gated by ``delegation.collab_v2.enabled`` (default true). Single Rail:
async completions never inject a parent turn. Body surfaces via autosurface
(wait woke / tool-tail reminder / cross-turn) plus optional ``collab_collect``.
``delegate_task`` background shares the same suppress_inject / mailbox gate.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from contextvars import ContextVar, Token
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from tools.registry import registry

logger = logging.getLogger(__name__)

_lock = threading.RLock()
# agent_id -> agent record
_agents: Dict[str, Dict[str, Any]] = {}
# parent_key -> next stable fan-out position.  Completion timing is scheduler
# dependent; synthesis order must follow dispatch order instead.
_spawn_sequence_by_parent: Dict[str, int] = {}
# parent_key -> ids already surfaced (agent_id and/or event_id)
_delivered_ids: Dict[str, Set[str]] = {}
_MAX_AUTOSURFACE_SUMMARY_CHARS = 4000
_COLLAB_REMINDER_OPEN = "\n\n<collab-reminder>\n"
_COLLAB_REMINDER_CLOSE = "\n</collab-reminder>"

# Bound by the agent tool executor for the duration of a tool call so
# registry-dispatched collab_* handlers can reach the parent AIAgent
# (delegate_task gets this via a dedicated intercept; collab uses this var).
_parent_agent_var: ContextVar[Any] = ContextVar("collab_parent_agent", default=None)


def bind_collab_parent_agent(agent: Any) -> Token:
    return _parent_agent_var.set(agent)


def reset_collab_parent_agent(token: Token) -> None:
    _parent_agent_var.reset(token)


def get_collab_parent_agent() -> Any:
    return _parent_agent_var.get()


def _resolve_parent_agent(explicit: Any = None) -> Any:
    return explicit if explicit is not None else get_collab_parent_agent()

_DEFAULT_MAX_CONCURRENT = 4
_DEFAULT_WAIT_MS = 30_000
_MIN_WAIT_MS = 1_000
_MAX_WAIT_MS = 3_600_000


def _reset_for_tests() -> None:
    with _lock:
        _agents.clear()
        _spawn_sequence_by_parent.clear()
        _delivered_ids.clear()
    from tools import collab_mailbox as mb

    mb._reset_for_tests()


def _load_delegation_config() -> dict:
    try:
        from hairball_cli.config import load_config

        cfg = load_config() or {}
        return dict(cfg.get("delegation") or {})
    except Exception:
        try:
            from tools.delegate_tool import _load_config

            return dict(_load_config() or {})
        except Exception:
            return {}


_DEFAULT_FORK_TURNS_MAX = 40
_DEFAULT_HISTORY_MAX = 60
_TASK_NAME_RE = re.compile(r"^[a-z0-9_]+$")


def get_collab_v2_config() -> dict:
    raw = _load_delegation_config().get("collab_v2") or {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "max_concurrent": int(raw.get("max_concurrent") or _DEFAULT_MAX_CONCURRENT),
        "default_wait_timeout_ms": int(
            raw.get("default_wait_timeout_ms") or _DEFAULT_WAIT_MS
        ),
        "min_wait_timeout_ms": int(raw.get("min_wait_timeout_ms") or _MIN_WAIT_MS),
        "max_wait_timeout_ms": int(raw.get("max_wait_timeout_ms") or _MAX_WAIT_MS),
        "fork_turns_max_messages": int(
            raw.get("fork_turns_max_messages") or _DEFAULT_FORK_TURNS_MAX
        ),
        "history_max_messages": int(
            raw.get("history_max_messages") or _DEFAULT_HISTORY_MAX
        ),
        # Single Rail autosurface (① reminder / ② wait body / ④ cross-turn).
        "autosurface_enabled": bool(raw.get("autosurface_enabled", True)),
    }


def is_collab_v2_enabled() -> bool:
    return bool(get_collab_v2_config().get("enabled"))


def is_autosurface_enabled() -> bool:
    if not is_collab_v2_enabled():
        return False
    return bool(get_collab_v2_config().get("autosurface_enabled", True))


def check_collab_v2_requirements() -> bool:
    """Registry check_fn — tools only appear when collab_v2 is enabled."""
    return is_collab_v2_enabled()


def _mark_results_delivered(parent_key: str, results: List[Dict[str, Any]]) -> None:
    if not parent_key or not results:
        return
    with _lock:
        bucket = _delivered_ids.setdefault(parent_key, set())
        for row in results:
            aid = str(row.get("agent_id") or "").strip()
            eid = str(row.get("event_id") or "").strip()
            if aid:
                bucket.add(aid)
            if eid:
                bucket.add(eid)


def _truncate_summary(text: Any) -> str:
    s = "" if text is None else str(text)
    if len(s) <= _MAX_AUTOSURFACE_SUMMARY_CHARS:
        return s
    return s[:_MAX_AUTOSURFACE_SUMMARY_CHARS] + "\n...[truncated]"


def format_collab_reminder_block(results: List[Dict[str, Any]]) -> str:
    """Model-facing reminder block (Grok-style system reminder, collab namespaced)."""
    if not results:
        return ""
    lines = [
        f"{len(results)} collab agent(s) completed "
        "(mailbox Single Rail — no inject turn). Summaries follow:"
    ]
    for row in results:
        aid = row.get("agent_id") or row.get("path") or "?"
        status = row.get("status") or "completed"
        spawn_index = row.get("spawn_index")
        summary = _truncate_summary(row.get("summary"))
        err = row.get("error")
        order = f" spawn_index={spawn_index}" if isinstance(spawn_index, int) else ""
        lines.append(f"- agent_id={aid}{order} status={status}")
        if summary:
            lines.append(f"  summary: {summary}")
        if err:
            lines.append(f"  error: {err}")
    lines.append(
        "Synthesize when ready. collab_collect is optional/idempotent if body "
        "was already surfaced."
    )
    return _COLLAB_REMINDER_OPEN + "\n".join(lines) + _COLLAB_REMINDER_CLOSE


def _piggyback_last_tool_message(messages: list, text: str) -> bool:
    if not text or not messages:
        return False
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        existing = msg.get("content", "")
        if isinstance(existing, str):
            if _COLLAB_REMINDER_OPEN.strip() in existing and text in existing:
                return True
            msg["content"] = existing + text
            return True
        try:
            blocks = list(existing) if existing else []
            blocks.append({"type": "text", "text": text})
            msg["content"] = blocks
            return True
        except Exception:
            return False
    return False


def autosurface_mailbox_to_parent(
    *,
    parent_agent: Any = None,
    session_key: str = "",
    reason: str = "reminder",
    event_id: str = "",
    drain_all: bool = True,
) -> Dict[str, Any]:
    """Drain mailbox via collect_results and mark DeliveredIds (unique XOR path)."""
    if not is_autosurface_enabled() and reason != "explicit_collect":
        return {"status": "skipped", "results": [], "reason": reason}
    out = collect_results(
        parent_agent=parent_agent,
        session_key=session_key,
        event_id=event_id,
        drain_all=drain_all,
    )
    if out.get("status") == "ok":
        parent_key = _parent_key(_resolve_parent_agent(parent_agent), session_key)
        _mark_results_delivered(parent_key, list(out.get("results") or []))
    out["autosurface_reason"] = reason
    return out


def apply_collab_completion_reminders(
    agent: Any,
    messages: list,
    num_tool_msgs: int,
) -> bool:
    """① Reminder: piggyback newly completed collab summaries onto last tool msg."""
    if num_tool_msgs <= 0 or not is_autosurface_enabled():
        return False
    try:
        drained = autosurface_mailbox_to_parent(
            parent_agent=agent,
            reason="reminder",
            drain_all=True,
        )
    except Exception:
        logger.debug("collab reminder autosurface failed", exc_info=True)
        return False
    rows = list(drained.get("results") or [])
    if not rows:
        return False
    block = format_collab_reminder_block(rows)
    ok = _piggyback_last_tool_message(messages, block)
    if ok:
        _notify_collab_results_surfaced(agent, rows)
    return ok


def apply_collab_cross_turn_autosurface(agent: Any, messages: list) -> bool:
    """④ Cross-turn / idle: surface pending completions before next API call."""
    if not is_autosurface_enabled():
        return False
    # Steer pattern: only drain when there is a tool message to piggyback onto.
    # Otherwise leave envelopes pending for wait/reminder/explicit collect.
    has_tool = any(
        isinstance(m, dict) and m.get("role") == "tool" for m in (messages or [])
    )
    if not has_tool:
        return False
    try:
        drained = autosurface_mailbox_to_parent(
            parent_agent=agent,
            reason="cross_turn",
            drain_all=True,
        )
    except Exception:
        logger.debug("collab cross-turn autosurface failed", exc_info=True)
        return False
    rows = list(drained.get("results") or [])
    if not rows:
        return False
    block = format_collab_reminder_block(rows)
    ok = _piggyback_last_tool_message(messages, block)
    if ok:
        _notify_collab_results_surfaced(agent, rows)
    return ok


def _notify_collab_results_surfaced(agent: Any, rows: List[Dict[str, Any]]) -> None:
    """Optional UI hook (hairball-ui chat_bridge) — Agents → collected."""
    if not rows or agent is None:
        return
    cb = getattr(agent, "collab_results_surfaced_callback", None)
    if not callable(cb):
        return
    try:
        cb(rows)
    except Exception:
        logger.debug("collab_results_surfaced_callback failed", exc_info=True)


def _parent_key(parent_agent: Any = None, session_key: str = "") -> str:
    if session_key:
        return str(session_key)
    if parent_agent is not None:
        sid = str(getattr(parent_agent, "session_id", "") or "")
        if sid:
            return sid
    try:
        from tools.approval import get_current_session_key

        return str(get_current_session_key(default="") or "")
    except Exception:
        return ""


def _new_agent_id(nickname: str = "") -> str:
    base = "".join(c for c in (nickname or "").lower() if c.isalnum() or c in "-_")[:24]
    suffix = uuid.uuid4().hex[:8]
    return f"{base}-{suffix}" if base else f"agent-{suffix}"


def _validate_task_name(task_name: str) -> Optional[str]:
    """Return error string if invalid; None if ok. Empty is allowed (optional)."""
    name = (task_name or "").strip()
    if not name:
        return None
    if name in {"root", ".", ".."} or "/" in name:
        return "task_name must be a single segment ([a-z0-9_]+), not a path"
    if not _TASK_NAME_RE.match(name):
        return "task_name must match [a-z0-9_]+"
    return None


def _path_for_task_name(task_name: str, nickname: str = "", agent_id: str = "") -> str:
    """Flat collab path: /{task_name} or /{nickname} or /{agent_id}."""
    name = (task_name or "").strip()
    if name:
        return f"/{name}"
    nick = "".join(
        c for c in (nickname or "").lower() if c.isalnum() or c == "_"
    )[:24]
    if nick:
        return f"/{nick}"
    return f"/{agent_id}" if agent_id else "/agent"


def _agent_matches_prefix(path: str, prefix: str) -> bool:
    """session-style: exact match or prefix/ child."""
    p = str(path or "")
    pref = str(prefix or "")
    if not pref:
        return True
    if p == pref:
        return True
    return p.startswith(pref.rstrip("/") + "/")


def _resolve_target(
    target: str,
    *,
    parent_key: str = "",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Resolve agent_id, path, or unique task_name → agent record.

    Returns (agent, error). Ambiguous task_name/path → error (no silent pick).
    """
    ref = (target or "").strip()
    if not ref:
        return None, "agent_id or target is required"
    with _lock:
        # Exact agent_id
        agent = _agents.get(ref)
        if agent is not None:
            if parent_key and agent.get("parent_key") != parent_key:
                return None, f"agent_id={ref} not in this parent session"
            return dict(agent), None

        matches: List[Dict[str, Any]] = []
        for a in _agents.values():
            if parent_key and a.get("parent_key") != parent_key:
                continue
            path = str(a.get("path") or "")
            task = str(a.get("task_name") or "")
            if ref == path or ref == task or (ref.startswith("/") and ref == path):
                matches.append(a)
            elif not ref.startswith("/") and task and task == ref:
                matches.append(a)
            elif not ref.startswith("/") and path.rstrip("/").endswith("/" + ref):
                matches.append(a)

        # Deduplicate by agent_id
        by_id = {m["agent_id"]: m for m in matches}
        matches = list(by_id.values())
        if len(matches) == 1:
            return dict(matches[0]), None
        if len(matches) > 1:
            ids = ", ".join(m["agent_id"] for m in matches)
            return None, f"ambiguous target={ref!r} matches: {ids}"
        return None, f"unknown agent_id/path={ref}"


def _parent_wait_wake_reason(parent_agent: Any) -> Optional[str]:
    """Observe parent steer/interrupt without draining steer text."""
    if parent_agent is None:
        return None
    if getattr(parent_agent, "_interrupt_requested", False):
        return "interrupted"
    steer = getattr(parent_agent, "_pending_steer", None)
    if steer:
        return "interrupted"
    return None


def _safe_history_messages(messages: Any, max_n: int) -> List[Dict[str, Any]]:
    """Keep role/content only; truncate to last max_n messages."""
    if not isinstance(messages, list) or max_n <= 0:
        return []
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in {"user", "assistant", "system", "tool"}:
            continue
        content = msg.get("content")
        if content is None:
            continue
        if isinstance(content, list):
            # Flatten multimodal to text-ish string for resume safety.
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(str(p.get("text") or ""))
                elif isinstance(p, str):
                    parts.append(p)
            content = "\n".join(parts)
        if not isinstance(content, str):
            content = str(content)
        row: Dict[str, Any] = {"role": role, "content": content}
        if role == "tool" and msg.get("tool_call_id"):
            row["tool_call_id"] = msg.get("tool_call_id")
        if role == "assistant" and msg.get("tool_calls"):
            row["tool_calls"] = msg.get("tool_calls")
        out.append(row)
    if len(out) > max_n:
        out = out[-max_n:]
    return out


def _format_queued_messages(queued: List[Any]) -> str:
    lines = []
    for item in queued:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("message") or "").strip()
            kind = str(item.get("kind") or "MESSAGE")
            if text:
                lines.append(f"[{kind}] {text}")
        else:
            text = str(item or "").strip()
            if text:
                lines.append(f"[MESSAGE] {text}")
    return "\n".join(lines)


def _fork_context_from_parent(
    parent_agent: Any,
    fork_turns: str,
    max_messages: int,
) -> str:
    """Read-only slice of parent history into a context prefix (no mutate)."""
    mode = (fork_turns or "none").strip().lower()
    if mode in {"", "none"}:
        return ""
    messages = None
    for attr in ("conversation_history", "messages", "_conversation_history"):
        candidate = getattr(parent_agent, attr, None)
        if isinstance(candidate, list) and candidate:
            messages = candidate
            break
    if messages is None:
        get_hist = getattr(parent_agent, "get_conversation_history", None)
        if callable(get_hist):
            try:
                candidate = get_hist()
                if isinstance(candidate, list):
                    messages = candidate
            except Exception:
                messages = None
    if not messages:
        return ""

    if mode == "all":
        n = max_messages
    else:
        try:
            n_turns = int(mode)
        except ValueError:
            return ""
        if n_turns < 1:
            return ""
        # Approximate N turns as 2N messages (user+assistant), capped.
        n = min(max_messages, n_turns * 2)

    sliced = _safe_history_messages(messages, n)
    if not sliced:
        return ""
    parts = ["Forked parent context (read-only snapshot):"]
    for msg in sliced:
        role = msg.get("role")
        content = str(msg.get("content") or "")
        if len(content) > 2000:
            content = content[:2000] + "…"
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def _running_count(parent_key: str = "") -> int:
    with _lock:
        return sum(
            1
            for a in _agents.values()
            if a.get("status") == "running"
            and (not parent_key or a.get("parent_key") == parent_key)
        )


def _stamp_async_record(delegation_id: str, agent: Dict[str, Any]) -> None:
    try:
        from tools import async_delegation as ad

        with ad._records_lock:
            rec = ad._records.get(delegation_id)
            if rec is not None:
                rec["collab_agent_id"] = agent["agent_id"]
                rec["collab_nickname"] = agent.get("nickname") or ""
                rec["collab_path"] = agent.get("path") or agent["agent_id"]
                # Single Rail — content via mailbox/collect, not inject.
                rec["delivery_channel"] = "collab_mailbox"
    except Exception:
        logger.debug("collab: could not stamp async record", exc_info=True)


def _update_agent_from_result(agent_id: str, result: Dict[str, Any], status: str) -> None:
    """Update agent record after child finish.

    Mailbox ``agent_completed`` doorbell is posted once from
    ``async_delegation._seal_mailbox_delivery`` (Single Rail) — do not post here.
    """
    with _lock:
        agent = _agents.get(agent_id)
        if agent is None:
            return
        agent["status"] = status
        agent["completed_at"] = time.time()
        agent["summary"] = result.get("summary")
        agent["error"] = result.get("error")
        agent["last_result"] = {
            "status": status,
            "summary": result.get("summary"),
            "error": result.get("error"),
        }
        cfg = get_collab_v2_config()
        hist_max = int(cfg.get("history_max_messages") or _DEFAULT_HISTORY_MAX)
        hist = _safe_history_messages(result.get("messages"), hist_max)
        if hist:
            agent["history_messages"] = hist
        child_sid = result.get("child_session_id") or result.get("session_id")
        if child_sid:
            agent["child_session_id"] = str(child_sid)
        pending_followups = list(agent.get("pending_followups") or [])
        agent["pending_followups"] = []
        queued = list(agent.get("pending_queue") or [])

    # TriggerTurn messages queued while running → auto-continue after idle.
    if pending_followups and status in {"completed", "interrupted", "error", "failed"}:
        msg = pending_followups[0]
        rest = pending_followups[1:]
        with _lock:
            agent = _agents.get(agent_id)
            if agent is not None:
                agent["pending_followups"] = rest
        cont = _start_continuation(agent_id, msg, include_queue=True)
        if cont.get("status") not in {"dispatched", "queued"}:
            logger.warning(
                "collab auto-followup failed for %s: %s",
                agent_id,
                cont.get("error") or cont,
            )
            with _lock:
                agent = _agents.get(agent_id)
                if agent is not None and rest:
                    agent["pending_followups"] = [msg] + rest
                elif agent is not None:
                    agent.setdefault("pending_followups", []).insert(0, msg)
    elif queued and status in {"completed", "interrupted"}:
        # QueueOnly alone does not start a turn (reference send_message).
        pass


def _wrap_runner(agent_id: str, runner: Callable[[], Dict[str, Any]]) -> Callable[[], Dict[str, Any]]:
    def wrapped() -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        status = "error"
        try:
            result = runner() or {}
            status = str(result.get("status") or "completed")
            if status == "completed":
                pass
            elif status in {"error", "failed", "interrupted"}:
                pass
            else:
                status = "completed" if not result.get("error") else "error"
            return result
        except Exception as exc:  # noqa: BLE001
            logger.exception("collab agent %s crashed", agent_id)
            result = {
                "status": "error",
                "summary": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            status = "error"
            return result
        finally:
            try:
                _update_agent_from_result(agent_id, result, status)
            except Exception:
                logger.debug("collab finalize hook failed", exc_info=True)

    return wrapped


def _resolve_origin_ui_session_id(parent_agent: Any = None) -> str:
    """Match delegate_task: stamp HAIRBALL_UI_SESSION_ID for UI ownership."""
    try:
        from gateway.session_context import get_session_env

        return str(get_session_env("HAIRBALL_UI_SESSION_ID", "") or "")
    except Exception:
        return ""


def _resolve_spawn_session_key(parent_agent: Any, session_key: str = "") -> str:
    """Match delegate_task routing for hairball-ui / tui / desktop."""
    key = str(session_key or "")
    try:
        from gateway.session_context import get_session_env
        from tools.approval import get_current_session_key

        if not key:
            key = str(get_current_session_key(default="") or "")
        source = get_session_env("HAIRBALL_SESSION_SOURCE", "")
        if source in {"tui", "desktop", "hairball-ui"} and parent_agent is not None:
            agent_sid = str(getattr(parent_agent, "session_id", "") or "")
            if agent_sid:
                key = agent_sid
    except Exception:
        pass
    if not key and parent_agent is not None:
        key = str(getattr(parent_agent, "session_id", "") or "")
    return key


def _build_default_runner(
    parent_agent: Any,
    goal: str,
    context: Optional[str],
    role: str,
    *,
    collab_agent_id: str = "",
    nickname: str = "",
    collab_path: str = "",
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> tuple[Callable[[], Dict[str, Any]], Callable[[], None], Any, Dict[str, Any]]:
    """Build child via existing delegate_tool helpers (same signature as delegate_task)."""
    import model_tools as _model_tools
    from tools.delegate_tool import (
        DEFAULT_MAX_ITERATIONS,
        _build_child_agent,
        _load_config,
        _resolve_delegation_credentials,
        _run_single_child,
    )

    cfg = _load_config()
    max_iter = int(cfg.get("max_iterations") or DEFAULT_MAX_ITERATIONS)
    try:
        creds = _resolve_delegation_credentials(cfg, parent_agent) or {}
    except Exception:
        creds = {}

    path = collab_path or collab_agent_id
    parent_tool_names = list(_model_tools._last_resolved_tool_names)
    child_ref: Dict[str, Any] = {}
    if collab_agent_id:
        child_ref["collab_agent_id"] = collab_agent_id
        child_ref["collab_path"] = path
        child_ref["collab_nickname"] = nickname or collab_agent_id
    prev_ref = getattr(parent_agent, "_delegate_batch_id_ref", None)
    try:
        parent_agent._delegate_batch_id_ref = child_ref
    except Exception:
        pass
    try:
        child = _build_child_agent(
            task_index=0,
            goal=goal,
            context=context,
            toolsets=None,
            model=creds.get("model"),
            max_iterations=max_iter,
            task_count=1,
            parent_agent=parent_agent,
            override_provider=creds.get("provider"),
            override_base_url=creds.get("base_url"),
            override_api_key=creds.get("api_key"),
            override_api_mode=creds.get("api_mode"),
            override_request_overrides=creds.get("request_overrides"),
            override_max_tokens=creds.get("max_output_tokens"),
            override_acp_command=creds.get("command"),
            override_acp_args=creds.get("args"),
            role=role or "leaf",
        )
        child._delegate_saved_tool_names = parent_tool_names
        if collab_agent_id:
            try:
                child._collab_agent_id = collab_agent_id
                child._collab_nickname = nickname or collab_agent_id
                child._collab_path = path
            except Exception:
                pass
    finally:
        _model_tools._last_resolved_tool_names = parent_tool_names
        try:
            parent_agent._delegate_batch_id_ref = prev_ref
        except Exception:
            pass

    ready = threading.Event()
    hist = list(conversation_history or [])

    def runner() -> Dict[str, Any]:
        ready.wait(timeout=5.0)
        return _run_single_child(
            task_index=0,
            goal=goal,
            child=child,
            parent_agent=parent_agent,
            conversation_history=hist or None,
        )

    def interrupt_fn() -> None:
        try:
            if hasattr(child, "interrupt"):
                child.interrupt("Collab interrupt")
            elif hasattr(child, "_interrupt_requested"):
                child._interrupt_requested = True
        except Exception:
            pass

    # Detach from parent turn interrupt list — lifecycle owned by async registry.
    if parent_agent is not None and hasattr(parent_agent, "_active_children"):
        lock = getattr(parent_agent, "_active_children_lock", None)
        try:
            if lock:
                with lock:
                    parent_agent._active_children.remove(child)
            else:
                parent_agent._active_children.remove(child)
        except ValueError:
            pass

    return runner, interrupt_fn, ready, child_ref


def spawn_agent(
    *,
    goal: str,
    context: Optional[str] = None,
    nickname: str = "",
    role: str = "leaf",
    task_name: str = "",
    fork_turns: str = "none",
    parent_agent: Any = None,
    session_key: str = "",
    origin_ui_session_id: str = "",
    runner: Optional[Callable[[], Dict[str, Any]]] = None,
    interrupt_fn: Optional[Callable[[], None]] = None,
    ready_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Fire-and-forget spawn. Returns handle immediately."""
    if not is_collab_v2_enabled():
        return {
            "status": "error",
            "error": "collab_v2 is disabled (set delegation.collab_v2.enabled=true).",
        }
    goal = (goal or "").strip()
    if not goal:
        return {"status": "error", "error": "goal is required"}
    task_err = _validate_task_name(task_name)
    if task_err:
        return {"status": "error", "error": task_err}
    fork_raw = (fork_turns or "none").strip().lower() or "none"
    if fork_raw not in {"none", "all"}:
        try:
            if int(fork_raw) < 1:
                return {
                    "status": "error",
                    "error": 'fork_turns must be "none", "all", or a positive integer',
                }
        except ValueError:
            return {
                "status": "error",
                "error": 'fork_turns must be "none", "all", or a positive integer',
            }

    cfg = get_collab_v2_config()
    parent_key = _parent_key(parent_agent, session_key)
    if _running_count(parent_key) >= max(1, cfg["max_concurrent"]):
        return {
            "status": "rejected",
            "error": (
                f"collab_v2 concurrency cap reached ({cfg['max_concurrent']}). "
                "Wait for an agent to finish or raise delegation.collab_v2.max_concurrent."
            ),
        }

    from tools.async_delegation import dispatch_async_delegation
    from tools.delegate_tool import _get_max_async_children

    parent_resolved = _resolve_parent_agent(parent_agent)
    parent_key = _resolve_spawn_session_key(parent_resolved, parent_key)
    origin = str(origin_ui_session_id or "").strip() or _resolve_origin_ui_session_id(
        parent_resolved
    )

    agent_id = _new_agent_id(nickname or task_name)
    path = _path_for_task_name(task_name, nickname=nickname, agent_id=agent_id)
    # Unique path if collision within parent
    with _lock:
        taken = {
            a.get("path")
            for a in _agents.values()
            if a.get("parent_key") == parent_key
        }
    if path in taken:
        path = f"{path}-{uuid.uuid4().hex[:6]}"

    child_context = context
    if parent_resolved is not None and fork_raw != "none":
        fork_prefix = _fork_context_from_parent(
            parent_resolved,
            fork_raw,
            int(cfg.get("fork_turns_max_messages") or _DEFAULT_FORK_TURNS_MAX),
        )
        if fork_prefix:
            child_context = (
                f"{fork_prefix}\n\n{context}" if context else fork_prefix
            )

    child_ready = ready_event
    child_ref: Dict[str, Any] = {}
    if runner is None:
        if parent_resolved is None:
            return {
                "status": "error",
                "error": "collab_spawn requires parent agent context (or a test runner).",
            }
        runner, interrupt_fn, child_ready, child_ref = _build_default_runner(
            parent_resolved,
            goal,
            child_context,
            role,
            collab_agent_id=agent_id,
            nickname=nickname or task_name or agent_id,
            collab_path=path,
        )

    wrapped = _wrap_runner(agent_id, runner)
    agent = {
        "agent_id": agent_id,
        "nickname": nickname or task_name or agent_id,
        "task_name": (task_name or "").strip(),
        "path": path,
        "goal": goal,
        "context": child_context,
        "role": role or "leaf",
        "fork_turns": fork_raw,
        "parent_key": parent_key,
        "origin_ui_session_id": origin,
        # Strong ref for same-process followup rebuild (fail closed if missing).
        "parent_agent": parent_resolved,
        "status": "running",
        "delegation_id": None,
        "pending_queue": [],
        "pending_followups": [],
        "history_messages": [],
        "closed": False,
        "collected": False,
        "dispatched_at": time.time(),
        "completed_at": None,
        "summary": None,
        "error": None,
        "last_result": None,
    }
    with _lock:
        spawn_index = _spawn_sequence_by_parent.get(parent_key, 0)
        _spawn_sequence_by_parent[parent_key] = spawn_index + 1
        agent["spawn_index"] = spawn_index
        _agents[agent_id] = agent

    parent_session_id = (
        getattr(parent_resolved, "session_id", None) if parent_resolved else None
    )
    dispatch = dispatch_async_delegation(
        goal=goal,
        context=child_context,
        toolsets=None,
        role=role or "leaf",
        model=None,
        session_key=parent_key,
        parent_session_id=parent_session_id,
        runner=wrapped,
        origin_ui_session_id=origin,
        interrupt_fn=interrupt_fn,
        max_async_children=_get_max_async_children(),
        collab_agent_id=agent_id,
        collab_nickname=agent["nickname"],
        collab_path=path,
    )
    if dispatch.get("status") != "dispatched":
        with _lock:
            _agents.pop(agent_id, None)
        return {
            "status": "rejected",
            "error": dispatch.get("error") or "async dispatch rejected",
        }

    delegation_id = str(dispatch.get("delegation_id") or "")
    with _lock:
        agent["delegation_id"] = delegation_id
    _stamp_async_record(delegation_id, agent)
    if isinstance(child_ref, dict):
        child_ref["delegation_id"] = delegation_id
        child_ref["collab_path"] = path
    if child_ready is not None:
        child_ready.set()

    # The child has actually been accepted by the async dispatcher.  Emit the
    # parent/child relationship here, before its eventual mailbox completion,
    # so the agent-tree projector never has to infer a spawn from UI state.
    native_events = getattr(parent_resolved, "_native_turn_events", None)
    if native_events is not None:
        native_events.subagent_spawned(
            agent_id,
            objective=goal,
            child_session_id=parent_key,
            depth=1,
            model="",
            provider="",
        )

    from tools import collab_mailbox as mb

    # Envelope only — do NOT doorbell wait (spawn wake poisoned parallel collect).
    mb.post_activity(
        parent_key,
        kind="agent_spawned",
        agent_id=agent_id,
        payload={
            "goal": goal,
            "nickname": agent["nickname"],
            "path": path,
            "spawn_index": spawn_index,
        },
        doorbell=False,
    )
    # Shape matches delegate_task background dispatch so hairball-ui chat_bridge
    # emits async_delegation_dispatched + Agents/Workspace cards.
    note = (
        "Agent is running in the background (collab mailbox — no inject turn). "
        "Continue local work; collab_wait for a doorbell (summaries may be "
        "included on wake). Reminder/cross-turn autosurface may also deliver "
        "bodies. collab_collect remains optional/idempotent."
    )
    return {
        "status": "dispatched",
        "mode": "background",
        "count": 1,
        "agent_id": agent_id,
        "spawn_index": spawn_index,
        "path": path,
        "task_name": agent.get("task_name") or None,
        "nickname": agent["nickname"],
        "collab_agent_id": agent_id,
        "collab_path": path,
        "collab_nickname": agent["nickname"],
        "delegation_id": delegation_id,
        "goals": [goal],
        "delegations": [
            {
                "task_index": 0,
                "goal": goal,
                "delegation_id": delegation_id,
                "role": role or "leaf",
                "collab_agent_id": agent_id,
                "spawn_index": spawn_index,
                "collab_path": path,
                "collab_nickname": agent["nickname"],
            }
        ],
        "delivery_channel": "collab_mailbox",
        "note": note,
    }


def _start_continuation(
    agent_id: str,
    message: str,
    *,
    include_queue: bool = False,
    parent_agent: Any = None,
) -> Dict[str, Any]:
    """Spawn a continuation turn for an existing agent_id (followup).

    Rebuilds a real child via ``_build_default_runner``. Fail closed (no echo
    stub) when parent agent context is missing.
    """
    with _lock:
        agent = _agents.get(agent_id)
        if agent is None:
            return {"status": "error", "error": f"unknown agent_id={agent_id}"}
        if agent.get("closed"):
            return {"status": "error", "error": "agent is closed"}
        if agent.get("status") == "running":
            return {"status": "error", "error": "agent still running"}
        queued = list(agent.get("pending_queue") or []) if include_queue else []
        if include_queue:
            agent["pending_queue"] = []
        prior = agent.get("last_result") or {}
        history = list(agent.get("history_messages") or [])
        parent_key = agent.get("parent_key") or ""
        nickname = agent.get("nickname") or agent_id
        path = agent.get("path") or agent_id
        origin = agent.get("origin_ui_session_id") or ""
        role = agent.get("role") or "leaf"
        stored_parent = agent.get("parent_agent")

    parent = _resolve_parent_agent(parent_agent) or stored_parent
    if parent is None:
        return {
            "status": "error",
            "error": (
                "collab_followup requires parent agent context to rebuild the "
                "child (no echo stub). Retry from an active parent turn."
            ),
        }

    parts = []
    # When we have real history, keep the user message lean (history carries prior).
    if not history and prior.get("summary"):
        parts.append(f"Prior result:\n{prior['summary']}")
    queued_block = _format_queued_messages(queued)
    if queued_block:
        parts.append("Queued messages (MESSAGE / QueueOnly):\n" + queued_block)
    parts.append(f"Follow-up task:\n{message}")
    goal = "\n\n".join(parts)

    runner, interrupt_fn, child_ready, child_ref = _build_default_runner(
        parent,
        goal,
        context=None,
        role=role,
        collab_agent_id=agent_id,
        nickname=nickname,
        collab_path=path,
        conversation_history=history,
    )
    wrapped = _wrap_runner(agent_id, runner)
    with _lock:
        agent = _agents.get(agent_id)
        if agent is None:
            return {"status": "error", "error": f"unknown agent_id={agent_id}"}
        agent["status"] = "running"
        agent["goal"] = message
        agent["completed_at"] = None
        agent["collected"] = False
        agent["parent_agent"] = parent
        agent["summary"] = None
        agent["error"] = None

    from tools.async_delegation import dispatch_async_delegation
    from tools.delegate_tool import _get_max_async_children

    parent_session_id = getattr(parent, "session_id", None)
    dispatch = dispatch_async_delegation(
        goal=goal,
        context=None,
        toolsets=None,
        role=role,
        model=None,
        session_key=parent_key,
        parent_session_id=parent_session_id,
        runner=wrapped,
        origin_ui_session_id=origin,
        interrupt_fn=interrupt_fn,
        max_async_children=_get_max_async_children(),
        collab_agent_id=agent_id,
        collab_nickname=nickname,
        collab_path=path,
    )
    if dispatch.get("status") != "dispatched":
        with _lock:
            agent = _agents.get(agent_id)
            if agent is not None:
                agent["status"] = "error"
                agent["error"] = dispatch.get("error")
        return {
            "status": "rejected",
            "error": dispatch.get("error") or "followup dispatch rejected",
        }
    delegation_id = str(dispatch.get("delegation_id") or "")
    with _lock:
        agent = _agents.get(agent_id)
        if agent is not None:
            agent["delegation_id"] = delegation_id
            _stamp_async_record(delegation_id, agent)
    if isinstance(child_ref, dict):
        child_ref["delegation_id"] = delegation_id
        child_ref["collab_path"] = path
    if child_ready is not None:
        child_ready.set()
    return {
        "status": "dispatched",
        "mode": "background",
        "count": 1,
        "agent_id": agent_id,
        "path": path,
        "nickname": nickname,
        "collab_agent_id": agent_id,
        "collab_path": path,
        "collab_nickname": nickname,
        "delegation_id": delegation_id,
        "goals": [message],
        "delegations": [
            {
                "task_index": 0,
                "goal": message,
                "delegation_id": delegation_id,
                "role": role,
                "collab_agent_id": agent_id,
                "collab_path": path,
                "collab_nickname": nickname,
            }
        ],
        "delivery": "trigger_turn",
        "delivery_channel": "collab_mailbox",
    }


def wait_agents(
    *,
    parent_agent: Any = None,
    session_key: str = "",
    timeout_ms: Optional[int] = None,
) -> Dict[str, Any]:
    if not is_collab_v2_enabled():
        return {"status": "error", "error": "collab_v2 is disabled"}
    cfg = get_collab_v2_config()
    ms = cfg["default_wait_timeout_ms"] if timeout_ms is None else int(timeout_ms)
    if ms < cfg["min_wait_timeout_ms"]:
        return {
            "status": "error",
            "error": f"timeout_ms must be at least {cfg['min_wait_timeout_ms']}",
        }
    if ms > cfg["max_wait_timeout_ms"]:
        return {
            "status": "error",
            "error": f"timeout_ms must be at most {cfg['max_wait_timeout_ms']}",
        }
    from tools import collab_mailbox as mb

    parent_resolved = _resolve_parent_agent(parent_agent)
    parent_key = _parent_key(parent_resolved, session_key)

    def _wake_check() -> Optional[str]:
        return _parent_wait_wake_reason(parent_resolved)

    outcome = mb.wait_activity(
        parent_key,
        ms,
        wake_check=_wake_check if parent_resolved is not None else None,
    )
    kind = None
    agent_id = None
    event_id = outcome.get("event_id")
    status = str(outcome.get("status") or "timeout")
    if event_id:
        peek = mb.peek_envelope(str(event_id))
        if peek:
            kind = peek.get("kind")
            agent_id = peek.get("agent_id") or None
    timed_out = status == "timeout"
    results: List[Dict[str, Any]] = []
    if status == "interrupted":
        message = "Wait interrupted by new input."
        note = (
            "Wait interrupted by parent steer/interrupt (no body drained). "
            "Handle the new input; call collab_wait again if still waiting on agents."
        )
    elif status == "woke":
        message = "Wait completed."
        # ② Autosurface body into wait tool result (unique collect_results path).
        if is_autosurface_enabled():
            drained = autosurface_mailbox_to_parent(
                parent_agent=parent_resolved,
                session_key=session_key,
                reason="wait_woke",
                # Drain all pending completion envelopes — do not pin to the
                # doorbell event_id (that would skip sibling envelopes).
                event_id="",
                drain_all=True,
            )
            results = list(drained.get("results") or [])
        still_running = _running_count(parent_key)
        if results:
            note = (
                "Doorbell woke; summaries included in results "
                "(mailbox Single Rail — no inject turn). "
                "collab_collect remains optional/idempotent."
            )
        else:
            note = (
                "Doorbell woke; no new summaries (already surfaced via reminder "
                "or collect, or empty). collab_collect is optional/idempotent."
            )
        if still_running:
            note += (
                f" Note: {still_running} collab agent(s) still running — "
                "collab_wait again for the rest before final synthesis."
            )
    else:
        message = "Wait timed out."
        still_running = _running_count(parent_key)
        note = "Wait timed out with no completion doorbell."
        if still_running:
            note += f" {still_running} agent(s) still running — wait again or collab_list."
    out: Dict[str, Any] = {
        "status": status,
        "event_id": event_id,
        "kind": kind,
        "agent_id": agent_id,
        "timed_out": timed_out,
        "message": message,
        "still_running": _running_count(parent_key) if status != "interrupted" else None,
        "note": note,
    }
    if results:
        out["results"] = results
    return out


def send_message(
    agent_id: str = "",
    message: str = "",
    *,
    target: str = "",
    parent_agent: Any = None,
) -> Dict[str, Any]:
    """QueueOnly — do not force a new child turn."""
    if not is_collab_v2_enabled():
        return {"status": "error", "error": "collab_v2 is disabled"}
    message = (message or "").strip()
    if not message:
        return {"status": "error", "error": "message is required"}
    parent_key = _parent_key(_resolve_parent_agent(parent_agent), "")
    ref = (target or agent_id or "").strip()
    agent, err = _resolve_target(ref, parent_key=parent_key)
    if err or agent is None:
        return {"status": "error", "error": err or "unknown agent"}
    aid = agent["agent_id"]
    with _lock:
        live = _agents.get(aid)
        if live is None:
            return {"status": "error", "error": f"unknown agent_id={aid}"}
        if live.get("closed"):
            return {"status": "error", "error": "agent is closed"}
        live.setdefault("pending_queue", []).append(
            {"kind": "MESSAGE", "text": message, "at": time.time()}
        )
        status = live.get("status")
        queued_count = len(live.get("pending_queue") or [])
        path = live.get("path") or aid
    return {
        "status": "queued",
        "agent_id": aid,
        "path": path,
        "delivery": "queue_only",
        "agent_status": status,
        "queued_count": queued_count,
        "note": (
            "MESSAGE queued (QueueOnly); will not start a turn until "
            "collab_followup (or auto-continue after a prior trigger)."
        ),
    }


def followup_task(
    agent_id: str = "",
    message: str = "",
    *,
    target: str = "",
    parent_agent: Any = None,
) -> Dict[str, Any]:
    """TriggerTurn — start a continuation when idle; queue if still running."""
    if not is_collab_v2_enabled():
        return {"status": "error", "error": "collab_v2 is disabled"}
    message = (message or "").strip()
    if not message:
        return {"status": "error", "error": "message is required"}
    parent_resolved = _resolve_parent_agent(parent_agent)
    parent_key = _parent_key(parent_resolved, "")
    ref = (target or agent_id or "").strip()
    agent, err = _resolve_target(ref, parent_key=parent_key)
    if err or agent is None:
        return {"status": "error", "error": err or "unknown agent"}
    aid = agent["agent_id"]
    with _lock:
        live = _agents.get(aid)
        if live is None:
            return {"status": "error", "error": f"unknown agent_id={aid}"}
        if live.get("closed"):
            return {"status": "error", "error": "agent is closed"}
        if live.get("status") == "running":
            live.setdefault("pending_followups", []).append(message)
            return {
                "status": "queued",
                "agent_id": aid,
                "path": live.get("path") or aid,
                "delivery": "trigger_turn",
                "note": "Agent busy; followup queued and will trigger after current turn ends.",
            }
    return _start_continuation(
        aid,
        message,
        include_queue=True,
        parent_agent=parent_resolved,
    )


def collect_results(
    *,
    parent_agent: Any = None,
    session_key: str = "",
    event_id: str = "",
    agent_id: str = "",
    drain_all: bool = True,
) -> Dict[str, Any]:
    """Drain completed collab envelopes into structured summaries (Single Rail)."""
    if not is_collab_v2_enabled():
        return {"status": "error", "error": "collab_v2 is disabled"}
    from tools import collab_mailbox as mb

    parent_resolved = _resolve_parent_agent(parent_agent)
    parent_key = _parent_key(parent_resolved, session_key)
    filter_aid = ""
    if agent_id:
        resolved, err = _resolve_target(str(agent_id), parent_key=parent_key)
        if err or resolved is None:
            return {
                "status": "error",
                "error": err or "unknown agent",
                "results": [],
            }
        filter_aid = resolved["agent_id"]
    collected: List[Dict[str, Any]] = []
    seen_agents: set = set()

    def _take_envelope(env: Dict[str, Any]) -> None:
        eid = str(env.get("event_id") or "")
        aid = str(env.get("agent_id") or "")
        if not eid:
            return
        claim = mb.claim_mailbox_delivery(eid, "collab_collect")
        if claim is None and env.get("delivery_state") == "delivered":
            return
        if claim is None:
            # Contended — skip; another collect may own it.
            return
        payload = dict(env.get("payload") or {})
        path = None
        delegation_id = ""
        with _lock:
            agent = _agents.get(aid) if aid else None
            if agent is not None:
                agent["collected"] = True
                status = agent.get("status") or payload.get("status") or "completed"
                summary = agent.get("summary")
                if summary is None:
                    summary = payload.get("summary")
                error = agent.get("error")
                if error is None:
                    error = payload.get("error")
                path = agent.get("path") or aid
                delegation_id = str(agent.get("delegation_id") or "")
                spawn_index = agent.get("spawn_index")
            else:
                status = payload.get("status") or "completed"
                summary = payload.get("summary")
                error = payload.get("error")
                spawn_index = payload.get("spawn_index")
        mb.complete_mailbox_delivery(eid, claim)
        if aid:
            seen_agents.add(aid)
        row = {
            "event_id": eid,
            "agent_id": aid or None,
            "spawn_index": spawn_index,
            "path": path,
            "kind": env.get("kind"),
            # UI terminal status after parent collect (Single Rail).
            "ui_status": "collected",
            "status": status,
            "summary": summary,
        }
        if delegation_id:
            row["delegation_id"] = delegation_id
        # Omit null error — string heuristic in _detect_tool_failure treats
        # literal '"error"' in JSON as failure even when status is ok.
        if error:
            row["error"] = error
        collected.append(row)

    if event_id:
        env = mb.get_envelope(str(event_id))
        if env is None:
            return {
                "status": "error",
                "error": f"unknown event_id={event_id}",
                "results": [],
            }
        if env.get("kind") == "agent_completed":
            _take_envelope(env)
        else:
            return {
                "status": "ok",
                "results": [],
                "note": (
                    f"event kind={env.get('kind')} has no completion body; "
                    "wait for agent_completed then collect again."
                ),
            }
    else:
        envs = mb.list_envelopes(
            parent_key,
            kind="agent_completed",
            agent_id=str(filter_aid or ""),
            include_delivered=False,
        )
        if not drain_all and envs:
            envs = envs[-1:]
        for env in envs:
            _take_envelope(env)

    # Also surface completed agents that still have summary but no pending
    # envelope (e.g. already woken + envelope drained elsewhere).
    if drain_all or filter_aid:
        with _lock:
            for a in _agents.values():
                if parent_key and a.get("parent_key") != parent_key:
                    continue
                aid = a["agent_id"]
                if filter_aid and aid != filter_aid:
                    continue
                if aid in seen_agents:
                    continue
                if a.get("status") not in {
                    "completed",
                    "error",
                    "failed",
                    "interrupted",
                }:
                    continue
                if a.get("collected"):
                    continue
                a["collected"] = True
                row = {
                    "event_id": None,
                    "agent_id": aid,
                    "spawn_index": a.get("spawn_index"),
                    "path": a.get("path") or aid,
                    "kind": "agent_completed",
                    "ui_status": "collected",
                    "status": a.get("status"),
                    "summary": a.get("summary"),
                    "delegation_id": a.get("delegation_id") or "",
                }
                if a.get("error"):
                    row["error"] = a.get("error")
                collected.append(row)
                seen_agents.add(aid)

    # Child completion time is nondeterministic.  Keep fan-in deterministic
    # without delaying partial results: only the rows ready in this drain are
    # ordered, using their immutable per-parent dispatch position.  Unknown
    # legacy rows stay at the end and retain mailbox order (stable sort).
    collected.sort(
        key=lambda row: (
            row.get("spawn_index")
            if isinstance(row.get("spawn_index"), int)
            else float("inf")
        )
    )

    still_running = _running_count(parent_key)
    if collected:
        _mark_results_delivered(parent_key, collected)
        note = (
            "Collected collab summaries. Synthesize once for the user when all "
            "needed agents are in; do not wait for an async inject turn."
        )
        if still_running:
            note += (
                f" {still_running} agent(s) still running — collab_wait again "
                "before the final synthesis if you need them."
            )
    else:
        note = (
            "No completed results ready (already autosurfaced, or none pending); "
            "keep working or collab_wait again."
        )
        if still_running:
            note += f" {still_running} agent(s) still running."
    return {
        "status": "ok",
        "results": collected,
        "still_running": still_running,
        "note": note,
    }


def interrupt_agent(
    agent_id: str = "",
    *,
    target: str = "",
    parent_agent: Any = None,
) -> Dict[str, Any]:
    if not is_collab_v2_enabled():
        return {"status": "error", "error": "collab_v2 is disabled"}
    parent_key = _parent_key(_resolve_parent_agent(parent_agent), "")
    ref = (target or agent_id or "").strip()
    agent, err = _resolve_target(ref, parent_key=parent_key)
    if err or agent is None:
        return {"status": "error", "error": err or "unknown agent"}
    aid = agent["agent_id"]
    with _lock:
        live = _agents.get(aid)
        if live is None:
            return {"status": "error", "error": f"unknown agent_id={aid}"}
        if live.get("closed"):
            return {"status": "error", "error": "agent is closed"}
        previous_status = live.get("status")
        delegation_id = live.get("delegation_id") or ""
        was_running = previous_status == "running"
        path = live.get("path") or aid
    if not was_running:
        return {
            "status": "ok",
            "agent_id": aid,
            "path": path,
            "previous_status": previous_status,
            "note": "Agent was not running; interrupt is a no-op (still not closed).",
        }
    try:
        from tools import async_delegation as ad

        with ad._records_lock:
            rec = ad._records.get(delegation_id)
            fn = rec.get("interrupt_fn") if rec else None
        if callable(fn):
            fn()
    except Exception as exc:
        return {"status": "error", "error": f"interrupt failed: {exc}"}
    return {
        "status": "ok",
        "agent_id": aid,
        "path": path,
        "previous_status": previous_status,
        "note": "Interrupt signaled; agent is not closed and may receive followup.",
    }


def list_agents(
    *,
    parent_agent: Any = None,
    session_key: str = "",
    path_prefix: str = "",
) -> Dict[str, Any]:
    if not is_collab_v2_enabled():
        return {"status": "error", "error": "collab_v2 is disabled"}
    parent_key = _parent_key(parent_agent, session_key)
    prefix = (path_prefix or "").strip()
    with _lock:
        rows = []
        for a in _agents.values():
            if parent_key and a.get("parent_key") != parent_key:
                continue
            path = a.get("path") or a["agent_id"]
            if prefix and not _agent_matches_prefix(str(path), prefix):
                continue
            status = a.get("status")
            rows.append(
                {
                    "agent_id": a["agent_id"],
                    "spawn_index": a.get("spawn_index"),
                    "path": path,
                    "task_name": a.get("task_name") or None,
                    "nickname": a.get("nickname"),
                    "status": status,
                    "agent_status": status,
                    "goal": a.get("goal"),
                    "delegation_id": a.get("delegation_id"),
                    "pending_queue_len": len(a.get("pending_queue") or []),
                    "pending_followups_len": len(a.get("pending_followups") or []),
                    "closed": bool(a.get("closed")),
                    "collected": bool(a.get("collected")),
                    "summary": a.get("summary"),
                    "error": a.get("error"),
                }
            )
    return {"status": "ok", "agents": rows}


def _json(handler_result: Dict[str, Any]) -> str:
    return json.dumps(handler_result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool schemas + registry (check_fn gated)
# ---------------------------------------------------------------------------

_SPAWN_SCHEMA = {
    "name": "collab_spawn",
    "description": (
        "Spawn a collaborative sub-agent that runs in the background and returns "
        "immediately with agent_id + path. Protocol: continue local work → "
        "collab_wait (doorbell; summaries often included on wake) → synthesize "
        "once. Runtime may also surface completions via tool-tail reminders. "
        "collab_collect is optional/idempotent. Example: "
        "collab_spawn(goal=\"draft section A\", task_name=\"writer\") → "
        "keep working → collab_wait → synthesize. Results do NOT re-enter via "
        "async inject. Prefer disjoint file writes. Prefer collab_* over "
        "delegate_task for collaborative parallel work."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "Task for the child agent."},
            "context": {"type": "string", "description": "Optional context."},
            "task_name": {
                "type": "string",
                "description": (
                    "Optional path segment [a-z0-9_]+ → path like /writer. "
                    "Use with collab_send/followup/list path_prefix."
                ),
            },
            "nickname": {"type": "string", "description": "Optional short label."},
            "fork_turns": {
                "type": "string",
                "description": (
                    'Parent history to seed child context: "none" (default), '
                    '"all" (capped), or a positive integer N (approx last N turns). '
                    "Does not mutate parent messages / prompt cache."
                ),
            },
            "role": {
                "type": "string",
                "enum": ["leaf", "orchestrator"],
                "description": "leaf (default) or orchestrator when depth allows.",
            },
        },
        "required": ["goal"],
    },
}

_WAIT_SCHEMA = {
    "name": "collab_wait",
    "description": (
        "Wait until an agent_completed doorbell (or parent steer/interrupt). "
        "Spawn does not wake wait. On wake, results may include drained "
        "summaries (autosurface). If still_running>0, wait again before final "
        "synthesis. collab_collect is optional. No async inject turn."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "timeout_ms": {
                "type": "integer",
                "description": "Wait timeout in milliseconds.",
            },
        },
    },
}

_COLLECT_SCHEMA = {
    "name": "collab_collect",
    "description": (
        "Optional/idempotent: collect completed collab summaries from the mailbox "
        "(XOR delivery). Usually unnecessary after collab_wait wake or runtime "
        "autosurface — returns [] if already drained. Use to explicitly drain. "
        "Do not wait for an inject turn. agent_id also accepts path/task_name."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "event_id": {
                "type": "string",
                "description": "Optional mailbox event_id from collab_wait.",
            },
            "agent_id": {
                "type": "string",
                "description": "Optional filter: agent_id, path, or task_name.",
            },
            "drain_all": {
                "type": "boolean",
                "description": "If true (default), drain all pending completions.",
            },
        },
    },
}

_SEND_SCHEMA = {
    "name": "collab_send",
    "description": (
        "Queue a MESSAGE to a collab agent without forcing a new turn "
        "(QueueOnly). Does not wake wait. Use collab_followup to trigger a turn. "
        "Target via agent_id or target (path/task_name)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "agent_id, or omit if target is set.",
            },
            "target": {
                "type": "string",
                "description": "Alias for agent_id/path/task_name.",
            },
            "message": {"type": "string"},
        },
        "required": ["message"],
    },
}

_FOLLOWUP_SCHEMA = {
    "name": "collab_followup",
    "description": (
        "TriggerTurn: continue the target agent (reuses prior conversation "
        "history when available). If still running, queues until idle. "
        "Interrupt does not close. Target via agent_id or target."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_id": {"type": "string"},
            "target": {
                "type": "string",
                "description": "Alias for agent_id/path/task_name.",
            },
            "message": {"type": "string"},
        },
        "required": ["message"],
    },
}

_INTERRUPT_SCHEMA = {
    "name": "collab_interrupt",
    "description": (
        "Interrupt a running collab agent turn. Returns previous_status. "
        "Does not close; collab_followup may continue afterward."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_id": {"type": "string"},
            "target": {
                "type": "string",
                "description": "Alias for agent_id/path/task_name.",
            },
        },
    },
}

_LIST_SCHEMA = {
    "name": "collab_list",
    "description": (
        "List collab agents for the current parent session (path, status, queues). "
        "Optional path_prefix filters like /writer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path_prefix": {
                "type": "string",
                "description": "Optional path prefix filter (exact or child paths).",
            },
        },
    },
}


def _tool_spawn(args, **kw):
    return _json(
        spawn_agent(
            goal=args.get("goal") or "",
            context=args.get("context"),
            nickname=args.get("nickname") or "",
            task_name=args.get("task_name") or "",
            fork_turns=args.get("fork_turns") or "none",
            role=args.get("role") or "leaf",
            parent_agent=_resolve_parent_agent(kw.get("parent_agent")),
        )
    )


def _tool_wait(args, **kw):
    return _json(
        wait_agents(
            parent_agent=_resolve_parent_agent(kw.get("parent_agent")),
            timeout_ms=args.get("timeout_ms"),
        )
    )


def _tool_collect(args, **kw):
    drain = args.get("drain_all")
    return _json(
        collect_results(
            parent_agent=_resolve_parent_agent(kw.get("parent_agent")),
            event_id=args.get("event_id") or "",
            agent_id=args.get("agent_id") or args.get("target") or "",
            drain_all=True if drain is None else bool(drain),
        )
    )


def _tool_send(args, **kw):
    return _json(
        send_message(
            args.get("agent_id") or "",
            args.get("message") or "",
            target=args.get("target") or "",
            parent_agent=_resolve_parent_agent(kw.get("parent_agent")),
        )
    )


def _tool_followup(args, **kw):
    return _json(
        followup_task(
            args.get("agent_id") or "",
            args.get("message") or "",
            target=args.get("target") or "",
            parent_agent=_resolve_parent_agent(kw.get("parent_agent")),
        )
    )


def _tool_interrupt(args, **kw):
    return _json(
        interrupt_agent(
            args.get("agent_id") or "",
            target=args.get("target") or "",
            parent_agent=_resolve_parent_agent(kw.get("parent_agent")),
        )
    )


def _tool_list(args, **kw):
    return _json(
        list_agents(
            parent_agent=_resolve_parent_agent(kw.get("parent_agent")),
            path_prefix=args.get("path_prefix") or "",
        )
    )


# Top-level registry.register(...) required for discover_builtin_tools AST scan
# (nested register inside for/def is ignored by tools/registry._module_registers_tools).
registry.register(
    name="collab_spawn",
    toolset="delegation",
    schema=_SPAWN_SCHEMA,
    handler=_tool_spawn,
    check_fn=check_collab_v2_requirements,
    emoji="📡",
)
registry.register(
    name="collab_wait",
    toolset="delegation",
    schema=_WAIT_SCHEMA,
    handler=_tool_wait,
    check_fn=check_collab_v2_requirements,
    emoji="📡",
)
registry.register(
    name="collab_collect",
    toolset="delegation",
    schema=_COLLECT_SCHEMA,
    handler=_tool_collect,
    check_fn=check_collab_v2_requirements,
    emoji="📡",
)
registry.register(
    name="collab_send",
    toolset="delegation",
    schema=_SEND_SCHEMA,
    handler=_tool_send,
    check_fn=check_collab_v2_requirements,
    emoji="📡",
)
registry.register(
    name="collab_followup",
    toolset="delegation",
    schema=_FOLLOWUP_SCHEMA,
    handler=_tool_followup,
    check_fn=check_collab_v2_requirements,
    emoji="📡",
)
registry.register(
    name="collab_interrupt",
    toolset="delegation",
    schema=_INTERRUPT_SCHEMA,
    handler=_tool_interrupt,
    check_fn=check_collab_v2_requirements,
    emoji="📡",
)
registry.register(
    name="collab_list",
    toolset="delegation",
    schema=_LIST_SCHEMA,
    handler=_tool_list,
    check_fn=check_collab_v2_requirements,
    emoji="📡",
)
