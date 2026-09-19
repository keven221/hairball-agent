"""Model-facing DAP ``debug`` tool (W3+ — OMP action parity).

Single tool with an action enum covering the OMP debug contract plus Hairball
extras (``status``, ``select``, ``terminate_all``, batch ``set_breakpoints``).
Hidden unless any adapter is available and ``tools.debug.enabled`` is not false.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

# OMP 28 + Hairball extras. Aliases: step_over→next, set_breakpoint↔set_breakpoints.
_ACTIONS = (
    "status",
    "sessions",
    "select",
    "launch",
    "attach",
    "set_breakpoint",
    "set_breakpoints",
    "remove_breakpoint",
    "set_instruction_breakpoint",
    "remove_instruction_breakpoint",
    "data_breakpoint_info",
    "set_data_breakpoint",
    "remove_data_breakpoint",
    "threads",
    "output",
    "stack_trace",
    "scopes",
    "variables",
    "evaluate",
    "disassemble",
    "read_memory",
    "write_memory",
    "modules",
    "loaded_sources",
    "custom_request",
    "continue",
    "next",
    "step_over",
    "step_in",
    "step_out",
    "pause",
    "terminate",
    "terminate_all",
)

DEBUG_SCHEMA = {
    "name": "debug",
    "description": (
        "Debug via DAP: Python (debugpy), JS/TS (js-debug), or native "
        "(lldb-dap / gdb). Adapter inferred from program extension or set via "
        "adapter=. Typical flow: launch|attach → set_breakpoint → "
        "stack_trace / threads / output / scopes / variables / evaluate → "
        "continue | step_over | step_in | step_out | pause → terminate. "
        "Also: disassemble / read_memory / write_memory / modules / "
        "loaded_sources / data breakpoints / custom_request when the adapter "
        "advertises them. Up to 3 concurrent sessions; pass session_id or use "
        "select. Not a substitute for `terminal` tests or `lsp` navigation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "Debug action to perform.",
            },
            "session_id": {
                "type": "string",
                "description": "Target debug session (default: active session).",
            },
            "adapter": {
                "type": "string",
                "enum": ["debugpy", "js-debug", "lldb", "gdb"],
                "description": (
                    "DAP adapter (default: infer — .py→debugpy, .js/.ts→js-debug, "
                    ".c/.cpp/.rs/…→lldb|gdb)."
                ),
            },
            "program": {
                "type": "string",
                "description": (
                    "Program for launch. Python attach may auto-spawn with "
                    "--connect; JS program-only attach maps to launch. "
                    "Required for launch; optional for attach."
                ),
            },
            "cwd": {
                "type": "string",
                "description": "Working directory (default: program dir).",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Program argv for launch/attach-spawned debuggee.",
            },
            "host": {
                "type": "string",
                "description": "Listen/connect host for attach (default 127.0.0.1).",
            },
            "port": {
                "type": "integer",
                "description": "Listen/connect port for attach (0/omit = ephemeral in listen mode).",
            },
            "pid": {
                "type": "integer",
                "description": (
                    "Process id — Python: debugpy --pid inject; JS: pwa-node processId."
                ),
            },
            "connect": {
                "type": "boolean",
                "description": "Attach by connecting to an already-listening debuggee at host:port.",
            },
            "stop_on_entry": {
                "type": "boolean",
                "description": "Stop at first line on launch (default false).",
            },
            "breakpoints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["line"],
                },
                "description": (
                    "Initial breakpoints for launch/attach, or used by "
                    "set_breakpoints when path+lines not given."
                ),
            },
            "path": {
                "type": "string",
                "description": "Source file for set/remove breakpoint.",
            },
            "file": {
                "type": "string",
                "description": "Alias of path (OMP set_breakpoint).",
            },
            "line": {
                "type": "integer",
                "minimum": 1,
                "description": "Source line for set/remove_breakpoint.",
            },
            "lines": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
                "description": "Breakpoint lines for set_breakpoints.",
            },
            "function": {
                "type": "string",
                "description": "Function name for function breakpoints.",
            },
            "condition": {
                "type": "string",
                "description": "Breakpoint condition expression.",
            },
            "hit_condition": {
                "type": "string",
                "description": "Breakpoint hit condition.",
            },
            "name": {
                "type": "string",
                "description": "Variable/data name for data_breakpoint_info.",
            },
            "data_id": {
                "type": "string",
                "description": "dataId from data_breakpoint_info.",
            },
            "access_type": {
                "type": "string",
                "description": "Data breakpoint access type (read/write/readWrite).",
            },
            "instruction_reference": {
                "type": "string",
                "description": "Instruction reference for instruction breakpoints / disassemble.",
            },
            "memory_reference": {
                "type": "string",
                "description": "Memory reference for disassemble / read_memory / write_memory.",
            },
            "offset": {
                "type": "integer",
                "description": "Byte offset for memory/instruction ops.",
            },
            "instruction_offset": {
                "type": "integer",
                "description": "Instruction offset for disassemble.",
            },
            "instruction_count": {
                "type": "integer",
                "description": "Instruction count for disassemble.",
            },
            "resolve_symbols": {
                "type": "boolean",
                "description": "Resolve symbols in disassemble.",
            },
            "count": {
                "type": "integer",
                "description": "Byte count for read_memory.",
            },
            "data": {
                "type": "string",
                "description": "Base64 data for write_memory.",
            },
            "allow_partial": {
                "type": "boolean",
                "description": "Allow partial write_memory.",
            },
            "start_module": {
                "type": "integer",
                "description": "modules pagination start.",
            },
            "module_count": {
                "type": "integer",
                "description": "modules pagination count.",
            },
            "command": {
                "type": "string",
                "description": "Raw DAP method for custom_request.",
            },
            "arguments": {
                "type": "object",
                "description": "Raw DAP arguments for custom_request.",
            },
            "thread_id": {
                "type": "integer",
                "description": "Thread id (defaults to last stopped thread).",
            },
            "frame_id": {
                "type": "integer",
                "description": "Stack frame id for scopes/evaluate/data_breakpoint_info.",
            },
            "variables_reference": {
                "type": "integer",
                "description": "variablesReference from scopes/variables/evaluate.",
            },
            "variable_ref": {
                "type": "integer",
                "description": "Alias of variables_reference (OMP).",
            },
            "scope_id": {
                "type": "integer",
                "description": "Alias of variables_reference for scopes (OMP).",
            },
            "expression": {
                "type": "string",
                "description": "Expression for evaluate.",
            },
            "levels": {
                "type": "integer",
                "description": "Max stack frames (default 20).",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout seconds (default 30; clamped 5–120).",
            },
        },
        "required": ["action"],
    },
}


def _tool_error(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


def _clamp_timeout(raw: Any) -> float:
    try:
        t = float(raw)
    except (TypeError, ValueError):
        t = 30.0
    return max(5.0, min(120.0, t))


def find_debugpy() -> bool:
    try:
        import debugpy  # noqa: F401

        return True
    except Exception:
        return False


def find_js_debug() -> bool:
    from agent.dap.adapters import find_js_debug as _find

    return _find()


def find_lldb() -> bool:
    from agent.dap.adapters import find_lldb as _find

    return _find()


def find_gdb() -> bool:
    from agent.dap.adapters import find_gdb as _find

    return _find()


def check_debug_requirements() -> bool:
    """Expose ``debug`` when any DAP adapter is available (unless disabled)."""
    try:
        from hairball_cli.config import load_config

        cfg = load_config()
    except Exception:
        cfg = None
    if isinstance(cfg, dict):
        tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
        dbg = tools_cfg.get("debug") if isinstance(tools_cfg.get("debug"), dict) else {}
        if dbg.get("enabled") is False:
            return False
    return find_debugpy() or find_js_debug() or find_lldb() or find_gdb()


def debug_tool(
    action: str,
    *,
    session_id: Optional[str] = None,
    adapter: Optional[str] = None,
    program: Optional[str] = None,
    cwd: Optional[str] = None,
    args: Optional[list] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    pid: Optional[int] = None,
    connect: Any = False,
    stop_on_entry: Any = False,
    breakpoints: Optional[list] = None,
    path: Optional[str] = None,
    file: Optional[str] = None,
    line: Optional[int] = None,
    lines: Optional[list] = None,
    function: Optional[str] = None,
    condition: Optional[str] = None,
    hit_condition: Optional[str] = None,
    name: Optional[str] = None,
    data_id: Optional[str] = None,
    access_type: Optional[str] = None,
    instruction_reference: Optional[str] = None,
    memory_reference: Optional[str] = None,
    offset: Optional[int] = None,
    instruction_offset: Optional[int] = None,
    instruction_count: Optional[int] = None,
    resolve_symbols: Any = None,
    count: Optional[int] = None,
    data: Optional[str] = None,
    allow_partial: Any = None,
    start_module: Optional[int] = None,
    module_count: Optional[int] = None,
    command: Optional[str] = None,
    arguments: Optional[dict] = None,
    thread_id: Optional[int] = None,
    frame_id: Optional[int] = None,
    variables_reference: Optional[int] = None,
    variable_ref: Optional[int] = None,
    scope_id: Optional[int] = None,
    expression: Optional[str] = None,
    levels: Optional[int] = None,
    timeout: Optional[float] = None,
    **_kwargs: Any,
) -> str:
    from agent.dap import session as dap_session
    from agent.dap.client import DapClientError

    act = str(action or "").strip().lower()
    if act not in _ACTIONS:
        return _tool_error(f"unknown action {action!r}; expected one of {', '.join(_ACTIONS)}")

    # OMP aliases
    if act == "step_over":
        act = "next"

    t = _clamp_timeout(timeout)
    sid = str(session_id).strip() if session_id else None
    adapter_s = str(adapter).strip() if adapter else None
    src_path = str(path or file or "").strip() or None
    var_ref = variables_reference if variables_reference is not None else variable_ref
    if var_ref is None:
        var_ref = scope_id
    mem_ref = memory_reference or instruction_reference

    try:
        if act == "status":
            sess = dap_session.get_session(sid)
            return json.dumps(
                {
                    "success": True,
                    "action": act,
                    "active": sess is not None and sess.status != "closed",
                    "session": sess.summary() if sess is not None else None,
                    "sessions": dap_session.list_sessions(),
                    "python": sys.executable,
                    "debugpy": find_debugpy(),
                    "js_debug": find_js_debug(),
                    "lldb": find_lldb(),
                    "gdb": find_gdb(),
                },
                ensure_ascii=False,
            )

        if act == "sessions":
            return json.dumps(
                {
                    "success": True,
                    "action": act,
                    "sessions": dap_session.list_sessions(),
                },
                ensure_ascii=False,
            )

        if act == "select":
            if not sid:
                return _tool_error("session_id is required for action=select")
            out = dap_session.select_session(sid)
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "terminate_all":
            out = dap_session.terminate_all(timeout=t)
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "terminate":
            out = dap_session.terminate(timeout=t, session_id=sid)
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "launch":
            if not program or not str(program).strip():
                return _tool_error("program is required for action=launch")
            argv = None
            if isinstance(args, list):
                argv = [str(a) for a in args]
            bps = breakpoints if isinstance(breakpoints, list) else None
            out = dap_session.initialize_and_launch(
                program=str(program).strip(),
                cwd=str(cwd).strip() if cwd else None,
                args=argv,
                stop_on_entry=bool(stop_on_entry),
                breakpoints=bps,
                timeout=t,
                adapter=adapter_s,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "attach":
            argv = None
            if isinstance(args, list):
                argv = [str(a) for a in args]
            bps = breakpoints if isinstance(breakpoints, list) else None
            out = dap_session.initialize_and_attach(
                program=str(program).strip() if program else None,
                cwd=str(cwd).strip() if cwd else None,
                args=argv,
                host=str(host).strip() if host else None,
                port=int(port) if port is not None else None,
                pid=int(pid) if pid is not None else None,
                connect=bool(connect),
                breakpoints=bps,
                timeout=t,
                adapter=adapter_s,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "set_breakpoint":
            out = dap_session.set_breakpoint(
                path=src_path,
                line=line,
                function=str(function).strip() if function else None,
                condition=str(condition) if condition else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "remove_breakpoint":
            out = dap_session.remove_breakpoint(
                path=src_path,
                line=line,
                function=str(function).strip() if function else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "set_breakpoints":
            line_list: list[Any] = []
            src = src_path or ""
            if isinstance(lines, list):
                line_list.extend(lines)
            if not line_list and isinstance(breakpoints, list):
                for bp in breakpoints:
                    if isinstance(bp, dict) and bp.get("line") is not None:
                        line_list.append(bp["line"])
                        if not src:
                            src = str(bp.get("file") or bp.get("path") or "").strip()
            if not src:
                sess = dap_session.get_session(sid)
                src = (sess.program if sess else "") or ""
            if not src or not line_list:
                return _tool_error("path and lines (or breakpoints) required for set_breakpoints")
            out = dap_session.set_breakpoints(
                path=src, lines=line_list, timeout=t, session_id=sid
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "set_instruction_breakpoint":
            if not instruction_reference:
                return _tool_error("instruction_reference is required")
            out = dap_session.set_instruction_breakpoint(
                instruction_reference=str(instruction_reference),
                offset=int(offset) if offset is not None else None,
                condition=str(condition) if condition else None,
                hit_condition=str(hit_condition) if hit_condition else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "remove_instruction_breakpoint":
            if not instruction_reference:
                return _tool_error("instruction_reference is required")
            out = dap_session.remove_instruction_breakpoint(
                instruction_reference=str(instruction_reference),
                offset=int(offset) if offset is not None else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "data_breakpoint_info":
            if not name:
                return _tool_error("name is required for data_breakpoint_info")
            out = dap_session.data_breakpoint_info(
                name=str(name),
                variables_reference=int(var_ref) if var_ref is not None else None,
                frame_id=int(frame_id) if frame_id is not None else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "set_data_breakpoint":
            if not data_id:
                return _tool_error("data_id is required for set_data_breakpoint")
            out = dap_session.set_data_breakpoint(
                data_id=str(data_id),
                access_type=str(access_type) if access_type else None,
                condition=str(condition) if condition else None,
                hit_condition=str(hit_condition) if hit_condition else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "remove_data_breakpoint":
            if not data_id:
                return _tool_error("data_id is required for remove_data_breakpoint")
            out = dap_session.remove_data_breakpoint(
                data_id=str(data_id), timeout=t, session_id=sid
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "stack_trace":
            out = dap_session.stack_trace(
                thread_id=int(thread_id) if thread_id is not None else None,
                levels=int(levels or 20),
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "threads":
            out = dap_session.threads(timeout=t, session_id=sid)
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "output":
            out = dap_session.captured_output(clear=False, session_id=sid)
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "scopes":
            if frame_id is None:
                return _tool_error("frame_id is required for scopes")
            out = dap_session.scopes(frame_id=int(frame_id), timeout=t, session_id=sid)
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "variables":
            if var_ref is None:
                return _tool_error("variables_reference (or variable_ref/scope_id) is required")
            out = dap_session.variables(
                variables_reference=int(var_ref),
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "evaluate":
            if not expression or not str(expression).strip():
                return _tool_error("expression is required for evaluate")
            out = dap_session.evaluate(
                expression=str(expression).strip(),
                frame_id=int(frame_id) if frame_id is not None else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "disassemble":
            if instruction_count is None:
                return _tool_error("instruction_count is required for disassemble")
            if not mem_ref:
                return _tool_error("memory_reference (or instruction_reference) is required")
            out = dap_session.disassemble(
                memory_reference=str(mem_ref),
                instruction_count=int(instruction_count),
                offset=int(offset) if offset is not None else None,
                instruction_offset=int(instruction_offset) if instruction_offset is not None else None,
                resolve_symbols=bool(resolve_symbols) if resolve_symbols is not None else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "read_memory":
            if not memory_reference:
                return _tool_error("memory_reference is required for read_memory")
            if count is None:
                return _tool_error("count is required for read_memory")
            out = dap_session.read_memory(
                memory_reference=str(memory_reference),
                count=int(count),
                offset=int(offset) if offset is not None else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "write_memory":
            if not memory_reference:
                return _tool_error("memory_reference is required for write_memory")
            if data is None:
                return _tool_error("data is required for write_memory")
            out = dap_session.write_memory(
                memory_reference=str(memory_reference),
                data=str(data),
                offset=int(offset) if offset is not None else None,
                allow_partial=bool(allow_partial) if allow_partial is not None else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "modules":
            out = dap_session.modules(
                start_module=int(start_module) if start_module is not None else None,
                module_count=int(module_count) if module_count is not None else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "loaded_sources":
            out = dap_session.loaded_sources(timeout=t, session_id=sid)
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "custom_request":
            if not command:
                return _tool_error("command is required for custom_request")
            args_obj = arguments if isinstance(arguments, dict) else None
            out = dap_session.custom_request(
                command=str(command),
                arguments=args_obj,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "continue":
            out = dap_session.continue_(
                thread_id=int(thread_id) if thread_id is not None else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "next":
            out = dap_session.next_(
                thread_id=int(thread_id) if thread_id is not None else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "step_in":
            out = dap_session.step_in(
                thread_id=int(thread_id) if thread_id is not None else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "step_out":
            out = dap_session.step_out(
                thread_id=int(thread_id) if thread_id is not None else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        if act == "pause":
            out = dap_session.pause(
                thread_id=int(thread_id) if thread_id is not None else None,
                timeout=t,
                session_id=sid,
            )
            return json.dumps({"success": True, "action": act, **out}, ensure_ascii=False)

        return _tool_error(f"unhandled action {act!r}")
    except DapClientError as exc:
        return _tool_error(str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("debug tool failed")
        return _tool_error(f"{type(exc).__name__}: {exc}")


registry.register(
    name="debug",
    toolset="debug",
    schema=DEBUG_SCHEMA,
    handler=lambda args, **kw: debug_tool(
        action=args.get("action", ""),
        session_id=args.get("session_id"),
        adapter=args.get("adapter"),
        program=args.get("program"),
        cwd=args.get("cwd"),
        args=args.get("args"),
        host=args.get("host"),
        port=args.get("port"),
        pid=args.get("pid"),
        connect=args.get("connect", False),
        stop_on_entry=args.get("stop_on_entry", False),
        breakpoints=args.get("breakpoints"),
        path=args.get("path"),
        file=args.get("file"),
        line=args.get("line"),
        lines=args.get("lines"),
        function=args.get("function"),
        condition=args.get("condition"),
        hit_condition=args.get("hit_condition"),
        name=args.get("name"),
        data_id=args.get("data_id"),
        access_type=args.get("access_type"),
        instruction_reference=args.get("instruction_reference"),
        memory_reference=args.get("memory_reference"),
        offset=args.get("offset"),
        instruction_offset=args.get("instruction_offset"),
        instruction_count=args.get("instruction_count"),
        resolve_symbols=args.get("resolve_symbols"),
        count=args.get("count"),
        data=args.get("data"),
        allow_partial=args.get("allow_partial"),
        start_module=args.get("start_module"),
        module_count=args.get("module_count"),
        command=args.get("command"),
        arguments=args.get("arguments"),
        thread_id=args.get("thread_id"),
        frame_id=args.get("frame_id"),
        variables_reference=args.get("variables_reference"),
        variable_ref=args.get("variable_ref"),
        scope_id=args.get("scope_id"),
        expression=args.get("expression"),
        levels=args.get("levels"),
        timeout=args.get("timeout"),
    ),
    check_fn=check_debug_requirements,
)
