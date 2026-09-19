"""Process-scoped debug sessions (W3 / W3+).

Python (debugpy): TCP adapter via ``python -m debugpy.adapter``.

JavaScript/TypeScript (js-debug): TCP ``node dapDebugServer.js <port>`` with
reverse ``runInTerminal`` / ``startDebugging`` (OMP-aligned child sessions).

Launch: initialize → launch → ``initialized`` → setBreakpoints →
configurationDone → wait launch response → optional ``stopped``.

Attach (debugpy listen): initialize → attach(listen) → wait
``debugpyWaitingForServer`` → optionally spawn ``python -m debugpy --connect``
→ ``initialized`` → setBreakpoints → configurationDone → wait attach.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from agent.dap.adapters import (
    ADAPTER_DEBUGPY,
    ADAPTER_GDB,
    ADAPTER_JS_DEBUG,
    ADAPTER_LLDB,
    find_gdb,
    find_js_debug,
    find_lldb,
    infer_adapter,
    resolve_gdb,
    resolve_lldb_dap,
)
from agent.dap.client import DapClient, DapClientError

logger = logging.getLogger("agent.dap.session")

_MAX_SESSIONS = 3
_lock = threading.RLock()
_sessions: dict[str, "DebugSession"] = {}
_active_id: Optional[str] = None


def _default_work_cwd() -> str:
    """Prefer TERMINAL_CWD (UI stamps repo root) over process getcwd()."""
    env = (os.environ.get("TERMINAL_CWD") or "").strip()
    if env:
        try:
            p = Path(env).expanduser().resolve()
            if p.is_dir():
                return str(p)
        except Exception:
            pass
    return os.getcwd()


def get_session(session_id: Optional[str] = None) -> Optional["DebugSession"]:
    with _lock:
        sid = (session_id or _active_id or "").strip() or None
        if not sid:
            return None
        return _sessions.get(sid)


def clear_session(session_id: Optional[str] = None) -> None:
    """Close and drop one session (active if ``session_id`` omitted).

    Also closes js-debug child sessions that list this id as parent.
    """
    global _active_id
    with _lock:
        sid = (session_id or _active_id or "").strip() or None
        if not sid:
            return
        child_ids = [
            cid
            for cid, s in _sessions.items()
            if getattr(s, "parent_session_id", None) == sid
        ]
    for cid in child_ids:
        clear_session(cid)
    with _lock:
        sess = _sessions.pop(sid, None)
        if _active_id == sid:
            _active_id = next(iter(_sessions), None)
        if sess is not None:
            try:
                sess.close()
            except Exception:
                logger.debug("debug session close failed", exc_info=True)


def clear_all_sessions() -> None:
    global _active_id
    with _lock:
        ids = list(_sessions.keys())
        _active_id = None
    for sid in ids:
        clear_session(sid)


def list_sessions() -> list[dict[str, Any]]:
    with _lock:
        out = []
        for sid, sess in _sessions.items():
            row = sess.summary()
            row["session_id"] = sid
            row["active"] = sid == _active_id
            out.append(row)
        return out


def select_session(session_id: str) -> dict[str, Any]:
    global _active_id
    sid = str(session_id or "").strip()
    with _lock:
        if sid not in _sessions:
            raise DapClientError(f"unknown debug session_id {sid!r}")
        _active_id = sid
        return {"ok": True, "session_id": sid, "session": _sessions[sid].summary()}


class DebugSession:
    def __init__(
        self,
        client: DapClient,
        *,
        session_id: str,
        adapter: str = ADAPTER_DEBUGPY,
        parent_session_id: Optional[str] = None,
    ) -> None:
        self.session_id = session_id
        self.client = client
        self.adapter = adapter
        self.parent_session_id = parent_session_id
        self.status = "idle"
        self.program: Optional[str] = None
        self.cwd: Optional[str] = None
        self.thread_id: Optional[int] = None
        self.last_stopped: Optional[dict[str, Any]] = None
        self.capabilities: dict[str, Any] = {}
        self.created_at = time.time()
        self.listen_host: Optional[str] = None
        self.listen_port: Optional[int] = None
        self.debuggee_proc: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self._pending_breakpoints: dict[str, list[dict[str, int]]] = {}
        self.output_chunks: list[str] = []
        # Live breakpoint registries for remove_* actions (OMP parity).
        self.source_breakpoints: dict[str, set[int]] = {}
        self.function_breakpoints: set[str] = set()
        self.instruction_breakpoints: list[dict[str, Any]] = []
        self.data_breakpoints: list[dict[str, Any]] = []

    def close(self) -> None:
        try:
            if self.debuggee_proc is not None and self.debuggee_proc.poll() is None:
                try:
                    self.debuggee_proc.terminate()
                    self.debuggee_proc.wait(timeout=2)
                except Exception:
                    try:
                        self.debuggee_proc.kill()
                    except Exception:
                        pass
            if self.status not in {"idle", "closed", "terminated"}:
                try:
                    self.client.request(
                        "disconnect",
                        {"terminateDebuggee": True},
                        timeout=2.0,
                    )
                except Exception:
                    pass
        finally:
            self.status = "closed"
            self.client.close()

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "session_id": self.session_id,
            "adapter": self.adapter,
            "status": self.status,
            "program": self.program,
            "cwd": self.cwd,
            "thread_id": self.thread_id,
            "last_stopped": self.last_stopped,
            "age_s": round(time.time() - self.created_at, 1),
        }
        if self.parent_session_id:
            out["parent_session_id"] = self.parent_session_id
        if self.pid is not None:
            out["pid"] = self.pid
        if self.listen_host is not None and self.listen_port is not None:
            out["listen"] = {"host": self.listen_host, "port": self.listen_port}
        if self.client.adapter_port is not None:
            out["adapter_port"] = self.client.adapter_port
        if self.debuggee_proc is not None:
            out["debuggee_pid"] = self.debuggee_proc.pid
            out["debuggee_alive"] = self.debuggee_proc.poll() is None
        return out


def _register_session(session: DebugSession) -> None:
    global _active_id
    with _lock:
        _sessions[session.session_id] = session
        _active_id = session.session_id


def _prepare_slot() -> str:
    """Allocate a session id; prune closed; enforce ``_MAX_SESSIONS`` on roots.

    js-debug child sessions (``parent_session_id`` set) do not count toward the
    cap — a root + its child is still one logical debug tree.
    """
    global _active_id
    while True:
        evict: Optional[str] = None
        with _lock:
            dead = [
                sid for sid, s in _sessions.items() if s.status in {"closed", "terminated"}
            ]
            for sid in dead:
                sess = _sessions.pop(sid, None)
                if _active_id == sid:
                    _active_id = None
                if sess is not None:
                    try:
                        sess.close()
                    except Exception:
                        pass
            roots = [sid for sid, s in _sessions.items() if not s.parent_session_id]
            if len(roots) >= _MAX_SESSIONS:
                candidates = [sid for sid in roots if sid != _active_id] or roots
                if candidates:
                    evict = min(candidates, key=lambda s: _sessions[s].created_at)
            else:
                if _active_id is None and _sessions:
                    _active_id = next(iter(_sessions))
                return uuid.uuid4().hex[:12]
        if evict:
            clear_session(evict)
            continue
        with _lock:
            if _active_id is None and _sessions:
                _active_id = next(iter(_sessions))
            return uuid.uuid4().hex[:12]


def ensure_idle_or_replace() -> None:
    """Back-compat: clear the active session before launch when callers expect replace."""
    clear_session()


def _pick_listen_port(host: str, port: Optional[int]) -> int:
    if port is not None and int(port) > 0:
        return int(port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _initialize_client(
    client: DapClient,
    *,
    timeout: float,
    adapter: str = ADAPTER_DEBUGPY,
) -> dict[str, Any]:
    js = adapter == ADAPTER_JS_DEBUG
    if js:
        adapter_id = "js-debug-adapter"
    elif adapter == ADAPTER_LLDB:
        adapter_id = "lldb-dap"
    elif adapter == ADAPTER_GDB:
        adapter_id = "gdb"
    else:
        adapter_id = "debugpy"
    resp = client.request(
        "initialize",
        {
            "clientID": "hairball",
            "clientName": "hairball",
            "adapterID": adapter_id,
            "pathFormat": "path",
            "linesStartAt1": True,
            "columnsStartAt1": True,
            "supportsVariableType": True,
            "supportsRunInTerminalRequest": js,
            "supportsStartDebuggingRequest": js,
        },
        timeout=timeout,
    )
    return dict(resp.get("body") or {})


def _wire_js_reverse_requests(session: DebugSession, *, timeout: float) -> None:
    """OMP-aligned reverse requests for vscode-js-debug."""

    def on_run_in_terminal(args: dict[str, Any]) -> dict[str, Any]:
        cmd = args.get("args")
        if not isinstance(cmd, list) or not cmd:
            raise DapClientError("runInTerminal request did not include a command")
        work = str(args.get("cwd") or session.cwd or _default_work_cwd())
        env = os.environ.copy()
        raw_env = args.get("env") if isinstance(args.get("env"), dict) else {}
        for key, val in raw_env.items():
            if val is None:
                env.pop(str(key), None)
            else:
                env[str(key)] = str(val)
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=work,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        session.debuggee_proc = proc
        return {"processId": proc.pid}

    def on_start_debugging(args: dict[str, Any]) -> dict[str, Any]:
        request = "attach" if args.get("request") == "attach" else "launch"
        configuration = (
            args.get("configuration") if isinstance(args.get("configuration"), dict) else {}
        )
        _start_js_child_session(
            session,
            request=request,
            configuration=configuration,
            timeout=timeout,
        )
        return {}

    session.client.on_reverse_request("runInTerminal", on_run_in_terminal)
    session.client.on_reverse_request("startDebugging", on_start_debugging)


def _filter_telemetry_output_enabled() -> bool:
    try:
        from hairball_cli.config import load_config

        cfg = load_config() or {}
        debug = cfg.get("debug") if isinstance(cfg, dict) else None
        if isinstance(debug, dict) and "filter_telemetry_output" in debug:
            return bool(debug.get("filter_telemetry_output"))
    except Exception:  # noqa: BLE001
        pass
    return True


def _should_keep_output_event(body: dict[str, Any]) -> bool:
    """Hairball upgrade: optionally drop DAP telemetry noise; keep stdout/stderr."""
    if not _filter_telemetry_output_enabled():
        return True
    category = str(body.get("category") or "").strip().lower()
    return category != "telemetry"


def _bind_stopped_events(session: DebugSession) -> None:
    def on_event(msg: dict[str, Any]) -> None:
        ev = msg.get("event")
        if ev == "output":
            body = msg.get("body") if isinstance(msg.get("body"), dict) else {}
            if not _should_keep_output_event(body):
                return
            text = str(body.get("output") or "")
            if text:
                session.output_chunks.append(text)
                joined = "".join(session.output_chunks)
                if len(joined) > 65536:
                    session.output_chunks = [joined[-65536:]]
            return
        if ev != "stopped":
            return
        body = msg.get("body") if isinstance(msg.get("body"), dict) else {}
        session.last_stopped = body
        try:
            session.thread_id = int(body.get("threadId"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
        session.status = "stopped"
        global _active_id
        with _lock:
            _active_id = session.session_id

    session.client.on_event(on_event)


def _start_js_child_session(
    parent: DebugSession,
    *,
    request: str,
    configuration: dict[str, Any],
    timeout: float,
) -> DebugSession:
    port = parent.client.adapter_port
    if port is None:
        raise DapClientError("js-debug parent has no adapter_port for child session")
    child_client = DapClient.connect_tcp("127.0.0.1", int(port), timeout=min(15.0, timeout))
    sid = uuid.uuid4().hex[:12]
    child = DebugSession(
        child_client,
        session_id=sid,
        adapter=ADAPTER_JS_DEBUG,
        parent_session_id=parent.session_id,
    )
    cfg_cwd = configuration.get("cwd")
    if isinstance(cfg_cwd, str) and cfg_cwd.strip():
        child.cwd = str(Path(cfg_cwd).expanduser().resolve())
    else:
        child.cwd = parent.cwd
    prog = configuration.get("program")
    if isinstance(prog, str) and prog.strip():
        try:
            child.program = str(Path(prog).expanduser().resolve())
        except Exception:
            child.program = prog
    child.status = "launching"
    _bind_stopped_events(child)
    _register_session(child)
    try:
        child.capabilities = _initialize_client(
            child_client, timeout=timeout, adapter=ADAPTER_JS_DEBUG
        )
        start_args = dict(configuration)
        if child.cwd and "cwd" not in start_args:
            start_args["cwd"] = child.cwd

        def after_initialized() -> None:
            child_client.wait_event("initialized", timeout=timeout)
            # Apply root breakpoints onto the child (where Node actually stops).
            bps = parent._pending_breakpoints or {}
            if bps:
                _apply_breakpoints(child_client, bps, timeout=timeout)
            child_client.request("configurationDone", {}, timeout=timeout)

        _run_start_request(
            child_client,
            request if request in {"launch", "attach"} else "launch",
            start_args,
            timeout=timeout,
            after_initialized=after_initialized,
        )
        child.status = "running"
        return child
    except Exception:
        clear_session(sid)
        raise


def _wait_js_tree_stopped(root: DebugSession, *, timeout: float) -> None:
    """Prefer a stopped child session (js-debug); fall back to root."""
    deadline = time.time() + min(12.0, max(1.0, timeout))
    while time.time() < deadline:
        with _lock:
            children = [
                s
                for s in _sessions.values()
                if s.parent_session_id == root.session_id and s.status == "stopped"
            ]
        if children:
            # Newest stopped child
            chosen = max(children, key=lambda s: s.created_at)
            select_session(chosen.session_id)
            return
        if root.status == "stopped" and root.last_stopped:
            select_session(root.session_id)
            return
        # Opportunistically drain parent stopped without long block.
        for ev in root.client.drain_events("stopped"):
            body = ev.get("body") if isinstance(ev.get("body"), dict) else {}
            root.last_stopped = body
            try:
                root.thread_id = int(body.get("threadId"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
            root.status = "stopped"
            select_session(root.session_id)
            return
        time.sleep(0.05)


def _normalize_breakpoint_line(value: Any) -> int:
    """Return a valid one-based DAP source line without lossy coercion."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DapClientError(
            f"breakpoint line must be a positive integer, got {value!r}"
        )
    return value


def _group_breakpoints(
    breakpoints: Optional[list[dict[str, Any]]],
    default_path: Optional[str],
) -> dict[str, list[dict[str, int]]]:
    by_file: dict[str, list[dict[str, int]]] = {}
    for bp in breakpoints or []:
        if not isinstance(bp, dict):
            continue
        line_i = _normalize_breakpoint_line(bp.get("line"))
        src = str(bp.get("file") or bp.get("path") or default_path or "").strip()
        if not src:
            continue
        try:
            src = str(Path(src).expanduser().resolve())
        except Exception:
            pass
        by_file.setdefault(src, []).append({"line": line_i})
    return by_file


def _apply_breakpoints(
    client: DapClient,
    by_file: dict[str, list[dict[str, int]]],
    *,
    timeout: float,
) -> list[dict[str, Any]]:
    bp_results: list[dict[str, Any]] = []
    for src, lines in by_file.items():
        resp = client.request(
            "setBreakpoints",
            {"source": {"path": src}, "breakpoints": lines},
            timeout=timeout,
        )
        bp_results.append({"file": src, "breakpoints": (resp.get("body") or {}).get("breakpoints")})
    return bp_results


def _capture_optional_stopped(session: DebugSession, *, timeout: float) -> None:
    try:
        stopped = session.client.wait_event("stopped", timeout=min(10.0, timeout))
    except DapClientError:
        return
    body = stopped.get("body") or {}
    session.last_stopped = body
    try:
        session.thread_id = int(body.get("threadId"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        session.thread_id = None
    session.status = "stopped"


def _run_start_request(
    client: DapClient,
    command: str,
    arguments: dict[str, Any],
    *,
    timeout: float,
    after_initialized,
) -> None:
    """Fire launch/attach on a worker; run ``after_initialized`` on the main thread."""
    box: dict[str, Any] = {"resp": None, "err": None}

    def _worker() -> None:
        try:
            box["resp"] = client.request(command, arguments, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            box["err"] = exc

    t = threading.Thread(target=_worker, name=f"dap-{command}", daemon=True)
    t.start()
    after_initialized()
    t.join(timeout=timeout)
    if box["err"] is not None:
        raise DapClientError(f"{command} failed: {box['err']}")
    if box["resp"] is None:
        raise DapClientError(f"{command} did not complete")


def initialize_and_launch(
    *,
    program: str,
    cwd: Optional[str] = None,
    args: Optional[list[str]] = None,
    stop_on_entry: bool = False,
    breakpoints: Optional[list[dict[str, Any]]] = None,
    python: Optional[str] = None,
    timeout: float = 30.0,
    adapter: Optional[str] = None,
) -> dict[str, Any]:
    """Launch under debugpy (Python) or js-debug (JS/TS). Optional breakpoints."""
    chosen = infer_adapter(program, adapter=adapter)
    if chosen == ADAPTER_JS_DEBUG:
        return _initialize_and_launch_js(
            program=program,
            cwd=cwd,
            args=args,
            stop_on_entry=stop_on_entry,
            breakpoints=breakpoints,
            timeout=timeout,
        )
    if chosen in {ADAPTER_LLDB, ADAPTER_GDB}:
        return _initialize_and_launch_native(
            program=program,
            cwd=cwd,
            args=args,
            stop_on_entry=stop_on_entry,
            breakpoints=breakpoints,
            timeout=timeout,
            adapter=chosen,
        )
    return _initialize_and_launch_python(
        program=program,
        cwd=cwd,
        args=args,
        stop_on_entry=stop_on_entry,
        breakpoints=breakpoints,
        python=python,
        timeout=timeout,
    )


def _initialize_and_launch_python(
    *,
    program: str,
    cwd: Optional[str] = None,
    args: Optional[list[str]] = None,
    stop_on_entry: bool = False,
    breakpoints: Optional[list[dict[str, Any]]] = None,
    python: Optional[str] = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    sid = _prepare_slot()
    prog = str(Path(program).expanduser().resolve())
    work = str(Path(cwd).expanduser().resolve()) if cwd else str(Path(prog).parent)
    client = DapClient.spawn_debugpy(python=python, timeout=min(15.0, timeout))
    session = DebugSession(client, session_id=sid, adapter=ADAPTER_DEBUGPY)
    session.program = prog
    session.cwd = work
    session.status = "launching"
    _register_session(session)

    try:
        session.capabilities = _initialize_client(
            client, timeout=timeout, adapter=ADAPTER_DEBUGPY
        )
        launch_args: dict[str, Any] = {
            "name": "hairball-python",
            "type": "python",
            "request": "launch",
            "program": prog,
            "cwd": work,
            "console": "internalConsole",
            "stopOnEntry": bool(stop_on_entry),
            "justMyCode": True,
        }
        if args:
            launch_args["args"] = [str(a) for a in args]

        by_file = _group_breakpoints(breakpoints, prog)
        bp_results: list[dict[str, Any]] = []

        def after_initialized() -> None:
            nonlocal bp_results
            client.wait_event("initialized", timeout=timeout)
            bp_results = _apply_breakpoints(client, by_file, timeout=timeout)
            client.request("configurationDone", {}, timeout=timeout)

        _run_start_request(
            client, "launch", launch_args, timeout=timeout, after_initialized=after_initialized
        )
        session.status = "running"
        _capture_optional_stopped(session, timeout=timeout)
        return {
            "ok": True,
            "session_id": sid,
            "adapter": ADAPTER_DEBUGPY,
            "session": session.summary(),
            "breakpoints": bp_results,
            "stopped": session.last_stopped,
        }
    except Exception:
        clear_session(sid)
        raise


def _initialize_and_launch_js(
    *,
    program: str,
    cwd: Optional[str] = None,
    args: Optional[list[str]] = None,
    stop_on_entry: bool = False,
    breakpoints: Optional[list[dict[str, Any]]] = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    if not find_js_debug(cwd):
        raise DapClientError(
            "js-debug unavailable — need node + dapDebugServer.js "
            "(HAIRBALL_JS_DEBUG_DAP_SERVER or ~/.local/opt/js-debug)"
        )
    sid = _prepare_slot()
    prog = str(Path(program).expanduser().resolve())
    work = str(Path(cwd).expanduser().resolve()) if cwd else str(Path(prog).parent)
    client = DapClient.spawn_js_debug(timeout=min(15.0, timeout), cwd=work)
    session = DebugSession(client, session_id=sid, adapter=ADAPTER_JS_DEBUG)
    session.program = prog
    session.cwd = work
    session.status = "launching"
    _bind_stopped_events(session)
    _wire_js_reverse_requests(session, timeout=timeout)
    _register_session(session)

    try:
        session.capabilities = _initialize_client(
            client, timeout=timeout, adapter=ADAPTER_JS_DEBUG
        )
        launch_args: dict[str, Any] = {
            "name": "hairball-js",
            "type": "pwa-node",
            "request": "launch",
            "program": prog,
            "cwd": work,
            "stopOnEntry": bool(stop_on_entry),
        }
        if args:
            launch_args["args"] = [str(a) for a in args]

        by_file = _group_breakpoints(breakpoints, prog)
        session._pending_breakpoints = by_file
        bp_results: list[dict[str, Any]] = []

        def after_initialized() -> None:
            nonlocal bp_results
            client.wait_event("initialized", timeout=timeout)
            # Root may not bind BPs; children re-apply via _pending_breakpoints.
            bp_results = _apply_breakpoints(client, by_file, timeout=timeout)
            client.request("configurationDone", {}, timeout=timeout)

        _run_start_request(
            client, "launch", launch_args, timeout=timeout, after_initialized=after_initialized
        )
        session.status = "running"
        _wait_js_tree_stopped(session, timeout=timeout)
        active = get_session() or session
        return {
            "ok": True,
            "session_id": active.session_id,
            "root_session_id": sid,
            "adapter": ADAPTER_JS_DEBUG,
            "session": active.summary(),
            "breakpoints": bp_results,
            "stopped": active.last_stopped,
        }
    except Exception:
        clear_session(sid)
        raise


def initialize_and_attach(
    *,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    program: Optional[str] = None,
    cwd: Optional[str] = None,
    args: Optional[list[str]] = None,
    breakpoints: Optional[list[dict[str, Any]]] = None,
    python: Optional[str] = None,
    timeout: float = 30.0,
    pid: Optional[int] = None,
    connect: bool = False,
    adapter: Optional[str] = None,
) -> dict[str, Any]:
    """Attach via debugpy or js-debug.

    debugpy modes (first match wins):
    - ``pid``: inject with ``python -m debugpy --listen host:port --pid`` then
      DAP attach ``connect``.
    - ``connect=True`` (or host+port without program): DAP attach to an already
      listening debuggee at host:port.
    - default listen: adapter listens; optional ``program`` spawns
      ``python -m debugpy --connect``.

    js-debug: ``program`` → launch; else attach to Node inspector (host/port/pid).
    """
    chosen = infer_adapter(program, adapter=adapter)
    if chosen == ADAPTER_JS_DEBUG:
        # program-only → launch; host/port/pid/connect → inspector attach.
        if (
            program
            and not connect
            and pid is None
            and (port is None or int(port) <= 0)
        ):
            return _initialize_and_launch_js(
                program=program,
                cwd=cwd,
                args=args,
                stop_on_entry=False,
                breakpoints=breakpoints,
                timeout=timeout,
            )
        return _initialize_and_attach_js(
            host=host,
            port=port,
            program=program,
            cwd=cwd,
            args=args,
            breakpoints=breakpoints,
            timeout=timeout,
            pid=pid,
        )
    if chosen in {ADAPTER_LLDB, ADAPTER_GDB}:
        if program and pid is None and not connect:
            return _initialize_and_launch_native(
                program=program,
                cwd=cwd,
                args=args,
                stop_on_entry=False,
                breakpoints=breakpoints,
                timeout=timeout,
                adapter=chosen,
            )
        return _initialize_and_attach_native(
            host=host,
            port=port,
            program=program,
            cwd=cwd,
            breakpoints=breakpoints,
            timeout=timeout,
            pid=pid,
            adapter=chosen,
        )

    sid = _prepare_slot()
    listen_host = (host or "127.0.0.1").strip() or "127.0.0.1"
    py = python or sys.executable
    client = DapClient.spawn_debugpy(python=py, timeout=min(15.0, timeout))
    session = DebugSession(client, session_id=sid, adapter=ADAPTER_DEBUGPY)
    session.status = "attaching"
    if pid is not None:
        session.pid = int(pid)
    if program:
        session.program = str(Path(program).expanduser().resolve())
        session.cwd = (
            str(Path(cwd).expanduser().resolve())
            if cwd
            else str(Path(session.program).parent)
        )
    elif cwd:
        session.cwd = str(Path(cwd).expanduser().resolve())
    _register_session(session)

    try:
        session.capabilities = _initialize_client(
            client, timeout=timeout, adapter=ADAPTER_DEBUGPY
        )
        by_file = _group_breakpoints(breakpoints, session.program)
        bp_results: list[dict[str, Any]] = []
        mode = "listen"
        wait_info: dict[str, Any] = {}

        # --- PID inject → connect attach ---
        if pid is not None:
            mode = "pid"
            inject_port = _pick_listen_port(listen_host, port)
            session.listen_host = listen_host
            session.listen_port = inject_port
            inj = subprocess.Popen(
                [
                    py,
                    "-m",
                    "debugpy",
                    "--listen",
                    f"{listen_host}:{inject_port}",
                    "--pid",
                    str(int(pid)),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            session.debuggee_proc = inj
            # Give inject a moment before DAP connect.
            time.sleep(0.4)
            attach_args: dict[str, Any] = {
                "name": "hairball-python-attach-pid",
                "type": "python",
                "request": "attach",
                "connect": {"host": listen_host, "port": inject_port},
                "justMyCode": True,
            }

            def after_initialized_pid() -> None:
                nonlocal bp_results
                client.wait_event("initialized", timeout=timeout)
                bp_results = _apply_breakpoints(client, by_file, timeout=timeout)
                client.request("configurationDone", {}, timeout=timeout)

            _run_start_request(
                client,
                "attach",
                attach_args,
                timeout=timeout,
                after_initialized=after_initialized_pid,
            )
            wait_info = {"host": listen_host, "port": inject_port}

        # --- Connect to already-listening debuggee ---
        elif connect or (port is not None and int(port) > 0 and not program):
            mode = "connect"
            connect_port = int(port) if port is not None and int(port) > 0 else 5678
            session.listen_host = listen_host
            session.listen_port = connect_port
            attach_args = {
                "name": "hairball-python-attach-connect",
                "type": "python",
                "request": "attach",
                "connect": {"host": listen_host, "port": connect_port},
                "justMyCode": True,
            }

            def after_initialized_connect() -> None:
                nonlocal bp_results
                client.wait_event("initialized", timeout=timeout)
                bp_results = _apply_breakpoints(client, by_file, timeout=timeout)
                client.request("configurationDone", {}, timeout=timeout)

            _run_start_request(
                client,
                "attach",
                attach_args,
                timeout=timeout,
                after_initialized=after_initialized_connect,
            )
            wait_info = {"host": listen_host, "port": connect_port}

        # --- Listen mode (existing) ---
        else:
            listen_port = _pick_listen_port(listen_host, port)
            session.listen_host = listen_host
            session.listen_port = listen_port
            attach_args = {
                "name": "hairball-python-attach",
                "type": "python",
                "request": "attach",
                "listen": {"host": listen_host, "port": listen_port},
                "justMyCode": True,
            }

            def after_initialized() -> None:
                nonlocal bp_results
                waiting = client.wait_event("debugpyWaitingForServer", timeout=timeout)
                body = waiting.get("body") or {}
                wait_host = str(body.get("host") or listen_host)
                try:
                    wait_port = int(body.get("port") or listen_port)
                except (TypeError, ValueError):
                    wait_port = listen_port
                session.listen_host = wait_host
                session.listen_port = wait_port
                wait_info["host"] = wait_host
                wait_info["port"] = wait_port

                if session.program:
                    cmd = [
                        py,
                        "-m",
                        "debugpy",
                        "--connect",
                        f"{wait_host}:{wait_port}",
                        session.program,
                    ]
                    if args:
                        cmd.extend(str(a) for a in args)
                    session.debuggee_proc = subprocess.Popen(
                        cmd,
                        cwd=session.cwd or None,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                client.wait_event("initialized", timeout=timeout)
                bp_results = _apply_breakpoints(client, by_file, timeout=timeout)
                client.request("configurationDone", {}, timeout=timeout)

            _run_start_request(
                client, "attach", attach_args, timeout=timeout, after_initialized=after_initialized
            )

        session.status = "running"
        _capture_optional_stopped(session, timeout=timeout)
        return {
            "ok": True,
            "session_id": sid,
            "adapter": ADAPTER_DEBUGPY,
            "mode": mode,
            "listen": wait_info or {"host": session.listen_host, "port": session.listen_port},
            "connect_hint": (
                f"python -m debugpy --connect {session.listen_host}:{session.listen_port} <script.py>"
                if mode == "listen" and not session.program
                else None
            ),
            "session": session.summary(),
            "breakpoints": bp_results,
            "stopped": session.last_stopped,
        }
    except Exception:
        clear_session(sid)
        raise


def _initialize_and_attach_js(
    *,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    program: Optional[str] = None,
    cwd: Optional[str] = None,
    args: Optional[list[str]] = None,
    breakpoints: Optional[list[dict[str, Any]]] = None,
    timeout: float = 30.0,
    pid: Optional[int] = None,
) -> dict[str, Any]:
    if not find_js_debug(cwd):
        raise DapClientError(
            "js-debug unavailable — need node + dapDebugServer.js "
            "(HAIRBALL_JS_DEBUG_DAP_SERVER or ~/.local/opt/js-debug)"
        )
    sid = _prepare_slot()
    listen_host = (host or "127.0.0.1").strip() or "127.0.0.1"
    work = str(Path(cwd).expanduser().resolve()) if cwd else None
    if program:
        prog = str(Path(program).expanduser().resolve())
        work = work or str(Path(prog).parent)
    else:
        prog = None
        work = work or _default_work_cwd()

    client = DapClient.spawn_js_debug(timeout=min(15.0, timeout), cwd=work)
    session = DebugSession(client, session_id=sid, adapter=ADAPTER_JS_DEBUG)
    session.program = prog
    session.cwd = work
    session.status = "attaching"
    if pid is not None:
        session.pid = int(pid)
    _bind_stopped_events(session)
    _wire_js_reverse_requests(session, timeout=timeout)
    _register_session(session)

    try:
        session.capabilities = _initialize_client(
            client, timeout=timeout, adapter=ADAPTER_JS_DEBUG
        )
        by_file = _group_breakpoints(breakpoints, prog)
        session._pending_breakpoints = by_file
        bp_results: list[dict[str, Any]] = []
        attach_port = int(port) if port is not None and int(port) > 0 else 9229
        session.listen_host = listen_host
        session.listen_port = attach_port
        attach_args: dict[str, Any] = {
            "name": "hairball-js-attach",
            "type": "pwa-node",
            "request": "attach",
            "address": listen_host,
            "port": attach_port,
            "cwd": work,
        }
        if pid is not None:
            attach_args["processId"] = int(pid)
        if args:
            attach_args["args"] = [str(a) for a in args]

        def after_initialized() -> None:
            nonlocal bp_results
            client.wait_event("initialized", timeout=timeout)
            bp_results = _apply_breakpoints(client, by_file, timeout=timeout)
            client.request("configurationDone", {}, timeout=timeout)

        _run_start_request(
            client, "attach", attach_args, timeout=timeout, after_initialized=after_initialized
        )
        session.status = "running"
        _wait_js_tree_stopped(session, timeout=timeout)
        active = get_session() or session
        return {
            "ok": True,
            "session_id": active.session_id,
            "root_session_id": sid,
            "adapter": ADAPTER_JS_DEBUG,
            "mode": "pid" if pid is not None else "connect",
            "listen": {"host": listen_host, "port": attach_port},
            "connect_hint": None,
            "session": active.summary(),
            "breakpoints": bp_results,
            "stopped": active.last_stopped,
        }
    except Exception:
        clear_session(sid)
        raise


def _native_stdio_command(adapter: str) -> list[str]:
    if adapter == ADAPTER_LLDB:
        path = resolve_lldb_dap()
        if not path:
            raise DapClientError(
                "lldb-dap not found — install LLVM (brew install llvm) or set HAIRBALL_LLDB_DAP"
            )
        return [path]
    if adapter == ADAPTER_GDB:
        path = resolve_gdb()
        if not path:
            raise DapClientError("gdb not found — install gdb or set HAIRBALL_GDB")
        return [path, "-i", "dap"]
    raise DapClientError(f"unsupported native adapter {adapter!r}")


def _initialize_and_launch_native(
    *,
    program: str,
    cwd: Optional[str] = None,
    args: Optional[list[str]] = None,
    stop_on_entry: bool = False,
    breakpoints: Optional[list[dict[str, Any]]] = None,
    timeout: float = 30.0,
    adapter: str = ADAPTER_LLDB,
) -> dict[str, Any]:
    """Launch a native binary via lldb-dap or gdb DAP (stdio)."""
    if adapter == ADAPTER_LLDB and not find_lldb():
        raise DapClientError(
            "lldb-dap unavailable — brew install llvm or set HAIRBALL_LLDB_DAP"
        )
    if adapter == ADAPTER_GDB and not find_gdb():
        raise DapClientError("gdb unavailable — install gdb or set HAIRBALL_GDB")

    sid = _prepare_slot()
    prog = str(Path(program).expanduser().resolve())
    work = str(Path(cwd).expanduser().resolve()) if cwd else str(Path(prog).parent)
    client = DapClient.spawn_stdio(_native_stdio_command(adapter), cwd=work)
    session = DebugSession(client, session_id=sid, adapter=adapter)
    session.program = prog
    session.cwd = work
    session.status = "launching"
    _bind_stopped_events(session)
    _register_session(session)

    try:
        session.capabilities = _initialize_client(client, timeout=timeout, adapter=adapter)
        # OMP lldb/gdb defaults use stopOnEntry; with breakpoints prefer entry
        # stop so configurationDone does not race a free-running process.
        want_entry = bool(stop_on_entry) or bool(breakpoints)
        if adapter == ADAPTER_GDB:
            launch_args: dict[str, Any] = {
                "name": "hairball-gdb",
                "type": "gdb",
                "request": "launch",
                "program": prog,
                "cwd": work,
                "stopOnEntry": want_entry,
                "stopAtBeginningOfMainSubprogram": True,
            }
        else:
            launch_args = {
                "name": "hairball-lldb",
                "type": "lldb-dap",
                "request": "launch",
                "program": prog,
                "cwd": work,
                "stopOnEntry": want_entry,
            }
        if args:
            launch_args["args"] = [str(a) for a in args]

        by_file = _group_breakpoints(breakpoints, prog)
        bp_results: list[dict[str, Any]] = []

        def after_initialized() -> None:
            nonlocal bp_results
            client.wait_event("initialized", timeout=timeout)
            bp_results = _apply_breakpoints(client, by_file, timeout=timeout)
            _configuration_done(client, timeout=timeout)

        _run_start_request(
            client, "launch", launch_args, timeout=timeout, after_initialized=after_initialized
        )
        session.status = "running"
        _capture_optional_stopped(session, timeout=timeout)
        return {
            "ok": True,
            "session_id": sid,
            "adapter": adapter,
            "session": session.summary(),
            "breakpoints": bp_results,
            "stopped": session.last_stopped,
        }
    except Exception:
        clear_session(sid)
        raise


def _configuration_done(client: DapClient, *, timeout: float) -> None:
    """Send configurationDone; tolerate lldb 'process already running'."""
    try:
        client.request("configurationDone", {}, timeout=timeout)
    except DapClientError as exc:
        msg = str(exc).lower()
        if "already running" in msg or "resume request failed" in msg:
            logger.debug("configurationDone ignored: %s", exc)
            return
        raise


def _initialize_and_attach_native(
    *,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    program: Optional[str] = None,
    cwd: Optional[str] = None,
    breakpoints: Optional[list[dict[str, Any]]] = None,
    timeout: float = 30.0,
    pid: Optional[int] = None,
    adapter: str = ADAPTER_LLDB,
) -> dict[str, Any]:
    """Attach lldb-dap/gdb to a running process (pid) or optional program path."""
    if adapter == ADAPTER_LLDB and not find_lldb():
        raise DapClientError(
            "lldb-dap unavailable — brew install llvm or set HAIRBALL_LLDB_DAP"
        )
    if adapter == ADAPTER_GDB and not find_gdb():
        raise DapClientError("gdb unavailable — install gdb or set HAIRBALL_GDB")
    if pid is None and not program:
        raise DapClientError("native attach requires pid or program")

    sid = _prepare_slot()
    work = str(Path(cwd).expanduser().resolve()) if cwd else _default_work_cwd()
    prog = str(Path(program).expanduser().resolve()) if program else None
    client = DapClient.spawn_stdio(_native_stdio_command(adapter), cwd=work)
    session = DebugSession(client, session_id=sid, adapter=adapter)
    session.program = prog
    session.cwd = work
    session.status = "attaching"
    if pid is not None:
        session.pid = int(pid)
    _bind_stopped_events(session)
    _register_session(session)

    try:
        session.capabilities = _initialize_client(client, timeout=timeout, adapter=adapter)
        by_file = _group_breakpoints(breakpoints, prog)
        bp_results: list[dict[str, Any]] = []
        attach_args: dict[str, Any] = {
            "name": f"hairball-{adapter}-attach",
            "type": "gdb" if adapter == ADAPTER_GDB else "lldb-dap",
            "request": "attach",
        }
        if pid is not None:
            attach_args["pid"] = int(pid)
        if prog:
            attach_args["program"] = prog
        if port is not None and int(port) > 0:
            attach_args["port"] = int(port)
            attach_args["host"] = (host or "127.0.0.1").strip() or "127.0.0.1"

        def after_initialized() -> None:
            nonlocal bp_results
            client.wait_event("initialized", timeout=timeout)
            bp_results = _apply_breakpoints(client, by_file, timeout=timeout)
            _configuration_done(client, timeout=timeout)

        _run_start_request(
            client, "attach", attach_args, timeout=timeout, after_initialized=after_initialized
        )
        session.status = "running"
        _capture_optional_stopped(session, timeout=timeout)
        return {
            "ok": True,
            "session_id": sid,
            "adapter": adapter,
            "mode": "pid" if pid is not None else "attach",
            "session": session.summary(),
            "breakpoints": bp_results,
            "stopped": session.last_stopped,
        }
    except Exception:
        clear_session(sid)
        raise


def require_session(session_id: Optional[str] = None) -> DebugSession:
    session = get_session(session_id)
    if session is None or session.status == "closed":
        raise DapClientError("no active debug session — call action=launch or attach first")
    return session


def set_breakpoints(
    *,
    path: str,
    lines: list[int],
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    src = str(Path(path).expanduser().resolve())
    normalized = [_normalize_breakpoint_line(x) for x in lines]
    bps = [{"line": line} for line in normalized]
    resp = session.client.request(
        "setBreakpoints",
        {"source": {"path": src}, "breakpoints": bps},
        timeout=timeout,
    )
    session.source_breakpoints[src] = set(normalized)
    return {"ok": True, "file": src, "breakpoints": (resp.get("body") or {}).get("breakpoints")}


def stack_trace(
    *,
    thread_id: Optional[int] = None,
    levels: int = 20,
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    tid = thread_id if thread_id is not None else session.thread_id
    if tid is None:
        raise DapClientError("no thread_id — wait for a stopped event or pass thread_id")
    resp = session.client.request(
        "stackTrace",
        {"threadId": int(tid), "startFrame": 0, "levels": max(1, min(50, int(levels)))},
        timeout=timeout,
    )
    frames = (resp.get("body") or {}).get("stackFrames") or []
    slim = []
    for f in frames:
        if not isinstance(f, dict):
            continue
        src = f.get("source") if isinstance(f.get("source"), dict) else {}
        slim.append(
            {
                "id": f.get("id"),
                "name": f.get("name"),
                "line": f.get("line"),
                "column": f.get("column"),
                "path": src.get("path"),
            }
        )
    return {"ok": True, "thread_id": tid, "stackFrames": slim}


def threads(*, timeout: float = 15.0, session_id: Optional[str] = None) -> dict[str, Any]:
    """DAP ``threads`` request — list known threads."""
    session = require_session(session_id)
    resp = session.client.request("threads", {}, timeout=timeout)
    raw = (resp.get("body") or {}).get("threads") or []
    slim = []
    for th in raw:
        if not isinstance(th, dict):
            continue
        slim.append({"id": th.get("id"), "name": th.get("name")})
    return {"ok": True, "threads": slim}


def captured_output(*, clear: bool = False, session_id: Optional[str] = None) -> dict[str, Any]:
    """Return buffered DAP ``output`` events (stdout/stderr/console)."""
    session = require_session(session_id)
    # Also fold any queued Output events still in the client queue.
    for ev in session.client.drain_events("output"):
        body = ev.get("body") if isinstance(ev.get("body"), dict) else {}
        if not _should_keep_output_event(body):
            continue
        text = str(body.get("output") or "")
        if text:
            session.output_chunks.append(text)
    text = "".join(session.output_chunks)
    if clear:
        session.output_chunks.clear()
    return {"ok": True, "output": text}


def scopes(*, frame_id: int, timeout: float = 15.0, session_id: Optional[str] = None) -> dict[str, Any]:
    session = require_session(session_id)
    resp = session.client.request("scopes", {"frameId": int(frame_id)}, timeout=timeout)
    return {"ok": True, "scopes": (resp.get("body") or {}).get("scopes") or []}


def variables(
    *,
    variables_reference: int,
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    resp = session.client.request(
        "variables",
        {"variablesReference": int(variables_reference)},
        timeout=timeout,
    )
    vars_ = (resp.get("body") or {}).get("variables") or []
    slim = []
    for v in vars_:
        if not isinstance(v, dict):
            continue
        slim.append(
            {
                "name": v.get("name"),
                "value": v.get("value"),
                "type": v.get("type"),
                "variablesReference": v.get("variablesReference"),
            }
        )
    return {"ok": True, "variables": slim}


def evaluate(
    *,
    expression: str,
    frame_id: Optional[int] = None,
    context: str = "repl",
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    args: dict[str, Any] = {"expression": str(expression), "context": context}
    if frame_id is not None:
        args["frameId"] = int(frame_id)
    resp = session.client.request("evaluate", args, timeout=timeout)
    body = resp.get("body") or {}
    return {
        "ok": True,
        "result": body.get("result"),
        "type": body.get("type"),
        "variablesReference": body.get("variablesReference"),
    }


def _step_or_continue(
    command: str,
    *,
    thread_id: Optional[int] = None,
    timeout: float = 30.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    tid = thread_id if thread_id is not None else session.thread_id
    if tid is None:
        raise DapClientError("no thread_id")
    session.client.drain_events("stopped")
    session.client.drain_events("terminated")
    session.client.drain_events("exited")
    session.status = "running"
    resp = session.client.request(command, {"threadId": int(tid)}, timeout=timeout)

    wait_s = min(15.0, timeout)
    try:
        outcome = session.client.wait_event_any(
            ("stopped", "terminated", "exited"),
            timeout=wait_s,
        )
    except DapClientError:
        session.status = "running"
        return {
            "ok": True,
            "command": command,
            "response": resp.get("body"),
            "stopped": None,
            "ended": False,
            "session": session.summary(),
        }

    event_name = str(outcome.get("event") or "")
    if event_name == "stopped":
        body = outcome.get("body") or {}
        session.last_stopped = body
        try:
            session.thread_id = int(body.get("threadId"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
        session.status = "stopped"
        return {
            "ok": True,
            "command": command,
            "response": resp.get("body"),
            "stopped": body,
            "ended": False,
            "session": session.summary(),
        }

    session.status = "terminated"
    return {
        "ok": True,
        "command": command,
        "response": resp.get("body"),
        "stopped": None,
        "ended": True,
        "end_event": event_name,
        "end_body": outcome.get("body"),
        "session": session.summary(),
    }


def continue_(
    *,
    thread_id: Optional[int] = None,
    timeout: float = 30.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    return _step_or_continue("continue", thread_id=thread_id, timeout=timeout, session_id=session_id)


def next_(
    *,
    thread_id: Optional[int] = None,
    timeout: float = 30.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    return _step_or_continue("next", thread_id=thread_id, timeout=timeout, session_id=session_id)


def step_in(
    *,
    thread_id: Optional[int] = None,
    timeout: float = 30.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    return _step_or_continue("stepIn", thread_id=thread_id, timeout=timeout, session_id=session_id)


def step_out(
    *,
    thread_id: Optional[int] = None,
    timeout: float = 30.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    return _step_or_continue("stepOut", thread_id=thread_id, timeout=timeout, session_id=session_id)


def pause(
    *,
    thread_id: Optional[int] = None,
    timeout: float = 30.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Request the debuggee to pause; wait for a ``stopped`` event."""
    session = require_session(session_id)
    tid = thread_id if thread_id is not None else session.thread_id
    if tid is None:
        # Some adapters accept pause without a prior stop; default thread 1.
        tid = 1
    session.client.drain_events("stopped")
    session.status = "running"
    resp = session.client.request("pause", {"threadId": int(tid)}, timeout=timeout)
    try:
        stopped = session.client.wait_event("stopped", timeout=min(15.0, timeout))
    except DapClientError:
        return {
            "ok": True,
            "command": "pause",
            "response": resp.get("body"),
            "stopped": None,
            "session": session.summary(),
        }
    body = stopped.get("body") or {}
    session.last_stopped = body
    try:
        session.thread_id = int(body.get("threadId"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pass
    session.status = "stopped"
    return {
        "ok": True,
        "command": "pause",
        "response": resp.get("body"),
        "stopped": body,
        "session": session.summary(),
    }


def terminate(*, timeout: float = 10.0, session_id: Optional[str] = None) -> dict[str, Any]:
    session = get_session(session_id)
    if session is None:
        return {"ok": True, "message": "no active session"}
    sid = session.session_id
    # Already finished — skip slow disconnect round-trip (21:25 chat was ~5s).
    if session.status in {"terminated", "closed"}:
        clear_session(sid)
        return {"ok": True, "message": "session already ended", "session_id": sid}
    try:
        session.client.request(
            "disconnect",
            {"terminateDebuggee": True},
            timeout=min(3.0, timeout),
        )
    except Exception as exc:
        logger.debug("disconnect: %s", exc)
    clear_session(sid)
    return {"ok": True, "message": "session terminated", "session_id": sid}


def terminate_all(*, timeout: float = 10.0) -> dict[str, Any]:
    with _lock:
        ids = list(_sessions.keys())
    results = []
    for sid in ids:
        results.append(terminate(timeout=timeout, session_id=sid))
    return {"ok": True, "terminated": len(results), "results": results}


def _require_capability(session: DebugSession, key: str, label: str) -> None:
    caps = session.capabilities if isinstance(session.capabilities, dict) else {}
    if not caps.get(key):
        raise DapClientError(f"adapter does not support {label} ({key}=false)")


def set_breakpoint(
    *,
    path: Optional[str] = None,
    line: Optional[int] = None,
    function: Optional[str] = None,
    condition: Optional[str] = None,
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """OMP-style single breakpoint (source line or function)."""
    session = require_session(session_id)
    if function:
        name = str(function).strip()
        if not name:
            raise DapClientError("function is required")
        session.function_breakpoints.add(name)
        bps = [{"name": n} for n in sorted(session.function_breakpoints)]
        if condition:
            for bp in bps:
                if bp["name"] == name:
                    bp["condition"] = str(condition)
        resp = session.client.request(
            "setFunctionBreakpoints", {"breakpoints": bps}, timeout=timeout
        )
        return {
            "ok": True,
            "function": name,
            "breakpoints": (resp.get("body") or {}).get("breakpoints"),
        }
    if not path or line is None:
        raise DapClientError("set_breakpoint requires file+line or function")
    src = str(Path(path).expanduser().resolve())
    lines = set(session.source_breakpoints.get(src) or set())
    lines.add(_normalize_breakpoint_line(line))
    return set_breakpoints(path=src, lines=sorted(lines), timeout=timeout, session_id=session_id)


def remove_breakpoint(
    *,
    path: Optional[str] = None,
    line: Optional[int] = None,
    function: Optional[str] = None,
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    if function:
        name = str(function).strip()
        session.function_breakpoints.discard(name)
        bps = [{"name": n} for n in sorted(session.function_breakpoints)]
        resp = session.client.request(
            "setFunctionBreakpoints", {"breakpoints": bps}, timeout=timeout
        )
        return {
            "ok": True,
            "removed": name,
            "breakpoints": (resp.get("body") or {}).get("breakpoints"),
        }
    if not path or line is None:
        raise DapClientError("remove_breakpoint requires file+line or function")
    src = str(Path(path).expanduser().resolve())
    lines = set(session.source_breakpoints.get(src) or set())
    lines.discard(_normalize_breakpoint_line(line))
    return set_breakpoints(path=src, lines=sorted(lines), timeout=timeout, session_id=session_id)


def set_instruction_breakpoint(
    *,
    instruction_reference: str,
    offset: Optional[int] = None,
    condition: Optional[str] = None,
    hit_condition: Optional[str] = None,
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    _require_capability(session, "supportsInstructionBreakpoints", "instruction breakpoints")
    entry: dict[str, Any] = {"instructionReference": str(instruction_reference)}
    if offset is not None:
        entry["offset"] = int(offset)
    if condition:
        entry["condition"] = str(condition)
    if hit_condition:
        entry["hitCondition"] = str(hit_condition)
    session.instruction_breakpoints.append(entry)
    resp = session.client.request(
        "setInstructionBreakpoints",
        {"breakpoints": list(session.instruction_breakpoints)},
        timeout=timeout,
    )
    return {"ok": True, "breakpoints": (resp.get("body") or {}).get("breakpoints")}


def remove_instruction_breakpoint(
    *,
    instruction_reference: str,
    offset: Optional[int] = None,
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    _require_capability(session, "supportsInstructionBreakpoints", "instruction breakpoints")
    ref = str(instruction_reference)
    kept = []
    for bp in session.instruction_breakpoints:
        if bp.get("instructionReference") != ref:
            kept.append(bp)
            continue
        if offset is not None and bp.get("offset") != int(offset):
            kept.append(bp)
    session.instruction_breakpoints = kept
    resp = session.client.request(
        "setInstructionBreakpoints",
        {"breakpoints": list(session.instruction_breakpoints)},
        timeout=timeout,
    )
    return {"ok": True, "breakpoints": (resp.get("body") or {}).get("breakpoints")}


def data_breakpoint_info(
    *,
    name: str,
    variables_reference: Optional[int] = None,
    frame_id: Optional[int] = None,
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    _require_capability(session, "supportsDataBreakpoints", "data breakpoints")
    args: dict[str, Any] = {"name": str(name)}
    if variables_reference is not None:
        args["variablesReference"] = int(variables_reference)
    if frame_id is not None:
        args["frameId"] = int(frame_id)
    resp = session.client.request("dataBreakpointInfo", args, timeout=timeout)
    return {"ok": True, "info": resp.get("body") or {}}


def set_data_breakpoint(
    *,
    data_id: str,
    access_type: Optional[str] = None,
    condition: Optional[str] = None,
    hit_condition: Optional[str] = None,
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    _require_capability(session, "supportsDataBreakpoints", "data breakpoints")
    entry: dict[str, Any] = {"dataId": str(data_id)}
    if access_type:
        entry["accessType"] = str(access_type)
    if condition:
        entry["condition"] = str(condition)
    if hit_condition:
        entry["hitCondition"] = str(hit_condition)
    session.data_breakpoints.append(entry)
    resp = session.client.request(
        "setDataBreakpoints",
        {"breakpoints": list(session.data_breakpoints)},
        timeout=timeout,
    )
    return {"ok": True, "breakpoints": (resp.get("body") or {}).get("breakpoints")}


def remove_data_breakpoint(
    *,
    data_id: str,
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    _require_capability(session, "supportsDataBreakpoints", "data breakpoints")
    did = str(data_id)
    session.data_breakpoints = [
        bp for bp in session.data_breakpoints if bp.get("dataId") != did
    ]
    resp = session.client.request(
        "setDataBreakpoints",
        {"breakpoints": list(session.data_breakpoints)},
        timeout=timeout,
    )
    return {"ok": True, "breakpoints": (resp.get("body") or {}).get("breakpoints")}


def disassemble(
    *,
    memory_reference: str,
    instruction_count: int,
    offset: Optional[int] = None,
    instruction_offset: Optional[int] = None,
    resolve_symbols: Optional[bool] = None,
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    _require_capability(session, "supportsDisassembleRequest", "disassembly")
    args: dict[str, Any] = {
        "memoryReference": str(memory_reference),
        "instructionCount": int(instruction_count),
    }
    if offset is not None:
        args["offset"] = int(offset)
    if instruction_offset is not None:
        args["instructionOffset"] = int(instruction_offset)
    if resolve_symbols is not None:
        args["resolveSymbols"] = bool(resolve_symbols)
    resp = session.client.request("disassemble", args, timeout=timeout)
    return {"ok": True, "instructions": (resp.get("body") or {}).get("instructions") or []}


def read_memory(
    *,
    memory_reference: str,
    count: int,
    offset: Optional[int] = None,
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    _require_capability(session, "supportsReadMemoryRequest", "memory reads")
    args: dict[str, Any] = {
        "memoryReference": str(memory_reference),
        "count": int(count),
    }
    if offset is not None:
        args["offset"] = int(offset)
    resp = session.client.request("readMemory", args, timeout=timeout)
    body = resp.get("body") or {}
    return {
        "ok": True,
        "address": body.get("address"),
        "data": body.get("data"),
        "unreadableBytes": body.get("unreadableBytes"),
    }


def write_memory(
    *,
    memory_reference: str,
    data: str,
    offset: Optional[int] = None,
    allow_partial: Optional[bool] = None,
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    _require_capability(session, "supportsWriteMemoryRequest", "memory writes")
    args: dict[str, Any] = {
        "memoryReference": str(memory_reference),
        "data": str(data),
    }
    if offset is not None:
        args["offset"] = int(offset)
    if allow_partial is not None:
        args["allowPartial"] = bool(allow_partial)
    resp = session.client.request("writeMemory", args, timeout=timeout)
    body = resp.get("body") or {}
    return {
        "ok": True,
        "bytesWritten": body.get("bytesWritten"),
        "offset": body.get("offset"),
    }


def modules(
    *,
    start_module: Optional[int] = None,
    module_count: Optional[int] = None,
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    _require_capability(session, "supportsModulesRequest", "module introspection")
    args: dict[str, Any] = {}
    if start_module is not None:
        args["startModule"] = int(start_module)
    if module_count is not None:
        args["moduleCount"] = int(module_count)
    resp = session.client.request("modules", args, timeout=timeout)
    body = resp.get("body") or {}
    return {"ok": True, "modules": body.get("modules") or []}


def loaded_sources(*, timeout: float = 15.0, session_id: Optional[str] = None) -> dict[str, Any]:
    session = require_session(session_id)
    _require_capability(session, "supportsLoadedSourcesRequest", "loaded sources")
    resp = session.client.request("loadedSources", {}, timeout=timeout)
    return {"ok": True, "sources": (resp.get("body") or {}).get("sources") or []}


def custom_request(
    *,
    command: str,
    arguments: Optional[dict[str, Any]] = None,
    timeout: float = 15.0,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    session = require_session(session_id)
    cmd = str(command or "").strip()
    if not cmd:
        raise DapClientError("command is required for custom_request")
    resp = session.client.request(cmd, arguments or {}, timeout=timeout)
    return {"ok": True, "command": cmd, "body": resp.get("body")}
