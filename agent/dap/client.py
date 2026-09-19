"""Minimal DAP client over TCP or stdio (debugpy / js-debug / lldb / gdb).

debugpy: ``python -m debugpy.adapter --host 127.0.0.1 --port 0`` via
``DEBUGPY_ADAPTER_ENDPOINTS``.

js-debug: ``node dapDebugServer.js <port> 127.0.0.1`` (OMP-aligned TCP),
plus reverse ``runInTerminal`` / ``startDebugging`` handlers.

lldb-dap / gdb: stdio pipes (OMP ``connectMode: "stdio"``).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO, Callable, Optional

from agent.dap.protocol import DapProtocolError, encode_message, try_extract_message

logger = logging.getLogger("agent.dap.client")

EventHandler = Callable[[dict[str, Any]], None]
ReverseRequestHandler = Callable[[dict[str, Any]], dict[str, Any]]


class DapClientError(Exception):
    """Request failed or adapter unavailable."""


class DapClient:
    """Synchronous DAP client with a background reader thread."""

    def __init__(
        self,
        sock: Optional[socket.socket] = None,
        *,
        proc: Optional[subprocess.Popen] = None,
        endpoints_path: Optional[Path] = None,
        adapter_port: Optional[int] = None,
        owns_process: bool = True,
        stdin: Optional[BinaryIO] = None,
        stdout: Optional[BinaryIO] = None,
    ) -> None:
        if sock is None and (stdin is None or stdout is None):
            raise DapClientError("DapClient requires sock or stdio pipes")
        self._sock = sock
        self._stdin = stdin
        self._stdout = stdout
        self._proc = proc
        self._endpoints_path = endpoints_path
        self.adapter_port = adapter_port
        self._owns_process = owns_process
        self._buf = bytearray()
        self._seq = 0
        self._lock = threading.Lock()
        self._pending: dict[int, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._event_handlers: list[EventHandler] = []
        self._reverse_handlers: dict[str, ReverseRequestHandler] = {}
        self._cv = threading.Condition(self._lock)
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop, name="dap-reader", daemon=True)
        self._reader.start()

    def _write_bytes(self, data: bytes) -> None:
        if self._sock is not None:
            self._sock.sendall(data)
            return
        assert self._stdin is not None
        self._stdin.write(data)
        self._stdin.flush()

    def _read_chunk(self) -> bytes:
        if self._sock is not None:
            return self._sock.recv(65536)
        assert self._stdout is not None
        return self._stdout.read(65536)

    @classmethod
    def spawn_debugpy(
        cls,
        *,
        python: Optional[str] = None,
        host: str = "127.0.0.1",
        timeout: float = 15.0,
    ) -> "DapClient":
        py = python or sys.executable
        fd, ep_name = tempfile.mkstemp(prefix="hairball-dap-", suffix=".json")
        os.close(fd)
        endpoints = Path(ep_name)
        endpoints.write_text("", encoding="utf-8")
        env = os.environ.copy()
        env["DEBUGPY_ADAPTER_ENDPOINTS"] = str(endpoints)
        try:
            proc = subprocess.Popen(
                [py, "-m", "debugpy.adapter", "--host", host, "--port", "0"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
        except OSError as exc:
            endpoints.unlink(missing_ok=True)
            raise DapClientError(f"failed to spawn debugpy adapter: {exc}") from exc

        port: Optional[int] = None
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                endpoints.unlink(missing_ok=True)
                raise DapClientError(
                    f"debugpy adapter exited early with code {proc.returncode}"
                )
            try:
                raw = endpoints.read_text(encoding="utf-8").strip()
                if raw:
                    info = json.loads(raw)
                    port = int(info["client"]["port"])
                    break
            except Exception:
                pass
            time.sleep(0.05)
        if port is None:
            proc.kill()
            endpoints.unlink(missing_ok=True)
            raise DapClientError("timed out waiting for debugpy adapter port")

        try:
            sock = socket.create_connection((host, port), timeout=timeout)
        except OSError as exc:
            proc.kill()
            endpoints.unlink(missing_ok=True)
            raise DapClientError(f"failed to connect to debugpy adapter: {exc}") from exc

        return cls(sock, proc=proc, endpoints_path=endpoints, adapter_port=port)

    @classmethod
    def spawn_js_debug(
        cls,
        *,
        host: str = "127.0.0.1",
        timeout: float = 15.0,
        cwd: Optional[str] = None,
        server_path: Optional[Path] = None,
        node: Optional[str] = None,
    ) -> "DapClient":
        """Spawn vscode-js-debug TCP DAP server and connect (OMP-aligned)."""
        from agent.dap.adapters import find_node, resolve_js_debug_server

        node_bin = node or find_node()
        if not node_bin:
            raise DapClientError("node not found on PATH (required for js-debug)")
        server = server_path or resolve_js_debug_server(cwd)
        if server is None:
            raise DapClientError(
                "js-debug dapDebugServer.js not found — set "
                "HAIRBALL_JS_DEBUG_DAP_SERVER or install to ~/.local/opt/js-debug"
            )
        # Reserve an ephemeral port, then hand it to dapDebugServer.js.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
            reserved.bind((host, 0))
            port = int(reserved.getsockname()[1])
        try:
            proc = subprocess.Popen(
                [node_bin, str(server), str(port), host],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=cwd or None,
            )
        except OSError as exc:
            raise DapClientError(f"failed to spawn js-debug adapter: {exc}") from exc

        deadline = time.time() + timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            if proc.poll() is not None:
                raise DapClientError(
                    f"js-debug adapter exited early with code {proc.returncode}"
                )
            try:
                sock = socket.create_connection((host, port), timeout=0.5)
                return cls(sock, proc=proc, adapter_port=port)
            except OSError as exc:
                last_err = exc
                time.sleep(0.05)
        proc.kill()
        raise DapClientError(f"timed out connecting to js-debug adapter: {last_err}")

    @classmethod
    def connect_tcp(
        cls,
        host: str,
        port: int,
        *,
        timeout: float = 15.0,
    ) -> "DapClient":
        """Connect to an already-running TCP DAP server (js-debug child session)."""
        try:
            sock = socket.create_connection((host, int(port)), timeout=timeout)
        except OSError as exc:
            raise DapClientError(f"failed to connect to DAP {host}:{port}: {exc}") from exc
        return cls(sock, proc=None, adapter_port=int(port), owns_process=False)

    @classmethod
    def spawn_stdio(
        cls,
        command: list[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> "DapClient":
        """Spawn a stdio DAP adapter (lldb-dap, ``gdb -i dap``, …)."""
        if not command:
            raise DapClientError("stdio adapter command is empty")
        try:
            proc = subprocess.Popen(
                [str(x) for x in command],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=cwd or None,
                env=env,
                bufsize=0,
            )
        except OSError as exc:
            raise DapClientError(f"failed to spawn stdio adapter {command[0]!r}: {exc}") from exc
        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise DapClientError("stdio adapter missing pipes")
        return cls(sock=None, proc=proc, stdin=proc.stdin, stdout=proc.stdout)

    def on_event(self, handler: EventHandler) -> None:
        with self._lock:
            self._event_handlers.append(handler)

    def on_reverse_request(self, command: str, handler: ReverseRequestHandler) -> None:
        with self._lock:
            self._reverse_handlers[str(command)] = handler

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cv.notify_all()
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
        else:
            for stream in (self._stdin, self._stdout):
                if stream is None:
                    continue
                try:
                    stream.close()
                except Exception:
                    pass
        if self._owns_process and self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._endpoints_path is not None:
            try:
                self._endpoints_path.unlink(missing_ok=True)
            except Exception:
                pass

    def request(
        self,
        command: str,
        arguments: Optional[dict[str, Any]] = None,
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise DapClientError("DAP client is closed")
            self._seq += 1
            seq = self._seq
            msg: dict[str, Any] = {"seq": seq, "type": "request", "command": command}
            if arguments is not None:
                msg["arguments"] = arguments
            self._pending[seq] = {"done": False, "message": None}
        try:
            self._write_bytes(encode_message(msg))
        except (OSError, BrokenPipeError, ValueError) as exc:
            with self._lock:
                self._pending.pop(seq, None)
            raise DapClientError(f"failed to send DAP request {command}: {exc}") from exc

        deadline = time.time() + timeout
        with self._cv:
            while True:
                slot = self._pending.get(seq)
                if slot is None:
                    raise DapClientError(f"DAP request {command} lost")
                if slot["done"]:
                    resp = slot["message"]
                    del self._pending[seq]
                    break
                remaining = deadline - time.time()
                if remaining <= 0:
                    self._pending.pop(seq, None)
                    raise DapClientError(f"DAP request {command} timed out after {timeout:.0f}s")
                self._cv.wait(timeout=min(0.5, remaining))

        if not isinstance(resp, dict):
            raise DapClientError(f"DAP request {command} returned empty response")
        if not resp.get("success", False):
            raise DapClientError(
                f"DAP {command} failed: {resp.get('message') or resp.get('body') or 'unknown error'}"
            )
        return resp

    def wait_event(
        self,
        event: str,
        *,
        timeout: float = 30.0,
        predicate: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> dict[str, Any]:
        return self.wait_event_any(
            (event,),
            timeout=timeout,
            predicate=predicate,
        )

    def wait_event_any(
        self,
        events: tuple[str, ...] | list[str],
        *,
        timeout: float = 30.0,
        predicate: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> dict[str, Any]:
        """Wait until any named DAP event arrives (or timeout / client closed)."""
        wanted = {str(e) for e in events if str(e).strip()}
        if not wanted:
            raise DapClientError("wait_event_any requires at least one event name")
        deadline = time.time() + timeout
        with self._cv:
            while True:
                for i, ev in enumerate(self._events):
                    if ev.get("event") in wanted and (predicate is None or predicate(ev)):
                        return self._events.pop(i)
                remaining = deadline - time.time()
                if remaining <= 0:
                    names = ", ".join(sorted(wanted))
                    raise DapClientError(f"timed out waiting for DAP event in {{{names}}}")
                if self._closed:
                    names = ", ".join(sorted(wanted))
                    raise DapClientError(f"DAP client closed while waiting for {{{names}}}")
                self._cv.wait(timeout=min(0.5, remaining))

    def drain_events(self, event: Optional[str] = None) -> list[dict[str, Any]]:
        with self._lock:
            if event is None:
                out = list(self._events)
                self._events.clear()
                return out
            keep: list[dict[str, Any]] = []
            matched: list[dict[str, Any]] = []
            for ev in self._events:
                if ev.get("event") == event:
                    matched.append(ev)
                else:
                    keep.append(ev)
            self._events = keep
            return matched

    def _read_loop(self) -> None:
        try:
            while True:
                try:
                    chunk = self._read_chunk()
                except (OSError, ValueError):
                    break
                if not chunk:
                    break
                reverse_jobs: list[dict[str, Any]] = []
                with self._cv:
                    self._buf.extend(chunk)
                    while True:
                        try:
                            msg = try_extract_message(self._buf)
                        except DapProtocolError as exc:
                            logger.warning("DAP framing error: %s", exc)
                            self._closed = True
                            self._cv.notify_all()
                            return
                        if msg is None:
                            break
                        if msg.get("type") == "request":
                            # Handle outside the lock — child TCP + waits must not deadlock.
                            reverse_jobs.append(msg)
                        else:
                            self._dispatch_unlocked(msg)
                for rev in reverse_jobs:
                    self._handle_reverse_request(rev)
        finally:
            with self._cv:
                self._closed = True
                self._cv.notify_all()

    def _handle_reverse_request(self, msg: dict[str, Any]) -> None:
        try:
            seq = int(msg.get("seq") or 0)
        except (TypeError, ValueError):
            return
        command = str(msg.get("command") or "")
        args = msg.get("arguments") if isinstance(msg.get("arguments"), dict) else {}
        with self._lock:
            handler = self._reverse_handlers.get(command)
        body: dict[str, Any] = {}
        success = True
        message = None
        if handler is not None:
            try:
                result = handler(args)
                if isinstance(result, dict):
                    body = result
            except Exception as exc:  # noqa: BLE001
                success = False
                message = str(exc)
                logger.debug("DAP reverse request %s failed: %s", command, exc, exc_info=True)
        # Unregistered reverse requests: empty success (debugpy-safe stub).
        reply: dict[str, Any] = {
            "seq": 0,
            "type": "response",
            "request_seq": seq,
            "success": success,
            "command": command,
            "body": body,
        }
        if message:
            reply["message"] = message
        try:
            self._write_bytes(encode_message(reply))
        except (OSError, BrokenPipeError, ValueError):
            pass

    def _dispatch_unlocked(self, msg: dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == "response":
            req_seq = msg.get("request_seq")
            try:
                req_seq_i = int(req_seq)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return
            slot = self._pending.get(req_seq_i)
            if slot is not None:
                slot["message"] = msg
                slot["done"] = True
                self._cv.notify_all()
            return
        if mtype == "event":
            self._events.append(msg)
            handlers = list(self._event_handlers)
            self._cv.notify_all()
            for handler in handlers:
                try:
                    handler(msg)
                except Exception:
                    logger.debug("DAP event handler failed", exc_info=True)
            return
