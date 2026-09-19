"""Service-level orchestration for LSP clients.

The :class:`LSPService` is the bridge between the synchronous
file_operations layer and the async :class:`agent.lsp.client.LSPClient`.

Design choices:

- A **single asyncio event loop** runs in a background thread.  All
  client work happens on that loop.  Synchronous callers from
  ``tools/file_operations.py`` use :meth:`get_diagnostics_sync` to
  open + wait + drain in one blocking call.

- One client per ``(server_id, workspace_root)`` key.  Lazy spawn:
  the first request for a key spawns the client; subsequent requests
  re-use it.

- A **broken map** records ``(server_id, workspace_root) → failed_at``
  for spawn/initialize failures.  Entries expire after
  ``broken_backoff_seconds`` (default 180s, OMP-style negative cache)
  so scratch ``/tmp`` trees recover without a process restart.
  ``reload`` / ``restart`` clear entries immediately.

- A **delta baseline** map keeps "diagnostics-as-of-the-last-snapshot"
  per file.  ``snapshot_baseline()`` is called BEFORE a write; the
  next ``get_diagnostics_sync()`` returns only diagnostics that
  weren't in the baseline.  This is the lift from Claude Code's
  ``beforeFileEdited`` / ``getNewDiagnostics`` pattern, except wired
  to the local LSP layer instead of MCP IDE RPC.

The service is **off by default** — call :meth:`is_active` to check
whether it's actually doing anything.  When LSP is disabled in
config, when no git workspace can be detected, when all configured
servers are missing binaries and auto-install is off, ``is_active``
returns False and the file_operations layer falls through to the
in-process syntax check.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

from agent.lsp import eventlog
from agent.lsp.client import (
    DIAGNOSTICS_DOCUMENT_WAIT,
    LSPClient,
    file_uri,
)
from agent.lsp.servers import (
    ServerContext,
    find_server_for_file,
    language_id_for,
)
from agent.lsp.workspace import (
    clear_cache,
    resolve_workspace_for_file,
)

logger = logging.getLogger("agent.lsp.manager")

DEFAULT_IDLE_TIMEOUT = 600  # seconds; servers idle for >10min get reaped
DEFAULT_BROKEN_BACKOFF = 180.0  # seconds; OMP initFailures-style TTL


class _BackgroundLoop:
    """A daemon thread that owns one asyncio event loop.

    Provides :meth:`run` for synchronous callers — submits a coroutine
    to the loop and blocks until it finishes (or a timeout fires).
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_forever,
            name="hairball-lsp-loop",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run_forever(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass

    def run(self, coro, *, timeout: Optional[float] = None) -> Any:
        """Submit a coroutine to the loop and block until done.

        Returns the coroutine's result, or raises its exception.
        """
        from agent.async_utils import safe_schedule_threadsafe
        if self._loop is None:
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError("background loop not started")
        fut = safe_schedule_threadsafe(coro, self._loop)
        if fut is None:
            raise RuntimeError("background loop not running")
        try:
            return fut.result(timeout=timeout)
        except Exception:
            fut.cancel()
            raise

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._loop = None
        self._thread = None


class LSPService:
    """The process-wide LSP service.

    Created once via :meth:`create_from_config`; the
    :func:`agent.lsp.get_service` accessor manages the singleton.
    Most callers should use that accessor rather than constructing
    :class:`LSPService` directly.
    """

    # ------------------------------------------------------------------
    # construction + factory
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        enabled: bool,
        wait_mode: str,
        wait_timeout: float,
        install_strategy: str,
        binary_overrides: Optional[Dict[str, List[str]]] = None,
        env_overrides: Optional[Dict[str, Dict[str, str]]] = None,
        init_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        disabled_servers: Optional[List[str]] = None,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        broken_backoff_seconds: float = DEFAULT_BROKEN_BACKOFF,
    ) -> None:
        self._enabled = enabled
        self._wait_mode = wait_mode if wait_mode in {"document", "full"} else "document"
        self._wait_timeout = wait_timeout
        self._install_strategy = install_strategy
        self._binary_overrides = binary_overrides or {}
        self._env_overrides = env_overrides or {}
        self._init_overrides = init_overrides or {}
        self._disabled_servers = set(disabled_servers or [])
        self._idle_timeout = idle_timeout
        self._broken_backoff = max(0.0, float(broken_backoff_seconds))

        self._loop = _BackgroundLoop()
        if self._enabled:
            self._loop.start()

        # Per-(server_id, workspace_root) state
        self._clients: Dict[Tuple[str, str], LSPClient] = {}
        # key → monotonic timestamp when marked broken (TTL via _broken_backoff)
        self._broken: Dict[Tuple[str, str], float] = {}
        self._spawning: Dict[Tuple[str, str], asyncio.Future] = {}
        self._last_used: Dict[Tuple[str, str], float] = {}
        self._state_lock = threading.Lock()

        # Delta baseline: file path → snapshot of diagnostics taken
        # immediately before a write.  ``get_diagnostics_sync`` filters
        # out anything in the baseline so the agent only sees errors
        # introduced by the current edit.
        self._delta_baseline: Dict[str, List[Dict[str, Any]]] = {}

    @classmethod
    def create_from_config(cls) -> Optional["LSPService"]:
        """Build a service from ``hairball_cli.config`` settings.

        Returns ``None`` if the config can't be loaded.  The service
        itself returns ``is_active()`` False when LSP is disabled.
        """
        try:
            from hairball_cli.config import load_config
            cfg = load_config()
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP config load failed: %s", e)
            return None

        lsp_cfg = (cfg.get("lsp") or {}) if isinstance(cfg, dict) else {}
        if not isinstance(lsp_cfg, dict):
            lsp_cfg = {}

        enabled = bool(lsp_cfg.get("enabled", True))
        wait_mode = lsp_cfg.get("wait_mode", "document")
        wait_timeout = float(lsp_cfg.get("wait_timeout", DIAGNOSTICS_DOCUMENT_WAIT))
        install_strategy = lsp_cfg.get("install_strategy", "auto")
        broken_backoff = float(
            lsp_cfg.get("broken_backoff_seconds", DEFAULT_BROKEN_BACKOFF)
        )
        servers_cfg = lsp_cfg.get("servers") or {}
        disabled = []
        binary_overrides: Dict[str, List[str]] = {}
        env_overrides: Dict[str, Dict[str, str]] = {}
        init_overrides: Dict[str, Dict[str, Any]] = {}
        if isinstance(servers_cfg, dict):
            for name, sub in servers_cfg.items():
                if not isinstance(sub, dict):
                    continue
                if sub.get("disabled"):
                    disabled.append(name)
                cmd = sub.get("command")
                if isinstance(cmd, list) and cmd:
                    binary_overrides[name] = cmd
                env = sub.get("env")
                if isinstance(env, dict):
                    env_overrides[name] = {k: str(v) for k, v in env.items()}
                init = sub.get("initialization_options")
                if isinstance(init, dict):
                    init_overrides[name] = init

        return cls(
            enabled=enabled,
            wait_mode=wait_mode,
            wait_timeout=wait_timeout,
            install_strategy=install_strategy,
            binary_overrides=binary_overrides,
            env_overrides=env_overrides,
            init_overrides=init_overrides,
            disabled_servers=disabled,
            broken_backoff_seconds=broken_backoff,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return True iff this service should be consulted at all."""
        return self._enabled

    def _prune_broken_unlocked(self) -> None:
        """Drop expired broken entries. Caller must hold ``_state_lock``."""
        if self._broken_backoff <= 0:
            return
        now = time.monotonic()
        expired = [
            key
            for key, at in self._broken.items()
            if (now - at) >= self._broken_backoff
        ]
        for key in expired:
            self._broken.pop(key, None)

    def _is_broken(self, key: Tuple[str, str]) -> bool:
        """Return True if ``key`` is still inside the broken backoff window."""
        with self._state_lock:
            self._prune_broken_unlocked()
            return key in self._broken

    def _mark_broken_key(self, key: Tuple[str, str]) -> bool:
        """Record a broken key. Returns True if this is a new mark (not refresh)."""
        with self._state_lock:
            self._prune_broken_unlocked()
            already = key in self._broken
            self._broken[key] = time.monotonic()
            return not already

    def _resolve_task_cwd(self, task_id: Optional[str] = None) -> str:
        """Prefer the registered session/task cwd over process getcwd()."""
        from agent.coding_workspace import get_active_coding_task_id

        tid = (str(task_id).strip() if task_id else None) or get_active_coding_task_id()
        if tid:
            try:
                from tools.file_tools import _registered_task_cwd_override

                registered = _registered_task_cwd_override(tid)
                if registered:
                    return registered
            except Exception:  # noqa: BLE001
                pass
        return os.getcwd()

    def _shutdown_clients(self, clients: List["LSPClient"]) -> None:
        """Best-effort shutdown for clients already removed from ``_clients``."""
        if not clients:
            return

        async def _shutdown_all() -> None:
            await asyncio.gather(
                *(c.shutdown() for c in clients),
                return_exceptions=True,
            )

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        lsp_loop = getattr(self._loop, "_loop", None)
        if running is not None and lsp_loop is not None and running is lsp_loop:
            running.create_task(_shutdown_all())
            return
        try:
            self._loop.run(_shutdown_all(), timeout=2.0)
        except Exception:  # noqa: BLE001
            logger.debug("LSP client shutdown failed", exc_info=True)

    def _evict_clients_matching(
        self, predicate
    ) -> List[Tuple[str, str]]:
        """Remove clients/broken entries whose workspace matches ``predicate(ws)``.

        Returns the removed ``(server_id, workspace_root)`` keys.
        """
        to_shutdown: List[LSPClient] = []
        removed: List[Tuple[str, str]] = []
        with self._state_lock:
            for key, client in list(self._clients.items()):
                if not predicate(key[1]):
                    continue
                self._clients.pop(key, None)
                self._last_used.pop(key, None)
                self._broken.pop(key, None)
                to_shutdown.append(client)
                removed.append(key)
            for key in list(self._broken.keys()):
                if predicate(key[1]):
                    self._broken.pop(key, None)
                    if key not in removed:
                        removed.append(key)
        self._shutdown_clients(to_shutdown)
        return removed

    def _evict_nested_clients(self, ancestor_root: str) -> List[Tuple[str, str]]:
        """Drop LSP clients whose workspace is a strict child of ``ancestor_root``."""
        from agent.coding_workspace import is_ancestor_root, normalize_coding_root

        ancestor = normalize_coding_root(ancestor_root)

        def _nested(ws: str) -> bool:
            ws_n = normalize_coding_root(ws)
            return ws_n != ancestor and is_ancestor_root(ancestor, ws_n)

        removed = self._evict_clients_matching(_nested)
        if removed:
            logger.info(
                "evicted %d nested LSP client(s) under %s: %s",
                len(removed),
                ancestor,
                ", ".join(f"{s}@{w}" for s, w in removed),
            )
        return removed

    def _evict_retired_coding_roots(
        self, retired_roots: List[str]
    ) -> List[Tuple[str, str]]:
        """Shut down clients keyed to coding roots removed by umbrella collapse."""
        from agent.coding_workspace import is_ancestor_root, normalize_coding_root

        if not retired_roots:
            return []
        retired = [normalize_coding_root(r) for r in retired_roots]

        def _match(ws: str) -> bool:
            ws_n = normalize_coding_root(ws)
            for r in retired:
                if ws_n == r or is_ancestor_root(r, ws_n):
                    return True
            return False

        removed = self._evict_clients_matching(_match)
        if removed:
            logger.info(
                "evicted %d LSP client(s) for retired coding root(s) %s: %s",
                len(removed),
                retired,
                ", ".join(f"{s}@{w}" for s, w in removed),
            )
        return removed

    def _resolve_and_pin(
        self, file_path: str, *, task_id: Optional[str] = None
    ) -> Tuple[Optional[str], bool]:
        """Resolve workspace for ``file_path`` and register the coding root.

        Uses the deepest registered root that contains the file (multi-tree
        sessions). Otherwise discovers (file-git / cwd-git / fallback),
        registers the result, and umbrellas sibling scratch trees to a safe
        LCA. Lifting / umbrella shuts down nested and retired-root clients.
        """
        from agent.coding_workspace import (
            get_active_coding_task_id,
            register_coding_root,
            select_coding_root_for_file,
        )

        tid = (str(task_id).strip() if task_id else None) or get_active_coding_task_id()
        cwd = self._resolve_task_cwd(tid)
        hint = select_coding_root_for_file(file_path, tid)
        ws_root, gated = resolve_workspace_for_file(
            file_path, cwd=cwd, coding_workspace=hint
        )
        if not (ws_root and gated):
            return ws_root, gated
        primary, umbrella, lifted, retired = register_coding_root(tid, ws_root)
        if retired:
            self._evict_retired_coding_roots(retired)
        if lifted or umbrella:
            self._evict_nested_clients(primary)
        # After umbrella collapse, prefer the registered root for this file.
        final = select_coding_root_for_file(file_path, tid) or primary
        return final, True

    def enabled_for(self, file_path: str) -> bool:
        """Return True iff LSP should run for this specific file.

        Gates on workspace detection (git worktree, else project/file-parent
        fallback unless ``lsp.require_git_workspace``), on whether any
        registered server matches the extension, and on whether the
        (server_id, workspace_root) pair is in the broken map from a
        recent spawn failure (TTL; see ``broken_backoff_seconds``).

        Broken pairs skip LSP until the backoff expires, or until
        ``reload`` / ``restart`` clears them.
        """
        if not self._enabled:
            return False
        srv = find_server_for_file(file_path)
        if srv is None or srv.server_id in self._disabled_servers:
            return False
        ws_root, gated_in = self._resolve_and_pin(file_path)
        if not (ws_root and gated_in):
            return False
        # Broken-set short-circuit.  Use the per-server root if we can
        # compute one cheaply; otherwise fall back to the workspace
        # root as the broken key (which is what _get_or_spawn would
        # have used anyway when it failed).
        try:
            per_server_root = srv.resolve_root(file_path, ws_root) or ws_root
        except Exception:  # noqa: BLE001
            per_server_root = ws_root
        if self._is_broken((srv.server_id, per_server_root)):
            return False
        return True

    def disabled_reason(self, file_path: str) -> Optional[str]:
        """Human-readable why ``enabled_for`` is False, or None if enabled."""
        if not self._enabled:
            return "lsp.enabled is false"
        srv = find_server_for_file(file_path)
        if srv is None:
            suffix = Path(file_path).suffix or "(no extension)"
            return f"no language server registered for {suffix}"
        if srv.server_id in self._disabled_servers:
            return f"language server {srv.server_id} is disabled"
        ws_root, gated_in = self._resolve_and_pin(file_path)
        if not (ws_root and gated_in):
            try:
                from agent.lsp.workspace import _require_git_workspace

                if _require_git_workspace():
                    return (
                        "no git worktree for this file "
                        "(lsp.require_git_workspace=true)"
                    )
            except Exception:  # noqa: BLE001
                pass
            return (
                "no acceptable workspace root for this file "
                "(home and / are denied as LSP workspaces)"
            )
        try:
            per_server_root = srv.resolve_root(file_path, ws_root) or ws_root
        except Exception:  # noqa: BLE001
            per_server_root = ws_root
        if self._is_broken((srv.server_id, per_server_root)):
            remaining = ""
            with self._state_lock:
                at = self._broken.get((srv.server_id, per_server_root))
            if at is not None and self._broken_backoff > 0:
                left = max(0, int(self._broken_backoff - (time.monotonic() - at)))
                remaining = f" (retry in ~{left}s)"
            return (
                f"language server {srv.server_id} previously failed for "
                f"{per_server_root}{remaining}; try lsp action=reload "
                f"(or wait for broken_backoff_seconds)"
            )
        return None

    def snapshot_baseline(self, file_path: str) -> None:
        """Snapshot current diagnostics for ``file_path`` as the delta baseline.

        Called BEFORE a write so the next ``get_diagnostics_sync()``
        can filter out pre-existing errors.  Best-effort — failures
        are silently swallowed so a flaky server can't break a write.

        Outer timeouts (e.g. server hangs during initialize) mark the
        (server_id, workspace_root) pair as broken so subsequent edits
        skip it instantly instead of re-paying the timeout cost.
        """
        if not self.enabled_for(file_path):
            return
        try:
            diags = self._loop.run(self._snapshot_async(file_path), timeout=8.0)
            self._delta_baseline[os.path.abspath(file_path)] = diags or []
        except Exception as e:  # noqa: BLE001
            logger.debug("baseline snapshot failed for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            self._delta_baseline[os.path.abspath(file_path)] = []

    def get_diagnostics_sync(
        self,
        file_path: str,
        *,
        delta: bool = True,
        timeout: Optional[float] = None,
        line_shift: Optional[Callable[[int], Optional[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronously open ``file_path`` in the right server, wait for
        diagnostics, return them.

        If ``delta`` is True (default), the result is filtered against
        any baseline previously captured via :meth:`snapshot_baseline`.
        Diagnostics present in the baseline are removed so the caller
        only sees errors introduced by the current edit.

        When ``line_shift`` is provided, baseline diagnostics are
        remapped through it before the set-difference.  This handles
        the case where the edit deleted or inserted lines, causing
        pre-existing diagnostics below the edit point to surface at
        different line numbers in the post-edit snapshot — without
        the shift, they'd all look "introduced by this edit".  Pass
        a callable built by
        :func:`agent.lsp.range_shift.build_line_shift` (pre_text,
        post_text).  Omit when pre/post content isn't available;
        the unshifted comparison still catches diagnostics that
        didn't move.

        Returns an empty list when LSP is disabled, when no workspace
        can be detected, when no server matches, or when the server
        can't be spawned.  Never raises.
        """
        if not self.enabled_for(file_path):
            return []

        # Resolve server_id eagerly so we can emit structured logs even
        # when the request errors out below.
        srv = find_server_for_file(file_path)
        server_id = srv.server_id if srv else "?"

        try:
            t = timeout if timeout is not None else self._wait_timeout + 2.0
            diags = self._loop.run(self._open_and_wait_async(file_path), timeout=t) or []
        except asyncio.TimeoutError as e:
            eventlog.log_timeout(server_id, file_path)
            logger.debug("LSP diagnostics timeout for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            return []
        except Exception as e:  # noqa: BLE001
            eventlog.log_server_error(server_id, file_path, e)
            logger.debug("LSP diagnostics fetch failed for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            return []

        abs_path = os.path.abspath(file_path)
        if delta:
            baseline = self._delta_baseline.get(abs_path) or []
            if baseline:
                if line_shift is not None:
                    # Remap baseline diagnostics into post-edit
                    # coordinates so shifted-but-otherwise-identical
                    # entries hash equal under _diag_key.  Entries
                    # that mapped into a deleted region drop out
                    # silently — they no longer apply.
                    from agent.lsp.range_shift import shift_baseline
                    baseline = shift_baseline(baseline, line_shift)
                seen = {_diag_key(d) for d in baseline}
                diags = [d for d in diags if _diag_key(d) not in seen]
            # Roll baseline forward — next call returns deltas relative
            # to the just-emitted state, mirroring claude-code's
            # diagnosticTracking.
            try:
                fresh = self._loop.run(self._current_diags_async(file_path), timeout=2.0) or []
            except Exception:  # noqa: BLE001
                fresh = []
            if fresh:
                self._delta_baseline[abs_path] = fresh

        if diags:
            eventlog.log_diagnostics(server_id, file_path, len(diags))
        else:
            eventlog.log_clean(server_id, file_path)
        return diags

    def reconcile_restored_files_sync(
        self,
        restored: Dict[str, bool],
        *,
        timeout: float = 5.0,
    ) -> List[str]:
        """Reconcile LSP document state after a filesystem rollback.

        ``restored`` maps file paths to whether each path existed when the
        transaction began. Existing files are reopened from their restored
        bytes; newly-created files removed by rollback are closed in every
        live client. This is best-effort and never raises so LSP enrichment
        cannot turn a successful filesystem rollback into a failed one.
        """
        if not self._enabled or not restored:
            return []
        normalized = {
            os.path.abspath(path): bool(existed)
            for path, existed in restored.items()
        }
        try:
            errors = self._loop.run(
                self._reconcile_restored_files_async(normalized),
                timeout=max(0.1, float(timeout)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("LSP rollback reconciliation failed: %s", exc)
            return [f"{type(exc).__name__}: {exc}"]
        finally:
            for path in normalized:
                self._delta_baseline.pop(path, None)
        return errors or []

    def format_document_sync(
        self,
        file_path: str,
        content: str,
        *,
        timeout: Optional[float] = None,
    ) -> str:
        """Best-effort ``textDocument/formatting``; return formatted text or ``content``.

        Never raises. Used by write_file format-on-write (opt-in). Servers
        without ``documentFormattingProvider`` or that return no edits
        leave ``content`` unchanged.
        """
        if not self.enabled_for(file_path):
            return content
        t = float(timeout) if timeout is not None else 5.0
        try:
            return self._loop.run(
                self._format_document_async(file_path, content, timeout=t),
                timeout=t + 1.0,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP format_document_sync failed for %s: %s", file_path, e)
            return content

    async def _format_document_async(
        self,
        file_path: str,
        content: str,
        *,
        timeout: float,
    ) -> str:
        client = await self._get_or_spawn(file_path)
        if client is None:
            return content
        if not client.supports_document_formatting():
            return content
        from agent.lsp.edits import apply_text_edits_to_string
        from agent.lsp.format_options import resolve_format_options

        lang = language_id_for(file_path)
        options = resolve_format_options(file_path, content)
        try:
            edits = await client.request_document_formatting(
                file_path,
                language_id=lang,
                options=options,
                timeout=timeout,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP formatting request failed for %s: %s", file_path, e)
            return content
        if not edits or not isinstance(edits, list):
            return content
        try:
            return apply_text_edits_to_string(content, edits)
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP apply TextEdit failed for %s: %s", file_path, e)
            return content

    def _mark_broken_for_file(self, file_path: str, exc: BaseException) -> None:
        """Mark the (server_id, workspace_root) pair as broken so subsequent
        edits skip it instantly instead of re-paying timeout cost.

        Called when the outer ``_loop.run`` timeout cancels an in-flight
        spawn/initialize that the inner ``_get_or_spawn`` task was still
        holding open.  Without this, every subsequent write would re-enter
        the spawn path and re-pay the full ``snapshot_baseline``
        timeout (8s) until the binary is fixed.

        Also kills any orphan client process that survived the cancelled
        future, and emits a single eventlog WARNING so the user knows
        which server gave up.

        ``exc`` is whatever exception the outer wrapper caught — used
        only for logging, never re-raised.
        """
        srv = find_server_for_file(file_path)
        if srv is None:
            return
        ws_root, gated = self._resolve_and_pin(file_path)
        if not (ws_root and gated):
            return
        try:
            per_server_root = srv.resolve_root(file_path, ws_root) or ws_root
        except Exception:  # noqa: BLE001
            per_server_root = ws_root
        key = (srv.server_id, per_server_root)
        already_broken = self._is_broken(key)
        self._mark_broken_key(key)

        # Kill any client we managed to spawn before the timeout.  The
        # cancelled future never reached the broken-set add inside
        # ``_get_or_spawn`` so the client may still be hanging in
        # ``_clients`` with a half-initialized state.
        with self._state_lock:
            client = self._clients.pop(key, None)
        if client is not None:
            try:
                # Fire-and-forget shutdown — give it a second to cleanup,
                # but don't block.  We're already on a slow path.
                self._loop.run(client.shutdown(), timeout=1.0)
            except Exception:  # noqa: BLE001
                pass

        if not already_broken:
            eventlog.log_spawn_failed(srv.server_id, per_server_root, exc)

    def clear_broken(
        self, *, file_path: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Drop broken-set entries so the next request can re-spawn.

        When ``file_path`` is set, only clear keys for that file's
        resolved per-server / workspace roots.  When omitted (or ``*``),
        clear the entire broken set.

        Returns the keys that were removed.
        """
        if not file_path or str(file_path).strip() in {"", "*"}:
            with self._state_lock:
                removed = list(self._broken.keys())
                self._broken.clear()
            return removed

        srv = find_server_for_file(file_path)
        ws_root, gated = self._resolve_and_pin(file_path)
        targets: set = set()
        if srv and ws_root and gated:
            try:
                per = srv.resolve_root(file_path, ws_root) or ws_root
            except Exception:  # noqa: BLE001
                per = ws_root
            targets.add((srv.server_id, per))
            targets.add((srv.server_id, ws_root))

        removed: List[Tuple[str, str]] = []
        with self._state_lock:
            for key in list(self._broken.keys()):
                if key in targets:
                    self._broken.pop(key, None)
                    removed.append(key)
        return removed

    def shutdown(self) -> None:
        """Tear down all clients and stop the background loop."""
        if not self._enabled:
            return
        try:
            self._loop.run(self._shutdown_async(), timeout=10.0)
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP shutdown error: %s", e)
        self._loop.stop()
        clear_cache()

    # ------------------------------------------------------------------
    # async internals
    # ------------------------------------------------------------------

    async def _snapshot_async(self, file_path: str) -> List[Dict[str, Any]]:
        client = await self._get_or_spawn(file_path)
        if client is None:
            return []
        try:
            version = await client.open_file(file_path, language_id=language_id_for(file_path))
            await client.wait_for_diagnostics(file_path, version, mode=self._wait_mode)
        except Exception as e:  # noqa: BLE001
            logger.debug("snapshot open/wait failed: %s", e)
            return []
        self._last_used[(client.server_id, client.workspace_root)] = time.time()
        return list(client.diagnostics_for(file_path))

    async def _open_and_wait_async(self, file_path: str) -> List[Dict[str, Any]]:
        client = await self._get_or_spawn(file_path)
        if client is None:
            return []
        try:
            version = await client.open_file(file_path, language_id=language_id_for(file_path))
            await client.save_file(file_path)
            await client.wait_for_diagnostics(file_path, version, mode=self._wait_mode)
        except Exception as e:  # noqa: BLE001
            logger.debug("open/wait failed for %s: %s", file_path, e)
            return []
        self._last_used[(client.server_id, client.workspace_root)] = time.time()
        return list(client.diagnostics_for(file_path))

    async def _current_diags_async(self, file_path: str) -> List[Dict[str, Any]]:
        ws, gated = self._resolve_and_pin(file_path)
        srv = find_server_for_file(file_path)
        if not (ws and gated and srv):
            return []
        try:
            per = srv.resolve_root(file_path, ws) or ws
        except Exception:  # noqa: BLE001
            per = ws
        with self._state_lock:
            client = self._clients.get((srv.server_id, per)) or self._clients.get(
                (srv.server_id, ws)
            )
        if client is None:
            return []
        return list(client.diagnostics_for(file_path))

    async def _reconcile_restored_files_async(
        self,
        restored: Dict[str, bool],
    ) -> List[str]:
        """Update live LSP clients to match files restored on disk."""
        errors: List[str] = []

        for path, existed in restored.items():
            if existed:
                client = await self._get_or_spawn(path)
                if client is None:
                    continue
                try:
                    await client.open_file(path, language_id=language_id_for(path))
                    await client.save_file(path)
                    self._last_used[(client.server_id, client.workspace_root)] = time.time()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("LSP restore reopen failed for %s: %s", path, exc)
                    errors.append(f"{path}: {type(exc).__name__}: {exc}")
                continue

            with self._state_lock:
                clients = list(self._clients.values())
            for client in clients:
                try:
                    await client.close_file(path)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("LSP rollback close failed for %s: %s", path, exc)
                    errors.append(f"{path}: {type(exc).__name__}: {exc}")
        return errors

    async def _get_or_spawn(self, file_path: str) -> Optional[LSPClient]:
        srv = find_server_for_file(file_path)
        if srv is None:
            return None
        if srv.server_id in self._disabled_servers:
            eventlog.log_disabled(srv.server_id, file_path, "disabled in config")
            return None
        ws_root, gated = self._resolve_and_pin(file_path)
        if not (ws_root and gated):
            eventlog.log_no_project_root(srv.server_id, file_path)
            return None
        per_server_root = srv.resolve_root(file_path, ws_root)
        if per_server_root is None:
            eventlog.log_disabled(
                srv.server_id, file_path, "exclude marker hit (server gated off)"
            )
            return None  # exclude marker hit, server gated off

        key = (srv.server_id, per_server_root)
        if self._is_broken(key):
            return None
        # Ensure workspace exists before spawn (snapshot_baseline may run
        # before write_file creates the target path's parents).
        try:
            os.makedirs(per_server_root, exist_ok=True)
        except OSError as e:
            eventlog.log_spawn_failed(
                srv.server_id,
                per_server_root,
                e,
            )
            self._mark_broken_key(key)
            return None
        with self._state_lock:
            client = self._clients.get(key)
            if client is not None and client.is_running:
                eventlog.log_active(srv.server_id, per_server_root)
                return client
            spawning = self._spawning.get(key)
        if spawning is not None:
            try:
                return await spawning
            except Exception:  # noqa: BLE001
                return None

        # Begin spawn
        loop = asyncio.get_running_loop()
        spawn_future: asyncio.Future = loop.create_future()
        with self._state_lock:
            self._spawning[key] = spawn_future
        try:
            ctx = ServerContext(
                workspace_root=per_server_root,
                install_strategy=self._install_strategy,
                binary_overrides=self._binary_overrides,
                env_overrides=self._env_overrides,
                init_overrides=self._init_overrides,
            )
            spec = srv.build_spawn(per_server_root, ctx)
            if spec is None:
                # ``build_spawn`` returns None when the binary can't be
                # located (auto-install disabled, manual-only server,
                # or install attempt failed).  Surface this once via
                # the structured logger so the user can act on it.
                eventlog.log_server_unavailable(srv.server_id, srv.server_id)
                self._mark_broken_key(key)
                spawn_future.set_result(None)
                return None
            client = LSPClient(
                server_id=srv.server_id,
                workspace_root=spec.workspace_root,
                command=spec.command,
                env=spec.env,
                cwd=spec.cwd,
                initialization_options=spec.initialization_options,
                seed_diagnostics_on_first_push=spec.seed_diagnostics_on_first_push or srv.seed_first_push,
            )
            try:
                await client.start()
            except Exception as e:  # noqa: BLE001
                eventlog.log_spawn_failed(srv.server_id, per_server_root, e)
                self._mark_broken_key(key)
                spawn_future.set_result(None)
                return None
            with self._state_lock:
                self._clients[key] = client
            self._last_used[key] = time.time()
            eventlog.log_active(srv.server_id, per_server_root)
            spawn_future.set_result(client)
            return client
        finally:
            with self._state_lock:
                self._spawning.pop(key, None)

    async def _shutdown_async(self) -> None:
        with self._state_lock:
            clients = list(self._clients.values())
            self._clients.clear()
            self._broken.clear()
            self._last_used.clear()
        await asyncio.gather(
            *(c.shutdown() for c in clients),
            return_exceptions=True,
        )

    # ------------------------------------------------------------------
    # status / introspection (used by ``hairball lsp status``)
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the service for the CLI status command."""
        from agent.coding_workspace import (
            get_active_coding_task_id,
            get_coding_roots,
            get_coding_workspace,
        )

        with self._state_lock:
            self._prune_broken_unlocked()
            clients = [
                {
                    "server_id": k[0],
                    "workspace_root": k[1],
                    "state": c.state,
                    "running": c.is_running,
                }
                for k, c in self._clients.items()
            ]
            broken = list(self._broken.keys())
        tid = get_active_coding_task_id()
        return {
            "enabled": self._enabled,
            "wait_mode": self._wait_mode,
            "wait_timeout": self._wait_timeout,
            "install_strategy": self._install_strategy,
            "broken_backoff_seconds": self._broken_backoff,
            "clients": clients,
            "broken": broken,
            "disabled_servers": sorted(self._disabled_servers),
            "coding_workspace": get_coding_workspace(tid),
            "coding_roots": get_coding_roots(tid),
            "coding_task_id": tid,
        }

    def request_sync(
        self,
        action: str,
        *,
        file_path: Optional[str] = None,
        line: Optional[int] = None,
        character: Optional[int] = None,
        new_name: Optional[str] = None,
        query: Optional[str] = None,
        timeout: Optional[float] = None,
        only: Optional[List[str]] = None,
        code_action: Optional[Dict[str, Any]] = None,
        apply: Optional[bool] = None,
        payload: Any = None,
        use_payload: bool = False,
    ) -> Any:
        """Synchronously run a model-facing LSP action.

        ``line``/``character`` are **0-based** (LSP wire). Never raises —
        returns ``{"error": "..."}`` on failure so tool handlers stay simple.

        For ``action=request``, ``use_payload=True`` means ``payload`` is the
        exact JSON-RPC params (including JSON ``null``). When false, params
        are auto-built from ``file_path`` / position.
        """
        t = float(timeout) if timeout is not None else 20.0
        try:
            return self._loop.run(
                self._request_async(
                    action,
                    file_path=file_path,
                    line=line,
                    character=character,
                    new_name=new_name,
                    query=query,
                    timeout=t,
                    only=only,
                    code_action=code_action,
                    apply=apply,
                    payload=payload,
                    use_payload=use_payload,
                ),
                timeout=t + 2.0,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP request_sync %s failed: %s", action, e)
            if file_path:
                self._mark_broken_for_file(file_path, e)
            return {"error": f"{type(e).__name__}: {e}"}

    async def _request_async(
        self,
        action: str,
        *,
        file_path: Optional[str],
        line: Optional[int],
        character: Optional[int],
        new_name: Optional[str],
        query: Optional[str],
        timeout: float,
        only: Optional[List[str]] = None,
        code_action: Optional[Dict[str, Any]] = None,
        apply: Optional[bool] = None,
        payload: Any = None,
        use_payload: bool = False,
    ) -> Any:
        if action == "workspace_symbol":
            # Prefer any live client; otherwise spawn from cwd probe file if given
            client = await self._any_client_or_spawn(file_path)
            if client is None:
                return {"error": "no language server available for workspace symbol"}
            return await client.request_workspace_symbol(query or "", timeout=timeout)

        if action == "reload":
            return await self._reload_async(file_path=file_path, timeout=timeout)

        if action == "restart":
            return await self._restart_async(file_path=file_path)

        if action == "rename_file":
            if not file_path or not new_name:
                return {"error": "rename_file requires file and new_name"}
            return await self._rename_file_async(
                file_path, new_name, apply=True if apply is None else bool(apply), timeout=timeout
            )

        if action == "request":
            return await self._raw_request_async(
                file_path=file_path,
                line=line,
                character=character,
                query=query,
                timeout=timeout,
                payload=payload,
                use_payload=use_payload,
            )

        if not file_path:
            return {"error": f"{action} requires file"}
        if not self.enabled_for(file_path):
            reason = self.disabled_reason(file_path) or "gated off"
            return {"error": f"LSP not enabled for {file_path}: {reason}"}

        client = await self._get_or_spawn(file_path)
        if client is None:
            return {"error": f"failed to spawn language server for {file_path}"}

        lang = language_id_for(file_path)
        pos_actions = {
            "definition",
            "type_definition",
            "implementation",
            "references",
            "hover",
            "rename",
            "prepare_rename",
            "code_actions",
        }
        if action in pos_actions:
            if line is None or character is None:
                return {"error": f"{action} requires line and character (0-based)"}
            if action == "definition":
                return await client.request_definition(
                    file_path, line, character, language_id=lang, timeout=timeout
                )
            if action == "type_definition":
                return await client.request_type_definition(
                    file_path, line, character, language_id=lang, timeout=timeout
                )
            if action == "implementation":
                return await client.request_implementation(
                    file_path, line, character, language_id=lang, timeout=timeout
                )
            if action == "references":
                return await client.request_references(
                    file_path, line, character, language_id=lang, timeout=timeout
                )
            if action == "hover":
                return await client.request_hover(
                    file_path, line, character, language_id=lang, timeout=timeout
                )
            if action == "prepare_rename":
                return await client.request_prepare_rename(
                    file_path, line, character, language_id=lang, timeout=timeout
                )
            if action == "rename":
                if not new_name:
                    return {"error": "rename requires new_name"}
                return await client.request_rename(
                    file_path, line, character, new_name, language_id=lang, timeout=timeout
                )
            if action == "code_actions":
                return await client.request_code_actions(
                    file_path,
                    line,
                    character,
                    language_id=lang,
                    only=only,
                    timeout=timeout,
                )

        if action == "code_action_resolve":
            if not isinstance(code_action, dict):
                return {"error": "code_action_resolve requires code_action dict"}
            return await client.request_code_action_resolve(code_action, timeout=timeout)

        if action == "execute_command":
            if not query:
                return {"error": "execute_command requires query=command name"}
            args: Optional[List[Any]] = None
            if isinstance(code_action, dict) and isinstance(code_action.get("arguments"), list):
                args = code_action["arguments"]
            return await client.request_execute_command(query, args, timeout=timeout)

        if action == "document_symbols":
            return await client.request_document_symbols(
                file_path, language_id=lang, timeout=timeout
            )

        if action == "diagnostics":
            # Absolute (non-delta) snapshot for the model tool
            return await self._open_and_wait_async(file_path)

        return {"error": f"unknown action: {action}"}

    async def _reload_async(
        self, *, file_path: Optional[str], timeout: float
    ) -> Dict[str, Any]:
        notes: List[str] = []
        # Allow recovery from a prior spawn failure without restarting
        # the whole UI process (model tool + write_file race).
        cleared = self.clear_broken(file_path=file_path)
        if cleared:
            notes.append(
                "cleared broken LSP entries: "
                + ", ".join(f"{sid}@{root}" for sid, root in cleared)
            )

        if file_path and str(file_path).strip() not in {"", "*"}:
            if not self.enabled_for(file_path):
                reason = self.disabled_reason(file_path) or "gated off"
                return {"error": f"LSP not enabled for {file_path}: {reason}"}
            client = await self._get_or_spawn(file_path)
            if client is None:
                return {"error": f"failed to spawn language server for {file_path}"}
            notes.append(await client.reload_workspace(timeout=timeout))
            return {"success": True, "messages": notes}

        # Workspace reload: all live clients (broken already cleared above)
        with self._state_lock:
            clients = [c for c in self._clients.values() if c.is_running]
        if not clients:
            notes.append("No live language servers to reload")
            return {"success": True, "messages": notes}
        for client in clients:
            notes.append(await client.reload_workspace(timeout=timeout))
        return {"success": True, "messages": notes}

    async def _restart_async(
        self, *, file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Tear down clients and clear broken so the next call re-spawns.

        Model-facing stand-in for ``hairball lsp restart`` (CLI often
        missing from the chat tool shell PATH).
        """
        cleared = self.clear_broken(file_path=file_path)
        notes: List[str] = []
        if cleared:
            notes.append(
                "cleared broken: "
                + ", ".join(f"{sid}@{root}" for sid, root in cleared)
            )

        # Shut down matching live clients (or all when no file filter).
        with self._state_lock:
            if not file_path or str(file_path).strip() in {"", "*"}:
                victims = list(self._clients.items())
                self._clients.clear()
            else:
                srv = find_server_for_file(file_path)
                ws_root, gated = self._resolve_and_pin(file_path)
                victims = []
                if srv and ws_root and gated:
                    try:
                        per = srv.resolve_root(file_path, ws_root) or ws_root
                    except Exception:  # noqa: BLE001
                        per = ws_root
                    targets = {(srv.server_id, per), (srv.server_id, ws_root)}
                    for key in list(self._clients):
                        if key in targets:
                            victims.append((key, self._clients.pop(key)))

        for key, client in victims:
            try:
                await client.shutdown()
                notes.append(f"stopped {key[0]}@{key[1]}")
            except Exception as e:  # noqa: BLE001
                notes.append(f"stop failed {key[0]}@{key[1]}: {e}")

        if not notes:
            notes.append("No LSP clients or broken entries to reset")
        return {"success": True, "messages": notes}

    async def _rename_file_async(
        self,
        source: str,
        dest: str,
        *,
        apply: bool,
        timeout: float,
    ) -> Dict[str, Any]:
        from agent.lsp.edits import apply_workspace_edit, flatten_workspace_text_edits
        from agent.lsp.file_rename import MAX_RENAME_PAIRS, enumerate_rename_pairs

        source = os.path.abspath(source)
        dest = os.path.abspath(dest)
        if source == dest:
            return {"error": "source and destination paths are identical"}
        if not os.path.exists(source):
            return {"error": f"source path does not exist: {source}"}
        if os.path.exists(dest):
            return {"error": f"destination already exists: {dest}"}

        pairs, is_dir, exceeded = enumerate_rename_pairs(source, dest)
        if exceeded:
            return {
                "error": (
                    f"directory contains more than {MAX_RENAME_PAIRS} files; "
                    "rename in smaller batches to keep LSP edits accurate"
                )
            }
        if not pairs:
            return {"error": "no files to rename"}

        # Spawn from source (or first nested file under a directory).
        probe = source if os.path.isfile(source) else None
        if probe is None:
            for pair in pairs:
                from agent.lsp.client import uri_to_path
                probe = uri_to_path(pair["oldUri"])
                break
        if probe is None or not self.enabled_for(probe):
            # Still allow filesystem rename without LSP edits
            client = None
        else:
            client = await self._get_or_spawn(probe)

        workspace_edit = None
        server_note = None
        if client is not None:
            try:
                workspace_edit = await client.request_will_rename_files(pairs, timeout=timeout)
            except Exception as e:  # noqa: BLE001
                server_note = f"{client.server_id}: {e}"

        file_count_label = (
            f"{len(pairs)} file{'s' if len(pairs) != 1 else ''} under {source}"
            if is_dir
            else source
        )

        if not apply:
            preview: Dict[str, Any] = {
                "success": True,
                "preview": True,
                "applied": False,
                "source": source,
                "dest": dest,
                "pairs": len(pairs),
                "label": f"Rename preview: {file_count_label} → {dest}",
            }
            if isinstance(workspace_edit, dict):
                flat = flatten_workspace_text_edits(workspace_edit)
                preview["edit_files"] = [
                    {"path": p, "edits": len(edits)} for p, edits in sorted(flat.items())
                ]
            else:
                preview["edit_files"] = []
                preview["note"] = "No LSP edits would be applied"
            if server_note:
                preview["server_note"] = server_note
            return preview

        summary_lines: List[str] = []
        edit_summary = None
        if isinstance(workspace_edit, dict):
            edit_summary = apply_workspace_edit(workspace_edit, dry_run=False)
            for f in edit_summary.get("files") or []:
                if f.get("changed"):
                    summary_lines.append(
                        f"  applied {f.get('edits', 0)} edit(s) to {f.get('path')}"
                    )

        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        os.rename(source, dest)
        summary_lines.append(f"  Renamed {source} → {dest}")

        if client is not None:
            # Close old URIs then notify didRename
            from agent.lsp.client import uri_to_path
            for pair in pairs:
                old_path = uri_to_path(pair["oldUri"])
                try:
                    await client.close_file(old_path)
                except Exception:  # noqa: BLE001
                    pass
            try:
                await client.notify_did_rename_files(pairs)
            except Exception as e:  # noqa: BLE001
                server_note = (server_note + "; " if server_note else "") + str(e)

        return {
            "success": True,
            "preview": False,
            "applied": True,
            "source": source,
            "dest": dest,
            "pairs": len(pairs),
            "label": f"Renamed {file_count_label} → {dest}",
            "summary": summary_lines,
            "edit": edit_summary,
            "server_note": server_note,
        }

    async def _raw_request_async(
        self,
        *,
        file_path: Optional[str],
        line: Optional[int],
        character: Optional[int],
        query: Optional[str],
        timeout: float,
        payload: Any,
        use_payload: bool,
    ) -> Any:
        """Bare LSP JSON-RPC request (model ``action=request``)."""
        method = (query or "").strip()
        if not method:
            return {
                "error": (
                    "action=request requires query to specify the LSP method "
                    "name (e.g. 'textDocument/formatting')"
                )
            }

        open_path: Optional[str] = None
        lang = "plaintext"
        if file_path:
            if not self.enabled_for(file_path):
                reason = self.disabled_reason(file_path) or "gated off"
                return {"error": f"LSP not enabled for {file_path}: {reason}"}
            client = await self._get_or_spawn(file_path)
            if client is None:
                return {"error": f"failed to spawn language server for {file_path}"}
            open_path = file_path
            lang = language_id_for(file_path)
        else:
            client = await self._any_client_or_spawn(None)
            if client is None:
                return {
                    "error": (
                        "no language server available "
                        "(provide file to select a server)"
                    )
                }

        if use_payload:
            request_params: Any = payload
        elif open_path:
            uri = file_uri(os.path.abspath(open_path))
            if line is not None and character is not None:
                request_params = {
                    "textDocument": {"uri": uri},
                    "position": {"line": int(line), "character": int(character)},
                }
            else:
                request_params = {"textDocument": {"uri": uri}}
        else:
            request_params = {}

        try:
            result = await client.send_raw_request(
                method,
                request_params,
                path=open_path,
                language_id=lang,
                timeout=timeout,
            )
        except Exception as e:  # noqa: BLE001
            preview_raw = json.dumps(request_params, ensure_ascii=False, default=str)
            if len(preview_raw) > 400:
                preview_raw = preview_raw[:397] + "..."
            return {
                "error": (
                    f"LSP error from {client.server_id} on {method}: "
                    f"{type(e).__name__}: {e}\n  params: {preview_raw}"
                ),
                "server": client.server_id,
                "method": method,
            }

        return {
            "success": True,
            "server": client.server_id,
            "method": method,
            "result": result,
        }

    async def _any_client_or_spawn(self, file_path: Optional[str]) -> Optional[LSPClient]:
        with self._state_lock:
            for client in self._clients.values():
                if client.is_running:
                    return client
        if file_path and self.enabled_for(file_path):
            return await self._get_or_spawn(file_path)
        return None


def _diag_key(d: Dict[str, Any]) -> str:
    """Content equality key used for cross-edit delta filtering.

    Includes the diagnostic's position range — when used together
    with :func:`agent.lsp.range_shift.shift_baseline`, the baseline
    is line-shifted into post-edit coordinates BEFORE this key is
    computed, so identical-but-shifted diagnostics hash equal.  Two
    genuinely distinct diagnostics at different lines (e.g. the same
    error class introduced at a second site) hash differently and
    are surfaced as new.

    Mirrors :func:`agent.lsp.client._diagnostic_key`; intentionally
    identical so the two layers agree on diagnostic identity.
    """
    rng = d.get("range") or {}
    start = rng.get("start") or {}
    end = rng.get("end") or {}
    code = d.get("code")
    if code is not None and not isinstance(code, str):
        code = str(code)
    return "\x00".join(
        [
            str(d.get("severity") or 1),
            str(code or ""),
            str(d.get("source") or ""),
            str(d.get("message") or "").strip(),
            f"{start.get('line', 0)}:{start.get('character', 0)}-{end.get('line', 0)}:{end.get('character', 0)}",
        ]
    )


__all__ = ["LSPService"]
