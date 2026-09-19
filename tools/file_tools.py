#!/usr/bin/env python3
"""File Tools Module - LLM agent file manipulation tools."""

import errno
import difflib
import json
import logging
import os
import posixpath
import re
import sys
import tempfile
import threading
from pathlib import Path, PurePosixPath

from agent.file_safety import get_read_block_error
from agent.plan_execution import PlanContractError
from tools.binary_extensions import has_binary_extension
from tools.file_operations import (
    ReadResult,
    ShellFileOperations,
    normalize_read_pagination,
    normalize_search_pagination,
)
from tools import file_state
from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)


_EXPECTED_WRITE_ERRNOS = {errno.EACCES, errno.EPERM, errno.EROFS}


def _expand_tilde(path: str) -> str:
    """Expand ``~`` using the effective profile home when available.

    In-process file tools share the gateway process's HOME, which may differ
    from the profile-specific HOME that interactive CLI sessions use.  This
    mirrors ``hairball_constants.get_subprocess_home()`` so that ``~`` resolves
    consistently regardless of whether the tool runs interactively or inside a
    gateway-driven cron job (#48552).
    """
    if not path or "~" not in path:
        return path
    try:
        from hairball_constants import get_subprocess_home

        home = get_subprocess_home()
    except Exception:
        home = None
    if home and (path == "~" or path.startswith("~/")):
        return home if path == "~" else os.path.join(home, path[2:])
    return os.path.expanduser(path)


# ---------------------------------------------------------------------------
# Read-size guard: cap the character count returned to the model.
# We're model-agnostic so we can't count tokens; characters are a safe proxy.
# 100K chars ≈ 25–35K tokens across typical tokenisers.  Files larger than
# this in a single read are a context-window hazard — the model should use
# offset+limit to read the relevant section.
#
# Configurable via config.yaml:  file_read_max_chars: 200000
# ---------------------------------------------------------------------------
_DEFAULT_MAX_READ_CHARS = 100_000
_max_read_chars_cached: int | None = None


def _get_max_read_chars() -> int:
    """Return the configured max characters per file read.

    Reads ``file_read_max_chars`` from config.yaml on first call, caches
    the result for the lifetime of the process.  Falls back to the
    built-in default if the config is missing or invalid.
    """
    global _max_read_chars_cached
    if _max_read_chars_cached is not None:
        return _max_read_chars_cached
    try:
        from hairball_cli.config import load_config
        cfg = load_config()
        val = cfg.get("file_read_max_chars")
        if isinstance(val, (int, float)) and val > 0:
            _max_read_chars_cached = int(val)
            return _max_read_chars_cached
    except Exception:
        pass
    _max_read_chars_cached = _DEFAULT_MAX_READ_CHARS
    return _max_read_chars_cached


def _truncate_to_char_budget(content: str, max_chars: int) -> tuple[str, int, bool]:
    """Trim line-numbered ``read_file`` content to fit a char budget.

    Where Hairball previously hard-rejected an oversized read
    (forcing the model to guess a smaller ``limit`` and burn a round-trip
    returning nothing), this trims the content to the last *complete line*
    that fits within ``max_chars`` and reports how many lines were kept so
    the caller can offer a ``next_offset`` continuation.

    ``content`` is the gutter-rendered text (``LINE_NUM|CONTENT`` joined by
    ``\\n``). Individual lines are already clamped to ``get_max_line_length()``
    upstream, so a single line never blows the whole budget on its own; the
    overflow this handles is the *accumulation* of many lines under the
    line-count limit (logs, wide CSV rows, minified data).

    Returns ``(kept_text, lines_kept, truncated)``. When ``content`` already
    fits, returns it unchanged with ``truncated=False``. If not even the
    first line fits, that single line is clamped on a code-point boundary
    (Python ``str`` slicing never splits a code point) so the read never
    returns empty and the cursor can still advance.
    """
    if len(content) <= max_chars:
        return content, (content.count("\n") + 1 if content else 0), False

    lines = content.split("\n")
    kept: list[str] = []
    running = 0
    for line in lines:
        # +1 for the "\n" that rejoins this line to the previous one.
        addition = len(line) + (1 if kept else 0)
        if running + addition > max_chars:
            break
        kept.append(line)
        running += addition

    if not kept:
        # First line alone exceeds the budget. Clamp on a code-point
        # boundary rather than emitting nothing.
        kept.append(lines[0][:max_chars])

    return "\n".join(kept), len(kept), True


# If the total file size exceeds this AND the caller didn't specify a narrow
# range (limit <= 200), we include a hint encouraging targeted reads.
_LARGE_FILE_HINT_BYTES = 512_000  # 512 KB

# ---------------------------------------------------------------------------
# Device path blocklist — reading these hangs the process (infinite output
# or blocking on input).  Checked by path only (no I/O).
# ---------------------------------------------------------------------------
_BLOCKED_DEVICE_PATHS = frozenset({
    # Infinite output — never reach EOF
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    # Blocks waiting for input
    "/dev/stdin", "/dev/tty", "/dev/console",
    # Nonsensical to read
    "/dev/stdout", "/dev/stderr",
    # fd aliases
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})


def _resolve_path(filepath: str, task_id: str = "default") -> Path | PurePosixPath:
    """Resolve a path relative to TERMINAL_CWD (the worktree base directory)
    instead of the main repository root.
    """
    return _resolve_path_for_task(filepath, task_id)


# Sentinel ``TERMINAL_CWD`` values that mean "not configured", NOT a literal
# directory to resolve against. A stale config / .env commonly leaves the
# literal "." here; "auto"/"cwd" are setup-wizard placeholders. Treating any of
# these as a real relative base silently anchors edits to the agent PROCESS cwd
# (e.g. the main repo while a worktree session is active), routing writes to the
# wrong checkout. The gateway sanitizes the same set at import time
# (gateway/run.py); the file/terminal-tool layer must do likewise so CLI
# sessions get the same protection. See references/worktree-cwd-discipline.md.
_TERMINAL_CWD_SENTINELS = frozenset({"", ".", "./", "auto", "cwd"})
_CONTAINER_PATH_BACKENDS_FALLBACK = frozenset({"docker", "singularity", "modal", "daytona"})


def _terminal_env_type_for_task(task_id: str = "default") -> str:
    """Best-effort terminal backend type for path-resolution decisions."""
    try:
        from tools.terminal_tool import (
            _active_environments,
            _env_lock,
            _get_env_config,
            _resolve_container_task_id,
        )

        try:
            container_key = _resolve_container_task_id(task_id)
        except Exception:
            container_key = task_id
        with _env_lock:
            env = _active_environments.get(container_key) or _active_environments.get(task_id)
        if env is not None:
            name = env.__class__.__name__.lower()
            if "local" in name:
                return "local"
            if "ssh" in name:
                return "ssh"
            if "docker" in name:
                return "docker"
            if "singularity" in name:
                return "singularity"
            if "modal" in name:
                return "modal"
            if "daytona" in name:
                return "daytona"
        cfg = _get_env_config()
        return str(cfg.get("env_type") or os.getenv("TERMINAL_ENV") or "local").lower()
    except Exception:
        return str(os.getenv("TERMINAL_ENV") or "local").lower()


def _uses_container_paths(task_id: str = "default") -> bool:
    try:
        from tools.terminal_tool import _CONTAINER_BACKENDS
        container_backends = _CONTAINER_BACKENDS
    except Exception:
        container_backends = _CONTAINER_PATH_BACKENDS_FALLBACK
    return _terminal_env_type_for_task(task_id) in container_backends


def _normalize_without_host_deref(path: str | Path | PurePosixPath) -> PurePosixPath:
    """Normalize path syntax without following host symlinks.

    Container backends use paths that are meaningful inside the sandbox. Calling
    ``Path.resolve()`` on the host can dereference a host-side symlink such as
    ``/workspace`` and rewrite the path before Docker sees it.
    """
    return PurePosixPath(posixpath.normpath(str(path)))


def _sentinel_free_abs_cwd(raw: str | None) -> str | None:
    """Normalize a cwd candidate to an absolute, sentinel-free anchor.

    Returns the expanded path only when *raw* is non-empty, not a sentinel (see
    ``_TERMINAL_CWD_SENTINELS``), and absolute. A relative anchor is meaningless
    without knowing which cwd it is relative to — exactly the ambiguity that
    misroutes worktree edits — so relative/sentinel/empty values yield ``None``.
    """
    raw = str(raw or "").strip()
    if raw.lower() in _TERMINAL_CWD_SENTINELS:
        return None
    expanded = _expand_tilde(raw)
    if not os.path.isabs(expanded):
        return None
    return expanded


def _configured_terminal_cwd() -> str | None:
    """Return ``$TERMINAL_CWD`` only when it names a real directory anchor.

    Sentinel values (see ``_TERMINAL_CWD_SENTINELS``) and relative paths are
    rejected — a relative anchor is meaningless without knowing which cwd it is
    relative to, which is exactly the ambiguity that misroutes worktree edits.
    Only an absolute, sentinel-free value is honored.
    """
    return _sentinel_free_abs_cwd(os.environ.get("TERMINAL_CWD"))


def _registered_task_cwd_override(task_id: str = "default") -> str | None:
    """Return a registered cwd override for the raw task id, when available.

    ``terminal_tool`` intentionally collapses CWD-only task overrides to the
    shared ``"default"`` environment so TUI/dashboard/ACP sessions do not spin
    up isolated sandboxes just because they have different workspaces. The cwd
    value itself is still keyed by the raw session/task id, so file tools must
    read that raw override before falling back to the collapsed container key.
    """
    try:
        from tools.terminal_tool import resolve_task_overrides

        overrides = resolve_task_overrides(task_id)
    except Exception:
        return None

    return _sentinel_free_abs_cwd(overrides.get("cwd"))


def _live_cwd_if_owned(env, task_id: str) -> str | None:
    """The env's live cwd, but only when THIS session owns it.

    The terminal env is shared (collapsed to the ``"default"`` container), so its
    ``cwd`` tracks the LAST session that ran a command. With two worktree
    sessions open, trusting it blindly routes one session's edits into the other
    session's checkout (the wrong-worktree-patch bug). ``terminal_tool`` stamps
    ``env.cwd_owner`` with the session that last drove the env; return its cwd
    only when that owner matches the resolving session, else ``None`` so the
    caller falls through to this session's own registered cwd override. Unknown
    owner / ``default`` keys keep the prior behavior (single-session / CLI).
    """
    if env is None:
        return None
    live = getattr(env, "cwd", None)
    if not live:
        return None
    owner = str(getattr(env, "cwd_owner", "") or "")
    tid = str(task_id or "")
    if owner and tid and owner != "default" and tid != "default" and owner != tid:
        return None
    return live


def _get_live_tracking_cwd(task_id: str = "default") -> str | None:
    """Return the task's live terminal cwd for bookkeeping when available."""
    try:
        from tools.terminal_tool import _resolve_container_task_id
        container_key = _resolve_container_task_id(task_id)
    except Exception:
        container_key = task_id

    with _file_ops_lock:
        # A session-scoped entry is more specific than the collapsed shared
        # container.  Prefer it when both exist so one long-lived ``default``
        # environment cannot route another session's relative file paths into
        # the wrong workspace.
        cached = _file_ops_cache.get(task_id) or _file_ops_cache.get(container_key)
    if cached is not None:
        env = getattr(cached, "env", None)
        live_cwd = _live_cwd_if_owned(env, task_id)
        if live_cwd:
            _remember_last_known_cwd(container_key, live_cwd)
            return live_cwd
        # Legacy: a cache entry carrying its own cwd with no env to own it.
        if env is None and getattr(cached, "cwd", None):
            legacy_cwd = getattr(cached, "cwd", None)
            _remember_last_known_cwd(container_key, legacy_cwd)
            return legacy_cwd

    try:
        from tools.terminal_tool import _active_environments, _env_lock

        with _env_lock:
            env = _active_environments.get(task_id) or _active_environments.get(container_key)
        live_cwd = _live_cwd_if_owned(env, task_id)
        if live_cwd:
            _remember_last_known_cwd(container_key, live_cwd)
            return live_cwd
    except Exception:
        pass

    return None


def _authoritative_workspace_root(task_id: str = "default") -> str | None:
    """Best-effort absolute workspace root for divergence checks.

    Prefers the live terminal cwd (the directory the agent is actually working
    in). When no terminal command has run yet — so the live registry is empty —
    falls back to a registered task/session cwd override (TUI/Desktop/ACP
    sessions register a raw-keyed cwd before any tool runs), then to a
    sentinel-free absolute ``$TERMINAL_CWD``. This is what lets a worktree or
    Desktop session warn about (and resolve into) its workspace from the very
    first ``write_file``/``patch``, before any ``cd`` has populated the live cwd.

    Returns ``None`` only when there is genuinely no reliable anchor, in which
    case callers fall back to the process cwd.
    """
    live = _get_live_tracking_cwd(task_id)
    if live:
        return live
    # A session-specific registered override (TUI/Desktop/ACP workspace cwd)
    # is more authoritative than the shared last-known anchor: it is keyed by
    # the raw session id, so when two worktree sessions share the single
    # "default" terminal env, a NON-owning session must resolve against its OWN
    # registered worktree — never the other session's leftover cwd. (Checked
    # before _last_known_cwd, which is keyed by the shared container id.)
    registered = _registered_task_cwd_override(task_id)
    if registered:
        return registered
    # When the terminal env was cleaned up mid-conversation, the live cwd is
    # gone but the directory the agent navigated to is still recorded in the
    # durable _last_known_cwd registry. Prefer it over the config/process
    # fallback so a relative-path write resolved BEFORE the env is rebuilt
    # still lands in the user's directory (root cause of #26211: write happens
    # via _resolve_path_for_task -> here, which runs before _get_file_ops
    # rebuilds the env). Keyed by the resolved container id, same as the save.
    preserved = _last_known_cwd_for(task_id)
    if preserved:
        return preserved
    return _configured_terminal_cwd()


def _resolve_base_dir(
    task_id: str = "default",
    *,
    container_paths: bool | None = None,
) -> Path | PurePosixPath:
    """Return the ABSOLUTE base directory for resolving relative paths.

    Resolution order:
      1. The task's live terminal cwd (the directory the agent is actually
         working in — e.g. a git worktree). Authoritative when known.
      2. A registered task/session cwd override (TUI/Desktop/ACP sessions
         register a raw-keyed workspace cwd before any terminal command runs).
      3. A sentinel-free, absolute ``$TERMINAL_CWD`` (the worktree path set by
         ``cli.py``/``main.py`` for ``-w`` sessions). Used even before any
         terminal command has populated the live cwd registry.
      4. The process cwd.

    The returned base is ALWAYS absolute. This is the core invariant that
    prevents the worktree-cwd divergence bug: a relative or sentinel
    ``TERMINAL_CWD`` (commonly the literal ``"."`` from a stale config) is
    meaningless as a resolution anchor — left to ``Path.resolve()`` it silently
    resolves against whatever the agent PROCESS cwd happens to be (e.g. the main
    repo while the terminal is in a worktree), routing edits to the wrong
    checkout. We therefore reject sentinel/relative ``TERMINAL_CWD`` values
    outright (rather than anchoring them to the process cwd) and fall through to
    the process cwd only as a last resort, deterministically.
    """
    root = _authoritative_workspace_root(task_id)
    if container_paths is None:
        container_paths = _uses_container_paths(task_id)
    if root:
        base_text = _expand_tilde(root)
    else:
        base_text = os.getcwd()
    if container_paths:
        if not posixpath.isabs(base_text):
            base_text = posixpath.join(os.getcwd(), base_text)
        return _normalize_without_host_deref(base_text)
    # Git Bash ``pwd -P`` reports ``/c/Users/...``; translate before Path so
    # relative file-tool paths don't anchor under a nonexistent ``\\c\\Users``.
    from tools.environments.local import _msys_to_windows_path

    base_text = _msys_to_windows_path(base_text)
    if sys.platform == "win32":
        import ntpath

        if not ntpath.isabs(base_text):
            base_text = ntpath.join(os.getcwd(), base_text)
        return Path(ntpath.normpath(base_text))
    base = Path(base_text)
    if not base.is_absolute():
        # Last-resort anchoring: a live cwd should already be absolute, but if a
        # terminal backend ever reports a relative cwd, anchor it to the process
        # cwd once, here, so the result no longer depends on cwd at resolve().
        base = Path(os.getcwd()) / base
    return base.resolve()


def _resolve_path_for_task(filepath: str, task_id: str = "default") -> Path | PurePosixPath:
    """Resolve *filepath* against the task's absolute base directory.

    See :func:`_resolve_base_dir` for how the base is chosen. Absolute input
    paths are returned resolved-but-unanchored.

    On native Windows, Git Bash / MSYS drive paths (``/c/Users/...``) are
    translated to ``C:\\Users\\...`` before resolution so file tools don't
    treat them as relative ``\\c\\Users\\...`` under the process cwd.
    """
    container_paths = _uses_container_paths(task_id)
    if container_paths:
        expanded = _expand_tilde(filepath)
        if posixpath.isabs(expanded):
            return _normalize_without_host_deref(expanded)
        resolved = _resolve_base_dir(task_id, container_paths=True) / expanded
        return _normalize_without_host_deref(resolved)

    # Host paths only — never rewrite Linux paths inside a container/WSL env.
    from tools.environments.local import _msys_to_windows_path

    expanded = _expand_tilde(_msys_to_windows_path(filepath))
    if sys.platform == "win32":
        import ntpath

        if ntpath.isabs(expanded):
            return Path(ntpath.normpath(expanded))
        joined = ntpath.join(str(_resolve_base_dir(task_id, container_paths=False)), expanded)
        return Path(ntpath.normpath(joined))

    p = Path(expanded)
    if p.is_absolute():
        return p.resolve()
    resolved = _resolve_base_dir(task_id, container_paths=False) / p
    return resolved.resolve()


def _path_resolution_warning(filepath: str, resolved: Path, task_id: str = "default") -> str | None:
    """Warn when a relative path resolved OUTSIDE the task's workspace root.

    Surfaces the worktree-cwd divergence the moment it would matter: if the
    agent passes a relative path but it resolves under a directory that is not
    the workspace root (i.e. the edit is about to land in a different checkout
    than the one the agent is working in), return a message naming the absolute
    target. ``None`` when the path is absolute, the base is unknown, or the
    resolved path is correctly under the workspace root.

    The workspace root is the live terminal cwd when known, else a registered
    task/session cwd override, else a sentinel-free absolute ``$TERMINAL_CWD``
    — so a worktree or Desktop session whose terminal registry is still empty
    (no ``cd`` run yet) is warned on the very first write.
    """
    try:
        if Path(_expand_tilde(filepath)).is_absolute():
            return None
        workspace_root = _authoritative_workspace_root(task_id)
        if not workspace_root:
            return None  # No authoritative workspace root to compare against.
        if _uses_container_paths(task_id):
            root = _normalize_without_host_deref(Path(_expand_tilde(workspace_root)))
        else:
            root = Path(_expand_tilde(workspace_root)).resolve()
        # Is `resolved` inside `root`?
        try:
            resolved.relative_to(root)
            return None  # Inside the workspace — expected.
        except ValueError:
            return (
                f"Relative path {filepath!r} resolved to {str(resolved)!r}, which is "
                f"OUTSIDE the active workspace ({str(root)!r}). The edit will land in "
                f"a different directory than the terminal's cwd. If this is not "
                f"intended (e.g. a git-worktree session writing into the main "
                f"checkout), pass an absolute path under the workspace instead."
            )
    except Exception:
        return None


def _is_blocked_device_path(path: str) -> bool:
    """Return True for concrete device/fd paths that can hang reads."""
    normalized = os.path.normpath(_expand_tilde(path))
    if normalized in _BLOCKED_DEVICE_PATHS:
        return True
    # /proc/self/fd/0-2 and /proc/<pid>/fd/0-2 are Linux aliases for stdio
    if normalized.startswith("/proc/") and normalized.endswith(
        ("/fd/0", "/fd/1", "/fd/2")
    ):
        return True
    # /proc/*/environ, /proc/*/cmdline, /proc/*/maps (and the maps variants
    # smaps, smaps_rollup, numa_maps) can leak secrets, command-line args, and
    # memory layout (ASLR bypass) from the host process (issue #4427).
    # /proc/*/mem exposes raw process memory; block it as defense-in-depth even
    # though it requires address knowledge to exploit usefully.
    # /proc/*/auxv leaks AT_RANDOM (stack canary seed) plus AT_BASE/AT_PHDR
    # load addresses — an ASLR oracle on par with maps. /proc/*/pagemap exposes
    # virtual->physical translation. Both are blocked alongside the maps family.
    # endswith matches both /proc/<pid>/X and /proc/<pid>/task/<tid>/X.
    if normalized.startswith("/proc/") and normalized.endswith(
        (
            "/environ",
            "/cmdline",
            "/maps",
            "/smaps",
            "/smaps_rollup",
            "/numa_maps",
            "/mem",
            "/auxv",
            "/pagemap",
        )
    ):
        return True
    return False


def _is_blocked_device(filepath: str, base_dir: str | Path | None = None) -> bool:
    """Return True if the path would hang the process (infinite output or blocking input).

    Check the literal path first so aliases like /dev/stdin are caught before
    they resolve to terminal-specific paths. Then check each symlink hop before
    the final resolved path so aliases to devices cannot bypass the guard.
    """
    expanded = _expand_tilde(filepath)
    if base_dir is not None and not os.path.isabs(expanded):
        expanded = os.path.join(os.fspath(base_dir), expanded)
    normalized = os.path.normpath(expanded)
    if _is_blocked_device_path(normalized):
        return True

    seen: set[str] = set()
    current = normalized
    for _ in range(20):
        try:
            target = os.readlink(current)
        except OSError:
            break
        if not os.path.isabs(target):
            target = os.path.join(os.path.dirname(current), target)
        target = os.path.normpath(target)
        if _is_blocked_device_path(target):
            return True
        if target in seen:
            break
        seen.add(target)
        current = target

    try:
        resolved = os.path.normpath(os.path.realpath(normalized))
    except (OSError, ValueError):
        return False
    if _is_blocked_device_path(resolved):
        return True
    return False


def _search_result_read_block_error(path: str, task_id: str = "default") -> str | None:
    """Return the read-safety error for a search result path.

    Search backends may return paths relative to the task cwd, while
    ``get_read_block_error`` expects an already-resolved path when the task cwd
    can differ from the Python process cwd. Mirror ``read_file_tool``'s path
    resolution before applying the shared read guard.
    """
    try:
        resolved = _resolve_path_for_task(path, task_id)
    except (OSError, ValueError, RuntimeError):
        return get_read_block_error(path)
    return get_read_block_error(str(resolved))


def _filter_read_blocked_search_results(result, task_id: str = "default") -> int:
    """Remove credential/cache/env paths from a SearchResult in-place."""
    omitted = 0

    if hasattr(result, "matches") and result.matches:
        allowed_matches = []
        for match in result.matches:
            if _search_result_read_block_error(match.path, task_id):
                omitted += 1
                continue
            allowed_matches.append(match)
        result.matches = allowed_matches

    if hasattr(result, "files") and result.files:
        allowed_files = []
        for file_path in result.files:
            if _search_result_read_block_error(file_path, task_id):
                omitted += 1
                continue
            allowed_files.append(file_path)
        result.files = allowed_files

    if hasattr(result, "counts") and result.counts:
        allowed_counts = {}
        for file_path, count in result.counts.items():
            if _search_result_read_block_error(file_path, task_id):
                omitted += 1
                continue
            allowed_counts[file_path] = count
        result.counts = allowed_counts

    return omitted


# Paths that file tools should refuse to write to without going through the
# terminal tool's approval system.  These match prefixes after os.path.realpath.
_SENSITIVE_PATH_PREFIXES = (
    "/etc/", "/boot/", "/usr/lib/systemd/",
    "/private/etc/", "/private/var/",
)
_SENSITIVE_EXACT_PATHS = {"/var/run/docker.sock", "/run/docker.sock"}

_hairball_config_resolved: str | None = None
_hairball_config_resolved_loaded = False


def _get_hairball_config_resolved() -> str | None:
    """Return the resolved absolute path of the Hairball config file (cached)."""
    global _hairball_config_resolved, _hairball_config_resolved_loaded
    if _hairball_config_resolved_loaded:
        return _hairball_config_resolved
    _hairball_config_resolved_loaded = True
    try:
        from hairball_cli.config import get_config_path
        _hairball_config_resolved = str(get_config_path().resolve())
    except Exception:
        try:
            _hairball_config_resolved = str(Path(_expand_tilde("~/.hairball/config.yaml")).resolve())
        except Exception:
            _hairball_config_resolved = None
    return _hairball_config_resolved


def _check_sensitive_path(filepath: str, task_id: str = "default") -> str | None:
    """Return an error message if the path targets a sensitive system location."""
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        resolved = filepath
    normalized = os.path.normpath(_expand_tilde(filepath))
    _err = (
        f"Refusing to write to sensitive system path: {filepath}\n"
        "Use the terminal tool with sudo if you need to modify system files."
    )
    try:
        runtime_temp = os.path.realpath(tempfile.gettempdir())
        resolved_in_runtime_temp = (
            os.path.commonpath((resolved, runtime_temp)) == runtime_temp
        )
    except (OSError, TypeError, ValueError):
        resolved_in_runtime_temp = False
    for prefix in _SENSITIVE_PATH_PREFIXES:
        if resolved.startswith(prefix) or normalized.startswith(prefix):
            # On macOS tempfile.gettempdir() resolves below /private/var.
            # Permit only the current runtime temp subtree.  A symlink from
            # there into another /private/var or /etc location resolves outside
            # this root and remains blocked.
            if resolved_in_runtime_temp:
                continue
            return _err
    if resolved in _SENSITIVE_EXACT_PATHS or normalized in _SENSITIVE_EXACT_PATHS:
        return _err
    # Prevent agents from modifying the Hairball config file directly.
    # approvals.mode and other security settings live here; a malicious or
    # prompt-injected agent could silently disable exec approval by writing to
    # this file.
    hairball_config = _get_hairball_config_resolved()
    if hairball_config and (resolved == hairball_config or normalized == hairball_config):
        return (
            f"Refusing to write to Hairball config file: {filepath}\n"
            "Agent cannot modify security-sensitive configuration. "
            "Edit ~/.hairball/config.yaml directly or use 'hairball config' instead."
        )
    return None


def _get_container_mirror_prefix_for_task(task_id: str = "default") -> str | None:
    """Return the container-side Hairball mirror prefix for Docker file tools."""
    try:
        from tools.terminal_tool import (
            _active_environments,
            _env_lock,
            _get_env_config,
            _resolve_container_task_id,
        )

        container_key = _resolve_container_task_id(task_id)
    except Exception:
        return None

    try:
        with _env_lock:
            env = _active_environments.get(container_key) or _active_environments.get(task_id)

        if env is not None:
            if env.__class__.__name__ == "DockerEnvironment" and bool(
                getattr(env, "_persistent", False)
            ):
                return "/root/.hairball"
            return None

        config = _get_env_config()
    except Exception:
        return None

    if config.get("env_type") == "docker" and config.get("container_persistent", True):
        return "/root/.hairball"
    return None


def _check_cross_profile_path(filepath: str, task_id: str = "default") -> str | None:
    """Return a soft-guard warning when ``filepath`` lands in another Hairball
    profile's scoped area, a host-side sandbox-mirror of authoritative profile
    state, or the Docker container's sandbox mirror of Hairball state.

    Three detectors run in order:

    * cross-profile — writes that hit another profile's
      ``skills/plugins/cron/memories`` directory.
    * sandbox-mirror (#32049) — writes that hit the
      ``…/sandboxes/<backend>/<task>/home/.hairball/…`` mirror created by a
      non-local terminal backend (Docker, Daytona, etc.), where the host
      Hairball process never reads the mirror and the authoritative file is
      left untouched.
    * container-mirror (#32049 follow-up) — writes from inside a Docker
      container whose bind-mounted home strips the ``sandboxes/`` prefix, so
      the agent sees a plain ``/root/.hairball/…`` path.

    Returns ``None`` when the write is in-scope or outside Hairball scope.
    All detectors are soft guards — the agent can override any by
    passing ``cross_profile=True`` to its write tool after explicit user
    direction. Defense-in-depth, NOT a security boundary — the terminal
    tool runs as the same OS user and can write any of these paths
    directly. See ``agent/file_safety.classify_cross_profile_target``,
    ``classify_sandbox_mirror_target`` and ``classify_container_mirror_target``
    for the detection rules.
    """
    try:
        from agent.file_safety import (
            get_container_mirror_warning,
            get_cross_profile_warning,
            get_sandbox_mirror_warning,
        )
    except Exception:
        # Fail open on import error — the existing sensitive-path guard
        # plus the write_denied list still apply.
        return None

    # Resolve via the task's cwd so a relative ``skills/foo/SKILL.md``
    # in a session that cd'd into ``~/.hairball/profiles/other/`` is
    # classified against the right base.
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        resolved = filepath

    warning = get_cross_profile_warning(resolved)
    if warning is not None:
        return warning

    warning = get_sandbox_mirror_warning(resolved)
    if warning is not None:
        return warning

    return get_container_mirror_warning(
        resolved,
        mirror_prefix=_get_container_mirror_prefix_for_task(task_id),
    )


def _is_expected_write_exception(exc: Exception) -> bool:
    """Return True for expected write denials that should not hit error logs."""
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError) and exc.errno in _EXPECTED_WRITE_ERRNOS:
        return True
    return False


_file_ops_lock = threading.Lock()
_file_ops_cache: dict = {}
# Per-task last-known CWD — preserved across env re-creation so
# relative-path file writes land in the right directory after the
# terminal environment is cleaned up and rebuilt (root cause of #26211).
_last_known_cwd: dict = {}


def _remember_last_known_cwd(task_id: str, cwd: str | None) -> None:
    """Mirror a live terminal cwd into the durable ``_last_known_cwd`` registry.

    Belt-and-suspenders for #26211: the cleanup thread can pop BOTH
    ``_file_ops_cache`` and ``_active_environments`` before ``_get_file_ops``
    reaches its stale-cache detection branch, in which case the old cwd is
    never saved and the rebuilt env falls back to the config default — exactly
    the silent-misplacement bug. By recording the cwd on every successful live
    read (which happens on every relative-path file resolution while the env is
    alive), the durable anchor no longer depends on the cleanup-detection
    branch firing, so it survives recreation regardless of pop ordering.
    """
    if not cwd:
        return
    with _file_ops_lock:
        if _last_known_cwd.get(task_id) != cwd:
            _last_known_cwd[task_id] = cwd


def _last_known_cwd_for(task_id: str = "default") -> str | None:
    """Read the durable last-known cwd for *task_id*, container-key aware.

    The registry is keyed by the resolved container id (the same key used by
    the save sites in ``_get_file_ops`` / ``_get_live_tracking_cwd``), so look
    up the resolved key first and fall back to the raw task id.
    """
    try:
        from tools.terminal_tool import _resolve_container_task_id
        container_key = _resolve_container_task_id(task_id)
    except Exception:
        container_key = task_id
    with _file_ops_lock:
        return _last_known_cwd.get(container_key) or _last_known_cwd.get(task_id)

# Track files read per task for diagnostics and optimistic write concurrency.
# Per task_id we store:
#   "last_key":     the key of the most recent read/search call (or None)
#   "consecutive":  how many times that exact call has been repeated in a row
#   "read_history": set of (path, offset, limit) tuples for get_read_files_summary
#   "read_timestamps": dict mapping resolved_path → modification-time float
#                      recorded when the file was last read (or written) by
#                      this task.  Used by write_file and patch to detect
#                      external changes between the agent's read and write.
#                      Updated after successful writes so consecutive edits
#                      by the same task don't trigger false warnings.
#   "content_hashes":  dict mapping resolved_path → 4-hex content fingerprint
#                      of the full on-disk file at last read/write.  Used by
#                      the content-hash edit gate (agent.hashline) so writes
#                      fail closed when bytes changed, not just mtime.
_read_tracker_lock = threading.Lock()
_read_tracker: dict = {}

# Track consecutive patch failures per (task_id, resolved_path).  Used to
# escalate the hint when the model repeatedly fails to patch the same file
# (typical cause: stale view of file contents, ambiguous old_string, or
# the file was modified externally between the agent's read and patch
# attempt).  Reset on a successful patch to that path.
_patch_failure_lock = threading.Lock()
_patch_failure_tracker: dict = {}  # {task_id: {resolved_path: count}}


def _record_patch_failure(task_id: str, resolved_path: str) -> int:
    """Increment and return the consecutive-failure count for this path."""
    with _patch_failure_lock:
        task_failures = _patch_failure_tracker.setdefault(task_id, {})
        # Cap dict size per task to avoid unbounded growth in long sessions
        # where the agent fails on many distinct files.  64 distinct
        # failing files per task is generous; older entries get evicted.
        if len(task_failures) >= 64 and resolved_path not in task_failures:
            try:
                first_key = next(iter(task_failures))
                del task_failures[first_key]
            except StopIteration:
                pass
        task_failures[resolved_path] = task_failures.get(resolved_path, 0) + 1
        return task_failures[resolved_path]


def _reset_patch_failures(task_id: str, resolved_paths: list) -> None:
    """Clear consecutive-failure counts for the given paths."""
    if not resolved_paths:
        return
    with _patch_failure_lock:
        task_failures = _patch_failure_tracker.get(task_id)
        if not task_failures:
            return
        for rp in resolved_paths:
            task_failures.pop(rp, None)


def _v4a_failed_source_paths(
    error: str,
    source_paths: list[str],
    resolved_paths: dict[str, str | None],
) -> list[tuple[str, str]]:
    """Return the V4A source headers named by a failed patch result.

    V4A errors carry their exact header path (for example
    ``"src/a.py: hunk 1 ..."`` or ``"Failed to update src/a.py: ..."``).
    Restricting this to existing Update/Delete/Move sources avoids telling the
    model to ``read_file`` an Add destination that does not yet exist.
    """
    failed: list[tuple[str, str]] = []
    for source_path in dict.fromkeys(source_paths):
        if f"{source_path}:" not in error:
            continue
        resolved = resolved_paths.get(source_path)
        if resolved:
            failed.append((source_path, resolved))
    return failed


def _format_v4a_failure_hint(
    error: str,
    failed_sources: list[tuple[str, str]],
    failure_count: int,
) -> str:
    """Give the model an actionable V4A recovery path without retrying it."""
    if "rollback was incomplete" in error.lower():
        atomicity = (
            "V4A apply reported an incomplete rollback. Do not retry blindly: "
            "re-read every affected source and verify the working tree first."
        )
    else:
        atomicity = (
            "V4A patches are atomic here: this failed batch did not commit any "
            "of its proposed changes."
        )

    if failed_sources:
        headers = ", ".join(repr(header) for header, _resolved in failed_sources)
        recovery = (
            f" Re-read the failing source header(s) with read_file: {headers}; "
            "then regenerate only the intended V4A transaction from the current "
            "content. Keep unrelated lines unchanged; this is not a full-file rewrite."
        )
    else:
        recovery = (
            " Inspect the named header(s) in the error, re-read the current "
            "content, then regenerate only the intended V4A transaction. Keep "
            "unrelated lines unchanged; this is not a full-file rewrite."
        )

    if failure_count >= 3:
        recovery += (
            f" This is failure #{failure_count} for the same source. Stop retrying "
            "small variations of the old hunk; use longer unique context, or "
            "write_file only when a deliberate full-file replacement is intended."
        )
    return atomicity + recovery

# Per-task bounds for the containers inside each _read_tracker[task_id].
# A CLI session uses one stable task_id for its lifetime; without these
# caps, a 10k-read session would accumulate ~1.5MB of dict/set state that
# is never referenced again (only the most recent reads matter for
# diagnostics and external-edit warnings).  Hard caps bound the
# accretion to a few hundred KB regardless of session length.
_READ_HISTORY_CAP = 500       # set; used only by get_read_files_summary
_READ_TIMESTAMPS_CAP = 1000   # dict; external-edit detection for write/patch
_CONTENT_HASHES_CAP = 1000    # dict; content-hash edit gate


def _cap_read_tracker_data(task_data: dict) -> None:
    """Enforce size caps on the per-task read-tracker sub-containers.

    Must be called with ``_read_tracker_lock`` held.  Eviction policy:

      * ``read_history`` (set): pop arbitrary entries on overflow.  This
        is fine because the set only feeds diagnostic summaries; losing
        old entries just trims the summary's tail.
      * ``read_timestamps`` (dict): pop oldest by insertion order
        (Python 3.7+ dicts).  Evicted entries lose the external-edit mtime
        comparison; the content-hash gate still protects recognized writes.
    """
    rh = task_data.get("read_history")
    if rh is not None and len(rh) > _READ_HISTORY_CAP:
        excess = len(rh) - _READ_HISTORY_CAP
        for _ in range(excess):
            try:
                rh.pop()
            except KeyError:
                break

    ts = task_data.get("read_timestamps")
    if ts is not None and len(ts) > _READ_TIMESTAMPS_CAP:
        excess = len(ts) - _READ_TIMESTAMPS_CAP
        for _ in range(excess):
            try:
                ts.pop(next(iter(ts)))
            except (StopIteration, KeyError):
                break

    hashes = task_data.get("content_hashes")
    if hashes is not None and len(hashes) > _CONTENT_HASHES_CAP:
        excess = len(hashes) - _CONTENT_HASHES_CAP
        for _ in range(excess):
            try:
                hashes.pop(next(iter(hashes)))
            except (StopIteration, KeyError):
                break


def _looks_like_read_file_line_numbered_content(content: str) -> bool:
    """Return True for content dominated by read_file's ``LINE_NUM|CONTENT`` display.

    ``read_file`` intentionally returns line-numbered text to the model. If
    that display format is echoed into ``write_file``, config/source files are
    silently corrupted with prefixes like `` 1|``.  We reject writes where the
    non-empty lines are mostly consecutive read_file-style numbered lines, while
    allowing sparse literal pipe content such as a single ``1|value`` line.
    """
    if not isinstance(content, str):
        return False

    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    numbered: list[int] = []
    for line in lines:
        stripped = line.lstrip()
        prefix, sep, _rest = stripped.partition("|")
        if sep and prefix.isdigit():
            numbered.append(int(prefix))

    if len(numbered) < 2:
        return False
    if len(numbered) / len(lines) < 0.6:
        return False

    consecutive_pairs = sum(
        1 for prev, current in zip(numbered, numbered[1:])
        if current == prev + 1
    )
    return consecutive_pairs >= len(numbered) - 1


_LINE_ANCHOR_DISPLAY_RE = re.compile(
    r"^(?P<line>[1-9]\d*):[a-z]{1,4}(?::[a-z]{1,4})?→"
)


def _looks_like_line_anchor_display_content(content: str) -> bool:
    """Detect an anchored read display echoed into a whole-file write."""
    if not isinstance(content, str):
        return False
    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    numbered = [
        int(match.group("line"))
        for line in lines
        if (match := _LINE_ANCHOR_DISPLAY_RE.match(line)) is not None
    ]
    if len(numbered) < 2 or len(numbered) / len(lines) < 0.6:
        return False
    return all(current == previous + 1 for previous, current in zip(numbered, numbered[1:]))


def _is_internal_file_tool_content(content: str) -> bool:
    """Return True when content is file-tool display text, not intended file bytes."""
    return (
        _looks_like_read_file_line_numbered_content(content)
        or _looks_like_line_anchor_display_content(content)
    )


def _get_file_ops(task_id: str = "default") -> ShellFileOperations:
    """Get or create ShellFileOperations for a terminal environment.

    Respects the TERMINAL_ENV setting -- if the task_id doesn't have an
    environment yet, creates one using the configured backend (local, docker,
    modal, etc.) rather than always defaulting to local.

    Thread-safe: uses the same per-task creation locks as terminal_tool to
    prevent duplicate sandbox creation from concurrent tool calls.

    Note: subagent task_ids are collapsed to "default" via
    ``_resolve_container_task_id`` so delegate_task children share the
    parent's container and its cached file_ops. RL/benchmark task_ids with
    a registered env override keep their isolation.
    """
    from tools.terminal_tool import (
        _active_environments, _env_lock, _create_environment,
        _get_env_config, _last_activity, _start_cleanup_thread,
        _creation_locks,
        _creation_locks_lock,
        _resolve_container_task_id,
        _is_unusable_container_cwd,
        _CONTAINER_BACKENDS,
        _container_config_from_env_config,
        _local_config_from_env_config,
    )
    import time

    raw_task_id = task_id or "default"
    task_id = _resolve_container_task_id(raw_task_id)

    # Fast path: check cache -- but also verify the underlying environment
    # is still alive (it may have been killed by the cleanup thread).
    with _file_ops_lock:
        cached = _file_ops_cache.get(task_id)
    if cached is not None:
        with _env_lock:
            if task_id in _active_environments:
                _last_activity[task_id] = time.time()
                return cached
            else:
                # Environment was cleaned up -- preserve the old cwd before
                # invalidating the stale cache entry (fixes #26211: silent
                # file-creation failures in long-running conversations).
                old_cwd = getattr(cached, "cwd", None)
                if old_cwd:
                    with _file_ops_lock:
                        _last_known_cwd[task_id] = old_cwd
                with _file_ops_lock:
                    _file_ops_cache.pop(task_id, None)

    # Need to ensure the environment exists before building file_ops.
    # Acquire per-task lock so only one thread creates the sandbox.
    with _creation_locks_lock:
        if task_id not in _creation_locks:
            _creation_locks[task_id] = threading.Lock()
        task_lock = _creation_locks[task_id]

    with task_lock:
        # Double-check: another thread may have created it while we waited
        with _env_lock:
            if task_id in _active_environments:
                _last_activity[task_id] = time.time()
                terminal_env = _active_environments[task_id]
            else:
                terminal_env = None

        if terminal_env is None:
            from tools.terminal_tool import resolve_task_overrides

            config = _get_env_config()
            env_type = config["env_type"]
            overrides = resolve_task_overrides(raw_task_id)

            if env_type == "docker":
                image = overrides.get("docker_image") or config["docker_image"]
            elif env_type == "singularity":
                image = overrides.get("singularity_image") or config["singularity_image"]
            elif env_type == "modal":
                image = overrides.get("modal_image") or config["modal_image"]
            elif env_type == "daytona":
                image = overrides.get("daytona_image") or config["daytona_image"]
            else:
                image = ""

            cwd = overrides.get("cwd") or _last_known_cwd.get(task_id) or config["cwd"]
            # Re-apply the container cwd guard that _get_env_config() already
            # ran on config["cwd"] (see #50636).  A per-task cwd override
            # registered by the gateway/TUI/ACP for workspace tracking is a
            # raw host path (e.g. a Desktop session's /Users/<me>/workspace or
            # C:\\Users\\<me>). On a container backend that reaches
            # ``docker run -w <host-path>`` and the container starts in a
            # directory that doesn't exist inside the sandbox, so search_files
            # and friends silently return empty results (#54447).  Sanitize it
            # back to the already-validated config["cwd"] so the override can't
            # bypass the guard.  Valid in-container override paths (RL/benchmark
            # sandboxes that set cwd to /workspace, /root, etc.) are absolute
            # non-host paths and pass through untouched.
            if env_type in _CONTAINER_BACKENDS and _is_unusable_container_cwd(cwd):
                if cwd != config["cwd"]:
                    logger.info(
                        "Ignoring host/relative cwd override %r for %s backend "
                        "(won't exist in sandbox). Using %r instead.",
                        cwd, env_type, config["cwd"],
                    )
                cwd = config["cwd"]
            logger.info("Creating new %s environment for task %s...", env_type, task_id[:8])

            container_config = None
            if env_type in {"docker", "singularity", "modal", "daytona"}:
                container_config = _container_config_from_env_config(config)

            ssh_config = None
            if env_type == "ssh":
                ssh_config = {
                    "host": config.get("ssh_host", ""),
                    "user": config.get("ssh_user", ""),
                    "port": config.get("ssh_port", 22),
                    "key": config.get("ssh_key", ""),
                    "persistent": config.get("ssh_persistent", False),
                }

            local_config = (
                _local_config_from_env_config(config)
                if env_type == "local" else None
            )

            terminal_env = _create_environment(
                env_type=env_type,
                image=image,
                cwd=cwd,
                timeout=config["timeout"],
                ssh_config=ssh_config,
                container_config=container_config,
                local_config=local_config,
                task_id=task_id,
                host_cwd=config.get("host_cwd"),
            )

            with _env_lock:
                _active_environments[task_id] = terminal_env
                _last_activity[task_id] = time.time()

            _start_cleanup_thread()
            logger.info("%s environment ready for task %s", env_type, task_id[:8])

    # Build file_ops from the (guaranteed live) environment and cache it
    file_ops = ShellFileOperations(terminal_env)
    with _file_ops_lock:
        _file_ops_cache[task_id] = file_ops
    return file_ops


def clear_file_ops_cache(task_id: str = None):
    """Clear the file operations cache."""
    with _file_ops_lock:
        if task_id:
            _file_ops_cache.pop(task_id, None)
        else:
            _file_ops_cache.clear()


def read_file_tool(
    path: str,
    offset: int = 1,
    limit: int = 500,
    task_id: str = "default",
    line_anchors: bool = False,
) -> str:
    """Read a file with pagination and line numbers."""
    try:
        offset, limit = normalize_read_pagination(offset, limit)

        # ── Device path guard ─────────────────────────────────────────
        # Block paths that would hang the process (infinite output,
        # blocking on input).  Pure path check — no I/O.
        device_base = None if Path(path).expanduser().is_absolute() else _resolve_base_dir(task_id)
        if _is_blocked_device(path, base_dir=device_base):
            return json.dumps({
                "error": (
                    f"Cannot read '{path}': this is a device file that would "
                    "block or produce infinite output."
                ),
            })

        _resolved = _resolve_path_for_task(path, task_id)

        # ── Structured-document extraction ────────────────────────────
        # Try before the binary-extension guard so .docx/.xlsx can render as text.
        # Malformed documents fall through to the normal path/binary guard.
        from tools.read_extract import ExtractionError, extract_document_text, is_extractable_document

        if is_extractable_document(str(_resolved)):
            if line_anchors:
                return tool_error(
                    "line_anchors is only available for editable plain-text files; "
                    "structured documents are extracted views."
                )
            try:
                extracted_text = extract_document_text(str(_resolved))
            except ExtractionError:
                logger.debug("document extraction failed for %s", path, exc_info=True)
            else:
                file_ops = _get_file_ops(task_id)
                lines = extracted_text.splitlines()
                total_lines = len(lines)
                end_line = offset + limit - 1
                page_text = "\n".join(lines[offset - 1:end_line])
                result_dict = {
                    "content": file_ops._add_line_numbers(page_text, offset) if page_text else "",
                    "total_lines": total_lines,
                    "file_size": os.path.getsize(_resolved),
                    "truncated": total_lines > end_line,
                    "extracted_document": True,
                }
                if result_dict["truncated"]:
                    result_dict["hint"] = (
                        f"Use offset={end_line + 1} to continue reading "
                        f"(showing {offset}-{min(end_line, total_lines)} of {total_lines} lines)"
                    )
                content_len = len(result_dict["content"])
                max_chars = _get_max_read_chars()
                if content_len > max_chars:
                    # Graceful character-budget truncation:
                    # trim to the last complete line that fits and offer a
                    # next_offset rather than rejecting the whole extraction.
                    trimmed, lines_kept, _ = _truncate_to_char_budget(
                        result_dict["content"], max_chars
                    )
                    next_offset = offset + lines_kept
                    shown_end = offset + lines_kept - 1
                    result_dict["content"] = trimmed
                    result_dict["truncated"] = True
                    result_dict["truncated_by"] = "bytes"
                    result_dict["next_offset"] = next_offset
                    result_dict["hint"] = (
                        f"Output truncated at the {max_chars:,}-char read budget "
                        f"after {lines_kept} line(s) (showing lines {offset}-"
                        f"{shown_end} of {total_lines}). Use offset={next_offset} "
                        "to continue."
                    )
                    if len(trimmed.split("\n", 1)[0]) >= max_chars:
                        result_dict["hint"] += (
                            " Note: the first line alone exceeded the budget and "
                            "was clamped mid-line; its remainder is not "
                            "retrievable via offset."
                        )
                if result_dict["content"]:
                    result_dict["content"] = redact_sensitive_text(result_dict["content"], file_read=True)
                try:
                    from agent.hashline import format_file_tag, hash_file_on_disk

                    _chash = hash_file_on_disk(str(_resolved))
                    if _chash:
                        _record_content_hash(str(_resolved), task_id, _chash)
                        result_dict["content_hash"] = _chash
                        result_dict["file_tag"] = format_file_tag(path, _chash)
                except Exception:
                    logger.debug("content hash on extracted read failed", exc_info=True)
                return json.dumps(result_dict, ensure_ascii=False)

        # ── Binary file guard ─────────────────────────────────────────
        # Block binary files by extension (no I/O).
        if has_binary_extension(str(_resolved)):
            _ext = _resolved.suffix.lower()
            return json.dumps({
                "error": (
                    f"Cannot read binary file '{path}' ({_ext}). "
                    "Use vision_analyze for images, or terminal to inspect binary files."
                ),
            })

        # ── Hairball internal path guard ────────────────────────────────
        # Prevent prompt injection via catalog or hub metadata files,
        # and block credential stores under HAIRBALL_HOME.  Pass the
        # already-resolved path so a relative-path read against
        # TERMINAL_CWD == HAIRBALL_HOME (e.g. "auth.json") still hits the
        # denylist — get_read_block_error's own resolve() runs against
        # the Python process cwd, which can differ.
        block_error = get_read_block_error(str(_resolved))
        if block_error:
            return json.dumps({"error": block_error})

        # ── Dedup check ───────────────────────────────────────────────
        resolved_str = str(_resolved)
        with _read_tracker_lock:
            task_data = _read_tracker.setdefault(task_id, {
                "last_key": None, "consecutive": 0,
                "read_history": set(), "read_timestamps": {},
                "content_hashes": {},
            })
            if "read_timestamps" not in task_data:
                task_data["read_timestamps"] = {}
            if "content_hashes" not in task_data:
                task_data["content_hashes"] = {}

        # ── Perform the read ──────────────────────────────────────────
        file_ops = _get_file_ops(task_id)
        _full_text_for_hash: str | None = None
        if line_anchors:
            from agent.hashline import render_anchored_page

            raw_result = file_ops.read_file_raw(resolved_str)
            if raw_result.error or raw_result.is_binary or raw_result.is_image:
                return json.dumps(raw_result.to_dict(), ensure_ascii=False)
            _full_text_for_hash = raw_result.content
            page = render_anchored_page(
                raw_result.content, offset=offset, limit=limit
            )
            result = ReadResult(
                content=page.content,
                total_lines=page.total_lines,
                file_size=raw_result.file_size,
                truncated=page.truncated,
                hint=(
                    f"Use offset={page.next_offset} to continue reading "
                    f"(showing from line {offset} of {page.total_lines})."
                    if page.next_offset is not None
                    else None
                ),
            )
            result_dict = result.to_dict()
            result_dict["line_anchors"] = True
            result_dict["anchor_format"] = "LINE:LOCAL:CONTEXT→CONTENT"
            if page.next_offset is not None:
                result_dict["next_offset"] = page.next_offset
        else:
            result = file_ops.read_file(path, offset, limit)
            result_dict = result.to_dict()

        # ── Character-count guard ─────────────────────────────────────
        # We're model-agnostic so we can't count tokens; characters are
        # the best proxy we have.  If the read produced an unreasonable
        # amount of content, reject it and tell the model to narrow down.
        # Note: we check the formatted content (with line-number prefixes),
        # not the raw file size, because that's what actually enters context.
        # Check BEFORE redaction to avoid expensive regex on huge content.
        content_len = len(result.content or "")
        file_size = result_dict.get("file_size", 0)
        max_chars = _get_max_read_chars()
        if content_len > max_chars:
            # Graceful character-budget truncation.
            # Instead of rejecting the whole read — which forces the model to
            # guess a smaller `limit` and wastes a round-trip returning nothing
            # — trim to the last complete line that fits and offer a
            # `next_offset` so the model can paginate forward. This rescues the
            # "few but very long lines" case (logs, wide CSVs, minified data)
            # that sails past the line-count `limit` but blows the char budget.
            total_lines = result_dict.get("total_lines", "unknown")
            trimmed, lines_kept, _ = _truncate_to_char_budget(
                result.content or "", max_chars
            )
            next_offset = offset + lines_kept
            shown_end = offset + lines_kept - 1
            result.content = trimmed
            result_dict["content"] = trimmed
            result_dict["truncated"] = True
            result_dict["truncated_by"] = "bytes"
            result_dict["next_offset"] = next_offset
            result_dict["hint"] = (
                f"Output truncated at the {max_chars:,}-char read budget after "
                f"{lines_kept} line(s) (showing lines {offset}-{shown_end} of "
                f"{total_lines}). Use offset={next_offset} to continue."
            )
            if len(trimmed.split("\n", 1)[0]) >= max_chars:
                result_dict["hint"] += (
                    " Note: the first line alone exceeded the budget and was "
                    "clamped mid-line; its remainder is not retrievable via "
                    "offset."
                )
            content_len = len(trimmed)

        # ── Redact secrets (after guard check to skip oversized content) ──
        if result.content:
            result.content = redact_sensitive_text(result.content, file_read=True)
            result_dict["content"] = result.content

        # Large-file hint: if the file is big and the caller didn't ask
        # for a narrow window, nudge toward targeted reads.
        if (file_size and file_size > _LARGE_FILE_HINT_BYTES
                and limit > 200
                and result_dict.get("truncated")):
            result_dict.setdefault("_hint", (
                f"This file is large ({file_size:,} bytes). "
                "Consider reading only the section you need with offset and limit "
                "to keep context usage efficient."
            ))

        # ── Track for consecutive-loop detection ──────────────────────
        read_key = ("read", path, offset, limit, bool(line_anchors))
        with _read_tracker_lock:
            task_data["read_history"].add((path, offset, limit))
            if task_data["last_key"] == read_key:
                task_data["consecutive"] += 1
            else:
                task_data["last_key"] = read_key
                task_data["consecutive"] = 1
            # Store mtime for write/patch staleness detection when another
            # process or agent changes the file after this read.
            try:
                _mtime_now = os.path.getmtime(resolved_str)
                task_data.setdefault("read_timestamps", {})[resolved_str] = _mtime_now
            except OSError:
                pass  # Can't stat — skip tracking for this entry

            # Bound the per-task containers so a long CLI session doesn't
            # accumulate megabytes of dict/set state.  See _cap_read_tracker_data.
            _cap_read_tracker_data(task_data)

        # Content fingerprint of the *full* on-disk file (not just this page).
        # Writes/patches validate against this so external byte changes fail closed.
        try:
            from agent.hashline import compute_file_hash, format_file_tag, hash_file_on_disk

            _disk_chash = hash_file_on_disk(resolved_str)
            _chash = _disk_chash or (
                compute_file_hash(_full_text_for_hash)
                if _full_text_for_hash is not None
                else None
            )
            if _chash:
                _record_content_hash(
                    resolved_str,
                    task_id,
                    _chash,
                    full_text=(None if _disk_chash else _full_text_for_hash),
                )
                result_dict["content_hash"] = _chash
                result_dict["file_tag"] = format_file_tag(path, _chash)
        except Exception:
            logger.debug("content hash record on read failed", exc_info=True)

        # Cross-agent file-state registry (separate from per-task read
        # tracker above): records that THIS agent has read this path so
        # write/patch can detect sibling-subagent writes that happened
        # after our read.  Partial read when offset>1 or the read was
        # truncated (large file with more content than limit covered).
        # Outside the _read_tracker_lock so the registry's own locking
        # isn't nested under ours.
        try:
            _partial = (offset > 1) or bool(result_dict.get("truncated"))
            file_state.record_read(task_id, resolved_str, partial=_partial)
        except Exception:
            logger.debug("file_state.record_read failed", exc_info=True)

        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))




def notify_other_tool_call(task_id: str = "default"):
    """Reset consecutive read/search counter for a task.

    Called by the tool dispatcher (model_tools.py) whenever a tool OTHER
    than read_file / search_files is executed.  This ensures we only warn
    or block on *truly consecutive* repeated reads — if the agent does
    anything else in between (write, patch, terminal, etc.) the counter
    resets and the next read is treated as fresh.
    """
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data:
            task_data["last_key"] = None
            task_data["consecutive"] = 0


def _record_content_hash(
    resolved: str,
    task_id: str,
    content_hash: str | None,
    *,
    full_text: str | None = None,
) -> None:
    """Store a content fingerprint for *resolved* under *task_id*.

    Also records the on-disk full text into the per-task SnapshotStore
    so mismatch handling can distinguish session tags
    from fabricated / cross-session hashes.
    """
    if not content_hash:
        return
    with _read_tracker_lock:
        task_data = _read_tracker.setdefault(task_id, {
            "last_key": None, "consecutive": 0,
            "read_history": set(), "read_timestamps": {},
            "content_hashes": {},
        })
        task_data.setdefault("content_hashes", {})[resolved] = content_hash.upper()
        _cap_read_tracker_data(task_data)
    try:
        from agent.hashline import get_snapshot_store

        if full_text is None:
            with open(resolved, "rb") as fh:
                raw = fh.read()
            text = raw.decode("utf-8", errors="surrogatepass")
        else:
            text = full_text
        get_snapshot_store(task_id).record(resolved, text)
    except Exception:
        logger.debug("snapshot record failed for %s", resolved, exc_info=True)


def _get_stored_content_hash(resolved: str, task_id: str) -> str | None:
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if not task_data:
            return None
        return task_data.get("content_hashes", {}).get(resolved)


def _check_content_hash(
    filepath: str,
    task_id: str,
    *,
    expected_hash: str | None = None,
) -> tuple[str | None, dict | None]:
    """Content-hash gate before write/patch.

    Returns ``(warning_or_none, error_payload_or_none)``.
    When *error_payload* is set the caller must not write to disk.
    """
    try:
        from agent.hashline import (
            evaluate_content_hash,
            format_file_tag,
            get_snapshot_store,
            hash_file_on_disk,
            resolve_content_hash_mode,
        )
    except Exception:
        logger.debug("content hash gate unavailable", exc_info=True)
        return None, None

    mode = resolve_content_hash_mode()
    # An explicit expected_hash is a caller-selected compare-and-swap
    # precondition, not merely a request for the ambient warn/off policy.
    # Silently allowing a mismatch would make the parameter a fake API and
    # lets execute_code/direct callers overwrite a file they proved stale.
    # Calls without an explicit hash continue to obey the configured mode.
    explicit_precondition = bool((expected_hash or "").strip())
    if mode == "off" and not explicit_precondition:
        return None, None
    effective_mode = "require" if explicit_precondition else mode

    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        return None, None

    stored = _get_stored_content_hash(resolved, task_id)
    expected = (expected_hash or stored or "").strip().upper() or None
    actual = hash_file_on_disk(resolved)
    hash_recognized = None
    if expected:
        try:
            snap = get_snapshot_store(task_id).by_hash(resolved, expected)
            # A snapshot hit recognizes the hash. Tracker fallback covers
            # hashes recorded before snapshot wiring or when record failed.
            hash_recognized = snap is not None or (
                bool(stored) and stored.upper() == expected
            )
        except Exception:
            hash_recognized = None
    decision = evaluate_content_hash(
        path=filepath,
        expected_hash=expected,
        actual_hash=actual,
        mode=effective_mode,
        hash_recognized=hash_recognized,
    )
    if not decision.ok:
        return None, decision.as_error_payload(filepath)
    if decision.message:
        tag = format_file_tag(filepath, decision.actual) if decision.actual else filepath
        return f"{decision.message} Current tag: {tag}.", None
    return None, None


def _update_read_timestamp(filepath: str, task_id: str) -> None:
    """Record the file's current modification time after a successful write.

    Called after write_file and patch so that consecutive edits by the
    same task don't trigger false staleness warnings — each write
    refreshes the stored timestamp to match the file's new state.

    Also refreshes the content-hash fingerprint from the on-disk file after
    the write.
    """
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
        current_mtime = os.path.getmtime(resolved)
    except (OSError, ValueError):
        return
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data is not None:
            task_data.setdefault("read_timestamps", {})[resolved] = current_mtime
            _cap_read_tracker_data(task_data)
    try:
        from agent.hashline import hash_file_on_disk

        _record_content_hash(resolved, task_id, hash_file_on_disk(resolved))
    except Exception:
        logger.debug("content hash refresh after write failed", exc_info=True)


def _check_file_staleness(filepath: str, task_id: str) -> str | None:
    """Check whether a file was modified since the agent last read it.

    Returns a warning string if the file is stale (mtime changed since
    the last read_file call for this task), or None if the file is fresh
    or was never read.  Does not block — the write still proceeds.
    Content-byte drift is handled separately by ``_check_content_hash``.
    """
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        return None
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if not task_data:
            return None
        read_mtime = task_data.get("read_timestamps", {}).get(resolved)
    if read_mtime is None:
        return None  # File was never read — nothing to compare against
    try:
        current_mtime = os.path.getmtime(resolved)
    except OSError:
        return None  # Can't stat — file may have been deleted, let write handle it
    if current_mtime != read_mtime:
        return (
            f"Warning: {filepath} was modified since you last read it "
            "(external edit or concurrent agent). The content you read may be "
            "stale. Consider re-reading the file to verify before writing."
        )
    return None


def _mark_verification_stale(
    task_id: str,
    resolved_paths: list[str],
    session_id: str | None = None,
) -> None:
    """Best-effort note that successful edits made prior verification stale."""
    paths = [p for p in resolved_paths if p]
    if not paths:
        return
    try:
        from agent.coding_context import project_facts_for
        from agent.verification_evidence import mark_workspace_edited

        cwd = None
        for path in paths:
            try:
                candidate = str(Path(path).parent)
            except Exception:
                continue
            if project_facts_for(candidate):
                cwd = candidate
                break
        if cwd is None:
            cwd = _authoritative_workspace_root(task_id)
        if cwd is None:
            try:
                cwd = str(Path(paths[0]).parent)
            except Exception:
                cwd = None
        mark_workspace_edited(session_id=session_id or task_id, cwd=cwd, paths=paths)
    except Exception:
        logger.debug("verification stale marker failed", exc_info=True)


def write_file_tool(path: str, content: str, task_id: str = "default",
                    cross_profile: bool = False,
                    session_id: str | None = None,
                    expected_hash: str | None = None) -> str:
    """Write content to a file.

    ``cross_profile`` opts out of the soft cross-Hairball-profile guard. The
    guard fires only on writes that land in another profile's
    skills/plugins/cron/memories directory; everything else is unaffected.
    Pass ``True`` after explicit user direction — same shape as ``force``
    on the terminal tool.

    ``expected_hash`` optionally asserts the 4-hex content fingerprint from a
    prior ``read_file`` (``content_hash`` / ``file_tag``). When omitted, the
    gate uses the fingerprint stored from the last read for this task.
    """
    sensitive_err = _check_sensitive_path(path, task_id)
    if sensitive_err:
        return tool_error(sensitive_err)
    if not cross_profile:
        cross_warning = _check_cross_profile_path(path, task_id)
        if cross_warning:
            return tool_error(cross_warning)
    if _is_internal_file_tool_content(content):
        return tool_error(
            "Refusing to write internal read_file display text as file content. "
            "Strip read_file line-number prefixes or reconstruct the intended "
            "file contents before writing."
        )
    # Plan drafts are session artifacts, not workspace edits.  They use the
    # exact-path capability propagated by the agent executor so the terminal
    # can keep the entire working tree read-only without making Plan mode
    # unable to maintain its canonical document.
    try:
        from agent.plan_execution import (
            is_plan_artifact_target,
            write_plan_artifact,
        )

        if is_plan_artifact_target(path):
            resolved_plan = str(Path(path).expanduser().resolve(strict=False))
            with file_state.lock_path(resolved_plan):
                hash_warning, hash_error = _check_content_hash(
                    path, task_id, expected_hash=expected_hash
                )
                if hash_error:
                    return json.dumps(hash_error, ensure_ascii=False)
                result_dict = write_plan_artifact(resolved_plan, content)
                if hash_warning:
                    result_dict["_warning"] = hash_warning
                _update_read_timestamp(path, task_id)
                file_state.note_write(task_id, resolved_plan)
            return json.dumps(result_dict, ensure_ascii=False)
    except PlanContractError as exc:
        return json.dumps(exc.to_dict(), ensure_ascii=False)
    except ImportError:
        pass
    try:
        # Resolve once for the registry lock + stale check.  Failures here
        # fall back to the legacy path — write proceeds, per-task staleness
        # check below still runs.
        try:
            _resolved = str(_resolve_path_for_task(path, task_id))
        except Exception:
            _resolved = None

        from agent.coding_workspace import coding_task_context

        if _resolved is None:
            hash_warning, hash_error = _check_content_hash(
                path, task_id, expected_hash=expected_hash
            )
            if hash_error:
                return json.dumps(hash_error, ensure_ascii=False)
            stale_warning = _check_file_staleness(path, task_id)
            file_ops = _get_file_ops(task_id)
            with coding_task_context(task_id):
                result = file_ops.write_file(path, content)
            result_dict = result.to_dict()
            effective_warning = hash_warning or stale_warning
            if effective_warning:
                result_dict["_warning"] = effective_warning
            if not result_dict.get("error"):
                _mark_verification_stale(task_id, [path], session_id=session_id)
            _update_read_timestamp(path, task_id)
            return json.dumps(result_dict, ensure_ascii=False)

        # Serialize the read→modify→write region per-path so concurrent
        # subagents can't interleave on the same file.  Different paths
        # remain fully parallel.
        with file_state.lock_path(_resolved):
            hash_warning, hash_error = _check_content_hash(
                path, task_id, expected_hash=expected_hash
            )
            if hash_error:
                return json.dumps(hash_error, ensure_ascii=False)
            # Cross-agent staleness wins over per-task warning when both
            # fire — its message names the sibling subagent.
            cross_warning = file_state.check_stale(task_id, _resolved)
            stale_warning = _check_file_staleness(path, task_id)
            # Workspace-divergence warning: relative path resolving outside the
            # terminal's cwd (the worktree-cwd bug). Lowest priority of the three.
            cwd_warning = _path_resolution_warning(path, Path(_resolved), task_id)
            file_ops = _get_file_ops(task_id)
            with coding_task_context(task_id):
                result = file_ops.write_file(_resolved, content)
            result_dict = result.to_dict()
            sibling_warning = (
                cross_warning
                if cross_warning and "sibling subagent" in cross_warning
                else None
            )
            if sibling_warning and hash_warning:
                effective_warning = f"{sibling_warning} | {hash_warning}"
            else:
                effective_warning = (
                    sibling_warning or hash_warning or cross_warning
                    or stale_warning or cwd_warning
                )
            if effective_warning:
                result_dict["_warning"] = effective_warning
            # Always report the ABSOLUTE path actually written, so a wrong-cwd
            # mismatch is visible in the response instead of silently routing
            # the edit to the wrong checkout.
            result_dict["resolved_path"] = _resolved
            if not result_dict.get("error"):
                result_dict["files_modified"] = [_resolved]
                _mark_verification_stale(task_id, [_resolved], session_id=session_id)
            # Refresh stamps after the successful write so consecutive
            # writes by this task don't trigger false staleness warnings.
            _update_read_timestamp(path, task_id)
            if not result_dict.get("error"):
                file_state.note_write(task_id, _resolved)
        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        if _is_expected_write_exception(e):
            logger.debug("write_file expected denial: %s: %s", type(e).__name__, e)
        else:
            logger.error("write_file error: %s: %s", type(e).__name__, e, exc_info=True)
        return tool_error(str(e))


def patch_tool(mode: str = "replace", path: str = None, old_string: str = None,
               new_string: str = None, replace_all: bool = False, patch: str = None,
               task_id: str = "default", cross_profile: bool = False,
               session_id: str | None = None,
               expected_hash: str | None = None,
               expected_hashes: dict[str, str] | None = None,
               edits: list[dict] | None = None) -> str:
    """Patch files by unique text, V4A transaction, or stable line anchors.

    ``cross_profile`` opts out of the soft cross-Hairball-profile guard for
    targets under another profile's skills/plugins/cron/memories
    directory. Same shape as ``write_file``'s flag.
    """
    # Check sensitive paths for both replace (explicit path) and V4A patch (extract paths)
    _paths_to_check = []
    # Only existing source files can have a read-time content hash.  Add-file
    # destinations and Move destinations deliberately do not accept one: a
    # hash for a path that is not read before the transaction would be a fake
    # compare-and-swap precondition.
    _hashable_v4a_paths: list[str] = []
    if path:
        _paths_to_check.append(path)
    if mode == "patch" and patch:
        import re as _re
        from tools.path_security import has_traversal_component
        def _reject_v4a_traversal(v4a_path: str) -> str | None:
            # V4A path headers come from patch CONTENT, not the explicit
            # ``path=`` arg — so they're more attacker-influenceable (skill
            # content, web extract, prompt injection). Reject ``..`` traversal
            # in V4A headers: a legitimate multi-file patch from a single cwd
            # can always emit absolute paths or paths relative to the agent's
            # cwd without ``..``. The explicit ``path=`` arg is unchanged
            # because the agent uses relative ``..`` paths legitimately
            # (e.g. ``patch path="../other_module/x.py"`` from a worktree).
            if has_traversal_component(v4a_path):
                return tool_error(
                    f"V4A patch header contains '..' traversal: {v4a_path!r}. "
                    "Use the agent's cwd-relative path (no '..') or an absolute "
                    "path in '*** Update File:' / '*** Add File:' / "
                    "'*** Delete File:' / '*** Move File:' headers."
                )
            return None

        # ``\s*`` (not ``\s+``) after ``***`` matches patch_parser leniency:
        # it accepts ``***Update File:`` with no space after the asterisks
        # (patch_parser.py uses ``\*\*\*\s*Update\s+File:``). Requiring a space
        # here let a no-space header parse + apply while skipping this check.
        for _m in _re.finditer(r'^\*\*\*\s*(?:Update|Add|Delete)\s+File:\s*(.+)$', patch, _re.MULTILINE):
            v4a_path = _m.group(1).strip()
            _err = _reject_v4a_traversal(v4a_path)
            if _err:
                return _err
            _paths_to_check.append(v4a_path)
            _header_kind = _re.match(r"^\*\*\*\s*(Update|Add|Delete)\b", _m.group(0), _re.IGNORECASE)
            if _header_kind and _header_kind.group(1).lower() in {"update", "delete"}:
                _hashable_v4a_paths.append(v4a_path)
        # ``*** Move File: src -> dst`` is a valid V4A op (patch_parser.py:114)
        # but was never extracted, so a Move targeting /etc/crontab skipped the
        # sensitive-path pre-check. Check BOTH endpoints, and run them through
        # the same ``..`` traversal rejection as the other headers.
        for _m in _re.finditer(r'^\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+)$', patch, _re.MULTILINE):
            for _index, v4a_path in enumerate((_m.group(1).strip(), _m.group(2).strip())):
                _err = _reject_v4a_traversal(v4a_path)
                if _err:
                    return _err
                _paths_to_check.append(v4a_path)
                if _index == 0:
                    _hashable_v4a_paths.append(v4a_path)

    # Bind every existing section to the content hash observed by the agent.
    # Hairball keeps V4A as its multi-file edit syntax and accepts the same
    # contract as an explicit per-header map. Exact header matching is
    # intentional: accepting aliases such as ``./a.py`` for ``a.py`` would
    # make it unclear which V4A target was protected.
    _v4a_expected_hashes: dict[str, str] = {}
    if expected_hashes is not None:
        if mode != "patch":
            return tool_error("expected_hashes is only valid for patch mode='patch'. Use expected_hash for replace mode.")
        if not isinstance(expected_hashes, dict):
            return tool_error("expected_hashes must be an object mapping each existing V4A header path to its 4-hex content_hash.")
        _hashable_set = set(_hashable_v4a_paths)
        for _map_path, _map_hash in expected_hashes.items():
            if not isinstance(_map_path, str) or _map_path not in _hashable_set:
                return tool_error(
                    "expected_hashes contains a path that is not an existing V4A "
                    f"Update/Delete/Move source header: {_map_path!r}."
                )
            if not isinstance(_map_hash, str) or not re.fullmatch(r"[0-9A-Fa-f]{4}", _map_hash.strip()):
                return tool_error(
                    f"expected_hashes[{_map_path!r}] must be the 4-hex content_hash returned by read_file."
                )
            _v4a_expected_hashes[_map_path] = _map_hash.strip().upper()
    for _p in _paths_to_check:
        sensitive_err = _check_sensitive_path(_p, task_id)
        if sensitive_err:
            return tool_error(sensitive_err)
        if not cross_profile:
            cross_warning = _check_cross_profile_path(_p, task_id)
            if cross_warning:
                return tool_error(cross_warning)

    # Keep Plan refinement on the same exact local-artifact transport as the
    # initial write.  Only the one-file replace form is admitted by the Plan
    # gate; multi-file/V4A/anchor modes continue through the ordinary path and
    # are denied before dispatch.
    try:
        from agent.plan_execution import (
            is_plan_artifact_target,
            write_plan_artifact,
        )

        if path and is_plan_artifact_target(path):
            normalized_mode = str(mode or "replace").strip().lower()
            if normalized_mode not in {"replace", "string", "str_replace"}:
                return tool_error("Plan artifact patch supports replace mode only")
            if old_string is None or new_string is None:
                return tool_error("old_string and new_string required")
            resolved_plan = str(Path(path).expanduser().resolve(strict=False))
            with file_state.lock_path(resolved_plan):
                hash_warning, hash_error = _check_content_hash(
                    path, task_id, expected_hash=expected_hash
                )
                if hash_error:
                    return json.dumps(hash_error, ensure_ascii=False)
                try:
                    current = Path(resolved_plan).read_text(encoding="utf-8")
                except OSError as exc:
                    return tool_error(f"Failed to read file: {resolved_plan}: {exc}")
                from tools.fuzzy_match import fuzzy_find_and_replace

                updated, match_count, _strategy, error = fuzzy_find_and_replace(
                    current,
                    old_string,
                    new_string,
                    bool(replace_all),
                )
                if error or match_count == 0:
                    return tool_error(error or f"Could not find match for old_string in {resolved_plan}")
                result_dict = write_plan_artifact(resolved_plan, updated)
                result_dict["diff"] = "".join(
                    difflib.unified_diff(
                        current.splitlines(keepends=True),
                        updated.splitlines(keepends=True),
                        fromfile=resolved_plan,
                        tofile=resolved_plan,
                    )
                )
                result_dict["matches"] = match_count
                if hash_warning:
                    result_dict["_warning"] = hash_warning
                _update_read_timestamp(path, task_id)
                file_state.note_write(task_id, resolved_plan)
            return json.dumps(result_dict, ensure_ascii=False)
    except PlanContractError as exc:
        return json.dumps(exc.to_dict(), ensure_ascii=False)
    except ImportError:
        pass
    try:
        # Resolve paths for locking.  Ordered + deduplicated so concurrent
        # callers lock in the same order — prevents deadlock on overlapping
        # multi-file V4A patches.
        _resolved_paths: list[str] = []
        _seen: set[str] = set()
        for _p in _paths_to_check:
            try:
                _r = str(_resolve_path_for_task(_p, task_id))
            except Exception:
                _r = None
            if _r and _r not in _seen:
                _resolved_paths.append(_r)
                _seen.add(_r)
        _resolved_paths.sort()

        # Acquire per-path locks in sorted order via ExitStack.  On single
        # path this degenerates to one lock; on empty list (unresolvable)
        # it's a no-op and execution falls through unchanged.
        from contextlib import ExitStack
        with ExitStack() as _locks:
            for _r in _resolved_paths:
                _locks.enter_context(file_state.lock_path(_r))

            # Collect warnings — content-hash gate first (may hard-reject),
            # then cross-agent registry (names sibling), then mtime fallback.
            stale_warnings: list[str] = []
            _path_to_resolved: dict[str, str] = {}
            for _p in _paths_to_check:
                try:
                    _r = str(_resolve_path_for_task(_p, task_id))
                except Exception:
                    _r = None
                _path_to_resolved[_p] = _r
                # ``expected_hash`` is the one-file replace contract;
                # ``expected_hashes`` carries the equivalent explicit
                # precondition per existing V4A source header.  An explicit
                # hash forces fail-closed behavior even when the ambient
                # content-hash policy is only ``warn``.
                _exp = (
                    expected_hash if (path and _p == path) else
                    _v4a_expected_hashes.get(_p)
                )
                _hash_warn, _hash_err = _check_content_hash(
                    _p, task_id, expected_hash=_exp
                )
                # Phase 3: replace-mode recovery — if the tagged snapshot still
                # contains old_string and live disk still has an unambiguous
                # match, proceed with a recovery warning instead of
                # hard-reject / stale-hash warn. write_file / V4A do not recover.
                if (
                    (_hash_err or _hash_warn)
                    and mode == "replace"
                    and path
                    and _p == path
                    and old_string
                    and _r
                ):
                    try:
                        from agent.hashline import (
                            get_snapshot_store,
                            try_recover_replace,
                        )

                        with open(_r, "rb") as _fh:
                            _live = _fh.read().decode("utf-8", errors="surrogatepass")
                        _stored = _get_stored_content_hash(_r, task_id)
                        _expect = (_exp or _stored or "").strip().upper() or None
                        if _expect:
                            _rec = try_recover_replace(
                                store=get_snapshot_store(task_id),
                                path=_r,
                                expected_hash=_expect,
                                old_string=old_string,
                                live_text=_live,
                                replace_all=bool(replace_all),
                            )
                            if _rec is not None:
                                _hash_err = None
                                _hash_warn = _rec.warning
                    except Exception:
                        logger.debug(
                            "replace recovery failed for %s", _p, exc_info=True
                        )
                if _hash_err:
                    return json.dumps(_hash_err, ensure_ascii=False)
                if _hash_warn:
                    stale_warnings.append(_hash_warn)
                _cross = file_state.check_stale(task_id, _r) if _r else None
                _sw = _cross or _check_file_staleness(_p, task_id)
                if not _sw and _r:
                    # Workspace-divergence warning (worktree-cwd bug): relative
                    # path resolving outside the terminal's cwd.
                    _sw = _path_resolution_warning(_p, Path(_r), task_id)
                if _sw:
                    stale_warnings.append(_sw)

            file_ops = _get_file_ops(task_id)

            if mode == "replace":
                if not path:
                    return tool_error("path required")
                if old_string is None or new_string is None:
                    return tool_error("old_string and new_string required")
                # Pass the resolved ABSOLUTE path to the shell layer so it
                # operates on the exact file the tool layer resolved — the
                # shell's own cwd may differ (worktree-cwd bug), and a relative
                # path would let the two layers disagree about which file is
                # being edited.
                _replace_target = _path_to_resolved.get(path) or path
                result = file_ops.patch_replace(_replace_target, old_string, new_string, replace_all)
            elif mode == "patch":
                if not patch:
                    return tool_error("patch content required")
                result = file_ops.patch_v4a(patch)
            elif mode == "anchors":
                if not path:
                    return tool_error("path required")
                if not isinstance(edits, list):
                    return tool_error("edits must be an array for mode='anchors'")
                from agent.coding_workspace import coding_task_context
                from agent.hashline import AnchorEditError, apply_anchor_edits

                _anchor_target = _path_to_resolved.get(path) or path
                _raw = file_ops.read_file_raw(_anchor_target)
                if _raw.error:
                    return json.dumps(
                        {
                            "success": False,
                            "error": _raw.error,
                            "code": "fileNotFound",
                            "retryable": False,
                        },
                        ensure_ascii=False,
                    )
                if _raw.is_binary or _raw.is_image:
                    return json.dumps(
                        {
                            "success": False,
                            "error": "Anchor edits require a plain-text file.",
                            "code": "invalidInput",
                            "retryable": False,
                        },
                        ensure_ascii=False,
                    )
                try:
                    _anchor_result = apply_anchor_edits(_raw.content, edits)
                except AnchorEditError as exc:
                    return json.dumps(exc.as_payload(), ensure_ascii=False)

                _diff = "".join(
                    difflib.unified_diff(
                        _raw.content.splitlines(keepends=True),
                        _anchor_result.content.splitlines(keepends=True),
                        fromfile=_anchor_target,
                        tofile=_anchor_target,
                    )
                )
                with coding_task_context(task_id):
                    _write = file_ops.write_file(
                        _anchor_target, _anchor_result.content
                    )
                _write_dict = _write.to_dict()
                _write_error = _write_dict.get("error")
                result_dict = {
                    **_write_dict,
                    "success": not bool(_write_error),
                    "diff": _diff,
                    "edits_applied": _anchor_result.edits_applied,
                    "affected_lines": _anchor_result.affected_lines,
                }
            else:
                return tool_error(f"Unknown mode: {mode}")

            if mode != "anchors":
                result_dict = result.to_dict()
            if stale_warnings:
                result_dict["_warning"] = stale_warnings[0] if len(stale_warnings) == 1 else " | ".join(stale_warnings)
            # Report the ABSOLUTE path(s) actually patched so a wrong-cwd
            # mismatch (e.g. a worktree session editing the main checkout) is
            # visible in the response instead of silently landing elsewhere.
            _resolved_modified = [
                _path_to_resolved.get(_p) or _p for _p in _paths_to_check
            ]
            # Refresh stored timestamps for all successfully-patched paths so
            # consecutive edits by this task don't trigger false warnings.
            if not result_dict.get("error"):
                result_dict["files_modified"] = _resolved_modified
                if len(_resolved_modified) == 1:
                    result_dict["resolved_path"] = _resolved_modified[0]
                _mark_verification_stale(task_id, _resolved_modified, session_id=session_id)
                for _p in _paths_to_check:
                    _update_read_timestamp(_p, task_id)
                    _r = _path_to_resolved.get(_p)
                    if _r:
                        file_state.note_write(task_id, _r)
                # Successful patch: clear any prior consecutive-failure
                # counters for the touched paths so a future failure on
                # the same path starts the escalation cycle fresh.
                _reset_patch_failures(task_id, [
                    _r for _r in (_path_to_resolved.get(_p) for _p in _paths_to_check) if _r
                ])
        error_text = str(result_dict.get("error") or "")
        # V4A validation failures already contain the precise parser error, but
        # previously gave no machine-visible recovery cue or repeated-failure
        # escalation. Reuse the existing per-task/per-path tracker; V4A stays
        # fully transactional and this branch never retries or edits anything.
        if mode == "patch" and error_text:
            failed_sources = _v4a_failed_source_paths(
                error_text, _hashable_v4a_paths, _path_to_resolved
            )
            failure_count = max(
                (_record_patch_failure(task_id, resolved)
                 for _header, resolved in failed_sources),
                default=0,
            )
            result_dict["_hint"] = _format_v4a_failure_hint(
                error_text, failed_sources, failure_count
            )
        # Hint when old_string not found — saves iterations where the agent
        # retries with stale content instead of re-reading the file.
        # Suppressed when patch_replace already attached a rich "Did you mean?"
        # snippet (which is strictly more useful than the generic hint).
        elif error_text and "Could not find" in error_text:
            # Track per-file consecutive failures for replace mode.  The
            # ``path`` arg only exists for replace mode; for V4A patches
            # we'd need to walk the headers, but in practice V4A failures
            # are far rarer and the existing _hint covers them adequately.
            failure_count = 0
            if mode == "replace" and path:
                resolved = _path_to_resolved.get(path) or path
                failure_count = _record_patch_failure(task_id, resolved)

            if failure_count >= 3:
                # Escalating hint after multiple consecutive failures on the
                # same path.  Most common cause is a stale view of the file —
                # the model is retrying with the same old_string against
                # content that has since changed.  Surface the failure count
                # so the model recognises it's in a loop and breaks out by
                # re-reading or falling back to write_file.
                result_dict["_hint"] = (
                    f"This is failure #{failure_count} patching {path!r}. "
                    "Stop retrying with variations of the same old_string. "
                    "Either: (1) re-read the file fresh to verify current "
                    "content, (2) use a longer / more unique old_string with "
                    "surrounding context lines, or (3) use write_file to "
                    "replace the entire file if the targeted region is hard "
                    "to anchor."
                )
            elif "Did you mean one of these sections?" not in str(result_dict["error"]):
                result_dict["_hint"] = (
                    "old_string not found. Use read_file to verify the current "
                    "content, or search_files to locate the text."
                )
        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))


def search_tool(pattern: str, target: str = "content", path: str = ".",
                file_glob: str = None, limit: int = 50, offset: int = 0,
                output_mode: str = "content", context: int = 0,
                case_insensitive: bool = False, fixed_string: bool = False,
                word: bool = False, multiline: bool = False,
                max_count: int = 0, context_before: int = 0,
                context_after: int = 0,
                task_id: str = "default",
                line_anchors: bool = False) -> str:
    """Search for content or files."""
    try:
        offset, limit = normalize_search_pagination(offset, limit)
        case_insensitive = bool(case_insensitive)
        fixed_string = bool(fixed_string)
        word = bool(word)
        multiline = bool(multiline)
        try:
            max_count = int(max_count or 0)
        except (TypeError, ValueError):
            max_count = 0
        max_count = max(0, max_count)
        try:
            context_before = max(0, int(context_before or 0))
        except (TypeError, ValueError):
            context_before = 0
        try:
            context_after = max(0, int(context_after or 0))
        except (TypeError, ValueError):
            context_after = 0

        try:
            resolved_path = _resolve_path_for_task(path, task_id)
        except (OSError, ValueError, RuntimeError):
            resolved_path = None
        block_error = get_read_block_error(str(resolved_path) if resolved_path else path)
        if block_error:
            return json.dumps({"error": block_error}, ensure_ascii=False)

        file_ops = _get_file_ops(task_id)
        result = file_ops.search(
            pattern=pattern, path=path, target=target, file_glob=file_glob,
            limit=limit, offset=offset, output_mode=output_mode, context=context,
            case_insensitive=case_insensitive, fixed_string=fixed_string, word=word,
            multiline=multiline, max_count=max_count,
            context_before=context_before, context_after=context_after,
        )
        omitted = _filter_read_blocked_search_results(result, task_id)
        _match_anchors: list[str | None] = []
        if line_anchors and target == "content" and output_mode == "content":
            from agent.hashline import build_line_anchors, format_anchor

            _anchor_cache: dict[str, list | None] = {}
            for match in getattr(result, "matches", []):
                anchors = _anchor_cache.get(match.path)
                if match.path not in _anchor_cache:
                    raw = file_ops.read_file_raw(match.path)
                    anchors = (
                        build_line_anchors(raw.content)
                        if not raw.error and not raw.is_binary and not raw.is_image
                        else None
                    )
                    _anchor_cache[match.path] = anchors
                if anchors is not None and 1 <= match.line_number <= len(anchors):
                    _match_anchors.append(format_anchor(anchors[match.line_number - 1]))
                else:
                    _match_anchors.append(None)
        if hasattr(result, 'matches'):
            for m in result.matches:
                if hasattr(m, 'content') and m.content:
                    m.content = redact_sensitive_text(m.content, file_read=True)
        result_dict = result.to_dict(densify=not line_anchors)
        if line_anchors and target == "content" and output_mode == "content":
            result_dict["line_anchors"] = True
            result_dict["anchor_format"] = "LINE:LOCAL:CONTEXT"
            for row, anchor in zip(result_dict.get("matches", []), _match_anchors):
                if anchor:
                    row["anchor"] = anchor

        if omitted:
            result_dict["_omitted"] = (
                f"{omitted} result(s) omitted because they target credential, "
                "token, cache, or secret-bearing environment files."
            )

        # Hint when results were truncated — explicit next offset is clearer
        # than relying on the model to infer it from total_count vs match count.
        if result_dict.get("truncated"):
            next_offset = offset + limit
            # Keep the tool result a single JSON document.  Downstream consumers
            # (MCP transports, context compression, remote execution, and model
            # tool dispatch) must not have to split a human-readable suffix off
            # before parsing the result.
            result_dict["next_offset"] = next_offset
            result_dict["_hint"] = (
                f"Showing paginated results (limit={limit}, offset={offset}). "
                f"Use offset={next_offset} for the next page, or narrow the "
                "search with a more specific pattern or file_glob."
            )
        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))




# ---------------------------------------------------------------------------
# Schemas + Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error


def _check_file_reqs():
    """Lazy wrapper to avoid circular import with tools/__init__.py."""
    from tools import check_file_requirements
    return check_file_requirements()

READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": "Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal. Output format: 'LINE_NUM|CONTENT'. Suggests similar filenames if not found. Use offset and limit for large files. Reads exceeding ~100K characters are truncated on a line boundary and return a next_offset; continue with offset to read the rest. Jupyter notebooks (.ipynb), Word documents (.docx), and Excel workbooks (.xlsx) are auto-extracted to readable text. Successful reads include content_hash (4-hex fingerprint of the full on-disk file) and file_tag like [path#ABCD] — write_file/patch use this to reject stale overwrites. NOTE: Cannot read images or other binary files — use vision_analyze for images.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (absolute, relative, or ~/path)"},
            "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed, default: 1)", "default": 1, "minimum": 1},
            "limit": {"type": "integer", "description": "Maximum number of lines to read (default: 500, max: 2000)", "default": 500, "maximum": 2000},
            "line_anchors": {
                "type": "boolean",
                "description": "Opt in to stable LINE:LOCAL:CONTEXT→CONTENT anchors. Use returned anchors with patch mode='anchors' for transactional line edits.",
                "default": False,
            },
        },
        "required": ["path"]
    }
}

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "Write content to a file, completely replacing existing content. Use this instead of echo/cat heredoc in terminal. Creates parent directories automatically. OVERWRITES the entire file — use 'patch' for targeted edits. Auto-runs syntax checks on .py/.json/.yaml/.toml and other linted languages; only NEW errors introduced by this write are surfaced (pre-existing errors are filtered out). If you previously read_file this path, a content_hash mismatch (file changed on disk) rejects the write with code content_hash_mismatch — re-read then retry.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (will be created if it doesn't exist, overwritten if it does)"},
            "content": {"type": "string", "description": "Complete content to write to the file"},
            "expected_hash": {
                "type": "string",
                "description": "Optional 4-hex content_hash from a prior read_file. When omitted, the fingerprint from the last read_file for this task is used.",
            },
            "cross_profile": {
                "type": "boolean",
                "description": "Opt out of the cross-profile soft guard. Defaults to false. Set true ONLY after explicit user direction to edit another Hairball profile's skills/plugins/cron/memories — by default these writes are blocked with a warning because they affect a different profile than the one this session is running under.",
                "default": False,
            },
        },
        "required": ["path", "content"]
    }
}

PATCH_SCHEMA = {
    "name": "patch",
    "description": (
        "Targeted and transactional edits in files. Use this instead of sed/awk in terminal. "
        "Uses fuzzy matching (9 strategies) so minor whitespace/indentation differences won't break it. "
        "Returns a unified diff. Auto-runs syntax checks after editing.\n\n"
        "REPLACE MODE (mode='replace', default): find a unique string and replace it. "
        "REQUIRED PARAMETERS: mode, path, old_string, new_string.\n"
        "PATCH MODE (mode='patch'): apply V4A multi-file patches for bulk changes. "
        "REQUIRED PARAMETERS: mode, patch.\n"
        "ANCHORS MODE (mode='anchors'): atomically apply one or more replace, "
        "insert_after, or whole-file write operations using anchors returned by "
        "read_file/search_files with line_anchors=true. Every anchor in one edits "
        "batch refers to the same original read snapshot; never adjust later anchor "
        "line numbers for earlier operations in that batch. REQUIRED PARAMETERS: "
        "mode, path, edits.\n\n"
        "If you previously read_file a target path, a content_hash mismatch rejects the "
        "replace edit with code content_hash_mismatch — re-read then retry. "
        "For multi-file V4A patch mode, pass expected_hashes={header_path: content_hash} "
        "for each Update/Delete/Move source that must fail closed if it changed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["replace", "patch", "anchors"],
                "description": "Edit mode. 'replace' requires path + old_string + new_string; 'patch' requires V4A patch content; 'anchors' requires path + edits from an anchored read/search.",
                "default": "replace",
            },
            "path": {
                "type": "string",
                "description": "REQUIRED when mode='replace'. File path to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "REQUIRED when mode='replace'. Exact text to find and replace. Must be unique in the file unless replace_all=true. Include surrounding context lines to ensure uniqueness.",
            },
            "new_string": {
                "type": "string",
                "description": "REQUIRED when mode='replace'. Replacement text. Pass empty string '' to delete the matched text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences instead of requiring a unique match (default: false)",
                "default": False,
            },
            "patch": {
                "type": "string",
                "description": "REQUIRED when mode='patch'. V4A format patch content. Format:\n*** Begin Patch\n*** Update File: path/to/file\n@@ context hint @@\n context line\n-removed line\n+added line\n*** End Patch",
            },
            "edits": {
                "type": "array",
                "description": "Atomic anchor operations. Copy every anchor unchanged from one anchored read/search snapshot; do not recalculate or shift later anchors for earlier operations in this same array. All anchors are validated against that one original snapshot before one write; any stale, missing, ambiguous, invalid, or overlapping operation rejects the entire batch.",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["replace", "insert_after", "write"],
                        },
                        "anchor": {
                            "type": "string",
                            "description": "LINE:LOCAL or LINE:LOCAL:CONTEXT anchor returned by read_file/search_files. Required for replace and insert_after.",
                        },
                        "endAnchor": {
                            "type": "string",
                            "description": "Optional inclusive end anchor for a multi-line replace range.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Clean replacement/insertion/full-file content without displayed anchor prefixes.",
                        },
                    },
                    "required": ["op", "content"],
                },
            },
            "expected_hash": {
                "type": "string",
                "description": "Optional 4-hex content_hash from a prior read_file for the replace-mode path. When omitted, the fingerprint from the last read_file for this task is used.",
            },
            "expected_hashes": {
                "type": "object",
                "description": "V4A patch mode only: optional map of each exact Update/Delete/Move source header path to its 4-hex content_hash from read_file. Each supplied entry is a fail-closed precondition even when content-hash policy is warn. Do not include Add or Move destination paths.",
                "additionalProperties": {"type": "string"},
            },
            "cross_profile": {
                "type": "boolean",
                "description": "Opt out of the cross-profile soft guard. Defaults to false. Set true ONLY after explicit user direction to edit another Hairball profile's skills/plugins/cron/memories.",
                "default": False,
            },
        },
        "required": ["mode"],
    },
}

SEARCH_FILES_SCHEMA = {
    "name": "search_files",
    "description": (
        "Search file contents or find files by name. Use this instead of "
        "grep/rg/find/ls in terminal. Ripgrep-backed.\n\n"
        "Content search (target='content'): regex inside files. Optional "
        "flags: case_insensitive, fixed_string, word, context, "
        "context_before, context_after, multiline, max_count.\n\n"
        "File search (target='files'): glob by name (e.g. '*.py'), sorted by mtime."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern for content search, or glob pattern (e.g., '*.py') for file search"},
            "target": {"type": "string", "enum": ["content", "files"], "description": "'content' searches inside file contents, 'files' searches for files by name", "default": "content"},
            "path": {"type": "string", "description": "Directory or file to search in (default: current working directory)", "default": "."},
            "file_glob": {"type": "string", "description": "Filter files by pattern in grep mode (e.g., '*.py' to only search Python files)"},
            "limit": {"type": "integer", "description": "Maximum number of results to return (default: 50)", "default": 50},
            "offset": {"type": "integer", "description": "Skip first N results for pagination (default: 0)", "default": 0},
            "output_mode": {"type": "string", "enum": ["content", "files_only", "count"], "description": "Output format for grep mode: 'content' shows matching lines with line numbers, 'files_only' lists file paths, 'count' shows match counts per file", "default": "content"},
            "context": {"type": "integer", "description": "Symmetric context lines (rg -C).", "default": 0},
            "context_before": {"type": "integer", "description": "Context lines before each match (rg -B).", "default": 0},
            "context_after": {"type": "integer", "description": "Context lines after each match (rg -A).", "default": 0},
            "case_insensitive": {"type": "boolean", "description": "Ignore case (rg -i).", "default": False},
            "fixed_string": {"type": "boolean", "description": "Literal pattern, not regex (rg -F).", "default": False},
            "word": {"type": "boolean", "description": "Whole-word match (rg -w).", "default": False},
            "multiline": {"type": "boolean", "description": "Match across lines (rg -U).", "default": False},
            "max_count": {"type": "integer", "description": "Max matches per file (rg --max-count). 0 = unlimited.", "default": 0},
            "line_anchors": {
                "type": "boolean",
                "description": "Opt in to a stable anchor on every content-search row so the result can be passed directly to patch mode='anchors'.",
                "default": False,
            },
        },
        "required": ["pattern"]
    }
}


def _handle_read_file(args, **kw):
    tid = kw.get("task_id") or "default"
    return read_file_tool(
        path=args.get("path", ""),
        offset=args.get("offset", 1),
        limit=args.get("limit", 500),
        line_anchors=bool(args.get("line_anchors", False)),
        task_id=tid,
    )


def _handle_write_file(args, **kw):
    tid = kw.get("task_id") or "default"
    if not args.get("path") or not isinstance(args.get("path"), str):
        return tool_error(
            "write_file: missing required field 'path'. Re-emit the tool call with "
            "both 'path' and 'content' set."
        )
    if "content" not in args:
        return tool_error(
            "write_file: missing required field 'content'. The tool call included a "
            "path but no content argument — this is almost always a dropped-arg bug "
            "under context pressure. Re-emit the tool call with the full content "
            "payload, or use execute_code with hairball_tools.write_file() for very "
            "large files."
        )
    if not isinstance(args["content"], str):
        return tool_error(
            f"write_file: 'content' must be a string, got "
            f"{type(args['content']).__name__}."
        )
    _eh = args.get("expected_hash")
    if _eh is not None and not isinstance(_eh, str):
        return tool_error("write_file: 'expected_hash' must be a string when provided.")
    return write_file_tool(
        path=args["path"], content=args["content"], task_id=tid,
        cross_profile=bool(args.get("cross_profile", False)),
        session_id=kw.get("session_id"),
        expected_hash=_eh if isinstance(_eh, str) else None,
    )


def _handle_patch(args, **kw):
    tid = kw.get("task_id") or "default"
    _eh = args.get("expected_hash")
    if _eh is not None and not isinstance(_eh, str):
        return tool_error("patch: 'expected_hash' must be a string when provided.")
    _ehs = args.get("expected_hashes")
    if _ehs is not None and not isinstance(_ehs, dict):
        return tool_error("patch: 'expected_hashes' must be an object when provided.")
    _edits = args.get("edits")
    if _edits is not None and not isinstance(_edits, list):
        return tool_error("patch: 'edits' must be an array when provided.")
    return patch_tool(
        mode=args.get("mode", "replace"), path=args.get("path"),
        old_string=args.get("old_string"), new_string=args.get("new_string"),
        replace_all=args.get("replace_all", False), patch=args.get("patch"), task_id=tid,
        edits=_edits if isinstance(_edits, list) else None,
        cross_profile=bool(args.get("cross_profile", False)),
        session_id=kw.get("session_id"),
        expected_hash=_eh if isinstance(_eh, str) else None,
        expected_hashes=_ehs if isinstance(_ehs, dict) else None,
    )


def _handle_search_files(args, **kw):
    tid = kw.get("task_id") or "default"
    target_map = {"grep": "content", "find": "files"}
    raw_target = args.get("target", "content")
    target = target_map.get(raw_target, raw_target)
    return search_tool(
        pattern=args.get("pattern", ""), target=target, path=args.get("path", "."),
        file_glob=args.get("file_glob"), limit=args.get("limit", 50), offset=args.get("offset", 0),
        output_mode=args.get("output_mode", "content"), context=args.get("context", 0),
        case_insensitive=bool(args.get("case_insensitive", False)),
        fixed_string=bool(args.get("fixed_string", False)),
        word=bool(args.get("word", False)),
        multiline=bool(args.get("multiline", False)),
        max_count=args.get("max_count", 0),
        context_before=args.get("context_before", 0),
        context_after=args.get("context_after", 0),
        line_anchors=bool(args.get("line_anchors", False)),
        task_id=tid)


registry.register(name="read_file", toolset="file", schema=READ_FILE_SCHEMA, handler=_handle_read_file, check_fn=_check_file_reqs, emoji="📖", max_result_size_chars=100_000)
registry.register(name="write_file", toolset="file", schema=WRITE_FILE_SCHEMA, handler=_handle_write_file, check_fn=_check_file_reqs, emoji="✍️", max_result_size_chars=100_000)
registry.register(name="patch", toolset="file", schema=PATCH_SCHEMA, handler=_handle_patch, check_fn=_check_file_reqs, emoji="🔧", max_result_size_chars=100_000)
registry.register(name="search_files", toolset="file", schema=SEARCH_FILES_SCHEMA, handler=_handle_search_files, check_fn=_check_file_reqs, emoji="🔎", max_result_size_chars=100_000)
