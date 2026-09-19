"""Workspace and project-root resolution for LSP.

Two concerns live here:

1. **Workspace gate** — whether LSP should run for a file at all.
   Prefer a git worktree that contains the **file**.  When the process
   cwd is a different checkout (common in chat: cwd=main repo, edits
   under ``/tmp/...``), fall back to project markers / package lift /
   file parent — never refuse solely because the agent cwd is git and
   the target is not.  Home and filesystem root remain denied.
   Set ``lsp.require_git_workspace: true`` to restore the old git-only gate.

2. **NearestRoot** — the per-server project-root walk.  Each language
   server cares about a different marker (``pyproject.toml`` for
   Python, ``Cargo.toml`` for Rust, ``go.mod`` for Go, etc.) and
   wants the directory containing that marker.  ``nearest_root()``
   walks up from a starting path looking for any of a list of marker
   files, optionally bailing if an exclude marker shows up first.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

logger = logging.getLogger("agent.lsp.workspace")

# Cache: cwd → (worktree_root, is_git) so repeated calls don't re-stat.
# Cleared on shutdown.  Keyed by absolute resolved path so symlink
# folds collapse to one entry.
_workspace_cache: dict = {}

# Markers that identify a non-git project root for fallback workspaces.
_PROJECT_MARKERS = (
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "Pipfile",
    "package.json",
    "tsconfig.json",
    "jsconfig.json",
    "Cargo.toml",
    "go.mod",
    "composer.json",
    "Gemfile",
    "mix.exs",
    "Makefile",
    "CMakeLists.txt",
    "sgconfig.yml",
)


def normalize_path(path: str) -> str:
    """Normalize a path for use as a stable map key.

    Resolves ``~``, makes absolute, and collapses ``.``/``..``.  We do
    NOT resolve symlinks here — symlink stability matters for some
    LSP servers (rust-analyzer cares about Cargo workspace identity)
    and we want the canonical path the user typed when possible.
    """
    return os.path.abspath(os.path.expanduser(path))


def find_git_worktree(start: str) -> Optional[str]:
    """Walk up from ``start`` looking for a ``.git`` entry (file or dir).

    Returns the directory containing ``.git``, or ``None`` if no git
    root is found before hitting the filesystem root.

    A ``.git`` *file* (not directory) means we're inside a git
    worktree set up via ``git worktree add`` — both forms count.
    """
    try:
        start_path = Path(normalize_path(start))
        if start_path.is_file():
            start_path = start_path.parent
    except (OSError, RuntimeError, ValueError):
        # Pathological input (loop in symlinks, encoding error, etc.) —
        # bail out rather than crash the lint hook.
        return None

    # Cache check
    cached = _workspace_cache.get(str(start_path))
    if cached is not None:
        root, _is_git = cached
        return root

    cur = start_path
    # Defensive cap: the deepest reasonable monorepo is well under 64
    # levels.  Caps the walk so a pathological cwd or a symlink cycle
    # we somehow traverse can't keep us looping.
    for _ in range(64):
        git_marker = cur / ".git"
        try:
            if git_marker.exists():
                resolved = str(cur)
                _workspace_cache[str(start_path)] = (resolved, True)
                return resolved
        except OSError:
            # Permission error on a parent dir — bail out cleanly.
            break
        parent = cur.parent
        if parent == cur:
            break
        cur = parent

    _workspace_cache[str(start_path)] = (None, False)
    return None


def is_inside_workspace(path: str, workspace_root: str) -> bool:
    """Return True iff ``path`` is inside (or equal to) ``workspace_root``.

    Uses absolute paths but does not resolve symlinks — a file accessed
    via a symlink that points outside the workspace still counts as
    outside.  This is the conservative interpretation; matches LSP
    behaviour where servers reject didOpen for unrelated files.
    """
    p = normalize_path(path)
    root = normalize_path(workspace_root)
    if p == root:
        return True
    # Use os.path.commonpath to handle case-insensitive filesystems
    # correctly on macOS/Windows.
    try:
        common = os.path.commonpath([p, root])
    except ValueError:
        # Different drives on Windows.
        return False
    return common == root


def nearest_root(
    start: str,
    markers: Iterable[str],
    *,
    excludes: Optional[Iterable[str]] = None,
    ceiling: Optional[str] = None,
) -> Optional[str]:
    """Walk up from ``start`` looking for any of the given marker files.

    Returns the **directory containing** the first matched marker, or
    ``None`` if no marker is found before hitting ``ceiling`` (or the
    filesystem root if no ceiling).

    If ``excludes`` is provided and an exclude marker matches *first*
    in the upward walk, returns ``None`` — the server is gated off
    for that file.  Mirrors OpenCode's NearestRoot exclude semantics
    (e.g. typescript skips deno projects when ``deno.json`` is found
    before ``package.json``).
    """
    start_path = Path(normalize_path(start))
    try:
        if start_path.is_file():
            start_path = start_path.parent
    except (OSError, RuntimeError, ValueError):
        return None
    ceiling_path = Path(normalize_path(ceiling)) if ceiling else None

    markers_list = list(markers)
    excludes_list = list(excludes) if excludes else []

    cur = start_path
    # Defensive cap matching ``find_git_worktree``.  Bounded walk
    # protects against pathological inputs even though the
    # parent-equality stop normally terminates within ~10 steps.
    for _ in range(64):
        # Check excludes first — if an exclude is found at this level,
        # the server is gated off for this file.
        for exc in excludes_list:
            try:
                if (cur / exc).exists():
                    return None
            except OSError:
                continue
        # Then check markers.
        for marker in markers_list:
            try:
                if (cur / marker).exists():
                    return str(cur)
            except OSError:
                continue
        # Stop conditions.
        if ceiling_path is not None and cur == ceiling_path:
            return None
        parent = cur.parent
        if parent == cur:
            return None
        cur = parent
    return None


def _require_git_workspace() -> bool:
    """Read ``lsp.require_git_workspace`` (default False)."""
    try:
        from hairball_cli.config import load_config

        cfg = load_config()
    except Exception:
        return False
    if not isinstance(cfg, dict):
        return False
    lsp = cfg.get("lsp") if isinstance(cfg.get("lsp"), dict) else {}
    return bool(lsp.get("require_git_workspace"))


def _is_denied_fallback_root(root: str) -> bool:
    """Refuse home / filesystem roots as synthetic LSP workspaces."""
    root_n = normalize_path(root)
    try:
        home_n = normalize_path(str(Path.home()))
    except Exception:
        home_n = ""
    deny = {home_n, "/", "/private"} if home_n else {"/", "/private"}
    return root_n in deny


def _dir_imported_as_package(dir_path: Path) -> bool:
    """True if a ``.py`` under ``dir_path`` imports ``dir_path.name`` as a package.

    Catches the hand-demo layout::

        /tmp/demo/demo_pkg/calc.py
        /tmp/demo/demo_pkg/test_calc.py   # from demo_pkg.calc import add

    with no ``__init__.py`` (namespace-style).  Without lifting the
    workspace to ``/tmp/demo``, pyright cannot resolve that import and
    rename/references miss the test file.
    """
    pkg = dir_path.name
    if not pkg.isidentifier():
        return False
    needles = (
        f"from {pkg}.",
        f"import {pkg}.",
        f"from {pkg} import",
        f"import {pkg}\n",
        f"import {pkg}\r",
        f"import {pkg} ",
    )
    try:
        for py in sorted(dir_path.glob("*.py")):
            try:
                text = py.read_text(encoding="utf-8", errors="ignore")[:12000]
            except OSError:
                continue
            if any(n in text for n in needles):
                return True
    except OSError:
        return False
    return False


def _lift_out_of_python_packages(start: Path) -> Path:
    """If ``start`` is inside a Python package, lift to its import root.

    Walks up while the current directory has ``__init__.py``, or (once)
    when sibling modules import the directory name as a package even
    without ``__init__.py``.  Stops before a denied root.

    Examples:
    - ``…/demo/hand_demo/calc.py`` + ``__init__.py`` → ``…/demo``
    - ``…/A3/demo_pkg/test_calc.py`` with ``from demo_pkg.calc`` → ``…/A3``
    """
    cur = start
    for _ in range(64):
        try:
            has_init = (cur / "__init__.py").exists()
        except OSError:
            break
        # Namespace / unmarked package: lift one level when local files
        # import this directory by name (see _dir_imported_as_package).
        if not has_init:
            if _dir_imported_as_package(cur):
                parent = cur.parent
                if parent != cur and not _is_denied_fallback_root(str(parent)):
                    cur = parent
            break
        parent = cur.parent
        if parent == cur or _is_denied_fallback_root(str(parent)):
            break
        cur = parent
    return cur


def fallback_workspace_root(file_path: str) -> Optional[str]:
    """Non-git workspace root for ``file_path``, or ``None`` if denied.

    Order: project markers → lift out of packages (``__init__.py`` or
    self-import as package name) → file's parent directory.  Never
    returns the user's home directory or ``/`` as a workspace.
    """
    try:
        path = Path(normalize_path(file_path))
    except (OSError, RuntimeError, ValueError):
        return None
    # Prefer treating path as a file when it has a suffix even if not
    # yet created (common for write-then-lsp flows).
    if path.suffix or path.is_file():
        start = path.parent
    elif path.is_dir():
        start = path
    else:
        start = path.parent

    marked = nearest_root(str(start), _PROJECT_MARKERS)
    if marked and not _is_denied_fallback_root(marked):
        return marked

    root_path = _lift_out_of_python_packages(start)
    root = str(root_path)
    if _is_denied_fallback_root(root):
        return None
    return root


def resolve_workspace_for_file(
    file_path: str,
    *,
    cwd: Optional[str] = None,
    coding_workspace: Optional[str] = None,
) -> Tuple[Optional[str], bool]:
    """Resolve the workspace root for a file.

    Returns ``(workspace_root, gated_in)`` where ``gated_in`` is True
    iff LSP should run for this file at all.

    Session coding-workspace pin (OMP session.cwd semantics) first:

    0. If ``coding_workspace`` is set and the file sits inside it, use
       that pin — unless package-lift / markers discover a **strict
       ancestor** of the pin (self-heal a too-narrow pin). When
       ``lsp.require_git_workspace`` and the pin is not a git worktree,
       fall through to discovery.

    Then file-anchored discovery:

    1. Git worktree that contains the **file** (walk from the file path)
    2. Else cwd's git worktree, **only if** the file sits inside it
    3. Unless ``lsp.require_git_workspace`` is true: project-marker /
       package-lift / file-parent fallback (see
       :func:`fallback_workspace_root`)

    Chat agents often keep process cwd on the main repo while writing
    under ``/tmp/...``; step 1+3 keep those trees from being refused or
    wrongly keyed to the agent cwd. Callers that pin the discovered root
    (see :mod:`agent.coding_workspace`) stabilize subsequent lookups.

    Returns ``(None, False)`` when no acceptable workspace is found.
    """
    cwd = cwd or os.getcwd()

    if coding_workspace:
        pin = normalize_path(coding_workspace)
        if is_inside_workspace(file_path, pin):
            require_git = _require_git_workspace()
            pin_ok = (not require_git) or (find_git_worktree(pin) is not None)
            if pin_ok:
                # Self-heal short pins: package-lift / markers may want a
                # strict ancestor of the current pin (e.g. demo_pkg → project).
                # Never shrink the pin — only lift.
                if not require_git:
                    healed = fallback_workspace_root(file_path)
                    if (
                        healed
                        and healed != pin
                        and is_inside_workspace(pin, healed)
                    ):
                        return healed, True
                return pin, True
            # require_git_workspace: pin is not git — fall through to
            # discover a git root (or refuse).

    # Anchor on the file first — never let an unrelated cwd git root
    # claim a path outside that worktree.
    file_root = find_git_worktree(file_path)
    if file_root is not None:
        return file_root, True

    cwd_root = find_git_worktree(cwd)
    if cwd_root is not None and is_inside_workspace(file_path, cwd_root):
        return cwd_root, True

    if _require_git_workspace():
        return None, False

    fallback = fallback_workspace_root(file_path)
    if fallback:
        return fallback, True
    return None, False


def clear_cache() -> None:
    """Clear the workspace-resolution cache.

    Called on service shutdown so a subsequent re-init doesn't pick
    up stale results from a previous session.
    """
    _workspace_cache.clear()


__all__ = [
    "find_git_worktree",
    "is_inside_workspace",
    "nearest_root",
    "normalize_path",
    "fallback_workspace_root",
    "resolve_workspace_for_file",
    "clear_cache",
]
