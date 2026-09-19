"""Task/session-scoped coding workspace root (OMP session.cwd semantics).

UI ``session.cwd`` still anchors relative paths and the project chip.
``coding_workspace`` is a separate pin used by LSP/AST: seeded from the
session cwd at stamp time, then updated when a file resolves outside the
current pin (discover → pin).

Multi-tree sessions (e.g. ``/tmp/hb-cap/A1`` + ``A2`` + ``A3``) register
each discovered root. Sibling scratch trees are umbrellad to their lowest
common ancestor when that LCA is safe (never ``/``, ``$HOME``, or bare
``/tmp``). Unrelated trees (repo + ``/tmp``) stay as separate roots; file
resolution picks the deepest matching root.

In-memory maps are the hot path. Durable state lives in
``$HAIRBALL_HOME/coding_workspaces.json`` (survives process restart).
UI sessions may also mirror fields onto ``SessionRecord``.

Not stored in ``register_task_env_overrides`` — that API replaces the
whole override dict on each stamp.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_coding_workspaces: Dict[str, str] = {}
# All known coding roots for a task (after umbrella collapse).
_coding_roots: Dict[str, Set[str]] = {}
_disk_lock = threading.RLock()
_persist_hooks: List[Callable[[str, Optional[str], List[str]], None]] = []

# Active task/session id for sync LSP paths that lack an explicit task_id
# (e.g. FileOperations.snapshot → LSPService).
_active_coding_task_id: ContextVar[Optional[str]] = ContextVar(
    "hairball_active_coding_task_id", default=None
)


def normalize_coding_root(path: str) -> str:
    """Absolute, expanded path suitable as a map key / LSP rootUri.

    Uses ``realpath`` when the path exists so macOS ``/tmp`` vs
    ``/private/tmp`` (and other symlink mounts) collapse to one identity
    for pin matching and orphan-client eviction.
    """
    expanded = os.path.abspath(os.path.expanduser(path))
    try:
        if os.path.exists(expanded):
            return os.path.realpath(expanded)
    except OSError:
        pass
    return expanded


def get_active_coding_task_id() -> Optional[str]:
    tid = _active_coding_task_id.get()
    if tid is None:
        return None
    tid = str(tid).strip()
    return tid or None


def set_active_coding_task_id(task_id: Optional[str]) -> Token:
    """Set the ContextVar; return the token for ``reset_active_coding_task_id``."""
    if task_id is None or not str(task_id).strip():
        return _active_coding_task_id.set(None)
    return _active_coding_task_id.set(str(task_id).strip())


def reset_active_coding_task_id(token: Token) -> None:
    _active_coding_task_id.reset(token)


@contextmanager
def coding_task_context(task_id: Optional[str]) -> Iterator[None]:
    """Bind ``task_id`` as the active coding task for the duration of the block."""
    if not task_id or not str(task_id).strip():
        yield
        return
    token = set_active_coding_task_id(task_id)
    try:
        yield
    finally:
        reset_active_coding_task_id(token)


def _resolve_tid(task_id: Optional[str]) -> Optional[str]:
    tid = (str(task_id).strip() if task_id else None) or get_active_coding_task_id()
    return tid or None


def get_coding_workspace(task_id: Optional[str] = None) -> Optional[str]:
    """Return the primary coding workspace pin for ``task_id``."""
    tid = _resolve_tid(task_id)
    if not tid:
        return None
    return _coding_workspaces.get(tid)


def get_coding_roots(task_id: Optional[str] = None) -> List[str]:
    """Return all registered coding roots for ``task_id`` (sorted)."""
    tid = _resolve_tid(task_id)
    if not tid:
        return []
    roots = _coding_roots.get(tid) or set()
    primary = _coding_workspaces.get(tid)
    if primary:
        roots = set(roots) | {primary}
    return sorted(roots)


def _disk_path() -> Path:
    from hairball_constants import get_hairball_home

    return get_hairball_home() / "coding_workspaces.json"


def register_coding_persist_hook(
    hook: Callable[[str, Optional[str], List[str]], None],
) -> None:
    """Register a listener invoked after durable pin changes (e.g. UI mirror)."""
    if hook not in _persist_hooks:
        _persist_hooks.append(hook)


def _notify_persist_hooks(task_id: str) -> None:
    primary = _coding_workspaces.get(task_id)
    roots = get_coding_roots(task_id)
    for hook in list(_persist_hooks):
        try:
            hook(task_id, primary, roots)
        except Exception:
            logger.debug("coding workspace persist hook failed", exc_info=True)


def persist_coding_workspace(task_id: Optional[str] = None) -> None:
    """Write the task's coding pin/roots to ``coding_workspaces.json``."""
    tid = _resolve_tid(task_id)
    if not tid:
        return
    primary = _coding_workspaces.get(tid)
    roots = get_coding_roots(tid)
    payload_entry = {
        "coding_workspace": primary,
        "coding_roots": roots,
    }
    path = _disk_path()
    with _disk_lock:
        data: Dict[str, Any] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data = raw
            except Exception:
                data = {}
        sessions = data.get("sessions")
        if not isinstance(sessions, dict):
            sessions = {}
            data["sessions"] = sessions
        if primary:
            sessions[tid] = payload_entry
        else:
            sessions.pop(tid, None)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            logger.debug("coding workspace disk persist failed", exc_info=True)
            return
    _notify_persist_hooks(tid)


def restore_coding_workspace(
    task_id: str,
    primary: str,
    roots: Optional[List[str]] = None,
    *,
    persist: bool = False,
) -> str:
    """Load a durable snapshot into memory (no umbrella recompute)."""
    tid = str(task_id or "").strip()
    if not tid:
        raise ValueError("task_id required")
    normalized = normalize_coding_root(primary)
    root_set: Set[str] = set()
    for r in roots or [normalized]:
        if r and str(r).strip():
            root_set.add(normalize_coding_root(str(r)))
    root_set.add(normalized)
    _coding_workspaces[tid] = normalized
    _coding_roots[tid] = root_set
    if persist:
        persist_coding_workspace(tid)
    return normalized


def restore_coding_workspace_from_disk(task_id: str) -> Optional[str]:
    """Restore pin/roots from disk if present. Returns primary or None."""
    tid = str(task_id or "").strip()
    if not tid:
        return None
    path = _disk_path()
    with _disk_lock:
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    sessions = raw.get("sessions") if isinstance(raw, dict) else None
    if not isinstance(sessions, dict):
        return None
    entry = sessions.get(tid)
    if not isinstance(entry, dict):
        return None
    primary = str(entry.get("coding_workspace") or "").strip()
    if not primary:
        return None
    roots_raw = entry.get("coding_roots") or []
    roots = [str(r).strip() for r in roots_raw if str(r).strip()] if isinstance(roots_raw, list) else []
    return restore_coding_workspace(tid, primary, roots, persist=False)


def ensure_coding_workspace_loaded(
    task_id: str,
    *,
    seed_cwd: str = "",
    fallback_primary: str = "",
    fallback_roots: Optional[List[str]] = None,
) -> Optional[str]:
    """Stamp helper: memory → disk → optional mirror → seed from session cwd.

    ``fallback_primary`` / ``fallback_roots`` cover UI ``SessionRecord`` mirrors
    when ``coding_workspaces.json`` is missing (export/import).
    """
    tid = str(task_id or "").strip()
    if not tid:
        return None
    existing = _coding_workspaces.get(tid)
    if existing:
        return existing
    restored = restore_coding_workspace_from_disk(tid)
    if restored:
        return restored
    mirror = str(fallback_primary or "").strip()
    if mirror:
        return restore_coding_workspace(
            tid, mirror, fallback_roots, persist=True
        )
    cwd = str(seed_cwd or "").strip()
    if cwd:
        return seed_coding_workspace(tid, cwd)
    return None


def set_coding_workspace(
    task_id: str,
    root: str,
    *,
    replace_roots: bool = False,
) -> str:
    """Set the primary pin and register it as a root.

    ``replace_roots`` is reserved for an explicit user project switch.  Normal
    model-driven discovery keeps the existing multi-root set, while a real
    project change must not leave LSP/AST attached to the previous project.
    """
    tid = str(task_id or "").strip()
    if not tid:
        raise ValueError("task_id required")
    normalized = normalize_coding_root(root)
    _coding_workspaces[tid] = normalized
    if replace_roots:
        _coding_roots[tid] = {normalized}
    else:
        _coding_roots.setdefault(tid, set()).add(normalized)
    persist_coding_workspace(tid)
    return normalized


def clear_coding_workspace(task_id: str) -> Optional[str]:
    """Remove the pin and all roots for ``task_id``. Returns the old primary."""
    tid = str(task_id or "").strip()
    if not tid:
        return None
    _coding_roots.pop(tid, None)
    old = _coding_workspaces.pop(tid, None)
    persist_coding_workspace(tid)
    return old


def seed_coding_workspace(task_id: str, root: str) -> str:
    """Initialize pin from session cwd only when unset.

    Subsequent stamps must not wipe a pin already lifted to ``/tmp/...``.
    """
    tid = str(task_id or "").strip()
    if not tid:
        raise ValueError("task_id required")
    existing = _coding_workspaces.get(tid)
    if existing:
        return existing
    # Prefer durable pin over re-seeding from cwd after restart.
    restored = restore_coding_workspace_from_disk(tid)
    if restored:
        return restored
    return set_coding_workspace(tid, root)


def is_ancestor_root(ancestor: str, descendant: str) -> bool:
    """True if ``ancestor`` is a strict ancestor of ``descendant`` (or equal)."""
    a = normalize_coding_root(ancestor)
    d = normalize_coding_root(descendant)
    if a == d:
        return True
    try:
        return os.path.commonpath([a, d]) == a
    except ValueError:
        return False


def lowest_common_ancestor(paths: List[str]) -> Optional[str]:
    """Return the lowest common ancestor directory of ``paths``, or None."""
    if not paths:
        return None
    norms = [normalize_coding_root(p) for p in paths]
    try:
        common = os.path.commonpath(norms)
    except ValueError:
        return None
    if not common:
        return None
    # commonpath may return a file if all paths are the same file — normalize
    # to a directory when possible.
    if os.path.isfile(common):
        common = os.path.dirname(common)
    return normalize_coding_root(common)


def _denied_umbrella_roots() -> Set[str]:
    deny = {"/", "/private", "/tmp", "/private/tmp", "/var", "/var/tmp"}
    try:
        deny.add(normalize_coding_root(str(Path.home())))
    except Exception:
        pass
    return deny


def is_safe_umbrella_root(root: str) -> bool:
    """True if ``root`` is acceptable as a multi-tree coding umbrella."""
    n = normalize_coding_root(root)
    if n in _denied_umbrella_roots():
        return False
    # Bare volume roots / single-segment paths are too broad.
    parts = Path(n).parts
    if len(parts) <= 2:  # '/', '/tmp', '/Users', '/private/tmp'
        return False
    return True


def select_coding_root_for_file(
    file_path: str, task_id: Optional[str] = None
) -> Optional[str]:
    """Deepest registered root that contains ``file_path``, or None."""
    tid = _resolve_tid(task_id)
    if not tid:
        return None
    roots = get_coding_roots(tid)
    if not roots:
        return None
    file_n = normalize_coding_root(file_path)
    matches = [r for r in roots if is_ancestor_root(r, file_n)]
    if not matches:
        return None
    # Deepest = longest normalized path (most specific).
    return max(matches, key=lambda r: (len(Path(r).parts), len(r)))


def _collapse_roots_under(tid: str, umbrella: str) -> None:
    """Replace roots under ``umbrella`` with ``umbrella`` itself."""
    umbrella = normalize_coding_root(umbrella)
    current = set(_coding_roots.get(tid) or set())
    primary = _coding_workspaces.get(tid)
    if primary:
        current.add(primary)
    kept: Set[str] = set()
    for r in current:
        if r == umbrella or is_ancestor_root(umbrella, r):
            continue
        kept.add(r)
    kept.add(umbrella)
    _coding_roots[tid] = kept
    _coding_workspaces[tid] = umbrella


def maybe_umbrella_coding_roots(task_id: Optional[str] = None) -> Optional[str]:
    """Collapse sibling scratch roots to a safe LCA when possible.

    Returns the umbrella root when a collapse happened, else None.
    Unrelated trees (e.g. repo + ``/tmp/...``) whose LCA is denied stay
    as separate roots.
    """
    tid = _resolve_tid(task_id)
    if not tid:
        return None
    roots = list(get_coding_roots(tid))
    if len(roots) < 2:
        return None

    # Greedy pairwise merge while any pair shares a safe LCA.
    current = set(roots)
    umbrella_hit: Optional[str] = None
    progress = True
    while progress:
        progress = False
        items = sorted(current, key=lambda p: (len(Path(p).parts), p))
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                lca = lowest_common_ancestor([a, b])
                if not lca or not is_safe_umbrella_root(lca):
                    continue
                # Only collapse when LCA is a real parent of at least one side
                # (or equal to one and parent of the other).
                if lca != a and lca != b and not (
                    is_ancestor_root(lca, a) and is_ancestor_root(lca, b)
                ):
                    continue
                if not (is_ancestor_root(lca, a) and is_ancestor_root(lca, b)):
                    continue
                # Replace every root under lca with lca.
                under = {r for r in current if is_ancestor_root(lca, r)}
                if len(under) < 2 and lca not in current:
                    continue
                current -= under
                current.add(lca)
                umbrella_hit = lca
                progress = True
                break
            if progress:
                break

    if umbrella_hit is None:
        return None
    _coding_roots[tid] = current
    # Primary becomes the umbrella when the old primary sits under it.
    old_primary = _coding_workspaces.get(tid)
    if old_primary and is_ancestor_root(umbrella_hit, old_primary):
        _coding_workspaces[tid] = umbrella_hit
    elif old_primary not in current:
        _coding_workspaces[tid] = umbrella_hit
    return umbrella_hit


def register_coding_root(
    task_id: Optional[str], root: str
) -> Tuple[str, Optional[str], bool, List[str]]:
    """Register ``root``, update primary pin, and try sibling umbrella.

    Returns ``(primary, umbrella_or_none, lifted, retired_roots)`` where
    ``lifted`` is True when the primary moved to a strict ancestor of the
    previous primary, and ``retired_roots`` are former coding roots removed
    by umbrella collapse (callers should shut down LSP clients for them).
    """
    normalized = normalize_coding_root(root)
    tid = _resolve_tid(task_id)
    if not tid:
        return normalized, None, False, []
    old_primary = _coding_workspaces.get(tid)
    before = set(get_coding_roots(tid))
    _coding_roots.setdefault(tid, set()).add(normalized)
    _coding_workspaces[tid] = normalized
    umbrella = maybe_umbrella_coding_roots(tid)
    primary = _coding_workspaces.get(tid) or normalized
    after = set(get_coding_roots(tid))
    retired = sorted(before - after)
    lifted = bool(
        old_primary
        and primary != old_primary
        and is_ancestor_root(primary, old_primary)
    )
    persist_coding_workspace(tid)
    return primary, umbrella, lifted, retired


def pin_coding_workspace(
    task_id: Optional[str], root: str
) -> Tuple[str, Optional[str], bool]:
    """Pin ``root`` for the task (registers + umbrellas).

    Returns ``(new_root, old_root, lifted)`` where ``lifted`` is True when the
    new primary is a strict ancestor of the previous primary.
    """
    tid = _resolve_tid(task_id)
    old = _coding_workspaces.get(tid) if tid else None
    primary, _umbrella, lifted, _retired = register_coding_root(task_id, root)
    if not tid:
        return primary, None, False
    return primary, old, lifted


def clear_all_coding_workspaces(*, wipe_disk: bool = True) -> None:
    """Test helper — drop every in-memory pin (and by default the disk file)."""
    _coding_workspaces.clear()
    _coding_roots.clear()
    if wipe_disk:
        path = _disk_path()
        with _disk_lock:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass


__all__ = [
    "normalize_coding_root",
    "get_active_coding_task_id",
    "set_active_coding_task_id",
    "reset_active_coding_task_id",
    "coding_task_context",
    "get_coding_workspace",
    "get_coding_roots",
    "set_coding_workspace",
    "clear_coding_workspace",
    "seed_coding_workspace",
    "register_coding_persist_hook",
    "persist_coding_workspace",
    "restore_coding_workspace",
    "restore_coding_workspace_from_disk",
    "ensure_coding_workspace_loaded",
    "is_ancestor_root",
    "lowest_common_ancestor",
    "is_safe_umbrella_root",
    "select_coding_root_for_file",
    "maybe_umbrella_coding_roots",
    "register_coding_root",
    "pin_coding_workspace",
    "clear_all_coding_workspaces",
]
