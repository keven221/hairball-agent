"""Turn-scoped workspace truth for coding completion.

This is Hairball's equivalent of Codex's ``TurnDiffTracker``.  Structured file
tools remain the fast path, but the final truth comes from the original Git
workspace so edits made through ``terminal``, IDE/LSP adapters, or scripts are
not lost.  The tracker is deliberately local and fail-open: remote/non-Git
workspaces keep their existing behavior instead of being guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import threading
from typing import Any, Sequence


_GIT_TIMEOUT_SECONDS = 4
_MAX_TRACKED_DIRTY_PATHS = 2000
_MAX_HASH_BYTES = 64 * 1024 * 1024
_TRACKER_LOCK = threading.RLock()

@dataclass(frozen=True)
class TurnWorkspaceState:
    root: str
    fingerprints: tuple[tuple[str, str], ...]
    digest: str
    complete: bool = True

    def as_map(self) -> dict[str, str]:
        return dict(self.fingerprints)


@dataclass(frozen=True)
class TurnWorkspaceOutcome:
    available: bool
    root: str = ""
    model_root: str = ""
    changed_paths: tuple[str, ...] = ()
    newly_changed_paths: tuple[str, ...] = ()
    digest: str = ""


def _git_bytes(cwd: Path, *args: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _git_root(cwd: str | os.PathLike[str] | None) -> Path | None:
    raw = str(cwd or "").strip()
    if not raw:
        return None
    try:
        candidate = Path(raw).expanduser()
        if not candidate.is_dir():
            return None
        output = _git_bytes(candidate, "rev-parse", "--show-toplevel")
        if output is None:
            return None
        root = Path(output.decode("utf-8", errors="replace").strip()).resolve()
        return root if root.is_dir() else None
    except (OSError, RuntimeError, ValueError):
        return None


def _default_workspace_cwd(agent: Any) -> str:
    for value in (
        getattr(agent, "_turn_workspace_cwd", None),
        os.environ.get("TERMINAL_CWD"),
        getattr(agent, "cwd", None),
    ):
        raw = str(value or "").strip()
        if raw and raw.lower() not in {"auto", "cwd", ".", "./"}:
            return raw
    try:
        return os.getcwd()
    except OSError:
        return ""


def _default_model_root(root: Path) -> str:
    env_type = str(os.environ.get("TERMINAL_ENV") or "local").casefold()
    mounted = str(
        os.environ.get("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE") or "false"
    ).casefold() in {"1", "true", "yes", "on"}
    if env_type == "docker" and mounted:
        return "/workspace"
    return str(root)


def _nul_paths(payload: bytes | None) -> list[str]:
    if not payload:
        return []
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in payload.split(b"\0")
        if item
    ]


def _dirty_paths(root: Path) -> tuple[list[str], bool]:
    tracked = _git_bytes(root, "diff", "--name-only", "-z", "HEAD", "--")
    if tracked is None:
        # An unborn repository has no HEAD. All indexed and untracked files are
        # its current state, and the same comparison still works turn-to-turn.
        tracked = _git_bytes(root, "ls-files", "-z")
        if tracked is None:
            return [], False
    untracked = _git_bytes(root, "ls-files", "--others", "--exclude-standard", "-z")
    if untracked is None:
        return [], False
    paths = sorted(set(_nul_paths(tracked)) | set(_nul_paths(untracked)))
    if len(paths) > _MAX_TRACKED_DIRTY_PATHS:
        return [], False
    return paths, True


def _path_fingerprint(root: Path, relative: str) -> str:
    path = root / relative
    try:
        stat = path.lstat()
    except OSError:
        return "missing"
    if path.is_symlink():
        try:
            target = os.readlink(path)
        except OSError:
            target = ""
        return f"symlink:{stat.st_mode:o}:{target}"
    if not path.is_file():
        return f"other:{stat.st_mode:o}:{stat.st_size}"
    if stat.st_size > _MAX_HASH_BYTES:
        return f"large:{stat.st_mode:o}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return f"unreadable:{stat.st_mode:o}:{stat.st_size}:{stat.st_mtime_ns}"
    return f"file:{stat.st_mode:o}:{stat.st_size}:{digest.hexdigest()}"


def _capture(root: Path) -> TurnWorkspaceState:
    paths, complete = _dirty_paths(root)
    rows = tuple((path, _path_fingerprint(root, path)) for path in paths)
    body = "\n".join(f"{path}\0{fingerprint}" for path, fingerprint in rows)
    digest = hashlib.sha256(body.encode("utf-8", errors="surrogateescape")).hexdigest()
    return TurnWorkspaceState(
        root=str(root),
        fingerprints=rows,
        digest=digest,
        complete=complete,
    )


def _changed_relative_paths(
    before: TurnWorkspaceState, after: TurnWorkspaceState
) -> tuple[str, ...]:
    if before.root != after.root or not before.complete or not after.complete:
        return ()
    old = before.as_map()
    new = after.as_map()
    return tuple(
        sorted(path for path in set(old) | set(new) if old.get(path) != new.get(path))
    )


def _relative_identity(raw: str, *, root: Path, model_root: str) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    normalized_model_root = str(PurePosixPath(model_root or "/workspace"))
    try:
        posix = PurePosixPath(text)
        model = PurePosixPath(normalized_model_root)
        if posix.is_absolute() and (posix == model or model in posix.parents):
            return str(posix.relative_to(model))
    except (TypeError, ValueError):
        pass
    try:
        host = Path(text).expanduser()
        if host.is_absolute():
            return str(host.resolve(strict=False).relative_to(root))
    except (OSError, RuntimeError, ValueError):
        return None
    if not os.path.isabs(text):
        normalized = os.path.normpath(text)
        if normalized not in {"", "."} and not normalized.startswith(".." + os.sep):
            return normalized.replace(os.sep, "/")
    return None


def begin_turn_workspace_tracking(
    agent: Any,
    *,
    cwd: str | os.PathLike[str] | None = None,
    model_root: str | None = None,
) -> TurnWorkspaceOutcome:
    """Capture the immutable per-turn baseline for a local Git workspace."""
    root = _git_root(cwd or _default_workspace_cwd(agent))
    if root is None:
        agent._turn_workspace_baseline = None
        agent._turn_workspace_latest = None
        agent._turn_workspace_outcome = TurnWorkspaceOutcome(available=False)
        return agent._turn_workspace_outcome
    state = _capture(root)
    if not state.complete:
        agent._turn_workspace_baseline = None
        agent._turn_workspace_latest = None
        agent._turn_workspace_outcome = TurnWorkspaceOutcome(available=False)
        return agent._turn_workspace_outcome
    outcome = TurnWorkspaceOutcome(
        available=True,
        root=str(root),
        model_root=str(model_root or _default_model_root(root)),
        digest=state.digest,
    )
    agent._turn_workspace_baseline = state
    agent._turn_workspace_latest = state
    agent._turn_workspace_outcome = outcome
    return outcome


def _reconcile_failed_mutations(
    agent: Any,
    *,
    root: Path,
    model_root: str,
    changed_relative: Sequence[str],
) -> None:
    failed = getattr(agent, "_turn_failed_file_mutations", None)
    if not isinstance(failed, dict) or not failed:
        return
    changed = {str(PurePosixPath(path)) for path in changed_relative}
    for raw in list(failed):
        identity = _relative_identity(str(raw), root=root, model_root=model_root)
        if identity is not None and str(PurePosixPath(identity)) in changed:
            failed.pop(raw, None)


def reconcile_turn_workspace(
    agent: Any,
    *,
    mark_stale: bool = True,
) -> TurnWorkspaceOutcome:
    """Refresh actual workspace state and feed existing verification rails.

    The function is idempotent and safe to call after every tool plus again at
    Stop/finalization.  ``changed_paths`` is always relative to the immutable
    turn baseline; ``newly_changed_paths`` is only the delta since the previous
    observation, which preserves verification ordering.
    """
    with _TRACKER_LOCK:
        baseline = getattr(agent, "_turn_workspace_baseline", None)
        latest = getattr(agent, "_turn_workspace_latest", None)
        prior_outcome = getattr(agent, "_turn_workspace_outcome", None)
        if not isinstance(baseline, TurnWorkspaceState) or not isinstance(
            latest, TurnWorkspaceState
        ):
            return TurnWorkspaceOutcome(available=False)
        root = Path(baseline.root)
        current = _capture(root)
        if not current.complete:
            outcome = TurnWorkspaceOutcome(available=False)
            agent._turn_workspace_outcome = outcome
            return outcome
        changed_relative = _changed_relative_paths(baseline, current)
        newly_relative = _changed_relative_paths(latest, current)
        changed_paths = tuple(str(root / path) for path in changed_relative)
        newly_changed_paths = tuple(str(root / path) for path in newly_relative)
        changed_from_baseline = set(changed_relative)
        verification_stale_paths = tuple(
            str(root / path)
            for path in newly_relative
            if path in changed_from_baseline
        )
        model_root = (
            prior_outcome.model_root
            if isinstance(prior_outcome, TurnWorkspaceOutcome)
            else _default_model_root(root)
        )
        outcome = TurnWorkspaceOutcome(
            available=True,
            root=str(root),
            model_root=model_root,
            changed_paths=changed_paths,
            newly_changed_paths=newly_changed_paths,
            digest=current.digest,
        )
        agent._turn_workspace_latest = current
        agent._turn_workspace_outcome = outcome
        observed = getattr(agent, "_turn_file_mutation_paths", None)
        if isinstance(observed, set):
            observed.update(changed_paths)
        _reconcile_failed_mutations(
            agent,
            root=root,
            model_root=model_root,
            changed_relative=changed_relative,
        )

        if verification_stale_paths and mark_stale:
            try:
                from agent.verification_evidence import mark_workspace_edited

                mark_workspace_edited(
                    session_id=getattr(agent, "session_id", None),
                    cwd=root,
                    paths=list(verification_stale_paths),
                )
            except Exception:
                pass
        if newly_changed_paths:
            try:
                runtime = getattr(agent, "_completion_contract_runtime", None)
                recorder = getattr(runtime, "record_workspace_observation", None)
                if callable(recorder):
                    recorder(
                        root=str(root),
                        changed_paths=newly_changed_paths,
                        state_digest=current.digest,
                    )
            except Exception:
                pass
        return outcome


def turn_workspace_changed_paths(agent: Any) -> tuple[str, ...]:
    """Return final observed paths, falling back for remote/non-Git sessions."""
    outcome = reconcile_turn_workspace(agent)
    if outcome.available:
        return outcome.changed_paths
    observed = getattr(agent, "_turn_file_mutation_paths", set()) or set()
    return tuple(sorted(str(path) for path in observed if str(path).strip()))


__all__ = [
    "TurnWorkspaceOutcome",
    "TurnWorkspaceState",
    "begin_turn_workspace_tracking",
    "reconcile_turn_workspace",
    "turn_workspace_changed_paths",
]
