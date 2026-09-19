"""Per-session Hairball snapshot store binding file tags to full text."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from agent.hashline.content_hash import compute_file_hash

DEFAULT_MAX_PATHS = 30
DEFAULT_MAX_VERSIONS_PER_PATH = 4
DEFAULT_MAX_TOTAL_BYTES = 64 * 1024 * 1024


@dataclass
class Snapshot:
    """One full-file version observed at a point in time."""

    path: str
    text: str
    hash: str
    recorded_at: float = field(default_factory=lambda: time.time() * 1000.0)
    seen_lines: Optional[set[int]] = None


def _merge_seen_lines(snapshot: Snapshot, lines: Optional[Iterable[int]]) -> None:
    if lines is None:
        return
    if snapshot.seen_lines is None:
        snapshot.seen_lines = set()
    for line in lines:
        snapshot.seen_lines.add(int(line))


class SnapshotStore:
    """Storage seam for full-file version snapshots (abstract)."""

    def head(self, path: str) -> Optional[Snapshot]:
        raise NotImplementedError

    def by_hash(self, path: str, content_hash: str) -> Optional[Snapshot]:
        raise NotImplementedError

    def by_content(self, path: str, full_text: str) -> Optional[Snapshot]:
        raise NotImplementedError

    def find_by_hash(self, content_hash: str) -> list[Snapshot]:
        return []

    def record(
        self,
        path: str,
        full_text: str,
        seen_lines: Optional[Iterable[int]] = None,
    ) -> str:
        raise NotImplementedError

    def record_seen_lines(
        self, path: str, content_hash: str, lines: Iterable[int]
    ) -> None:
        raise NotImplementedError

    def invalidate(self, path: str) -> None:
        raise NotImplementedError

    def relocate(self, from_path: str, to_path: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class InMemorySnapshotStore(SnapshotStore):
    """In-memory SnapshotStore with path LRU + per-path version ring."""

    def __init__(
        self,
        *,
        max_paths: int = DEFAULT_MAX_PATHS,
        max_versions_per_path: int = DEFAULT_MAX_VERSIONS_PER_PATH,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        self._versions: OrderedDict[str, list[Snapshot]] = OrderedDict()
        self._max_paths = max(1, int(max_paths))
        self._max_versions_per_path = max(1, int(max_versions_per_path))
        self._max_total_bytes = max(1, int(max_total_bytes))

    def _touch(self, path: str) -> None:
        if path in self._versions:
            self._versions.move_to_end(path)

    def _history_size(self, history: list[Snapshot]) -> int:
        total = 1
        for version in history:
            total += len(version.text)
        return total

    def _total_size(self) -> int:
        return sum(self._history_size(h) for h in self._versions.values())

    def _evict(self) -> None:
        while len(self._versions) > self._max_paths:
            self._versions.popitem(last=False)
        while self._total_size() > self._max_total_bytes and self._versions:
            self._versions.popitem(last=False)

    def head(self, path: str) -> Optional[Snapshot]:
        history = self._versions.get(path)
        if not history:
            return None
        self._touch(path)
        return history[0]

    def by_hash(self, path: str, content_hash: str) -> Optional[Snapshot]:
        needle = str(content_hash or "").strip().upper()
        history = self._versions.get(path)
        if not history:
            return None
        self._touch(path)
        for version in history:
            if version.hash == needle:
                return version
        return None

    def by_content(self, path: str, full_text: str) -> Optional[Snapshot]:
        history = self._versions.get(path)
        if not history:
            return None
        self._touch(path)
        for version in history:
            if version.text == full_text:
                return version
        return None

    def find_by_hash(self, content_hash: str) -> list[Snapshot]:
        needle = str(content_hash or "").strip().upper()
        matches: list[Snapshot] = []
        for history in self._versions.values():
            for version in history:
                if version.hash == needle:
                    matches.append(version)
        return matches

    def record(
        self,
        path: str,
        full_text: str,
        seen_lines: Optional[Iterable[int]] = None,
    ) -> str:
        content_hash = compute_file_hash(full_text)
        history = list(self._versions.get(path) or [])
        existing = next(
            (
                version
                for version in history
                if version.hash == content_hash and version.text == full_text
            ),
            None,
        )
        if existing is not None:
            existing.recorded_at = time.time() * 1000.0
            _merge_seen_lines(existing, seen_lines)
            if not history or history[0] is not existing:
                history = [existing, *[v for v in history if v is not existing]]
            self._versions[path] = history[: self._max_versions_per_path]
            self._touch(path)
            self._evict()
            return content_hash

        snapshot = Snapshot(
            path=path,
            text=full_text,
            hash=content_hash,
            recorded_at=time.time() * 1000.0,
        )
        _merge_seen_lines(snapshot, seen_lines)
        history = [snapshot, *history][: self._max_versions_per_path]
        self._versions[path] = history
        self._touch(path)
        self._evict()
        return content_hash

    def record_seen_lines(
        self, path: str, content_hash: str, lines: Iterable[int]
    ) -> None:
        version = self.by_hash(path, content_hash)
        if version is not None:
            _merge_seen_lines(version, lines)

    def invalidate(self, path: str) -> None:
        self._versions.pop(path, None)

    def relocate(self, from_path: str, to_path: str) -> None:
        source = self._versions.get(from_path)
        if not source:
            return
        relocated = [
            Snapshot(
                path=to_path,
                text=v.text,
                hash=v.hash,
                recorded_at=v.recorded_at,
                seen_lines=set(v.seen_lines) if v.seen_lines else None,
            )
            for v in source
        ]
        dest = self._versions.get(to_path)
        if dest is None:
            self._versions[to_path] = relocated[: self._max_versions_per_path]
        else:
            seen: set[str] = set()
            merged: list[Snapshot] = []
            for version in [*relocated, *dest]:
                if version.hash in seen:
                    continue
                seen.add(version.hash)
                merged.append(version)
            self._versions[to_path] = merged[: self._max_versions_per_path]
        self._versions.pop(from_path, None)
        self._touch(to_path)
        self._evict()

    def clear(self) -> None:
        self._versions.clear()
