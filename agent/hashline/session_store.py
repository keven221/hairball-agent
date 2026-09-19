"""Per-task SnapshotStore registry for the live content-hash gate."""

from __future__ import annotations

import threading
from typing import Dict

from agent.hashline.snapshots import InMemorySnapshotStore, SnapshotStore

_lock = threading.Lock()
_stores: Dict[str, SnapshotStore] = {}


def get_snapshot_store(task_id: str | None) -> SnapshotStore:
    """Return (and lazily create) the in-memory store for *task_id*."""
    key = str(task_id or "default")
    with _lock:
        store = _stores.get(key)
        if store is None:
            store = InMemorySnapshotStore()
            _stores[key] = store
        return store


def clear_snapshot_store(task_id: str | None = None) -> None:
    """Drop one task store, or all stores when *task_id* is None."""
    with _lock:
        if task_id is None:
            _stores.clear()
            return
        _stores.pop(str(task_id or "default"), None)
