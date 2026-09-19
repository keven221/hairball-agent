"""Fail-closed replace recovery over Hairball's per-session snapshots.

Recovery proceeds only when the replace target remains unambiguous on live
disk and was present in the tagged snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent.hashline.messages import recovery_warning_for
from agent.hashline.snapshots import SnapshotStore


@dataclass(frozen=True)
class RecoverDecision:
    """Successful replace-mode recovery; caller may proceed with the patch."""

    warning: str
    expected_hash: str
    snapshot_count: int
    live_count: int


def _count_occurrences(haystack: str, needle: str) -> int:
    """Non-overlapping occurrence count (matches ``str.replace`` semantics)."""
    if not needle:
        return 0
    count = 0
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        count += 1
        start = idx + len(needle)
    return count


def try_recover_replace(
    *,
    store: SnapshotStore,
    path: str,
    expected_hash: str,
    old_string: str,
    live_text: str,
    replace_all: bool = False,
) -> Optional[RecoverDecision]:
    """Attempt fail-closed recovery for a stale-hash replace patch.

    Returns ``None`` when recovery is unsafe — caller must surface mismatch.
    """
    expected = str(expected_hash or "").strip().upper()
    if not expected or old_string is None or old_string == "":
        return None
    snap = store.by_hash(path, expected)
    if snap is None:
        return None
    snap_count = _count_occurrences(snap.text, old_string)
    if snap_count < 1:
        return None
    live_count = _count_occurrences(live_text, old_string)
    if replace_all:
        if live_count < 1 or live_count != snap_count:
            return None
    elif live_count != 1:
        return None
    warning = recovery_warning_for(store, path, expected)
    if not warning:
        return None
    return RecoverDecision(
        warning=warning,
        expected_hash=expected,
        snapshot_count=snap_count,
        live_count=live_count,
    )
