"""Product-neutral recovery warnings for Hairball file edits."""

from __future__ import annotations

from typing import Optional

from agent.hashline.snapshots import SnapshotStore

# Keep the warning contract stable for downstream result classifiers.
RECOVERY_EXTERNAL_WARNING = (
    "Recovered from a stale file hash using a previous read snapshot "
    "(file changed externally between read and edit)."
)

RECOVERY_SESSION_CHAIN_WARNING = (
    "Recovered from a stale file hash using an earlier in-session snapshot "
    "(a prior edit in this session advanced the hash)."
)

RECOVERY_LINE_REMAP_WARNING = (
    "Recovered by remapping stale line anchors to unchanged current lines "
    "(file changed since the tagged read). Verify the diff matches your intent."
)


def recovery_warning_for(
    store: SnapshotStore,
    path: str,
    expected_hash: str,
) -> Optional[str]:
    """Pick the external-change or in-session-chain warning."""
    snap = store.by_hash(path, expected_hash)
    if snap is None:
        return None
    if store.head(path) is snap:
        return RECOVERY_EXTERNAL_WARNING
    return RECOVERY_SESSION_CHAIN_WARNING
