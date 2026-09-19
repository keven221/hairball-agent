"""File fingerprints, snapshots, stable line anchors, and edit recovery."""

from agent.hashline.anchors import (
    AnchorConfig,
    AnchorEditError,
    AnchorEditResult,
    AnchoredPage,
    DEFAULT_CONFIG,
    LineAnchor,
    apply_anchor_edits,
    build_line_anchors,
    format_anchor,
    normalize_anchor_line,
    render_anchored_page,
)

from agent.hashline.content_hash import (
    CONTENT_HASH_LENGTH,
    ContentHashDecision,
    compute_file_hash,
    evaluate_content_hash,
    format_file_tag,
    hash_file_on_disk,
    normalize_file_hash_text,
    resolve_content_hash_mode,
)
from agent.hashline.messages import (
    RECOVERY_EXTERNAL_WARNING,
    RECOVERY_LINE_REMAP_WARNING,
    RECOVERY_SESSION_CHAIN_WARNING,
    recovery_warning_for,
)
from agent.hashline.mismatch import format_mismatch_message, rejection_header
from agent.hashline.replace_recovery import RecoverDecision, try_recover_replace
from agent.hashline.session_store import clear_snapshot_store, get_snapshot_store
from agent.hashline.snapshots import InMemorySnapshotStore, Snapshot, SnapshotStore

__all__ = [
    "AnchorConfig",
    "AnchorEditError",
    "AnchorEditResult",
    "AnchoredPage",
    "CONTENT_HASH_LENGTH",
    "ContentHashDecision",
    "DEFAULT_CONFIG",
    "InMemorySnapshotStore",
    "LineAnchor",
    "RECOVERY_EXTERNAL_WARNING",
    "RECOVERY_LINE_REMAP_WARNING",
    "RECOVERY_SESSION_CHAIN_WARNING",
    "RecoverDecision",
    "Snapshot",
    "SnapshotStore",
    "apply_anchor_edits",
    "build_line_anchors",
    "clear_snapshot_store",
    "compute_file_hash",
    "evaluate_content_hash",
    "format_file_tag",
    "format_anchor",
    "format_mismatch_message",
    "get_snapshot_store",
    "hash_file_on_disk",
    "normalize_file_hash_text",
    "normalize_anchor_line",
    "recovery_warning_for",
    "rejection_header",
    "resolve_content_hash_mode",
    "render_anchored_page",
    "try_recover_replace",
]
