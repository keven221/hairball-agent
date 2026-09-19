"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})


# Tools whose interrupted/dangling execution is safe to discard because they
# cannot mutate either external state or Hairball session state. Unknown/plugin/
# MCP tools stay effect-capable by default.
NO_EFFECT_TOOL_NAMES = frozenset({
    "read_file", "search_files", "session_search", "skill_view", "skills_list",
    "web_extract", "web_search", "vision_analyze", "browser_snapshot",
    "browser_get_images", "browser_console", "read_terminal",
})


def tool_may_have_side_effect(tool_name: str) -> bool:
    return tool_name not in NO_EFFECT_TOOL_NAMES


def file_mutation_result_landed(tool_name: str, result: Any) -> bool:
    """Return True when a file mutation result proves the write landed."""
    if tool_name not in FILE_MUTATING_TOOL_NAMES:
        return False
    data = tool_result_mapping(result)
    if data is None or data.get("error"):
        return False
    if tool_name == "write_file":
        return "bytes_written" in data
    if tool_name == "patch":
        return data.get("success") is True
    return False


def tool_result_mapping(result: Any) -> Mapping[str, Any] | None:
    """Decode a canonical structured tool result without display prose."""
    if isinstance(result, Mapping):
        return result
    if not isinstance(result, str):
        return None
    try:
        data = json.loads(result.strip())
    except Exception:
        return None
    return data if isinstance(data, Mapping) else None


def workspace_mutation_paths(tool_name: str, result: Any) -> tuple[str, ...]:
    """Return structured workspace paths reported by a successful edit tool.

    This is the common mutation waist for verification, completion contracts,
    and turn-diff reconciliation.  Arbitrary plugin JSON is not trusted: only
    Hairball's known editing tools and their existing success markers qualify.
    Terminal/execute-code mutations are discovered by ``TurnWorkspaceTracker``
    from the actual workspace rather than inferred from command text.
    """
    data = tool_result_mapping(result)
    if data is None or data.get("error"):
        return ()
    name = str(tool_name or "")
    landed = (
        (name == "write_file" and "bytes_written" in data)
        or (name == "patch" and data.get("success") is True)
        or (name == "ast_edit" and data.get("applied") is True)
        or (name == "lsp" and data.get("applied") is True)
    )
    if not landed:
        return ()
    paths: list[str] = []
    for key in ("files_modified", "files_created", "files_deleted", "files_touched"):
        value = data.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            paths.extend(str(path) for path in value if path)
    for key in ("resolved_path", "file"):
        value = data.get(key)
        if value:
            paths.append(str(value))
    return tuple(dict.fromkeys(paths))


def classify_side_effect_outcome(
    tool_name: str,
    result: Any,
    *,
    failed: bool,
    cancelled: bool = False,
    timed_out: bool = False,
) -> str:
    """Return the conservative durable outcome for one dispatched effect.

    Only failures whose implementation proves atomic non-commit are
    retryable.  Unknown/plugin/MCP/remote failures, interruption, and timeout
    remain ambiguous because the effect may have landed before the response
    was lost.
    """

    if cancelled or timed_out:
        return "unknown"
    if not failed:
        return "succeeded"
    try:
        text = result if isinstance(result, str) else json.dumps(result, default=str)
    except (TypeError, ValueError):
        text = str(result)
    lowered = str(text or "").casefold()
    if "rollback was incomplete" in lowered:
        return "unknown"
    if str(tool_name or "") == "write_file" and any(
        marker in lowered
        for marker in (
            "failed to write file:",
            "content hash mismatch",
            "modified since you last read",
            "refusing to write",
        )
    ):
        return "failed_retryable"
    if str(tool_name or "") == "patch" and any(
        marker in lowered
        for marker in (
            "failed batch did not commit",
            "old_string not found",
            "content hash mismatch",
            "modified since you last read",
            "no changes were applied",
        )
    ):
        return "failed_retryable"
    return "unknown"
