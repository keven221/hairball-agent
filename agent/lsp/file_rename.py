"""Filesystem rename paired with LSP ``willRenameFiles`` / ``didRenameFiles``.

Contract mirrors oh-my-pi ``enumerateRenamePairs`` + ``rename_file`` action
at the I/O level (single primary server per Hairball spawn key).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agent.lsp.client import file_uri

MAX_RENAME_PAIRS = 1000


def enumerate_rename_pairs(
    source: str,
    dest: str,
) -> Tuple[List[Dict[str, str]], bool, bool]:
    """Return ``(pairs, is_directory, exceeded)``.

    Each pair is ``{"oldUri": ..., "newUri": ...}``.
    """
    src = Path(os.path.abspath(source))
    dst = Path(os.path.abspath(dest))
    if not src.exists():
        return [], False, False
    if src.is_file():
        return (
            [{"oldUri": file_uri(str(src)), "newUri": file_uri(str(dst))}],
            False,
            False,
        )
    pairs: List[Dict[str, str]] = []
    for root, _dirs, files in os.walk(src):
        for name in files:
            if len(pairs) >= MAX_RENAME_PAIRS:
                return pairs, True, True
            abs_old = os.path.join(root, name)
            rel = os.path.relpath(abs_old, str(src))
            abs_new = os.path.join(str(dst), rel)
            pairs.append({
                "oldUri": file_uri(abs_old),
                "newUri": file_uri(abs_new),
            })
    return pairs, True, False


def format_code_action(action: Dict[str, Any], index: int) -> str:
    """One-line listing for ``code_actions`` list mode (OMP formatCodeAction)."""
    if "command" in action and isinstance(action.get("command"), str):
        kind = "command"
        title = action.get("title") or action.get("command") or "?"
        preferred = ""
        disabled = ""
    else:
        kind = action.get("kind") or "action"
        title = action.get("title") or "?"
        preferred = " (preferred)" if action.get("isPreferred") else ""
        disabled_obj = action.get("disabled")
        if isinstance(disabled_obj, dict) and disabled_obj.get("reason"):
            disabled = f" (disabled: {disabled_obj['reason']})"
        else:
            disabled = ""
    return f"{index}: [{kind}] {title}{preferred}{disabled}"
