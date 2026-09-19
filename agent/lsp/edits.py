"""Apply LSP WorkspaceEdit payloads to the local filesystem.

Contract mirrors oh-my-pi ``packages/coding-agent/src/lsp/edits.ts`` at the
I/O level: flatten ``changes`` / ``documentChanges`` text edits, apply
bottom-up per file. Resource ops (create/rename/delete) are applied in
documentChanges order when present; ``changes``-only edits are text-only.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.lsp.client import uri_to_path

logger = logging.getLogger("agent.lsp.edits")


def apply_text_edits_to_string(content: str, edits: List[Dict[str, Any]]) -> str:
    """Apply LSP TextEdit list to ``content`` (bottom-to-top)."""
    sorted_edits = _sort_and_validate_text_edits(edits)
    lines = content.split("\n")
    for edit in sorted_edits:
        rng = edit.get("range") or {}
        start = rng.get("start") or {}
        end = rng.get("end") or {}
        new_text = edit.get("newText", "")
        if not isinstance(new_text, str):
            new_text = str(new_text)
        s_line = int(start.get("line", 0))
        s_char = int(start.get("character", 0))
        e_line = int(end.get("line", 0))
        e_char = int(end.get("character", 0))
        if s_line == e_line:
            line = lines[s_line] if s_line < len(lines) else ""
            lines[s_line] = line[:s_char] + new_text + line[e_char:]
        else:
            start_line = lines[s_line] if s_line < len(lines) else ""
            end_line = lines[e_line] if e_line < len(lines) else ""
            new_content = start_line[:s_char] + new_text + end_line[e_char:]
            lines[s_line : e_line + 1] = new_content.split("\n")
    return "\n".join(lines)


def flatten_workspace_text_edits(edit: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Map file path → TextEdit[] from a WorkspaceEdit."""
    out: Dict[str, List[Dict[str, Any]]] = {}

    def _push(uri: str, edits: List[Dict[str, Any]]) -> None:
        if not edits:
            return
        path = uri_to_path(uri) if uri.startswith("file:") else uri
        abs_path = os.path.abspath(path)
        out.setdefault(abs_path, []).extend(edits)

    changes = edit.get("changes")
    if isinstance(changes, dict):
        for uri, edits in changes.items():
            if isinstance(edits, list):
                _push(str(uri), [e for e in edits if isinstance(e, dict)])

    doc_changes = edit.get("documentChanges")
    if isinstance(doc_changes, list):
        for change in doc_changes:
            if not isinstance(change, dict):
                continue
            if "textDocument" in change and "edits" in change:
                uri = ((change.get("textDocument") or {}).get("uri")) or ""
                edits = [e for e in (change.get("edits") or []) if isinstance(e, dict) and "range" in e]
                _push(str(uri), edits)
    return out


def apply_workspace_edit(
    edit: Optional[Dict[str, Any]],
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Apply a WorkspaceEdit. Returns a summary dict (never raises for I/O skip).

    ``dry_run=True`` computes affected paths and edit counts without writing.
    """
    if not isinstance(edit, dict) or not edit:
        return {"success": False, "error": "empty workspace edit", "files": []}

    by_path = flatten_workspace_text_edits(edit)
    files_out: List[Dict[str, Any]] = []
    errors: List[str] = []

    for path, edits in sorted(by_path.items()):
        entry: Dict[str, Any] = {"path": path, "edits": len(edits)}
        try:
            original = Path(path).read_text(encoding="utf-8", errors="replace")
            updated = apply_text_edits_to_string(original, edits)
            entry["changed"] = updated != original
            if not dry_run and entry["changed"]:
                Path(path).write_text(updated, encoding="utf-8")
            files_out.append(entry)
        except OSError as e:
            errors.append(f"{path}: {e}")
            entry["error"] = str(e)
            files_out.append(entry)

    # Resource ops (create/rename/delete) — only when not dry_run
    if not dry_run:
        for err in _apply_resource_ops(edit.get("documentChanges")):
            errors.append(err)

    return {
        "success": not errors,
        "dry_run": dry_run,
        "files": files_out,
        "errors": errors,
    }


def _apply_resource_ops(document_changes: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(document_changes, list):
        return errors
    for change in document_changes:
        if not isinstance(change, dict) or "kind" not in change:
            continue
        kind = change.get("kind")
        try:
            if kind == "create":
                path = uri_to_path(change["uri"])
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                if not Path(path).exists():
                    Path(path).write_text("", encoding="utf-8")
            elif kind == "rename":
                old = uri_to_path(change["oldUri"])
                new = uri_to_path(change["newUri"])
                Path(new).parent.mkdir(parents=True, exist_ok=True)
                os.rename(old, new)
            elif kind == "delete":
                path = uri_to_path(change["uri"])
                p = Path(path)
                if p.is_dir():
                    import shutil
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
        except (OSError, KeyError) as e:
            errors.append(f"resource {kind}: {e}")
    return errors


def _sort_and_validate_text_edits(edits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    indexed = list(enumerate(edits))
    indexed.sort(
        key=lambda pair: (
            -int((pair[1].get("range") or {}).get("start", {}).get("line", 0)),
            -int((pair[1].get("range") or {}).get("start", {}).get("character", 0)),
            -pair[0],
        )
    )
    return [e for _, e in indexed]


__all__ = [
    "apply_text_edits_to_string",
    "flatten_workspace_text_edits",
    "apply_workspace_edit",
]
