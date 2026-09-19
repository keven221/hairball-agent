"""Model-facing LSP tool — navigation, symbols, diagnostics, rename.

Contracts align with oh-my-pi ``lsp`` tool actions (W1 + W1.5):
diagnostics / definition / type_definition / implementation / references /
hover / symbols / status / capabilities / rename / rename_file /
code_actions / reload / request.

Wire RPCs live in ``agent.lsp``; this module is schema + position resolve +
WorkspaceEdit apply glue. Hidden when ``lsp.enabled`` or ``lsp.model_tools``
is false (check_fn).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import registry, tool_error

_READ_ACTIONS = frozenset({
    "diagnostics",
    "definition",
    "type_definition",
    "implementation",
    "references",
    "hover",
    "symbols",
    "status",
    "capabilities",
    "code_actions",  # list mode is read; apply=true is write-tier
    "reload",
    "restart",
    "workspace",
})
_WRITE_ACTIONS = frozenset({"rename", "rename_file", "request", "workspace_set"})
_ALL_ACTIONS = sorted(_READ_ACTIONS | _WRITE_ACTIONS)

# Bare identifier → require word boundaries (OMP findSymbolMatchIndexes).
_BARE_IDENTIFIER_RE = re.compile(r"^[$A-Za-z_][\w$]*$")
_IDENTIFIER_CHAR_RE = re.compile(r"[A-Za-z0-9_$]")
# Prefer declaration-looking hits when symbol is used without an explicit line.
_DECL_LINE_RE = re.compile(
    r"(?m)^[ \t]*(?:async[ \t]+)?(?:def|class|function|const|let|var|fn|pub[ \t]+fn|"
    r"export[ \t]+(?:async[ \t]+)?function|export[ \t]+(?:const|class|type|interface))\b"
)
_SYMBOL_SPEC_RE = re.compile(r"^(.+)#(\d+)$")


def _parse_symbol_spec(spec: str) -> Tuple[str, int]:
    """Parse ``name`` or ``name#N`` (N = 1-based occurrence). Mirrors OMP parseSymbolSpec."""
    m = _SYMBOL_SPEC_RE.match(spec)
    if not m:
        return spec, 1
    return m.group(1), max(1, int(m.group(2)))


def _find_symbol_indexes(line_text: str, symbol: str, *, case_insensitive: bool = False) -> List[int]:
    """Indexes of ``symbol`` in ``line_text``; bare identifiers use word boundaries."""
    if not symbol:
        return []
    haystack = line_text.lower() if case_insensitive else line_text
    needle = symbol.lower() if case_insensitive else symbol
    require_boundary = bool(_BARE_IDENTIFIER_RE.match(symbol))
    indexes: List[int] = []
    from_index = 0
    while from_index <= len(haystack) - len(needle):
        match_index = haystack.find(needle, from_index)
        if match_index < 0:
            break
        if require_boundary:
            before = haystack[match_index - 1] if match_index > 0 else ""
            after_idx = match_index + len(needle)
            after = haystack[after_idx] if after_idx < len(haystack) else ""
            if (before and _IDENTIFIER_CHAR_RE.match(before)) or (
                after and _IDENTIFIER_CHAR_RE.match(after)
            ):
                from_index = match_index + 1
                continue
        indexes.append(match_index)
        from_index = match_index + len(needle)
    return indexes


def _hit_declaration_score(line_text: str, symbol: str, char: int) -> int:
    """Higher score = better default pick for definition-oriented lookups."""
    score = 0
    if _DECL_LINE_RE.match(line_text or ""):
        score += 10
    # Prefer the identifier that starts the declared name near def/class …
    stripped = (line_text or "").lstrip()
    if stripped.startswith(("def ", "async def ", "class ", "function ")):
        score += 5
    # Exact token at char (already boundary-matched)
    if 0 <= char < len(line_text) and line_text[char : char + len(symbol)] == symbol:
        score += 1
    return score


def _resolve_position(
    file_path: str,
    line: Optional[int],
    symbol: Optional[str],
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Return (0-based line, 0-based character, error).

    Symbol matching follows oh-my-pi: bare identifiers require word boundaries
    so ``lsp_tool`` does not land inside ``check_lsp_tool_requirements``.
    """
    path = Path(file_path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return None, None, f"cannot read {file_path}: {e}"

    lines = text.split("\n")

    if symbol:
        explicit_occurrence = bool(_SYMBOL_SPEC_RE.match(symbol))
        needle, match_index = _parse_symbol_spec(symbol)
        hits: List[Tuple[int, int, int]] = []  # line, char, score

        def _collect(line_idx: int, line_text: str) -> None:
            indexes = _find_symbol_indexes(line_text, needle)
            if not indexes:
                indexes = _find_symbol_indexes(line_text, needle, case_insensitive=True)
            for pos in indexes:
                hits.append((line_idx, pos, _hit_declaration_score(line_text, needle, pos)))

        if line is not None:
            want = int(line) - 1
            if want < 0 or want >= len(lines):
                return None, None, f"line {line} out of range (file has {len(lines)} lines)"
            _collect(want, lines[want])
            if not hits:
                return None, None, f"symbol {needle!r} not found on line {line}"
        else:
            for i, ln in enumerate(lines):
                _collect(i, ln)
            if not hits:
                return None, None, f"symbol {needle!r} not found in {file_path}"

        # Explicit #N → stable file order. Default → prefer declaration hits.
        if explicit_occurrence:
            ordered: List[Tuple[int, int]] = []
            seen = set()
            scan_lines = [int(line) - 1] if line is not None else range(len(lines))
            for i in scan_lines:
                if i < 0 or i >= len(lines):
                    continue
                indexes = _find_symbol_indexes(lines[i], needle)
                if not indexes:
                    indexes = _find_symbol_indexes(lines[i], needle, case_insensitive=True)
                for pos in indexes:
                    key = (i, pos)
                    if key not in seen:
                        seen.add(key)
                        ordered.append(key)
            if match_index > len(ordered):
                return None, None, f"symbol {needle!r} has only {len(ordered)} match(es)"
            hit_line, hit_char = ordered[match_index - 1]
            return hit_line, hit_char, None

        hits.sort(key=lambda h: (-h[2], h[0], h[1]))
        hit_line, hit_char, _score = hits[0]
        return hit_line, hit_char, None

    if line is None:
        return None, None, "line or symbol is required"
    zero = int(line) - 1
    if zero < 0 or zero >= len(lines):
        return None, None, f"line {line} out of range (file has {len(lines)} lines)"
    ln = lines[zero]
    char = len(ln) - len(ln.lstrip())
    return zero, char, None


def _normalize_locations(result: Any) -> List[Dict[str, Any]]:
    if result is None:
        return []
    items = result if isinstance(result, list) else [result]
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        # LocationLink
        if "targetUri" in item:
            uri = item.get("targetUri")
            rng = item.get("targetSelectionRange") or item.get("targetRange") or {}
        else:
            uri = item.get("uri")
            rng = item.get("range") or {}
        start = (rng.get("start") or {}) if isinstance(rng, dict) else {}
        path = uri
        if isinstance(uri, str) and uri.startswith("file:"):
            from agent.lsp.client import uri_to_path
            path = uri_to_path(uri)
        loc = {
            "path": path,
            "line": int(start.get("line", 0)) + 1,
            "character": int(start.get("character", 0)) + 1,
        }
        key = (loc["path"], loc["line"], loc["character"])
        if key in seen:
            continue
        seen.add(key)
        out.append(loc)
    return out


def _format_hover(result: Any) -> str:
    if not isinstance(result, dict):
        return "No hover information"
    contents = result.get("contents")
    if contents is None:
        return "No hover information"
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        return str(contents.get("value") or contents)
    if isinstance(contents, list):
        parts = []
        for c in contents:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict):
                parts.append(str(c.get("value") or c))
        return "\n\n".join(parts) if parts else "No hover information"
    return str(contents)


def _format_symbols(result: Any, *, limit: int = 80) -> str:
    if not result:
        return "No symbols found"
    if not isinstance(result, list):
        return json.dumps(result, ensure_ascii=False, indent=2)[:4000]

    lines: List[str] = []

    def walk(nodes: List[Any], indent: int = 0) -> None:
        for node in nodes:
            if len(lines) >= limit:
                return
            if not isinstance(node, dict):
                continue
            name = node.get("name") or "?"
            kind = node.get("kind")
            loc = node.get("location") or {}
            rng = node.get("range") or loc.get("range") or {}
            start = (rng.get("start") or {}) if isinstance(rng, dict) else {}
            path = loc.get("uri")
            if isinstance(path, str) and path.startswith("file:"):
                from agent.lsp.client import uri_to_path
                path = uri_to_path(path)
            prefix = "  " * indent
            loc_s = ""
            if path:
                loc_s = f" @ {path}:{int(start.get('line', 0)) + 1}"
            elif start:
                loc_s = f" :{int(start.get('line', 0)) + 1}"
            lines.append(f"{prefix}{name} (kind={kind}){loc_s}")
            children = node.get("children")
            if isinstance(children, list):
                walk(children, indent + 1)

    walk(result)
    extra = ""
    if isinstance(result, list) and len(lines) >= limit:
        extra = f"\n… truncated at {limit} symbols"
    return "\n".join(lines) + extra


def _format_diagnostics(diags: Any) -> str:
    if not isinstance(diags, list) or not diags:
        return "No diagnostics"
    from agent.lsp.reporter import format_diagnostic
    parts = []
    for d in diags[:50]:
        if isinstance(d, dict):
            parts.append(format_diagnostic(d))
    if len(diags) > 50:
        parts.append(f"… and {len(diags) - 50} more")
    return "\n".join(parts) if parts else "No diagnostics"


def _apply_code_action(
    svc: Any,
    action: Dict[str, Any],
    *,
    abs_file: str,
    timeout: float,
) -> Optional[Dict[str, Any]]:
    """Apply a CodeAction or Command (OMP applyCodeAction contract)."""
    from agent.lsp.edits import apply_workspace_edit

    # Pure Command item: title + command string
    if isinstance(action.get("command"), str):
        cmd = action["command"]
        args = action.get("arguments") if isinstance(action.get("arguments"), list) else None
        raw = svc.request_sync(
            "execute_command",
            file_path=abs_file,
            query=cmd,
            code_action={"arguments": args} if args is not None else None,
            timeout=timeout,
        )
        if isinstance(raw, dict) and raw.get("error"):
            return {"title": action.get("title") or cmd, "error": raw["error"],
                    "edits": [], "executed_commands": []}
        return {
            "title": action.get("title") or cmd,
            "edits": [],
            "executed_commands": [cmd],
            "output": f'Applied "{action.get("title") or cmd}" (command)',
        }

    resolved = action
    if not resolved.get("edit"):
        resolved_raw = svc.request_sync(
            "code_action_resolve",
            file_path=abs_file,
            code_action=action,
            timeout=timeout,
        )
        if isinstance(resolved_raw, dict) and not resolved_raw.get("error"):
            resolved = resolved_raw

    edit_paths: List[str] = []
    edit_summary = None
    if isinstance(resolved.get("edit"), dict):
        edit_summary = apply_workspace_edit(resolved["edit"], dry_run=False)
        for f in edit_summary.get("files") or []:
            if f.get("changed") or f.get("edits"):
                edit_paths.append(
                    f"{f.get('path')}: {f.get('edits', 0)} edit(s)"
                )

    executed: List[str] = []
    cmd_obj = resolved.get("command")
    if isinstance(cmd_obj, dict) and isinstance(cmd_obj.get("command"), str):
        cmd = cmd_obj["command"]
        args = cmd_obj.get("arguments") if isinstance(cmd_obj.get("arguments"), list) else None
        raw = svc.request_sync(
            "execute_command",
            file_path=abs_file,
            query=cmd,
            code_action={"arguments": args} if args is not None else None,
            timeout=timeout,
        )
        if not (isinstance(raw, dict) and raw.get("error")):
            executed.append(cmd)

    if not edit_paths and not executed:
        return None

    title = resolved.get("title") or action.get("title") or "?"
    summary_lines: List[str] = []
    if edit_paths:
        summary_lines.append("  Workspace edit:")
        summary_lines.extend(f"    {p}" for p in edit_paths)
    if executed:
        summary_lines.append("  Executed command(s):")
        summary_lines.extend(f"    {c}" for c in executed)
    return {
        "title": title,
        "edits": edit_paths,
        "executed_commands": executed,
        "edit": edit_summary,
        "output": f'Applied "{title}":\n' + "\n".join(summary_lines),
    }


LSP_SCHEMA = {
    "name": "lsp",
    "description": (
        "Language-server intelligence: go-to-definition, references, hover, "
        "document/workspace symbols, diagnostics, project-aware rename, "
        "file rename (willRenameFiles), code actions, server reload/restart, and "
        "raw request for extension methods. Prefer this over search_files/"
        "patch for symbol navigation and cross-file renames. Lines are "
        "1-indexed. Bare identifiers match on word boundaries. For rename/"
        "rename_file, set apply=false to preview without writing. For "
        "code_actions, list first; apply=true with query=index or title "
        "substring. For request, query is the LSP method name and optional "
        "payload is JSON params (returns JSON only — does not apply edits). "
        "If a server is marked broken after a spawn failure, use action=reload "
        "or action=restart (file='*' clears all) instead of shelling out to CLI. "
        "action=workspace returns the session coding workspace pin; "
        "action=workspace_set with file=<dir> pins it explicitly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": _ALL_ACTIONS,
                "description": "LSP operation to run.",
            },
            "file": {
                "type": "string",
                "description": (
                    "Path to the source file (required for most actions). "
                    "Use '*' for workspace symbols, reload-all, or restart-all. For "
                    "action=request, selects the language server and opens "
                    "the document when set. For action=workspace_set, the "
                    "directory to pin as the session coding workspace."
                ),
            },
            "line": {
                "type": "integer",
                "description": "1-indexed line number for position-based actions.",
            },
            "symbol": {
                "type": "string",
                "description": (
                    "Identifier to locate (word-boundary match for bare names). "
                    "Optional when line is set. Use suffix #N for the Nth match "
                    "in file order (1-based)."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Workspace symbol search (symbols+file='*'); code action "
                    "kind filter when listing; index/title when applying "
                    "code_actions; or LSP method name for action=request "
                    "(e.g. textDocument/formatting, textDocument/signatureHelp)."
                ),
            },
            "new_name": {
                "type": "string",
                "description": (
                    "New identifier for action=rename, or destination path "
                    "for action=rename_file."
                ),
            },
            "apply": {
                "type": "boolean",
                "description": (
                    "For rename/rename_file/code_actions: true applies "
                    "(default for rename/rename_file); false previews/lists only."
                ),
            },
            "payload": {
                "type": "string",
                "description": (
                    "For action=request: JSON string of raw LSP params. "
                    "When omitted, params are built from file (+ line/symbol "
                    "→ position) or {}."
                ),
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (default 20; clamped 5–300).",
            },
        },
        "required": ["action"],
    },
}


def check_lsp_tool_requirements() -> bool:
    """Expose the tool only when the LSP subsystem and model-tools gate are on."""
    try:
        from hairball_cli.config import load_config
        cfg = load_config()
    except Exception:
        return True
    lsp_cfg = cfg.get("lsp") if isinstance(cfg, dict) else None
    if not isinstance(lsp_cfg, dict):
        return True
    if not bool(lsp_cfg.get("enabled", True)):
        return False
    return bool(lsp_cfg.get("model_tools", True))


def _clamp_timeout(raw: Any) -> float:
    try:
        t = float(raw)
    except (TypeError, ValueError):
        t = 20.0
    return max(5.0, min(300.0, t))


def _resolve_lsp_path(filepath: Optional[str], task_id: str = "default") -> Optional[str]:
    """Resolve ``filepath`` against the task workspace (not bare process cwd).

    Reuses ``tools.file_tools._resolve_path_for_task`` so UI/TUI/ACP
    ``register_task_env_overrides`` pins relative paths the same way as
    read_file/patch. ``file='*'`` is left unchanged.
    """
    if filepath is None:
        return None
    raw = str(filepath).strip()
    if not raw:
        return None
    if raw == "*":
        return "*"
    from tools.file_tools import _resolve_path_for_task

    tid = (task_id or "default").strip() or "default"
    return str(_resolve_path_for_task(raw, tid))


def _format_raw_request_result(result: Any) -> str:
    if result is None:
        return "null"
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        return str(result)


def lsp_tool(
    action: str,
    file: Optional[str] = None,
    line: Optional[int] = None,
    symbol: Optional[str] = None,
    query: Optional[str] = None,
    new_name: Optional[str] = None,
    apply: Optional[bool] = None,
    timeout: Optional[float] = None,
    payload: Optional[str] = None,
    task_id: Optional[str] = None,
    **_kwargs: Any,
) -> str:
    action = (action or "").strip().lower()
    if action not in _ALL_ACTIONS:
        return tool_error(f"unsupported action {action!r}; choose from {', '.join(_ALL_ACTIONS)}")

    t = _clamp_timeout(timeout)
    tid = (task_id or _kwargs.get("task_id") or "default")
    if not isinstance(tid, str) or not tid.strip():
        tid = "default"
    else:
        tid = tid.strip()

    from agent.coding_workspace import coding_task_context

    with coding_task_context(tid):
        return _lsp_tool_body(
            action=action,
            file=file,
            line=line,
            symbol=symbol,
            query=query,
            new_name=new_name,
            apply=apply,
            timeout=t,
            payload=payload,
            tid=tid,
        )


def _lsp_tool_body(
    *,
    action: str,
    file: Optional[str],
    line: Optional[int],
    symbol: Optional[str],
    query: Optional[str],
    new_name: Optional[str],
    apply: Optional[bool],
    timeout: float,
    payload: Optional[str],
    tid: str,
) -> str:
    t = timeout
    from agent.lsp import get_service
    from agent.coding_workspace import (
        get_coding_roots,
        get_coding_workspace,
        pin_coding_workspace,
    )

    svc = get_service()
    if action == "workspace":
        root = get_coding_workspace(tid)
        roots = get_coding_roots(tid)
        return json.dumps({
            "success": True,
            "action": "workspace",
            "task_id": tid,
            "coding_workspace": root,
            "coding_roots": roots,
            "output": root or "(unset)",
        }, ensure_ascii=False)

    if action == "workspace_set":
        if not file or not str(file).strip() or str(file).strip() == "*":
            return tool_error("workspace_set requires file=<directory to pin>")
        target = _resolve_lsp_path(file, tid)
        if not target:
            return tool_error("workspace_set: could not resolve directory path")
        path = Path(target)
        if path.is_file():
            path = path.parent
        if not path.exists() or not path.is_dir():
            return tool_error(f"workspace_set: not a directory: {path}")
        old = get_coding_workspace(tid)
        new_root, _, lifted = pin_coding_workspace(tid, str(path))
        if svc is not None and (lifted or (old and old != new_root)):
            try:
                svc._evict_nested_clients(new_root)  # noqa: SLF001
            except Exception:
                pass
        return json.dumps({
            "success": True,
            "action": "workspace_set",
            "task_id": tid,
            "coding_workspace": new_root,
            "coding_roots": get_coding_roots(tid),
            "previous": old,
            "lifted": lifted,
            "output": f"coding workspace pinned to {new_root}",
        }, ensure_ascii=False)

    if action == "status":
        if svc is None:
            return json.dumps({"success": True, "action": "status", "active": False})
        return json.dumps({"success": True, "action": "status", **svc.get_status()}, ensure_ascii=False)

    if action == "capabilities":
        if svc is None:
            return json.dumps({"success": True, "action": "capabilities", "servers": []})
        status = svc.get_status()
        return json.dumps({
            "success": True,
            "action": "capabilities",
            "clients": status.get("clients") or [],
            "disabled_servers": status.get("disabled_servers") or [],
        }, ensure_ascii=False)

    if svc is None:
        return tool_error("LSP service is inactive (disabled in config or failed to start)")

    if action == "request":
        method = (query or "").strip()
        if not method:
            return tool_error(
                "action=request requires query to specify the LSP method name "
                "(e.g. 'textDocument/formatting')"
            )
        use_payload = payload is not None
        parsed_payload: Any = None
        if use_payload:
            try:
                parsed_payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                return tool_error(f"invalid JSON in payload: {exc}")

        abs_file: Optional[str] = None
        zero_line: Optional[int] = None
        zero_char: Optional[int] = None
        if file and str(file).strip() and str(file).strip() != "*":
            abs_file = _resolve_lsp_path(file, tid) or ""
            if not use_payload and (line is not None or symbol):
                zero_line, zero_char, err = _resolve_position(abs_file, line, symbol)
                if err:
                    return tool_error(err)

        raw = svc.request_sync(
            "request",
            file_path=abs_file,
            line=zero_line,
            character=zero_char,
            query=method,
            timeout=t,
            payload=parsed_payload,
            use_payload=use_payload,
        )
        if isinstance(raw, dict) and raw.get("error"):
            return tool_error(str(raw["error"]))
        server = (raw or {}).get("server") or ""
        result = (raw or {}).get("result")
        formatted = _format_raw_request_result(result)
        return json.dumps({
            "success": True,
            "action": "request",
            "server": server,
            "method": method,
            "result": result,
            "output": f"{server} ← {method}:\n{formatted}",
        }, ensure_ascii=False)

    if action == "reload":
        reload_file = _resolve_lsp_path(file, tid)
        raw = svc.request_sync(
            "reload",
            file_path=None if not reload_file or reload_file == "*" else reload_file,
            timeout=t,
        )
        if isinstance(raw, dict) and raw.get("error"):
            return tool_error(str(raw["error"]))
        return json.dumps({
            "success": True,
            "action": "reload",
            "output": "\n".join((raw or {}).get("messages") or ["Reloaded"]),
            **{k: v for k, v in (raw or {}).items() if k != "error"},
        }, ensure_ascii=False)

    if action == "restart":
        restart_file = _resolve_lsp_path(file, tid)
        raw = svc.request_sync(
            "restart",
            file_path=None if not restart_file or restart_file == "*" else restart_file,
            timeout=t,
        )
        if isinstance(raw, dict) and raw.get("error"):
            return tool_error(str(raw["error"]))
        return json.dumps({
            "success": True,
            "action": "restart",
            "output": "\n".join((raw or {}).get("messages") or ["Restarted"]),
            **{k: v for k, v in (raw or {}).items() if k != "error"},
        }, ensure_ascii=False)

    if action == "rename_file":
        if not file or not new_name:
            return tool_error("rename_file requires file (source) and new_name (destination path)")
        do_apply = True if apply is None else bool(apply)
        src = _resolve_lsp_path(file, tid)
        dest = _resolve_lsp_path(new_name, tid)
        raw = svc.request_sync(
            "rename_file",
            file_path=src,
            new_name=dest,
            apply=do_apply,
            timeout=t,
        )
        if isinstance(raw, dict) and raw.get("error"):
            return tool_error(str(raw["error"]))
        return json.dumps({
            "success": bool((raw or {}).get("success", True)),
            "action": "rename_file",
            **(raw or {}),
        }, ensure_ascii=False)

    # symbols: file='*' + query → workspace/symbol; else documentSymbol
    if action == "symbols":
        if file and str(file).strip() == "*":
            raw = svc.request_sync(
                "workspace_symbol",
                file_path=None,
                query=query or "",
                timeout=t,
            )
            if isinstance(raw, dict) and raw.get("error"):
                return tool_error(str(raw["error"]))
            return json.dumps({
                "success": True,
                "action": "symbols",
                "scope": "workspace",
                "output": _format_symbols(raw),
            }, ensure_ascii=False)
        if not file:
            return tool_error("symbols requires file (or file='*' with query for workspace search)")
        abs_file = _resolve_lsp_path(file, tid) or ""
        raw = svc.request_sync("document_symbols", file_path=abs_file, timeout=t)
        if isinstance(raw, dict) and raw.get("error"):
            return tool_error(str(raw["error"]))
        return json.dumps({
            "success": True,
            "action": "symbols",
            "scope": "document",
            "file": abs_file,
            "output": _format_symbols(raw),
        }, ensure_ascii=False)

    if action == "diagnostics":
        if not file:
            return tool_error("diagnostics requires file")
        abs_file = _resolve_lsp_path(file, tid) or ""
        raw = svc.request_sync("diagnostics", file_path=abs_file, timeout=t)
        if isinstance(raw, dict) and raw.get("error"):
            return tool_error(str(raw["error"]))
        return json.dumps({
            "success": True,
            "action": "diagnostics",
            "file": abs_file,
            "count": len(raw) if isinstance(raw, list) else 0,
            "output": _format_diagnostics(raw),
        }, ensure_ascii=False)

    if not file:
        return tool_error(f"{action} requires file")
    abs_file = _resolve_lsp_path(file, tid) or ""
    zero_line, zero_char, err = _resolve_position(abs_file, line, symbol)
    if err:
        return tool_error(err)

    if action == "rename":
        if not new_name:
            return tool_error("rename requires new_name")
        do_apply = True if apply is None else bool(apply)
        raw = svc.request_sync(
            "rename",
            file_path=abs_file,
            line=zero_line,
            character=zero_char,
            new_name=new_name,
            timeout=t,
        )
        if isinstance(raw, dict) and raw.get("error"):
            return tool_error(str(raw["error"]))
        if not isinstance(raw, dict):
            return tool_error("rename returned no WorkspaceEdit")
        from agent.lsp.edits import apply_workspace_edit
        summary = apply_workspace_edit(raw, dry_run=not do_apply)
        return json.dumps({
            "success": bool(summary.get("success")),
            "action": "rename",
            "applied": do_apply and bool(summary.get("success")),
            "preview": not do_apply,
            "new_name": new_name,
            "file": abs_file,
            "line": (zero_line or 0) + 1,
            "character": (zero_char or 0) + 1,
            "edit": summary,
        }, ensure_ascii=False)

    if action == "code_actions":
        only = None
        # When listing (apply not true), query filters CodeActionKind
        if apply is not True and query:
            only = [query.strip()]
        raw = svc.request_sync(
            "code_actions",
            file_path=abs_file,
            line=zero_line,
            character=zero_char,
            only=only,
            timeout=t,
        )
        if isinstance(raw, dict) and raw.get("error"):
            return tool_error(str(raw["error"]))
        actions = raw if isinstance(raw, list) else []
        from agent.lsp.file_rename import format_code_action

        if apply is True:
            normalized = (query or "").strip()
            if not normalized:
                return tool_error("query parameter required when apply=true for code_actions")
            selected = None
            if normalized.isdigit():
                idx = int(normalized)
                if 0 <= idx < len(actions):
                    selected = actions[idx]
            else:
                needle = normalized.lower()
                for item in actions:
                    if isinstance(item, dict) and needle in str(item.get("title") or "").lower():
                        selected = item
                        break
            if selected is None:
                lines = [
                    format_code_action(a, i)
                    for i, a in enumerate(actions)
                    if isinstance(a, dict)
                ]
                return json.dumps({
                    "success": False,
                    "action": "code_actions",
                    "error": f'No code action matches "{normalized}"',
                    "output": (
                        f'No code action matches "{normalized}". Available actions:\n'
                        + ("\n".join(f"  {ln}" for ln in lines) if lines else "  (none)")
                    ),
                }, ensure_ascii=False)

            applied = _apply_code_action(svc, selected, abs_file=abs_file, timeout=t)
            if applied is None:
                return tool_error(
                    f'Action "{selected.get("title")}" has no workspace edit or command to apply'
                )
            return json.dumps({
                "success": True,
                "action": "code_actions",
                "applied": True,
                "file": abs_file,
                "line": (zero_line or 0) + 1,
                **applied,
            }, ensure_ascii=False)

        if not actions:
            output = "No code actions available"
        else:
            lines = [
                format_code_action(a, i)
                for i, a in enumerate(actions)
                if isinstance(a, dict)
            ]
            output = f"{len(lines)} code action(s):\n" + "\n".join(f"  {ln}" for ln in lines)
        return json.dumps({
            "success": True,
            "action": "code_actions",
            "file": abs_file,
            "line": (zero_line or 0) + 1,
            "count": len(actions) if isinstance(actions, list) else 0,
            "actions": actions[:50],
            "output": output,
        }, ensure_ascii=False)

    wire_action = {
        "definition": "definition",
        "type_definition": "type_definition",
        "implementation": "implementation",
        "references": "references",
        "hover": "hover",
    }[action]
    raw = svc.request_sync(
        wire_action,
        file_path=abs_file,
        line=zero_line,
        character=zero_char,
        timeout=t,
    )
    if isinstance(raw, dict) and raw.get("error"):
        return tool_error(str(raw["error"]))

    if action == "hover":
        output = _format_hover(raw)
        return json.dumps({
            "success": True,
            "action": "hover",
            "file": abs_file,
            "line": (zero_line or 0) + 1,
            "output": output,
        }, ensure_ascii=False)

    locations = _normalize_locations(raw)
    label = {
        "definition": "definition(s)",
        "type_definition": "type definition(s)",
        "implementation": "implementation(s)",
        "references": "reference(s)",
    }[action]
    if not locations:
        output = f"No {label} found"
    else:
        output = f"Found {len(locations)} {label}:\n" + "\n".join(
            f"  {loc['path']}:{loc['line']}:{loc['character']}" for loc in locations[:100]
        )
    return json.dumps({
        "success": True,
        "action": action,
        "file": abs_file,
        "line": (zero_line or 0) + 1,
        "locations": locations[:100],
        "output": output,
    }, ensure_ascii=False)


def _handle_lsp(args: dict, **kwargs) -> str:
    return lsp_tool(
        action=args.get("action", ""),
        file=args.get("file"),
        line=args.get("line"),
        symbol=args.get("symbol"),
        query=args.get("query"),
        new_name=args.get("new_name"),
        apply=args.get("apply"),
        timeout=args.get("timeout"),
        payload=args.get("payload"),
        task_id=kwargs.get("task_id") or "",
    )


registry.register(
    name="lsp",
    toolset="lsp",
    schema=LSP_SCHEMA,
    handler=_handle_lsp,
    check_fn=check_lsp_tool_requirements,
    emoji="🧭",
    max_result_size_chars=100_000,
)
