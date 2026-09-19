"""Model-facing structural search/edit via the ``ast-grep`` / ``sg`` CLI (W2).

``ast_grep`` is read-only search. ``ast_edit`` (same module helpers / sibling
tool) defaults to dry-run preview; ``apply=true`` writes via ``sg -U``.
Hidden when the binary is absent (``check_fn``). Mounted on coding edges like
``lsp`` — not core, not messaging defaults.
See ``docs/design/2026-08-04-hairball-independent-core-knowledge.md``
section 4.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

_MAX_MATCHES = 50
_MAX_OUTLINE_FILES = 40
_MAX_OUTLINE_ITEMS = 120
_MAX_PATHS = 20
_MAX_GLOBS = 20
_MAX_OUTLINE_TYPES = 20
_MAX_OUTLINE_MATCH_CHARS = 512
_DEFAULT_TIMEOUT = 30.0
_OUTLINE_ITEMS = frozenset({"auto", "structure", "exports", "imports", "all"})
_OUTLINE_VIEWS = frozenset({"auto", "names", "signatures", "digest", "expanded"})

_EXT_LANG = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".lua": "lua",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".html": "html",
    ".css": "css",
    ".json": "json",
}

AST_GREP_SCHEMA = {
    "name": "ast_grep",
    "description": (
        "Structural code search with ast-grep patterns (AST shape, not plain text), "
        "or action=outline to extract symbols/classes/members. "
        "Use search when syntax matters: calls, declarations, class/method forms. "
        "Pattern metavariables: $NAME (one node), $$$ (zero-or-more). "
        "Prefer a narrow path. For identifier rename across files use `lsp` "
        "action=rename — not this tool. Text/regex search stays on `search_files`. "
        "Structural rewrites use `ast_edit`."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "outline"],
                "description": (
                    "search (default): pattern match. "
                    "outline: extract top-level symbols/members (no pattern required)."
                ),
            },
            "pattern": {
                "type": "string",
                "description": (
                    "One ast-grep pattern that parses as a single AST node "
                    "(e.g. 'console.log($A)', "
                    "'def $NAME($$$ARGS) -> $RET: $$$BODY' for typed Python, "
                    "'def $NAME($$$ARGS): $$$BODY' only when there is no return annotation). "
                    "Required for action=search."
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "File or directory to search (task workspace relative or absolute). "
                    "Defaults to the session coding workspace (same pin as LSP), "
                    "else the task working directory. Prefer a narrow path. "
                    "Ignored when paths is non-empty."
                ),
            },
            "paths": {
                "type": "array",
                "description": (
                    f"Multiple file/directory scopes (max {_MAX_PATHS}), "
                    "same idea as OMP paths[]. Each scope is searched and results merged."
                ),
                "items": {"type": "string"},
            },
            "glob": {
                "type": "string",
                "description": (
                    "Include/exclude glob for directory scans (passed to ast-grep --globs). "
                    "Prefix with ! to exclude. Prefer with a directory path."
                ),
            },
            "globs": {
                "type": "array",
                "description": f"Multiple --globs filters (max {_MAX_GLOBS}).",
                "items": {"type": "string"},
            },
            "lang": {
                "type": "string",
                "description": (
                    "Language id for ast-grep (python, typescript, go, rust, …). "
                    "Inferred from file extension when omitted for a single file."
                ),
            },
            "items": {
                "type": "string",
                "enum": ["auto", "structure", "exports", "imports", "all"],
                "description": "outline only: which top-level items to include (default auto).",
            },
            "match": {
                "type": "string",
                "description": (
                    "outline only: regex matched against symbol names, signatures, "
                    "first source lines, and import/export signatures. Use this to "
                    "retrieve a named symbol without returning an entire directory outline."
                ),
            },
            "symbol_types": {
                "type": "array",
                "description": (
                    f"outline only: keep these symbol types (max {_MAX_OUTLINE_TYPES}), "
                    "for example ['class', 'function', 'interface']."
                ),
                "items": {"type": "string"},
                "maxItems": _MAX_OUTLINE_TYPES,
            },
            "view": {
                "type": "string",
                "enum": ["auto", "names", "signatures", "digest", "expanded"],
                "description": (
                    "outline only: output detail. names/signatures omit members; "
                    "digest/expanded include member context (default auto)."
                ),
            },
            "public_members": {
                "type": "boolean",
                "description": "outline only: when true, omit non-public members.",
            },
            "skip": {
                "type": "integer",
                "description": "Skip the first N matches before returning (default 0; search only).",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (default 30; clamped 5–120).",
            },
        },
        "required": [],
    },
}


def _tool_error(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


def find_ast_grep_bin() -> Optional[str]:
    """Return path to ``ast-grep`` or ``sg``, else None."""
    for name in ("ast-grep", "sg"):
        found = shutil.which(name)
        if found:
            return found
    return None


def check_ast_grep_requirements() -> bool:
    """Expose the tool only when an ast-grep CLI is on PATH."""
    try:
        from hairball_cli.config import load_config

        cfg = load_config()
    except Exception:
        cfg = None
    if isinstance(cfg, dict):
        tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
        ast_cfg = tools_cfg.get("ast_grep") if isinstance(tools_cfg.get("ast_grep"), dict) else {}
        if ast_cfg.get("enabled") is False:
            return False
    return find_ast_grep_bin() is not None


def _clamp_timeout(raw: Any) -> float:
    try:
        t = float(raw)
    except (TypeError, ValueError):
        t = _DEFAULT_TIMEOUT
    return max(5.0, min(120.0, t))


def _infer_lang(path: Path, explicit: Optional[str]) -> Optional[str]:
    if explicit and str(explicit).strip():
        return str(explicit).strip().lower()
    if path.is_file():
        return _EXT_LANG.get(path.suffix.lower())
    return None


def _normalize_search_paths(path: Any, paths: Any) -> tuple[Optional[list[str]], Optional[str]]:
    if paths is not None:
        if not isinstance(paths, list) or not paths:
            return None, "paths must be a non-empty array of strings"
        if len(paths) > _MAX_PATHS:
            return None, f"paths exceeds cap of {_MAX_PATHS}"
        out: list[str] = []
        for i, item in enumerate(paths):
            s = str(item or "").strip()
            if not s:
                return None, f"paths[{i}] is empty"
            out.append(s)
        return out, None
    if path is None or str(path).strip() == "":
        return ["."], None
    return [str(path).strip()], None


def _normalize_globs(glob: Any, globs: Any) -> tuple[Optional[list[str]], Optional[str]]:
    out: list[str] = []
    if globs is not None:
        if isinstance(globs, str):
            s = globs.strip()
            if s:
                out.append(s)
        elif isinstance(globs, list):
            for i, item in enumerate(globs):
                s = str(item or "").strip()
                if not s:
                    return None, f"globs[{i}] is empty"
                out.append(s)
        else:
            return None, "globs must be a string or array of strings"
    if glob is not None and str(glob).strip():
        out.append(str(glob).strip())
    if len(out) > _MAX_GLOBS:
        return None, f"globs exceeds cap of {_MAX_GLOBS}"
    return out, None


def _normalize_outline_types(raw: Any) -> tuple[Optional[list[str]], Optional[str]]:
    """Normalize the upstream outline ``--type`` list without shell parsing."""
    if raw is None:
        return [], None
    values = [raw] if isinstance(raw, str) else raw
    if not isinstance(values, list):
        return None, "symbol_types must be an array of strings"
    out: list[str] = []
    for index, item in enumerate(values):
        if not isinstance(item, str):
            return None, f"symbol_types[{index}] must be a string"
        for part in item.split(","):
            value = part.strip()
            if not value:
                return None, f"symbol_types[{index}] is empty"
            if len(value) > 64:
                return None, f"symbol_types[{index}] exceeds 64 characters"
            if value not in out:
                out.append(value)
    if len(out) > _MAX_OUTLINE_TYPES:
        return None, f"symbol_types exceeds cap of {_MAX_OUTLINE_TYPES}"
    return out, None


def _resolve_search_root(path: Optional[str], task_id: str) -> Path:
    """Resolve an AST search/edit root for ``task_id``.

    Default ``.`` prefers the session coding-workspace pin (same root LSP
    uses). Explicit relative/absolute paths still resolve via the task
    session cwd so UI project-relative paths keep working.
    """
    from tools.file_tools import _resolve_path_for_task

    tid = (task_id or "default").strip() or "default"
    raw = str(path or "").strip()
    if not raw or raw in {".", "./"}:
        try:
            from agent.coding_workspace import get_coding_workspace

            pin = get_coding_workspace(tid)
            if pin:
                return Path(pin)
        except Exception:
            pass
        resolved = _resolve_path_for_task(".", tid)
        return Path(resolved)
    return Path(_resolve_path_for_task(raw, tid))


def _flatten_meta_variables(raw: Any) -> Optional[dict[str, str]]:
    """Flatten ast-grep CLI / native metaVariables into ``{NAME: text}``.

    CLI JSON nests captures under ``single`` / ``multi`` / ``transformed``.
    OMP surfaces a flat map for the model (``includeMeta: true``).
    """
    if not isinstance(raw, dict) or not raw:
        return None

    def _text_of(node: Any) -> Optional[str]:
        if isinstance(node, str):
            return node
        if isinstance(node, dict) and node.get("text") is not None:
            return str(node.get("text"))
        return None

    # Already flat string map
    if all(isinstance(v, str) for v in raw.values()) and not any(
        k in raw for k in ("single", "multi", "transformed")
    ):
        out = {str(k): str(v) for k, v in raw.items() if str(k).strip()}
        return out or None

    out: dict[str, str] = {}
    single = raw.get("single")
    if isinstance(single, dict):
        for key, node in single.items():
            text = _text_of(node)
            if text is not None and str(key).strip():
                out[str(key)] = text
    multi = raw.get("multi")
    if isinstance(multi, dict):
        for key, node in multi.items():
            if not str(key).strip():
                continue
            if isinstance(node, list):
                parts = [t for t in (_text_of(item) for item in node) if t is not None]
                out[str(key)] = "\n".join(parts)
            else:
                text = _text_of(node)
                if text is not None:
                    out[str(key)] = text
    transformed = raw.get("transformed")
    if isinstance(transformed, dict):
        for key, node in transformed.items():
            text = _text_of(node)
            if text is not None and str(key).strip() and str(key) not in out:
                out[str(key)] = text
    return out or None


def _normalize_match(raw: Any, *, root: Path) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    file_path = str(raw.get("file") or raw.get("path") or "").strip()
    if not file_path:
        return None
    try:
        file_resolved = Path(file_path).expanduser().resolve()
        root_resolved = root.expanduser().resolve()
        # Single-file search: relative_to(self) is ".", which is useless for the model.
        if root_resolved.is_file():
            rel = root_resolved.name if file_resolved == root_resolved else str(file_resolved)
        else:
            rel = str(file_resolved.relative_to(root_resolved))
    except Exception:
        try:
            import os

            root_resolved = root.expanduser().resolve()
            if root_resolved.is_file():
                file_resolved = Path(file_path).expanduser().resolve()
                rel = (
                    root_resolved.name
                    if file_resolved == root_resolved
                    else str(file_resolved)
                )
            else:
                rel = os.path.relpath(str(Path(file_path).resolve()), str(root_resolved))
                if rel.startswith(".."):
                    rel = file_path
        except Exception:
            rel = file_path
    start = raw.get("range", {}).get("start") if isinstance(raw.get("range"), dict) else {}
    if not isinstance(start, dict):
        start = {}
    line = start.get("line")
    try:
        line_1 = int(line) + 1 if line is not None else None
    except (TypeError, ValueError):
        line_1 = None
    text = str(raw.get("text") or raw.get("lines") or "").rstrip()
    if len(text) > 400:
        text = text[:400] + "…"
    out: dict[str, Any] = {"file": rel, "text": text}
    if line_1 is not None:
        out["line"] = line_1
    lang = raw.get("language") or raw.get("lang")
    if lang:
        out["lang"] = str(lang)
    if "replacement" in raw:
        repl = str(raw.get("replacement") if raw.get("replacement") is not None else "")
        if len(repl) > 400:
            repl = repl[:400] + "…"
        out["replacement"] = repl
    meta = _flatten_meta_variables(raw.get("metaVariables") or raw.get("meta_variables"))
    if meta:
        # Cap capture values so tool results stay lean.
        capped: dict[str, str] = {}
        for key, val in list(meta.items())[:40]:
            s = val if len(val) <= 200 else val[:200] + "…"
            capped[key] = s
        out["meta"] = capped
    return out


def _parse_cli_json(stdout: str, *, root: Path) -> list[dict[str, Any]]:
    text = (stdout or "").strip()
    if not text:
        return []
    matches: list[dict[str, Any]] = []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        for item in parsed:
            norm = _normalize_match(item, root=root)
            if norm:
                matches.append(norm)
        return matches
    if isinstance(parsed, dict):
        # Some builds wrap under "matches"
        inner = parsed.get("matches") if isinstance(parsed.get("matches"), list) else [parsed]
        for item in inner:
            norm = _normalize_match(item, root=root)
            if norm:
                matches.append(norm)
        return matches
    # NDJSON fallback
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        norm = _normalize_match(item, root=root)
        if norm:
            matches.append(norm)
    return matches


def _normalize_outline_item(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    start = {}
    rng = raw.get("range")
    if isinstance(rng, dict):
        start = rng.get("start") if isinstance(rng.get("start"), dict) else {}
    line = start.get("line")
    try:
        line_1 = int(line) + 1 if line is not None else None
    except (TypeError, ValueError):
        line_1 = None
    out: dict[str, Any] = {
        "name": name,
        "symbol_type": str(raw.get("symbolType") or raw.get("symbol_type") or ""),
        "role": str(raw.get("role") or "item"),
    }
    sig = str(raw.get("signature") or "").strip()
    if sig:
        if len(sig) > 200:
            sig = sig[:200] + "…"
        out["signature"] = sig
    if line_1 is not None:
        out["line"] = line_1
    kind = raw.get("astKind") or raw.get("ast_kind")
    if kind:
        out["ast_kind"] = str(kind)
    members_raw = raw.get("members")
    if isinstance(members_raw, list) and members_raw:
        members: list[dict[str, Any]] = []
        for m in members_raw[:40]:
            nm = _normalize_outline_item(m)
            if nm:
                members.append(nm)
        if members:
            out["members"] = members
    return out


def _parse_outline_json(stdout: str, *, root: Path) -> list[dict[str, Any]]:
    text = (stdout or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    files: list[dict[str, Any]] = []
    for entry in parsed[:_MAX_OUTLINE_FILES]:
        if not isinstance(entry, dict):
            continue
        file_path = str(entry.get("path") or entry.get("file") or "").strip()
        if not file_path:
            continue
        try:
            file_resolved = Path(file_path).expanduser().resolve()
            root_resolved = root.expanduser().resolve()
            if root_resolved.is_file():
                rel = root_resolved.name if file_resolved == root_resolved else str(file_resolved)
            else:
                rel = str(file_resolved.relative_to(root_resolved))
        except Exception:
            rel = file_path
        items_out: list[dict[str, Any]] = []
        for item in (entry.get("items") or [])[:_MAX_OUTLINE_ITEMS]:
            norm = _normalize_outline_item(item)
            if norm:
                items_out.append(norm)
        files.append(
            {
                "file": rel,
                "lang": str(entry.get("language") or entry.get("lang") or ""),
                "item_count": len(items_out),
                "items": items_out,
            }
        )
    return files


def _is_empty_json_array(stdout: str) -> bool:
    text = (stdout or "").strip()
    if text != "[]":
        return False
    try:
        return json.loads(text) == []
    except json.JSONDecodeError:
        return False


def run_ast_outline_cli(
    *,
    root: Path,
    lang: Optional[str],
    items: str,
    timeout: float,
    globs: Optional[list[str]] = None,
    match: Optional[str] = None,
    symbol_types: Optional[list[str]] = None,
    view: str = "auto",
    public_members: bool = False,
    bin_path: Optional[str] = None,
) -> tuple[list[dict[str, Any]], str]:
    """Execute ``ast-grep outline`` and return (files, stderr_or_hint)."""
    binary = bin_path or find_ast_grep_bin()
    if not binary:
        return [], "ast-grep CLI not found (install `ast-grep` or `sg` on PATH)"

    item_mode = items if items in _OUTLINE_ITEMS else "auto"
    cmd = [binary, "outline", "--json=compact", "--items", item_mode]
    if lang:
        cmd.extend(["--lang", lang])
    if match:
        cmd.extend(["--match", match])
    if symbol_types:
        cmd.extend(["--type", ",".join(symbol_types)])
    if view in _OUTLINE_VIEWS and view != "auto":
        cmd.extend(["--view", view])
    if public_members:
        cmd.append("--pub-members")
    for g in globs or []:
        cmd.extend(["--globs", g])
    cmd.append(str(root))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], f"ast-grep outline timed out after {timeout:.0f}s"
    except OSError as exc:
        return [], f"failed to run ast-grep outline: {exc}"

    if proc.returncode != 0 and "outline" in (proc.stderr or "").lower() and "unrecognized" in (
        proc.stderr or ""
    ).lower():
        return [], (
            "ast-grep outline is unavailable on this CLI version; "
            "upgrade ast-grep/sg, or use action=search with a pattern"
        )

    files = _parse_outline_json(proc.stdout or "", root=root)
    err = (proc.stderr or "").strip()
    if proc.returncode != 0 and not files:
        if _is_empty_json_array(proc.stdout or ""):
            return [], err
        return [], err or f"ast-grep outline exited with code {proc.returncode}"
    return files, err


def run_ast_grep_cli(
    *,
    pattern: str,
    root: Path,
    lang: Optional[str],
    timeout: float,
    globs: Optional[list[str]] = None,
    bin_path: Optional[str] = None,
) -> tuple[list[dict[str, Any]], str]:
    """Execute ast-grep and return (matches, stderr_or_hint)."""
    binary = bin_path or find_ast_grep_bin()
    if not binary:
        return [], "ast-grep CLI not found (install `ast-grep` or `sg` on PATH)"

    cmd = [binary, "run", "--pattern", pattern]
    if lang:
        cmd.extend(["--lang", lang])
    for g in globs or []:
        cmd.extend(["--globs", g])
    # Prefer compact JSON when supported; older CLIs may ignore unknown flags.
    cmd.append("--json=compact")
    cmd.append(str(root))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], f"ast-grep timed out after {timeout:.0f}s"
    except OSError as exc:
        return [], f"failed to run ast-grep: {exc}"

    # Retry without --json=compact if the flag is rejected.
    if proc.returncode != 0 and "json" in (proc.stderr or "").lower():
        cmd2 = [binary, "run", "--pattern", pattern]
        if lang:
            cmd2.extend(["--lang", lang])
        for g in globs or []:
            cmd2.extend(["--globs", g])
        cmd2.extend(["--json", str(root)])
        try:
            proc = subprocess.run(
                cmd2,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except Exception as exc:
            return [], f"failed to run ast-grep: {exc}"

    matches = _parse_cli_json(proc.stdout or "", root=root)
    err = (proc.stderr or "").strip()
    # ast-grep 0.44+: a successful query with zero hits often exits 1 with
    # stdout `[]`. That is empty success, not a parse failure.
    if proc.returncode != 0 and not matches:
        if _is_empty_json_array(proc.stdout or ""):
            return [], err
        return [], err or f"ast-grep exited with code {proc.returncode}"
    return matches, err


def ast_grep(
    pattern: Optional[str] = None,
    path: Optional[str] = None,
    lang: Optional[str] = None,
    skip: Optional[int] = None,
    timeout: Optional[float] = None,
    task_id: Optional[str] = None,
    action: Optional[str] = None,
    items: Optional[str] = None,
    match: Optional[str] = None,
    symbol_types: Any = None,
    view: Optional[str] = None,
    public_members: Optional[bool] = None,
    paths: Any = None,
    glob: Any = None,
    globs: Any = None,
    **_kwargs: Any,
) -> str:
    mode = str(action or _kwargs.get("action") or "search").strip().lower() or "search"
    if mode not in {"search", "outline"}:
        return _tool_error("action must be 'search' or 'outline'")

    path_specs, path_err = _normalize_search_paths(path, paths)
    if path_err or not path_specs:
        return _tool_error(path_err or "path is required")
    glob_list, glob_err = _normalize_globs(glob, globs)
    if glob_err is not None:
        return _tool_error(glob_err)

    t = _clamp_timeout(timeout)
    tid = (task_id or _kwargs.get("task_id") or "default")
    if not isinstance(tid, str) or not tid.strip():
        tid = "default"
    else:
        tid = tid.strip()

    from agent.coding_workspace import coding_task_context

    with coding_task_context(tid):
        return _ast_grep_run(
            mode=mode,
            path_specs=path_specs,
            glob_list=glob_list,
            pattern=pattern,
            lang=lang,
            skip=skip,
            timeout=t,
            tid=tid,
            items=items,
            outline_match=match,
            symbol_types=symbol_types,
            view=view,
            public_members=public_members,
        )


def _ast_grep_run(
    *,
    mode: str,
    path_specs: list[str],
    glob_list: Optional[list[str]],
    pattern: Optional[str],
    lang: Optional[str],
    skip: Optional[int],
    timeout: float,
    tid: str,
    items: Optional[str],
    outline_match: Optional[str],
    symbol_types: Any,
    view: Optional[str],
    public_members: Optional[bool],
) -> str:
    t = timeout
    roots: list[Path] = []
    for spec in path_specs:
        try:
            root = _resolve_search_root(spec, tid)
        except Exception as exc:
            return _tool_error(f"invalid path {spec!r}: {exc}")
        if not root.exists():
            return _tool_error(f"path not found: {root}")
        roots.append(root)

    primary = roots[0]
    language = _infer_lang(primary, lang)

    if mode == "outline":
        item_mode = str(items or "auto").strip().lower() or "auto"
        if item_mode not in _OUTLINE_ITEMS:
            return _tool_error(
                "items must be one of: auto, structure, exports, imports, all"
            )
        match_value = str(outline_match or "").strip()
        if len(match_value) > _MAX_OUTLINE_MATCH_CHARS:
            return _tool_error(
                f"match exceeds cap of {_MAX_OUTLINE_MATCH_CHARS} characters"
            )
        type_values, type_error = _normalize_outline_types(symbol_types)
        if type_error is not None:
            return _tool_error(type_error)
        view_mode = str(view or "auto").strip().lower() or "auto"
        if view_mode not in _OUTLINE_VIEWS:
            return _tool_error(
                "view must be one of: auto, names, signatures, digest, expanded"
            )
        if public_members is not None and not isinstance(public_members, bool):
            return _tool_error("public_members must be a boolean")
        public_only = bool(public_members)
        all_files: list[dict[str, Any]] = []
        for root in roots:
            lang_i = _infer_lang(root, lang)
            files, hint = run_ast_outline_cli(
                root=root,
                lang=lang_i,
                items=item_mode,
                timeout=t,
                globs=glob_list,
                match=match_value or None,
                symbol_types=type_values,
                view=view_mode,
                public_members=public_only,
            )
            if hint and not files and "not found" in hint.lower():
                return _tool_error(hint)
            if hint and not files and "timed out" in hint.lower():
                return _tool_error(hint)
            if hint and not files and (
                "exited" in hint.lower() or "unavailable" in hint.lower()
            ):
                return _tool_error(hint[:800])
            all_files.extend(files)
        all_files = all_files[:_MAX_OUTLINE_FILES]
        total_items = sum(int(f.get("item_count") or 0) for f in all_files)
        focused_outline = bool(
            match_value or type_values or view_mode != "auto" or public_only
        )
        if focused_outline:
            hint = (
                "Outline lists symbols/members (read-only). "
                "Use match/symbol_types to keep directory results focused; "
                "use action=search for AST pattern match, ast_edit for rewrites, "
                "and lsp for rename/navigation."
            )
        else:
            hint = (
                "Outline lists symbols/members (read-only). "
                "Use action=search for pattern match; ast_edit for rewrites; "
                "lsp for rename/navigation."
            )
        payload: dict[str, Any] = {
            "success": True,
            "action": "outline",
            "path": str(primary),
            "paths": [str(r) for r in roots],
            "globs": glob_list,
            "lang": language,
            "items_mode": item_mode,
            "file_count": len(all_files),
            "item_count": total_items,
            "files": all_files,
            "hint": hint,
        }
        if match_value:
            payload["match"] = match_value
        if type_values:
            payload["symbol_types"] = type_values
        if view_mode != "auto":
            payload["view"] = view_mode
        if public_only:
            payload["public_members"] = True
        return json.dumps(payload, ensure_ascii=False)

    outline_only = {
        "items": items,
        "match": outline_match,
        "symbol_types": symbol_types,
        "view": view,
        "public_members": public_members,
    }
    used_outline_only = [
        name for name, value in outline_only.items()
        if value is not None and value != "" and value != []
    ]
    if used_outline_only:
        return _tool_error(
            "outline-only parameter(s) require action='outline': "
            + ", ".join(used_outline_only)
        )

    pat = str(pattern or "").strip()
    if not pat:
        return _tool_error("pattern is required for action=search")

    try:
        skip_n = max(0, int(skip or 0))
    except (TypeError, ValueError):
        skip_n = 0

    matches: list[dict[str, Any]] = []
    fatal_hint = ""
    for root in roots:
        lang_i = _infer_lang(root, lang)
        part, hint = run_ast_grep_cli(
            pattern=pat, root=root, lang=lang_i, timeout=t, globs=glob_list
        )
        if hint and not part and "not found" in hint.lower():
            return _tool_error(hint)
        if hint and not part and "timed out" in hint.lower():
            return _tool_error(hint)
        if hint and not part and "exited" in hint.lower():
            return _tool_error(
                "ast-grep query failed (pattern may not parse as a single AST node, "
                f"or language may be wrong): {hint[:800]}"
            )
        if hint and "ERROR node" in hint:
            fatal_hint = hint
        for row in part:
            item = dict(row)
            item["scope"] = str(root)
            matches.append(item)

    sliced = matches[skip_n : skip_n + _MAX_MATCHES]
    truncated = len(matches) > skip_n + len(sliced)
    hint_out = (
        "Narrow `path`/`paths` / fix pattern if empty; parse errors are not absence. "
        "For symbol rename use lsp action=rename. "
        "For structure extraction use action=outline."
    )
    if fatal_hint and "ERROR node" in fatal_hint:
        hint_out = f"{fatal_hint[:400]} | {hint_out}"
    elif not matches and language == "python" and "->" not in pat and "def " in pat:
        hint_out = (
            "No matches. Typed Python defs need a return annotation in the pattern "
            "(e.g. 'def $NAME($$$ARGS) -> $RET: $$$BODY'). " + hint_out
        )
    return json.dumps(
        {
            "success": True,
            "action": "search",
            "pattern": pat,
            "path": str(primary),
            "paths": [str(r) for r in roots],
            "globs": glob_list,
            "lang": language,
            "skip": skip_n,
            "match_count": len(sliced),
            "total_before_cap": len(matches),
            "truncated": truncated,
            "matches": sliced,
            "hint": hint_out,
        },
        ensure_ascii=False,
    )


def _handle_ast_grep(args: dict, **kwargs) -> str:
    return ast_grep(
        pattern=args.get("pattern") or args.get("pat") or "",
        path=args.get("path"),
        paths=args.get("paths"),
        glob=args.get("glob"),
        globs=args.get("globs"),
        lang=args.get("lang") or args.get("language"),
        skip=args.get("skip"),
        timeout=args.get("timeout"),
        action=args.get("action"),
        items=args.get("items"),
        match=args.get("match"),
        symbol_types=args.get("symbol_types"),
        view=args.get("view"),
        public_members=args.get("public_members"),
        task_id=kwargs.get("task_id") or "",
    )


registry.register(
    name="ast_grep",
    toolset="ast",
    schema=AST_GREP_SCHEMA,
    handler=_handle_ast_grep,
    check_fn=check_ast_grep_requirements,
    emoji="🌳",
    max_result_size_chars=80_000,
)
