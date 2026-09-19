"""Model-facing structural rewrite via ``ast-grep`` / ``sg`` CLI (W2 slice 2).

Default is dry-run preview (``--json``, no ``-U``). ``apply=true`` runs a
second pass with ``-U`` and **without** ``--json`` (combined flags do not
persist writes on ast-grep 0.44). Same binary gate as ``ast_grep``.
See ``docs/design/2026-08-04-hairball-independent-core-knowledge.md``
section 4.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

from tools.ast_grep_tool import (
    _MAX_MATCHES,
    _clamp_timeout,
    _infer_lang,
    _is_empty_json_array,
    _parse_cli_json,
    _resolve_search_root,
    _tool_error,
    check_ast_grep_requirements,
    find_ast_grep_bin,
)
from tools.registry import registry

logger = logging.getLogger(__name__)

_MAX_APPLY_FILES = 20
_MAX_OPS = 10
_MAX_PATHS = 20
_APPLIED_RE = re.compile(r"Applied\s+(\d+)\s+changes?", re.IGNORECASE)

AST_EDIT_SCHEMA = {
    "name": "ast_edit",
    "description": (
        "Structural code rewrite with ast-grep (pattern → rewrite). "
        "Default is preview only (applied=false); set apply=true to write. "
        "Pass pattern+rewrite, or ops=[{pattern,rewrite},…] for multi-rule "
        "rewrites (applied in listed order). Optional paths=[…] for multi-file "
        "scopes. Prefer a narrow file path. Empty rewrite deletes the matched "
        "node. For cross-file identifier rename use `lsp` action=rename — not "
        "this. One-off text edits stay on `patch`/`write_file`. For structure "
        "extraction (symbols/members) use `ast_grep` action=outline."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "ast-grep pattern for a single AST node "
                    "(e.g. 'print($A)', 'console.log($MSG)'). "
                    "Omit when using ops."
                ),
            },
            "rewrite": {
                "type": "string",
                "description": (
                    "Replacement pattern using the same metavariables "
                    "(e.g. 'logger.info($A)'). Empty string deletes the match. "
                    "Omit when using ops."
                ),
            },
            "ops": {
                "type": "array",
                "description": (
                    "Multi-rule rewrites: each item needs pattern/pat and "
                    "rewrite/out (empty out deletes). Max "
                    f"{_MAX_OPS} ops. When set, top-level pattern/rewrite "
                    "are ignored."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "pat": {"type": "string"},
                        "rewrite": {"type": "string"},
                        "out": {"type": "string"},
                        "replacement": {"type": "string"},
                    },
                },
            },
            "path": {
                "type": "string",
                "description": (
                    "File or directory to edit. Prefer a single file. "
                    "Defaults to the task working directory. Ignored when "
                    "paths is non-empty."
                ),
            },
            "paths": {
                "type": "array",
                "description": (
                    f"Multiple file/directory scopes (max {_MAX_PATHS}). "
                    "Each path is rewritten with the same op set."
                ),
                "items": {"type": "string"},
            },
            "lang": {
                "type": "string",
                "description": (
                    "Language id (python, typescript, …). "
                    "Inferred from file extension when omitted for a single file."
                ),
            },
            "apply": {
                "type": "boolean",
                "description": (
                    "If false (default), preview replacements only. "
                    "If true, write changes to disk."
                ),
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (default 30; clamped 5–120).",
            },
        },
        "required": [],
    },
}


def check_ast_edit_requirements() -> bool:
    """Same CLI gate as ``ast_grep``, plus optional ``tools.ast_edit.enabled``."""
    if not check_ast_grep_requirements():
        return False
    try:
        from hairball_cli.config import load_config

        cfg = load_config()
    except Exception:
        return True
    if not isinstance(cfg, dict):
        return True
    tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
    edit_cfg = tools_cfg.get("ast_edit") if isinstance(tools_cfg.get("ast_edit"), dict) else {}
    if edit_cfg.get("enabled") is False:
        return False
    return True


def _coerce_apply(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    if isinstance(raw, (int, float)):
        return bool(raw)
    s = str(raw).strip().lower()
    return s in {"1", "true", "yes", "on"}


def _op_pair(raw: Any) -> Optional[tuple[str, str]]:
    if not isinstance(raw, dict):
        return None
    pat = str(raw.get("pattern") or raw.get("pat") or "").strip()
    if not pat:
        return None
    if "rewrite" in raw:
        repl = raw.get("rewrite")
    elif "out" in raw:
        repl = raw.get("out")
    elif "replacement" in raw:
        repl = raw.get("replacement")
    else:
        return None
    if repl is None:
        return None
    return pat, str(repl)


def normalize_ast_edit_ops(
    pattern: Any,
    rewrite: Any,
    ops: Any,
) -> tuple[Optional[list[tuple[str, str]]], Optional[str]]:
    """Return (ops, error). Either explicit ``ops`` or single pattern+rewrite."""
    if ops is not None:
        if not isinstance(ops, list) or not ops:
            return None, "ops must be a non-empty array of {pattern, rewrite}"
        if len(ops) > _MAX_OPS:
            return None, f"ops exceeds cap of {_MAX_OPS}"
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for i, item in enumerate(ops):
            pair = _op_pair(item)
            if pair is None:
                return None, f"ops[{i}] needs pattern/pat and rewrite/out"
            pat, _repl = pair
            if pat in seen:
                return None, f"duplicate pattern in ops: {pat}"
            seen.add(pat)
            pairs.append(pair)
        return pairs, None

    pat = str(pattern or "").strip()
    if not pat:
        return None, "pattern is required (or pass ops)"
    if rewrite is None:
        return None, "rewrite is required (or pass ops)"
    return [(pat, str(rewrite))], None


def normalize_ast_edit_paths(path: Any, paths: Any) -> tuple[Optional[list[str]], Optional[str]]:
    """Return (path specs, error). ``paths`` wins when non-empty."""
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


def run_ast_edit_preview(
    *,
    pattern: str,
    rewrite: str,
    root: Path,
    lang: Optional[str],
    timeout: float,
    bin_path: Optional[str] = None,
) -> tuple[list[dict[str, Any]], str]:
    """Dry-run: ``--rewrite`` + ``--json=compact``, never ``-U``."""
    binary = bin_path or find_ast_grep_bin()
    if not binary:
        return [], "ast-grep CLI not found (install `ast-grep` or `sg` on PATH)"

    cmd = [binary, "run", "--pattern", pattern, "--rewrite", rewrite]
    if lang:
        cmd.extend(["--lang", lang])
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

    matches = _parse_cli_json(proc.stdout or "", root=root)
    err = (proc.stderr or "").strip()
    if proc.returncode != 0 and not matches:
        if _is_empty_json_array(proc.stdout or ""):
            return [], err
        return [], err or f"ast-grep exited with code {proc.returncode}"
    return matches, err


def run_ast_edit_apply(
    *,
    pattern: str,
    rewrite: str,
    root: Path,
    lang: Optional[str],
    timeout: float,
    bin_path: Optional[str] = None,
) -> tuple[int, str]:
    """Write pass: ``-U`` without ``--json`` (json+U does not persist on 0.44)."""
    binary = bin_path or find_ast_grep_bin()
    if not binary:
        return 0, "ast-grep CLI not found (install `ast-grep` or `sg` on PATH)"

    cmd = [binary, "run", "--pattern", pattern, "--rewrite", rewrite, "-U"]
    if lang:
        cmd.extend(["--lang", lang])
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
        return 0, f"ast-grep timed out after {timeout:.0f}s"
    except OSError as exc:
        return 0, f"failed to run ast-grep: {exc}"

    out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    applied = 0
    for line in out.splitlines():
        m = _APPLIED_RE.search(line)
        if m:
            try:
                applied = int(m.group(1))
            except ValueError:
                applied = 0
            break
    if proc.returncode != 0 and applied == 0:
        return 0, out or f"ast-grep exited with code {proc.returncode}"
    return applied, out


def _sensitive_block(root: Path, task_id: str) -> Optional[str]:
    from tools.file_tools import _check_sensitive_path

    return _check_sensitive_path(str(root), task_id)


def _note_writes(task_id: str, changes: list[dict[str, Any]], root: Path) -> None:
    try:
        from tools import file_state
    except Exception:
        return
    seen: set[str] = set()
    for ch in changes:
        rel = str(ch.get("file") or "").strip()
        if not rel or rel in seen:
            continue
        seen.add(rel)
        try:
            if root.is_file():
                target = root
            else:
                target = (root / rel).resolve()
            file_state.note_write(task_id, str(target))
        except Exception:
            logger.debug("file_state.note_write failed for ast_edit", exc_info=True)


def ast_edit(
    pattern: Optional[str] = None,
    rewrite: Optional[str] = None,
    path: Optional[str] = None,
    lang: Optional[str] = None,
    apply: Any = False,
    timeout: Optional[float] = None,
    task_id: Optional[str] = None,
    ops: Any = None,
    paths: Any = None,
    **_kwargs: Any,
) -> str:
    pairs, op_err = normalize_ast_edit_ops(pattern, rewrite, ops)
    if op_err or not pairs:
        return _tool_error(op_err or "pattern is required (or pass ops)")
    path_specs, path_err = normalize_ast_edit_paths(path, paths)
    if path_err or not path_specs:
        return _tool_error(path_err or "path is required")

    do_apply = _coerce_apply(apply)
    t = _clamp_timeout(timeout)
    tid = (task_id or _kwargs.get("task_id") or "default")
    if not isinstance(tid, str) or not tid.strip():
        tid = "default"
    else:
        tid = tid.strip()

    from agent.coding_workspace import coding_task_context

    with coding_task_context(tid):
        return _ast_edit_run(
            pairs=pairs,
            path_specs=path_specs,
            lang=lang,
            do_apply=do_apply,
            timeout=t,
            tid=tid,
        )


def _ast_edit_run(
    *,
    pairs: list,
    path_specs: list[str],
    lang: Optional[str],
    do_apply: bool,
    timeout: float,
    tid: str,
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

    if do_apply:
        for root in roots:
            blocked = _sensitive_block(root, tid)
            if blocked:
                return _tool_error(blocked)

    all_changes: list[dict[str, Any]] = []
    fatal_hint = ""
    for root in roots:
        language = _infer_lang(root, lang)
        for pat, repl in pairs:
            changes, hint = run_ast_edit_preview(
                pattern=pat, rewrite=repl, root=root, lang=language, timeout=t
            )
            if hint and not changes and "not found" in hint.lower():
                return _tool_error(hint)
            if hint and not changes and "timed out" in hint.lower():
                return _tool_error(hint)
            if hint and not changes and "exited" in hint.lower():
                return _tool_error(
                    "ast-grep rewrite preview failed (pattern/rewrite may not parse, "
                    f"or language may be wrong): {hint[:800]}"
                )
            if hint and "ERROR node" in hint:
                fatal_hint = hint
            for ch in changes:
                row = dict(ch)
                row["pattern"] = pat
                row["rewrite"] = repl
                row["scope"] = str(root)
                all_changes.append(row)

    sliced = all_changes[:_MAX_MATCHES]
    truncated = len(all_changes) > len(sliced)
    files = sorted({str(c.get("file") or "") for c in all_changes if c.get("file")})
    primary_pat, primary_repl = pairs[0]
    primary_root = roots[0]
    primary_lang = _infer_lang(primary_root, lang)

    if do_apply and not all_changes:
        return json.dumps(
            {
                "success": True,
                "applied": False,
                "pattern": primary_pat,
                "rewrite": primary_repl,
                "ops": [{"pattern": p, "rewrite": r} for p, r in pairs],
                "path": str(primary_root),
                "paths": [str(r) for r in roots],
                "lang": primary_lang,
                "match_count": 0,
                "changes": [],
                "hint": "No matches to apply. Refine pattern/path/lang.",
            },
            ensure_ascii=False,
        )

    if do_apply and len(files) > _MAX_APPLY_FILES:
        return _tool_error(
            f"Refusing apply: preview touches {len(files)} files "
            f"(cap {_MAX_APPLY_FILES}). Narrow `path`/`paths` or preview only."
        )

    applied = False
    applied_count = 0
    apply_detail = ""
    if do_apply:
        for root in roots:
            language = _infer_lang(root, lang)
            for pat, repl in pairs:
                count, detail = run_ast_edit_apply(
                    pattern=pat, rewrite=repl, root=root, lang=language, timeout=t
                )
                applied_count += count
                if detail:
                    apply_detail = detail
        if applied_count == 0:
            detail = apply_detail or "ast-grep apply wrote 0 changes"
            return _tool_error(f"ast-grep apply failed: {detail[:800]}")
        applied = True
        for root in roots:
            _note_writes(tid, sliced, root)

    hint_out = (
        "Preview only. Set apply=true to write. "
        "For symbol rename use lsp action=rename; one-off text edits use patch. "
        "For structure extraction use ast_grep action=outline."
    )
    if applied:
        hint_out = (
            f"Applied {applied_count} change(s) on disk. "
            "For symbol rename prefer lsp action=rename."
        )
    if fatal_hint and "ERROR node" in fatal_hint:
        hint_out = f"{fatal_hint[:300]} | {hint_out}"

    return json.dumps(
        {
            "success": True,
            "applied": applied,
            "pattern": primary_pat,
            "rewrite": primary_repl,
            "ops": [{"pattern": p, "rewrite": r} for p, r in pairs],
            "path": str(primary_root),
            "paths": [str(r) for r in roots],
            "lang": primary_lang,
            "match_count": len(sliced),
            "total_before_cap": len(all_changes),
            "truncated": truncated,
            "files_touched": files if applied else files[:_MAX_APPLY_FILES],
            "applied_count": applied_count if applied else 0,
            "changes": sliced,
            "hint": hint_out,
        },
        ensure_ascii=False,
    )


def _handle_ast_edit(args: dict, **kwargs) -> str:
    rewrite = args.get("rewrite")
    if rewrite is None:
        rewrite = args.get("out")
    if rewrite is None:
        rewrite = args.get("replacement")
    return ast_edit(
        pattern=args.get("pattern") or args.get("pat") or None,
        rewrite=rewrite,
        path=args.get("path"),
        paths=args.get("paths"),
        ops=args.get("ops"),
        lang=args.get("lang") or args.get("language"),
        apply=args.get("apply"),
        timeout=args.get("timeout"),
        task_id=kwargs.get("task_id") or "",
    )


registry.register(
    name="ast_edit",
    toolset="ast",
    schema=AST_EDIT_SCHEMA,
    handler=_handle_ast_edit,
    check_fn=check_ast_edit_requirements,
    emoji="✏️",
    max_result_size_chars=80_000,
)
