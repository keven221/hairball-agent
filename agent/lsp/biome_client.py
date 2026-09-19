"""Biome CLI-based linter (OMP-style — not an LSP ServerDef).

Biome's LSP path historically produces stale diagnostics; the reference
coding agent therefore shells out to ``biome lint --reporter=json`` and
maps the result into LSP-shaped diagnostic dicts. Hairball mirrors that
contract so post-write diagnostics can include Biome without spawning a
Biome language server.

See ``docs/design/2026-08-04-hairball-independent-core-knowledge.md``
section 4 (Biome adapter).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("agent.lsp.biome")

# Extensions Biome commonly lint/format. Keep narrow — Python/Go stay on
# language servers + shell linters.
BIOME_EXTENSIONS = frozenset(
    {
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".mjs",
        ".cjs",
        ".mts",
        ".cts",
        ".json",
        ".jsonc",
        ".css",
    }
)

_SEVERITY = {
    "error": 1,
    "warning": 2,
    "info": 3,
    "hint": 4,
}

_reported_failures: set[str] = set()


def biome_enabled(config: Optional[dict[str, Any]] = None) -> bool:
    """Return whether Biome CLI lint/format is allowed by config.

    Default True when unset — still no-ops without a binary on PATH.
    """
    if config is None:
        try:
            from hairball_cli.config import load_config

            config = load_config()
        except Exception:  # noqa: BLE001
            config = {}
    lsp_cfg = (config or {}).get("lsp") if isinstance(config, dict) else None
    if not isinstance(lsp_cfg, dict):
        return True
    return bool(lsp_cfg.get("biome", True))


@lru_cache(maxsize=1)
def find_biome() -> Optional[str]:
    """Resolve ``biome`` executable, or None."""
    return shutil.which("biome")


def supports_path(file_path: str) -> bool:
    ext = os.path.splitext(file_path or "")[1].lower()
    return ext in BIOME_EXTENSIONS


def _warn_once(key: str, message: str, **meta: Any) -> None:
    if key in _reported_failures:
        return
    _reported_failures.add(key)
    logger.warning("%s %s", message, meta or "")


def _run_biome(
    args: Sequence[str],
    *,
    cwd: str,
    command: Optional[str] = None,
    timeout: float = 30.0,
) -> tuple[str, str, int]:
    cmd = command or find_biome() or "biome"
    try:
        proc = subprocess.run(
            [cmd, *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.stdout or "", proc.stderr or "", int(proc.returncode)
    except FileNotFoundError:
        return "", f"biome not found: {cmd}", 127
    except subprocess.TimeoutExpired:
        return "", "biome timed out", 124
    except Exception as exc:  # noqa: BLE001
        return "", str(exc), 1


def offsets_to_positions(
    source: str, offsets: Sequence[int]
) -> Dict[int, tuple[int, int]]:
    """Map UTF-8 byte offsets → 1-indexed (line, column)."""
    sorted_offs = sorted(set(int(o) for o in offsets if o is not None))
    result: Dict[int, tuple[int, int]] = {}
    if not sorted_offs:
        return result
    line = 1
    column = 1
    byte_index = 0
    next_i = 0
    for ch in source:
        if next_i >= len(sorted_offs):
            break
        cp = ord(ch)
        if cp < 0x80:
            byte_len = 1
        elif cp < 0x800:
            byte_len = 2
        elif cp < 0x10000:
            byte_len = 3
        else:
            byte_len = 4
        while next_i < len(sorted_offs) and byte_index + byte_len > sorted_offs[next_i]:
            result[sorted_offs[next_i]] = (line, column)
            next_i += 1
        if ch == "\n":
            line += 1
            column = 1
        else:
            column += 1
        byte_index += byte_len
    while next_i < len(sorted_offs):
        result[sorted_offs[next_i]] = (line, column)
        next_i += 1
    return result


def parse_biome_json(json_output: str, target_file: str, *, cwd: str) -> List[Dict[str, Any]]:
    """Parse ``biome lint --reporter=json`` into LSP diagnostic dicts.

    Supports both shapes:
    - Biome 2.x: ``message`` + ``location.path`` (string) + ``start``/``end``
      line/column (1-indexed).
    - Legacy/OMP: ``description`` + ``location.path.file`` + byte ``span`` +
      ``sourceCode``.
    """
    diagnostics: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(json_output)
    except json.JSONDecodeError:
        _warn_once(f"parse:{cwd}", "Failed to parse Biome JSON output")
        return diagnostics

    target = os.path.abspath(target_file)
    raw_diags = parsed.get("diagnostics") if isinstance(parsed, dict) else None
    if not isinstance(raw_diags, list):
        return diagnostics

    # Legacy path: batch span offsets per source text.
    relevant_legacy: list[dict[str, Any]] = []
    offsets_by_source: dict[str, list[int]] = {}

    for diag in raw_diags:
        if not isinstance(diag, dict):
            continue
        location = diag.get("location") if isinstance(diag.get("location"), dict) else None
        if not location:
            continue

        # Resolve target path (Biome 2 string vs legacy {file: ...}).
        path_field = location.get("path")
        if isinstance(path_field, dict):
            file_field = path_field.get("file")
        else:
            file_field = path_field
        if not file_field:
            continue
        diag_file = (
            str(file_field)
            if os.path.isabs(str(file_field))
            else os.path.join(cwd, str(file_field))
        )
        if os.path.abspath(diag_file) != target:
            continue

        # Biome 2.x: line/column already present (1-indexed).
        start = location.get("start") if isinstance(location.get("start"), dict) else None
        end = location.get("end") if isinstance(location.get("end"), dict) else None
        if start and "line" in start and "column" in start:
            start_line = int(start.get("line") or 1)
            start_col = int(start.get("column") or 1)
            end_line = int((end or start).get("line") or start_line)
            end_col = int((end or start).get("column") or start_col)
            severity = _SEVERITY.get(str(diag.get("severity") or "").lower(), 2)
            msg = diag.get("message") or diag.get("description") or ""
            diagnostics.append(
                {
                    "range": {
                        "start": {"line": max(0, start_line - 1), "character": max(0, start_col - 1)},
                        "end": {"line": max(0, end_line - 1), "character": max(0, end_col - 1)},
                    },
                    "severity": severity,
                    "message": str(msg),
                    "source": "biome",
                    "code": diag.get("category"),
                }
            )
            continue

        # Legacy span + sourceCode path (OMP / older reporters).
        relevant_legacy.append(diag)
        span = location.get("span")
        source_code = location.get("sourceCode")
        if (
            isinstance(span, (list, tuple))
            and len(span) >= 2
            and isinstance(source_code, str)
        ):
            offs = offsets_by_source.setdefault(source_code, [])
            offs.append(int(span[0]))
            offs.append(int(span[1]))

    if not relevant_legacy:
        return diagnostics

    positions_by_source = {
        src: offsets_to_positions(src, offs) for src, offs in offsets_by_source.items()
    }

    for diag in relevant_legacy:
        location = diag.get("location") or {}
        start_line = start_col = end_line = end_col = 1
        span = location.get("span") if isinstance(location, dict) else None
        source_code = location.get("sourceCode") if isinstance(location, dict) else None
        if (
            isinstance(span, (list, tuple))
            and len(span) >= 2
            and isinstance(source_code, str)
        ):
            positions = positions_by_source.get(source_code) or {}
            start_pos = positions.get(int(span[0]))
            end_pos = positions.get(int(span[1]))
            if start_pos:
                start_line, start_col = start_pos
            if end_pos:
                end_line, end_col = end_pos
        severity = _SEVERITY.get(str(diag.get("severity") or "").lower(), 2)
        msg = diag.get("description") or diag.get("message") or ""
        diagnostics.append(
            {
                "range": {
                    "start": {"line": start_line - 1, "character": start_col - 1},
                    "end": {"line": end_line - 1, "character": end_col - 1},
                },
                "severity": severity,
                "message": str(msg),
                "source": "biome",
                "code": diag.get("category"),
            }
        )
    return diagnostics


def lint_file(
    file_path: str,
    *,
    cwd: Optional[str] = None,
    command: Optional[str] = None,
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """Run Biome lint on ``file_path``; return LSP-shaped diagnostics."""
    abs_path = os.path.abspath(file_path)
    work = cwd or os.path.dirname(abs_path) or "."
    binary = command or find_biome()
    if not binary:
        return []
    stdout, stderr, code = _run_biome(
        ["lint", "--reporter=json", abs_path],
        cwd=work,
        command=binary,
        timeout=timeout,
    )
    # Biome exits non-zero when diagnostics exist; empty stdout = real failure.
    if code != 0 and not stdout.strip():
        _warn_once(f"run:{work}", "Biome lint failed; reporting no diagnostics", stderr=stderr[:500])
        return []
    return parse_biome_json(stdout, abs_path, cwd=work)


def format_file(
    file_path: str,
    content: str,
    *,
    cwd: Optional[str] = None,
    command: Optional[str] = None,
    timeout: float = 30.0,
) -> str:
    """Format ``file_path`` via ``biome format --write``; return new content.

    Writes ``content`` to disk first (caller already wrote / is about to).
    On failure returns the original ``content``.
    """
    abs_path = os.path.abspath(file_path)
    work = cwd or os.path.dirname(abs_path) or "."
    binary = command or find_biome()
    if not binary:
        return content
    try:
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write(content)
    except OSError:
        return content
    stdout, stderr, code = _run_biome(
        ["format", "--write", abs_path],
        cwd=work,
        command=binary,
        timeout=timeout,
    )
    if code != 0:
        _warn_once(f"format:{work}", "Biome format failed", stderr=stderr[:500], stdout=stdout[:200])
        return content
    try:
        with open(abs_path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return content


def clear_caches() -> None:
    """Test helper — drop find_biome + one-shot warn state."""
    cached = find_biome
    if hasattr(cached, "cache_clear"):
        cached.cache_clear()
    _reported_failures.clear()
