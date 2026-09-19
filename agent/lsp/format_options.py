"""Per-file LSP ``FormattingOptions`` resolution.

Near-isomorphic to oh-my-pi ``lsp/format-options.ts``:
``.editorconfig`` (indent_style / indent_size / tab_width) → content sniff →
fallback. Full EditorConfig glob dialect is not ported; common ``*.ext`` /
basename patterns via ``fnmatch`` cover the write path.
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict


FALLBACK_TAB_SIZE = 2
FALLBACK_INSERT_SPACES = True
_MIN_TAB = 1
_MAX_TAB = 16

_TRIM_OPTIONS = {
    "trimTrailingWhitespace": True,
    "insertFinalNewline": True,
    "trimFinalNewlines": True,
}


class DetectedIndent(TypedDict, total=False):
    tabSize: int
    insertSpaces: bool


def _gcd(a: int, b: int) -> int:
    x, y = a, b
    while y:
        x, y = y, x % y
    return x


def _clamp_tab(n: int) -> int:
    return max(_MIN_TAB, min(_MAX_TAB, int(n)))


def detect_indent_from_content(content: str) -> DetectedIndent:
    """Sniff ``insertSpaces`` and indent unit from ``content``.

    First indented line decides spaces vs tabs; for space indents, GCD of
    leading-space widths is the stride.
    """
    if not content:
        return {}

    insert_spaces: Optional[bool] = None
    unit = 0

    for line in content.split("\n"):
        if not line or not line.strip():
            continue
        first = line[0]
        if first not in (" ", "\t"):
            continue
        if insert_spaces is None:
            insert_spaces = first == " "
        if first == "\t":
            continue
        n = 0
        while n < len(line) and line[n] == " ":
            n += 1
        if n == 0:
            continue
        unit = n if unit == 0 else _gcd(unit, n)

    result: DetectedIndent = {}
    if insert_spaces is not None:
        result["insertSpaces"] = insert_spaces
    if unit > 0 and insert_spaces is True:
        result["tabSize"] = unit
    return result


def _pattern_matches(pattern: str, rel_posix: str, basename: str) -> bool:
    pat = pattern.replace("\\", "/").lstrip("/")
    if not pat:
        return False
    candidates = [pat]
    if "/" not in pat and not pat.startswith("**/"):
        candidates.append(f"**/{pat}")
    for cand in candidates:
        if fnmatch.fnmatch(rel_posix, cand) or fnmatch.fnmatch(basename, cand):
            return True
        # fnmatch does not treat ** like globstar; also try basename-only *.ext
        if cand.startswith("**/") and fnmatch.fnmatch(basename, cand[3:]):
            return True
    return False


def _parse_editorconfig(content: str) -> tuple[bool, list[tuple[str, dict[str, str]]]]:
    root = False
    sections: list[tuple[str, dict[str, str]]] = []
    current: Optional[dict[str, str]] = None
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]") and len(line) >= 2:
            pat = line[1:-1].strip()
            if not pat:
                current = None
                continue
            props: dict[str, str] = {}
            sections.append((pat, props))
            current = props
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().lower()
        val = val.strip().lower()
        if not key:
            continue
        if current is not None:
            current[key] = val
        elif key == "root":
            root = val == "true"
    return root, sections


def get_editorconfig_formatting(file_path: str) -> DetectedIndent:
    """Best-effort ``indent_style`` / size from nearest ``.editorconfig`` chain.

    Walks parents until ``root = true`` or filesystem root. Later matching
    sections in a file override earlier ones; nearer configs override parents
    (OMP precedence: editorconfig wins over content sniff).
    """
    if not file_path or not str(file_path).strip():
        return {}
    try:
        target = Path(file_path).expanduser().resolve()
    except Exception:
        return {}
    if any(len(part.encode("utf-8", "surrogatepass")) > 255 for part in target.parts):
        return {}

    chain: list[tuple[Path, list[tuple[str, dict[str, str]]]]] = []
    cursor = target.parent
    seen: set[Path] = set()
    while cursor not in seen:
        seen.add(cursor)
        cfg_path = cursor / ".editorconfig"
        try:
            if cfg_path.is_file():
                text = cfg_path.read_text(encoding="utf-8", errors="replace")
                is_root, sections = _parse_editorconfig(text)
                chain.append((cursor, sections))
                if is_root:
                    break
        except OSError:
            pass
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent

    if not chain:
        return {}

    # Apply root→leaf so nearer configs win.
    merged: dict[str, str] = {}
    for base_dir, sections in reversed(chain):
        try:
            rel = os.path.relpath(str(target), str(base_dir)).replace("\\", "/")
        except ValueError:
            rel = target.name
        if rel.startswith(".."):
            rel = target.name
        basename = target.name
        for pattern, props in sections:
            if _pattern_matches(pattern, rel, basename):
                merged.update(props)

    out: DetectedIndent = {}
    style = merged.get("indent_style")
    if style == "space":
        out["insertSpaces"] = True
    elif style == "tab":
        out["insertSpaces"] = False

    size_raw = merged.get("indent_size")
    tab_raw = merged.get("tab_width")
    size: Optional[int] = None
    if size_raw and size_raw.isdigit() and int(size_raw) > 0:
        size = _clamp_tab(int(size_raw))
    elif size_raw == "tab" and tab_raw and tab_raw.isdigit() and int(tab_raw) > 0:
        size = _clamp_tab(int(tab_raw))
    if size is None and tab_raw and tab_raw.isdigit() and int(tab_raw) > 0:
        size = _clamp_tab(int(tab_raw))
    if size is not None and out.get("insertSpaces") is not False:
        out["tabSize"] = size
    elif size is not None and "tabSize" not in out and out.get("insertSpaces") is False:
        # tab-indented: tabWidth is display; still expose as tabSize for LSP.
        out["tabSize"] = size
    return out


def resolve_format_options(file_path: str, content: str) -> Dict[str, Any]:
    """Build ``FormattingOptions`` for ``textDocument/formatting``.

    Precedence: ``.editorconfig`` → content sniff → hardcoded fallback.
    """
    from_config = get_editorconfig_formatting(file_path)
    detected = detect_indent_from_content(content)
    return {
        "tabSize": from_config.get("tabSize", detected.get("tabSize", FALLBACK_TAB_SIZE)),
        "insertSpaces": from_config.get(
            "insertSpaces", detected.get("insertSpaces", FALLBACK_INSERT_SPACES)
        ),
        **_TRIM_OPTIONS,
    }
