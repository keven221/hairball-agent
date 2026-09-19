"""Text normalize for vision-frame glyphs (OMP snapcompact subset)."""

from __future__ import annotations

import re
import unicodedata

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b].*?(?:\x07|\x1b\\)|\x1b.")
_NEWLINE_GLYPH = "\u2588"  # full block — oracle NEWLINE_GLYPH

_EMOJI_FOLD = {
    "✅": "[OK]",
    "❌": "[FAIL]",
    "⚠": "[WARN]",
    "⚠️": "[WARN]",
    "ℹ": "[INFO]",
    "ℹ️": "[INFO]",
    "✔": "[OK]",
    "✖": "[FAIL]",
}

_CHAR_FOLD = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u2190": "<-",
        "\u2192": "->",
        "\u2191": "^",
        "\u2193": "v",
    }
)


def _fold_box(ch: str) -> str:
    cp = ord(ch)
    if not (0x2500 <= cp <= 0x257F):
        return ch
    # Light approximation: horizontals -, verticals |, corners/junctions +
    if cp in {0x2500, 0x2501, 0x2504, 0x2505, 0x2508, 0x2509, 0x254C, 0x254D}:
        return "-"
    if cp in {0x2502, 0x2503, 0x2506, 0x2507, 0x250A, 0x250B}:
        return "|"
    return "+"


def normalize(text: str) -> str:
    """Strip ANSI, fold whitespace/box/emoji, keep Unicode for CJK fonts."""
    if not text:
        return ""
    s = text
    if "\x1b" in s:
        s = _ANSI_RE.sub("", s)
    # Collapse whitespace: newline runs → block glyph; other → space
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in "\r\n":
            while i < n and s[i] in "\r\n":
                i += 1
            out.append(_NEWLINE_GLYPH)
            continue
        if ch.isspace() or unicodedata.category(ch) == "Cf":
            while i < n and (s[i].isspace() or unicodedata.category(s[i]) == "Cf"):
                if s[i] in "\r\n":
                    break
                i += 1
            out.append(" ")
            continue
        out.append(ch)
        i += 1
    s = "".join(out).strip(" \u2588")
    folded: list[str] = []
    for ch in s:
        if ch in _EMOJI_FOLD:
            folded.append(_EMOJI_FOLD[ch])
            continue
        ch = ch.translate(_CHAR_FOLD)
        if len(ch) == 1 and 0x2500 <= ord(ch) <= 0x257F:
            folded.append(_fold_box(ch))
            continue
        folded.append(ch)
    s = "".join(folded)
    s = re.sub(r" {2,}", " ", s)
    return s.strip(" \u2588")


def wrap_lines(text: str, cols: int, *, wide_cells: bool = True) -> list[str]:
    """Wrap normalized text into fixed-width rows (wide CJK = 2 cells)."""
    cols = max(1, cols)
    lines: list[str] = []
    for raw in (text or "").split(_NEWLINE_GLYPH):
        if not raw:
            lines.append("")
            continue
        row = ""
        width = 0
        for ch in raw:
            w = 2 if (wide_cells and is_wide_char(ch)) else 1
            if width + w > cols and row:
                lines.append(row)
                row = ch
                width = w
            else:
                row += ch
                width += w
        lines.append(row)
    return lines


def is_wide_char(ch: str) -> bool:
    from agent.snapcompact.shapes import is_wide_codepoint

    if not ch:
        return False
    return is_wide_codepoint(ord(ch))
