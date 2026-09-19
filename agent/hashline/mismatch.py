"""Hairball hashline mismatch diagnostics and anchored context formatting."""

from __future__ import annotations

from typing import Optional, Sequence

from agent.hashline.content_hash import (
    HL_FILE_HASH_SEP,
    HL_FILE_PREFIX,
    HL_FILE_SUFFIX,
)

MISMATCH_CONTEXT = 2


def format_numbered_line(line_num: int, text: str) -> str:
    """Format one ``N|TEXT`` diagnostic row."""
    return f"{line_num}|{text}"


def format_anchored_context(
    anchor_lines: Sequence[int],
    file_lines: Sequence[str],
) -> list[str]:
    """Format compact context around the requested line numbers."""
    display: set[int] = set()
    for line in anchor_lines:
        if line < 1 or line > len(file_lines):
            continue
        lo = max(1, line - MISMATCH_CONTEXT)
        hi = min(len(file_lines), line + MISMATCH_CONTEXT)
        for line_num in range(lo, hi + 1):
            display.add(line_num)
    anchor_set = set(anchor_lines)
    rows: list[str] = []
    previous = -1
    for line_num in sorted(display):
        if previous != -1 and line_num > previous + 1:
            rows.append("...")
        previous = line_num
        marker = "*" if line_num in anchor_set else " "
        rows.append(
            f"{marker}{format_numbered_line(line_num, file_lines[line_num - 1] if line_num - 1 < len(file_lines) else '')}"
        )
    return rows


def rejection_header(
    *,
    path: Optional[str],
    expected_file_hash: str,
    actual_file_hash: str,
    hash_recognized: bool = True,
) -> list[str]:
    """Port of ``MismatchError.rejectionHeader``."""
    path_text = f" for {path}" if path else ""
    expected = str(expected_file_hash or "").strip().upper()
    actual = str(actual_file_hash or "").strip().upper()
    if not hash_recognized:
        return [
            (
                f"Edit rejected{path_text}: hash {HL_FILE_HASH_SEP}{expected} "
                f"is not from this session."
            ),
            (
                f"The current file hashes to {HL_FILE_HASH_SEP}{actual}. "
                f"Re-read the file with `read_file` to copy a current "
                f"{HL_FILE_PREFIX}path{HL_FILE_HASH_SEP}tag{HL_FILE_SUFFIX} header "
                f"— never invent the tag and never reuse one from a prior session."
            ),
        ]
    return [
        f"Edit rejected{path_text}: file changed between read and edit.",
        (
            f"Section is bound to {HL_FILE_HASH_SEP}{expected}, but the current "
            f"file hashes to {HL_FILE_HASH_SEP}{actual}. If a prior edit in this "
            f"session modified this file, copy the "
            f"{HL_FILE_PREFIX}path{HL_FILE_HASH_SEP}newhash{HL_FILE_SUFFIX} header "
            f"from that edit's response; otherwise re-read the file with "
            f"`read_file` to refresh the tag before retrying."
        ),
    ]


def format_mismatch_message(
    *,
    path: Optional[str],
    expected_file_hash: str,
    actual_file_hash: str,
    hash_recognized: bool = True,
    file_lines: Optional[Sequence[str]] = None,
    anchor_lines: Optional[Sequence[int]] = None,
) -> str:
    """Port of ``MismatchError.formatMessage``."""
    lines = rejection_header(
        path=path,
        expected_file_hash=expected_file_hash,
        actual_file_hash=actual_file_hash,
        hash_recognized=hash_recognized,
    )
    if file_lines and anchor_lines:
        context = format_anchored_context(anchor_lines, file_lines)
        if context:
            lines.extend(["", *context])
    return "\n".join(lines)
