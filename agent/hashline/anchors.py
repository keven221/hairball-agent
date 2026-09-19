"""Stable line anchors and transactional line-oriented edits.

The public seam is intentionally small: build/render anchors, then apply a
validated batch to an in-memory text snapshot.  Disk access, locking, path
policy, linting, and verification stay in the existing file-tool layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence


_FNV_OFFSET = 2_166_136_261
_FNV_PRIME = 16_777_619
_CHECKPOINT_SEED = 635_107_967
_ANCHOR_RE = re.compile(r"^(?P<line>[1-9]\d*):(?P<local>[a-z]{1,4})(?::(?P<context>[a-z]{1,4}))?$")
_DISPLAY_PREFIX_RE = re.compile(r"(?m)^\d+:[a-z]{1,4}(?::[a-z]{1,4})?→")
_ASCII_WS_RE = re.compile(r"[\t\n\v\f\r ]+")


AnchorScheme = Literal["content_only_v1", "chunk_v1", "checkpoint_v1"]


@dataclass(frozen=True)
class AnchorConfig:
    """Runtime-compatible anchor settings.

    The default constructor recovered from the shipped runtime is
    ``chunk`` with a three-letter hash and eight-line chunks.
    """

    hash_len: int = 3
    chunk_size: int = 8
    scheme: AnchorScheme = "chunk_v1"
    relocation_window: int = 10

    def __post_init__(self) -> None:
        if not 1 <= int(self.hash_len) <= 4:
            raise ValueError(f"hash_len must be between 1 and 4, got {self.hash_len}")
        if int(self.chunk_size) <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")
        if self.scheme not in {"content_only_v1", "chunk_v1", "checkpoint_v1"}:
            raise ValueError(f"unknown anchor scheme: {self.scheme}")
        if int(self.relocation_window) < 0:
            raise ValueError("relocation_window must be >= 0")


DEFAULT_CONFIG = AnchorConfig()


@dataclass(frozen=True)
class LineAnchor:
    line: int
    local: str
    context: str | None


@dataclass(frozen=True)
class AnchoredPage:
    content: str
    total_lines: int
    truncated: bool
    next_offset: int | None = None


@dataclass(frozen=True)
class AnchorEditResult:
    content: str
    edits_applied: int
    affected_lines: list[int] = field(default_factory=list)


class AnchorEditError(ValueError):
    """Structured, non-committing anchor-edit failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})
        self.retryable = (
            code in {"anchorStale", "anchorNotFound", "anchorAmbiguous"}
            if retryable is None
            else bool(retryable)
        )

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "success": False,
            "error": self.message,
            "code": self.code,
            "retryable": self.retryable,
        }
        if self.details:
            payload["details"] = self.details
        if self.retryable:
            payload["hint"] = (
                "Re-read the affected file with line_anchors=true and retry "
                "with the fresh anchor values. Every operation in one edits "
                "batch is resolved against that same read snapshot; never "
                "shift later anchor line numbers to account for earlier "
                "insert or replace operations in the batch."
            )
        return payload


@dataclass(frozen=True)
class _ParsedAnchor:
    line: int
    local: str
    context: str | None
    raw: str


@dataclass(frozen=True)
class _ResolvedEdit:
    index: int
    op: str
    start: int
    end: int
    content_lines: tuple[str, ...]


def normalize_anchor_line(line: str) -> str:
    """Collapse ASCII whitespace exactly as the anchor hash contract does."""

    return _ASCII_WS_RE.sub(" ", str(line)).strip(" \t\n\v\f\r")


def _fnv1a(data: bytes, *, seed: int = _FNV_OFFSET) -> int:
    value = seed & 0xFFFFFFFF
    for byte in data:
        value = ((value ^ byte) * _FNV_PRIME) & 0xFFFFFFFF
    return value


def _local_hash(line: str) -> int:
    return _fnv1a(normalize_anchor_line(line).encode("utf-8"))


def _render_hash(value: int, length: int) -> str:
    return "".join(
        chr(((value >> (8 * index)) & 0xFF) % 26 + ord("a"))
        for index in range(length)
    )


def _split_lines(text: str) -> tuple[list[str], str, bool]:
    if "\r\n" in text:
        ending = "\r\n"
    elif "\r" in text and "\n" not in text:
        ending = "\r"
    else:
        ending = "\n"
    had_final = bool(text) and text.endswith(("\n", "\r"))
    return text.splitlines(), ending, had_final


def _join_lines(lines: Sequence[str], ending: str, had_final: bool) -> str:
    if not lines:
        return ""
    joined = ending.join(lines)
    return joined + ending if had_final else joined


def _content_lines(content: str) -> tuple[str, ...]:
    if content == "":
        return ()
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    pieces = normalized.split("\n")
    if pieces and pieces[-1] == "":
        pieces.pop()
    return tuple(pieces)


def _chunk_context(lines: Sequence[str], index: int, config: AnchorConfig) -> int:
    start = (index // config.chunk_size) * config.chunk_size
    end = min(len(lines), start + config.chunk_size)
    value = 0
    for line in lines[start:end]:
        raw_hash = _fnv1a(line.encode("utf-8"))
        value = (_FNV_PRIME * ((value + raw_hash) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return value


def _checkpoint_context(lines: Sequence[str], index: int, config: AnchorConfig) -> int:
    start = (index // config.chunk_size) * config.chunk_size
    value = _CHECKPOINT_SEED
    for line in lines[start : index + 1]:
        value = (_FNV_PRIME * (value ^ _local_hash(line))) & 0xFFFFFFFF
    return value


def build_line_anchors(
    text: str, config: AnchorConfig = DEFAULT_CONFIG
) -> list[LineAnchor]:
    lines, _ending, _had_final = _split_lines(text)
    anchors: list[LineAnchor] = []
    for index, line in enumerate(lines):
        local = _render_hash(_local_hash(line), config.hash_len)
        if config.scheme == "content_only_v1":
            context = None
        elif config.scheme == "checkpoint_v1":
            context = _render_hash(
                _checkpoint_context(lines, index, config), config.hash_len
            )
        else:
            context = _render_hash(_chunk_context(lines, index, config), config.hash_len)
        anchors.append(LineAnchor(index + 1, local, context))
    return anchors


def format_anchor(anchor: LineAnchor, *, include_context: bool = True) -> str:
    if include_context and anchor.context:
        return f"{anchor.line}:{anchor.local}:{anchor.context}"
    return f"{anchor.line}:{anchor.local}"


def render_anchored_page(
    text: str,
    *,
    offset: int = 1,
    limit: int = 500,
    config: AnchorConfig = DEFAULT_CONFIG,
) -> AnchoredPage:
    if offset < 1:
        raise ValueError("offset must be >= 1")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    lines, _ending, _had_final = _split_lines(text)
    anchors = build_line_anchors(text, config)
    start = offset - 1
    end = min(len(lines), start + limit)
    rendered = [
        f"{format_anchor(anchors[index])}→{lines[index]}" for index in range(start, end)
    ]
    truncated = end < len(lines)
    return AnchoredPage(
        content="\n".join(rendered),
        total_lines=len(lines),
        truncated=truncated,
        next_offset=end + 1 if truncated else None,
    )


def _parse_anchor(value: Any, *, field_name: str) -> _ParsedAnchor:
    if not isinstance(value, str):
        raise AnchorEditError("invalidInput", f"{field_name} must be a string anchor.")
    match = _ANCHOR_RE.fullmatch(value.strip())
    if match is None:
        raise AnchorEditError(
            "invalidInput",
            f"Invalid {field_name}. Expected LINE:LOCAL or LINE:LOCAL:CONTEXT.",
            details={"field": field_name, "anchor": value},
            retryable=False,
        )
    return _ParsedAnchor(
        line=int(match.group("line")),
        local=match.group("local"),
        context=match.group("context"),
        raw=value.strip(),
    )


def _matches(parsed: _ParsedAnchor, candidate: LineAnchor) -> bool:
    return parsed.local == candidate.local and (
        parsed.context is None or parsed.context == candidate.context
    )


def _resolve_anchor(
    parsed: _ParsedAnchor,
    anchors: Sequence[LineAnchor],
    config: AnchorConfig,
    *,
    field_name: str,
) -> int:
    if parsed.line > len(anchors):
        raise AnchorEditError(
            "lineOutOfRange",
            f"{field_name} line {parsed.line} exceeds the current file length ({len(anchors)}).",
            details={"field": field_name, "line": parsed.line, "total_lines": len(anchors)},
            retryable=True,
        )
    exact = anchors[parsed.line - 1]
    if _matches(parsed, exact):
        return parsed.line

    lower = max(1, parsed.line - config.relocation_window)
    upper = min(len(anchors), parsed.line + config.relocation_window)
    matches = [
        candidate
        for candidate in anchors[lower - 1 : upper]
        if candidate.line != parsed.line and _matches(parsed, candidate)
    ]
    if len(matches) == 1:
        shifted = matches[0]
        fresh = format_anchor(shifted, include_context=parsed.context is not None)
        raise AnchorEditError(
            "anchorStale",
            f"{field_name} moved from line {parsed.line} to line {shifted.line}; no edit was applied.",
            details={
                "field": field_name,
                "original_anchor": parsed.raw,
                "shifted_line": shifted.line,
                "fresh_anchor": fresh,
            },
        )
    if len(matches) > 1:
        raise AnchorEditError(
            "anchorAmbiguous",
            f"Multiple nearby lines match {field_name}; no edit was applied.",
            details={
                "field": field_name,
                "anchor": parsed.raw,
                "matching_lines": [candidate.line for candidate in matches],
            },
        )
    raise AnchorEditError(
        "anchorNotFound",
        f"{field_name} no longer matches the current file; no edit was applied.",
        details={"field": field_name, "anchor": parsed.raw},
    )


def _require_clean_content(content: Any, *, op: str) -> str:
    if not isinstance(content, str):
        raise AnchorEditError("invalidInput", f"{op} requires string field 'content'.")
    if _DISPLAY_PREFIX_RE.search(content):
        raise AnchorEditError(
            "invalidInput",
            f"{op} content contains line-anchor display prefixes. Use clean file content.",
            retryable=False,
        )
    return content


def _validate_operations(
    lines: Sequence[str], edits: Sequence[Mapping[str, Any]], config: AnchorConfig
) -> tuple[list[_ResolvedEdit], AnchorEditResult | None]:
    if not edits:
        raise AnchorEditError("invalidInput", "No edit operations provided.", retryable=False)
    if not all(isinstance(edit, Mapping) for edit in edits):
        raise AnchorEditError("invalidInput", "Every edit must be an object.", retryable=False)

    write_ops = [edit for edit in edits if edit.get("op") == "write"]
    if write_ops:
        if len(edits) != 1:
            raise AnchorEditError(
                "invalidInput",
                "Write operation must be the only operation. Batch edits with other operations are not allowed.",
                retryable=False,
            )
        content = _require_clean_content(write_ops[0].get("content"), op="write")
        return [], AnchorEditResult(content=content, edits_applied=1, affected_lines=list(range(1, len(lines) + 1)))

    anchors = build_line_anchors(_join_lines(lines, "\n", False), config)
    resolved: list[_ResolvedEdit] = []
    for index, edit in enumerate(edits):
        op = edit.get("op")
        if op not in {"replace", "insert_after"}:
            raise AnchorEditError(
                "invalidInput",
                "Edit op must be one of: replace, insert_after, write.",
                details={"index": index, "op": op},
                retryable=False,
            )
        content = _require_clean_content(edit.get("content"), op=str(op))
        start_anchor = _parse_anchor(edit.get("anchor"), field_name="anchor")
        start = _resolve_anchor(start_anchor, anchors, config, field_name="anchor")
        end = start
        if op == "replace" and edit.get("endAnchor") is not None:
            end_anchor = _parse_anchor(edit.get("endAnchor"), field_name="endAnchor")
            end = _resolve_anchor(end_anchor, anchors, config, field_name="endAnchor")
            if end < start:
                raise AnchorEditError(
                    "invalidInput",
                    f"endAnchor line {end} is before start anchor line {start}.",
                    details={"start_line": start, "end_line": end},
                    retryable=False,
                )
        resolved.append(
            _ResolvedEdit(index, str(op), start, end, _content_lines(content))
        )

    replacements = [edit for edit in resolved if edit.op == "replace"]
    for left_index, left in enumerate(replacements):
        for right in replacements[left_index + 1 :]:
            if left.start <= right.end and right.start <= left.end:
                raise AnchorEditError(
                    "overlappingEdits",
                    f"Edit ranges overlap: {left.start}-{left.end} and {right.start}-{right.end}.",
                    details={"left": [left.start, left.end], "right": [right.start, right.end]},
                    retryable=False,
                )
    for insert in (edit for edit in resolved if edit.op == "insert_after"):
        for replace in replacements:
            if replace.start <= insert.start <= replace.end:
                raise AnchorEditError(
                    "overlappingEdits",
                    f"Insert after line {insert.start} overlaps replace range {replace.start}-{replace.end}.",
                    details={"insert_line": insert.start, "replace": [replace.start, replace.end]},
                    retryable=False,
                )
    return resolved, None


def apply_anchor_edits(
    text: str,
    edits: Sequence[Mapping[str, Any]],
    config: AnchorConfig = DEFAULT_CONFIG,
) -> AnchorEditResult:
    """Validate every operation against one snapshot, then apply all or none."""

    if isinstance(edits, (str, bytes)) or not isinstance(edits, Sequence):
        raise AnchorEditError("invalidInput", "edits must be an array.", retryable=False)
    lines, ending, had_final = _split_lines(text)
    resolved, direct = _validate_operations(lines, edits, config)
    if direct is not None:
        return direct

    output = list(lines)
    affected: set[int] = set()
    # Descending source positions keep all already-resolved original indexes
    # valid.  Reverse input order for equal-position inserts so their final
    # file order matches the request order.
    ordered = sorted(
        resolved,
        key=lambda edit: (edit.start, 1 if edit.op == "replace" else 0, edit.index),
        reverse=True,
    )
    for edit in ordered:
        if edit.op == "replace":
            output[edit.start - 1 : edit.end] = edit.content_lines
            affected.update(range(edit.start, edit.end + 1))
        else:
            output[edit.start : edit.start] = edit.content_lines
            affected.add(edit.start)
    return AnchorEditResult(
        content=_join_lines(output, ending, had_final),
        edits_applied=len(resolved),
        affected_lines=sorted(affected),
    )


__all__ = [
    "AnchorConfig",
    "AnchorEditError",
    "AnchorEditResult",
    "AnchoredPage",
    "DEFAULT_CONFIG",
    "LineAnchor",
    "apply_anchor_edits",
    "build_line_anchors",
    "format_anchor",
    "normalize_anchor_line",
    "render_anchored_page",
]
