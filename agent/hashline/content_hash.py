"""Hairball content fingerprints for optimistic file-edit concurrency.

Third-party attribution and provenance live in ``THIRD_PARTY_NOTICES.md``;
the runtime contract here remains product-neutral.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, Optional

import xxhash

# Stable file-tag delimiters.
HL_FILE_HASH_LENGTH = 4
HL_FILE_PREFIX = "["
HL_FILE_SUFFIX = "]"
HL_FILE_HASH_SEP = "#"

CONTENT_HASH_LENGTH = HL_FILE_HASH_LENGTH  # alias for existing callers
ContentHashMode = Literal["off", "warn", "require"]

_TRAILING_WS_RE = re.compile(r"[ \t\r]+(?=\n|$)")


def normalize_file_hash_text(text: str) -> str:
    """Trim trailing ``[ \\t\\r]`` from every line (and the final line) so CRLF
    endings and display-trimmed lines do not invalidate a tag.
    """
    return _TRAILING_WS_RE.sub("", text)


def compute_file_hash(text: str) -> str:
    """Return a 4-hex uppercase xxHash32 fingerprint of normalized text.
    """
    normalized = normalize_file_hash_text(text)
    # The cross-runtime contract hashes UTF-8 bytes with seed zero.
    low16 = xxhash.xxh32(normalized.encode("utf-8"), seed=0).intdigest() & 0xFFFF
    return f"{low16:0{HL_FILE_HASH_LENGTH}X}"


def hash_file_on_disk(path: str) -> Optional[str]:
    """Hash the full on-disk file as UTF-8 text. None if unreadable."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    text = raw.decode("utf-8", errors="surrogatepass")
    return compute_file_hash(text)


def format_file_tag(path: str, content_hash: str) -> str:
    """Format a stable ``[path#ABCD]`` file tag."""
    return f"{HL_FILE_PREFIX}{path}{HL_FILE_HASH_SEP}{content_hash}{HL_FILE_SUFFIX}"


def resolve_content_hash_mode() -> ContentHashMode:
    """Resolve ``agent.content_hash_mode`` from config (default: warn)."""
    raw = "warn"
    try:
        from hairball_cli.config import load_config

        cfg = load_config() or {}
        agent = cfg.get("agent") if isinstance(cfg, dict) else None
        if isinstance(agent, dict) and agent.get("content_hash_mode") is not None:
            raw = agent.get("content_hash_mode")
    except Exception:
        pass
    env = os.environ.get("HAIRBALL_CONTENT_HASH_MODE")
    if env:
        raw = env
    mode = str(raw or "warn").strip().lower()
    if mode in ("off", "warn", "require"):
        return mode  # type: ignore[return-value]
    return "warn"


@dataclass(frozen=True)
class ContentHashDecision:
    """Result of a pre-write content-hash check."""

    ok: bool
    mode: ContentHashMode
    expected: Optional[str] = None
    actual: Optional[str] = None
    message: Optional[str] = None
    code: Optional[str] = None
    hash_recognized: Optional[bool] = None

    def as_error_payload(self, path: str) -> dict:
        from agent.hashline.mismatch import format_mismatch_message

        recognized = True if self.hash_recognized is None else bool(self.hash_recognized)
        message = self.message
        if self.expected and self.actual:
            message = format_mismatch_message(
                path=path,
                expected_file_hash=self.expected,
                actual_file_hash=self.actual,
                hash_recognized=recognized,
            )
        return {
            "error": message or "content hash mismatch",
            "code": self.code or "content_hash_mismatch",
            "path": path,
            "expected_hash": self.expected,
            "actual_hash": self.actual,
            "hash_recognized": recognized,
            "file_tag": format_file_tag(path, self.actual) if self.actual else None,
            "hint": (
                "Re-read the file with read_file, then retry the edit using the "
                f"new content_hash ({self.actual}). Do not overwrite blindly — "
                "the on-disk content changed since your last read."
                if self.actual
                else "Re-read the file with read_file, then retry the edit."
            ),
        }


def evaluate_content_hash(
    *,
    path: str,
    expected_hash: Optional[str],
    actual_hash: Optional[str],
    mode: Optional[ContentHashMode] = None,
    hash_recognized: Optional[bool] = None,
) -> ContentHashDecision:
    """Compare expected vs live content hash under the active mode."""
    active = mode or resolve_content_hash_mode()
    if active == "off":
        return ContentHashDecision(ok=True, mode=active)
    if not expected_hash:
        return ContentHashDecision(ok=True, mode=active)
    if actual_hash is None:
        return ContentHashDecision(ok=True, mode=active, expected=expected_hash)
    expected = expected_hash.strip().upper()
    actual = actual_hash.strip().upper()
    if expected == actual:
        return ContentHashDecision(
            ok=True, mode=active, expected=expected, actual=actual
        )
    recognized = True if hash_recognized is None else bool(hash_recognized)
    from agent.hashline.mismatch import format_mismatch_message

    msg = format_mismatch_message(
        path=path,
        expected_file_hash=expected,
        actual_file_hash=actual,
        hash_recognized=recognized,
    )
    if active == "require":
        return ContentHashDecision(
            ok=False,
            mode=active,
            expected=expected,
            actual=actual,
            message=msg,
            code="content_hash_mismatch",
            hash_recognized=recognized,
        )
    return ContentHashDecision(
        ok=True,
        mode=active,
        expected=expected,
        actual=actual,
        message=msg,
        code="content_hash_mismatch",
        hash_recognized=recognized,
    )
