"""DAP Content-Length framing (same wire family as LSP).

DAP uses Content-Length framed JSON messages over TCP/stdio. Bodies are not
JSON-RPC 2.0 envelopes — they use DAP ``type`` / ``command`` / ``event`` fields.
"""

from __future__ import annotations

import json
from typing import Any, Optional


class DapProtocolError(Exception):
    """Framing or envelope violation."""


def encode_message(obj: dict[str, Any]) -> bytes:
    body = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def try_extract_message(buf: bytearray) -> Optional[dict[str, Any]]:
    """If ``buf`` holds a full framed message, consume and return it; else None."""
    sep = b"\r\n\r\n"
    idx = buf.find(sep)
    if idx < 0:
        if len(buf) > 8192:
            raise DapProtocolError("DAP header block exceeded 8 KiB")
        return None
    header = bytes(buf[:idx])
    length: Optional[int] = None
    for line in header.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            try:
                length = int(line.split(b":", 1)[1].strip())
            except ValueError as exc:
                raise DapProtocolError(f"invalid Content-Length: {line!r}") from exc
    if length is None:
        raise DapProtocolError("Content-Length header missing")
    if length < 0 or length > 16_000_000:
        raise DapProtocolError(f"Content-Length out of range: {length}")
    start = idx + len(sep)
    if len(buf) < start + length:
        return None
    body = bytes(buf[start : start + length])
    del buf[: start + length]
    try:
        parsed = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DapProtocolError(f"invalid DAP JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DapProtocolError("DAP message must be a JSON object")
    return parsed
