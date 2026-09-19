"""Append-only vision-frame savings journal (OMP snapcompact-savings contract)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("agent.snapcompact.journal")


def savings_journal_path() -> Path:
    try:
        from hairball_constants import get_hairball_home

        return get_hairball_home() / "snapcompact-savings.jsonl"
    except Exception:  # noqa: BLE001
        return Path.home() / ".hairball" / "snapcompact-savings.jsonl"


def append_savings(
    records: list[dict[str, Any]],
    *,
    journal_path: Optional[Path] = None,
) -> None:
    """Fire-and-forget NDJSON append. Never raises into the request path."""
    lines: list[str] = []
    ts = int(time.time() * 1000)
    for rec in records or []:
        saved = int(rec.get("savedTokens") or rec.get("saved_tokens") or 0)
        if saved <= 0:
            continue
        lines.append(
            json.dumps(
                {
                    "ts": ts,
                    "session": str(rec.get("session") or ""),
                    "provider": str(rec.get("provider") or ""),
                    "model": str(rec.get("model") or ""),
                    "toolCallId": str(rec.get("toolCallId") or rec.get("tool_call_id") or "compress"),
                    "savedTokens": saved,
                },
                ensure_ascii=False,
            )
        )
    if not lines:
        return
    path = journal_path or savings_journal_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.debug("snapcompact savings journal append failed: %s", exc)


def read_savings(journal_path: Optional[Path] = None) -> list[dict[str, Any]]:
    path = journal_path or savings_journal_path()
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except Exception:  # noqa: BLE001
        return []
    return out
