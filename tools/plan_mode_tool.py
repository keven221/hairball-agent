"""Hairball Plan-mode transition tool.

The tool does not approve a plan or mutate the session.  It only proves that
the designated plan file exists and is non-empty; product surfaces then ask
the user whether to enter Build mode.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agent.plan_execution import (
    PlanContractError,
    resolve_plan_file,
    resolve_plan_title,
)
from tools.registry import registry


PLAN_EXIT_SCHEMA = {
    "name": "plan_exit",
    "description": (
        "Signal that the Hairball plan file is complete and ready for user "
        "review. Call this only after writing the full plan in Plan mode."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title for the completed plan review.",
            }
        },
        "additionalProperties": False,
    },
}


def plan_exit(
    *,
    session_id: str,
    db: Any = None,
    cwd: str | None = None,
    title: Any = None,
) -> str:
    """Validate the session's designated plan file without changing state."""

    owned_db = None
    try:
        if db is None:
            from hairball_state import SessionDB

            owned_db = SessionDB()
            db = owned_db
        try:
            path = resolve_plan_file(session_id, db=db, cwd=cwd)
        except PlanContractError as exc:
            return json.dumps(exc.to_dict(), ensure_ascii=False)
        try:
            content = Path(path).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return json.dumps(
                {
                    "success": False,
                    "code": "plan_file_missing",
                    "error": "Write the designated Hairball plan file before calling plan_exit.",
                    "plan_file": str(path),
                },
                ensure_ascii=False,
            )
        except OSError as exc:
            return json.dumps(
                {
                    "success": False,
                    "code": "plan_file_unreadable",
                    "error": str(exc),
                    "plan_file": str(path),
                },
                ensure_ascii=False,
            )
        if not content:
            return json.dumps(
                {
                    "success": False,
                    "code": "plan_file_empty",
                    "error": "The designated Hairball plan file is empty.",
                    "plan_file": str(path),
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "success": True,
                "status": "ready_for_approval",
                "plan_file": str(path),
                "title": resolve_plan_title(content, path, title),
                "digest": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            },
            ensure_ascii=False,
        )
    finally:
        if owned_db is not None:
            owned_db.close()


registry.register(
    name="plan_exit",
    toolset="planning",
    schema=PLAN_EXIT_SCHEMA,
    handler=lambda _args, **kw: plan_exit(
        session_id=str(kw.get("session_id") or kw.get("task_id") or ""),
        db=kw.get("session_db"),
        cwd=kw.get("cwd") or kw.get("working_dir"),
        title=(_args or {}).get("title"),
    ),
    emoji="📋",
)


__all__ = ["PLAN_EXIT_SCHEMA", "plan_exit"]
