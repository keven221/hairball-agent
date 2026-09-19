"""Audit log for dashboard-auth events.

Profile-aware location: ``$HAIRBALL_HOME/logs/dashboard-auth.log``.
Format: one JSON object per line. Token-like fields are stripped before
serialisation to avoid leaking refresh tokens or JWTs to disk.

This module deliberately keeps a minimal dependency surface — no imports
from ``hairball_constants`` or other hairball_cli modules — so it can be
imported safely from middleware code that loads early in the startup
sequence.
"""
from __future__ import annotations

import datetime as _dt
import enum
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)
_write_lock = threading.Lock()

# Field names that must never appear in the log raw. Any kwarg matching
# these is silently dropped.
_REDACTED_FIELDS: frozenset = frozenset({
    "access_token", "refresh_token", "code", "code_verifier",
    "state", "ticket", "cookie", "Authorization", "authorization",
})


class AuditEvent(enum.Enum):
    """Event types written to dashboard-auth.log.

    Values are the literal ``event`` field on the JSON line.
    """

    LOGIN_START = "login_start"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    REFRESH_SUCCESS = "refresh_success"
    REFRESH_FAILURE = "refresh_failure"
    REVOKE = "revoke"
    SESSION_VERIFY_FAILURE = "session_verify_failure"
    WS_TICKET_MINTED = "ws_ticket_minted"
    WS_TICKET_REJECTED = "ws_ticket_rejected"
    TOKEN_AUTH_SUCCESS = "token_auth_success"
    TOKEN_AUTH_FAILURE = "token_auth_failure"


def _resolve_log_path() -> Path:
    """``$HAIRBALL_HOME/logs/dashboard-auth.log`` with the standard fallback.

    Mirrors ``hairball_constants.get_hairball_home`` semantics: env var wins,
    else ``~/.hairball``. A local copy avoids an import cycle with the
    middleware which lives below ``hairball_cli``.
    """
    home = os.environ.get("HAIRBALL_HOME") or str(Path.home() / ".hairball")
    return Path(home) / "logs" / "dashboard-auth.log"


def audit_log(event: AuditEvent, **fields: Any) -> None:
    """Append one event to the audit log.

    Token-like fields are dropped. Missing log directory is created.
    Write failures are logged at WARNING but never raise — auth must not
    fail because the audit logger broke.
    """
    safe_fields = {
        k: v for k, v in fields.items()
        if k not in _REDACTED_FIELDS
    }
    try:
        from hairball_cli.governance import redact_audit_payload

        safe_fields = redact_audit_payload(safe_fields, policy={})
    except Exception:
        # The legacy top-level deny set remains the minimum safety boundary if
        # the shared recursive redactor is unavailable during early startup.
        pass
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "event": event.value,
        **safe_fields,
    }
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    path = _resolve_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        _log.warning("dashboard-auth audit log write failed: %s", e)

    # When enterprise governance is explicitly enabled, project the same auth
    # decision into the profile-local tamper-evident ledger. Keep the legacy
    # JSONL surface for compatibility; the governance ledger is an opt-in
    # durable projection, not a replacement that could break login.
    try:
        from hairball_cli.governance import audit_event_if_enabled

        audit_event_if_enabled(
            {
                "event": f"auth.{event.value}",
                "actor": str(safe_fields.get("user_id") or safe_fields.get("email") or ""),
                "subject": str(safe_fields.get("provider") or "dashboard"),
                "payload": safe_fields,
            }
        )
    except Exception:
        # ``governance.audit.required`` deliberately propagates from the shared
        # helper. Ordinary optional audit remains fail-open like the legacy log.
        try:
            from hairball_cli.config import load_config_readonly

            audit_cfg = ((load_config_readonly() or {}).get("governance") or {}).get("audit") or {}
        except Exception:
            audit_cfg = {}
        if isinstance(audit_cfg, dict) and audit_cfg.get("required"):
            raise
        _log.warning("governance auth audit projection failed", exc_info=True)
