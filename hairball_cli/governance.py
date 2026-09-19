"""Hairball-native managed policy, audit, retention, and export contracts.

The module is inert unless a managed configuration explicitly enables
``governance``.  It contains no model tool and never mutates conversation
messages or the system prompt.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hairball_cli.managed_scope import ManagedScopeError


_CAPABILITY_AXES = ("toolsets", "skills", "plugins", "providers")
_log = logging.getLogger(__name__)


def _names(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        value = [value]
    try:
        return {str(item).strip() for item in value if str(item).strip()}
    except TypeError:
        return set()


def _axis(policy: Mapping[str, Any] | None, name: str) -> tuple[set[str], set[str]]:
    raw = (policy or {}).get(name)
    if not isinstance(raw, Mapping):
        return set(), set()
    return _names(raw.get("allow")), _names(raw.get("deny"))


@dataclass(frozen=True)
class PolicyValidation:
    valid: bool
    violations: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class EffectivePolicy:
    policy: dict[str, Any]
    violations: tuple[dict[str, str], ...] = ()
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimePolicyScope:
    enabled_toolsets: list[str] | None
    disabled_toolsets: list[str] | None
    visible_toolsets: list[str] | None = None
    policy_active: bool = False


@dataclass(frozen=True)
class CapabilityDecision:
    axis: str
    value: str
    allowed: bool
    reason: str
    policy_active: bool


@dataclass(frozen=True)
class ManagedDecision:
    key: str
    requested_value: Any
    effective_value: Any
    allowed: bool
    source: str
    reason: str


@dataclass(frozen=True)
class AuditCursor:
    sequence: int
    digest: str


@dataclass(frozen=True)
class AuditPage:
    events: tuple[dict[str, Any], ...]
    next_cursor: AuditCursor | None
    has_more: bool
    chain_valid: bool


@dataclass(frozen=True)
class AuditVerification:
    valid: bool
    checked: int
    broken_sequence: int | None = None


@dataclass(frozen=True)
class RetentionOutcome:
    scanned: int
    deleted: int
    retained: int
    legal_hold: bool
    cutoff: str | None = None


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    sha256: str
    event_count: int


class PolicyDenied(PermissionError):
    """A managed policy rejected a runtime capability before side effects."""

    def __init__(self, axis: str, value: str, reason: str):
        super().__init__(reason)
        self.axis = axis
        self.value = value
        self.reason = reason


def validate_policy_narrowing(
    parent: Mapping[str, Any] | None,
    child: Mapping[str, Any] | None,
) -> PolicyValidation:
    """Report attempts by *child* to expand an already effective policy."""
    violations: list[dict[str, str]] = []
    for axis_name in _CAPABILITY_AXES:
        parent_allow, _ = _axis(parent, axis_name)
        child_allow, _ = _axis(child, axis_name)
        if parent_allow and child_allow and not child_allow.issubset(parent_allow):
            violations.append(
                {
                    "axis": axis_name,
                    "kind": "allow_expansion",
                    "detail": ",".join(sorted(child_allow - parent_allow)),
                }
            )
        # Denylists are additive by contract. A child list names new denies;
        # omission does not express removal, and the union below makes removal
        # impossible. Reporting omitted parent values as a violation would be
        # a false positive for normal layered policy authoring.
    return PolicyValidation(valid=not violations, violations=tuple(violations))


def _merge_policy_pair(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for axis_name in _CAPABILITY_AXES:
        parent_allow, parent_deny = _axis(parent, axis_name)
        child_allow, child_deny = _axis(child, axis_name)
        if parent_allow and child_allow:
            allow = parent_allow & child_allow
        else:
            allow = parent_allow or child_allow
        deny = parent_deny | child_deny
        allow -= deny
        merged[axis_name] = {"allow": sorted(allow), "deny": sorted(deny)}
    return merged


def merge_policy_layers(
    system: Mapping[str, Any] | None,
    organization: Mapping[str, Any] | None,
    project: Mapping[str, Any] | None,
    user: Mapping[str, Any] | None,
    session: Mapping[str, Any] | None,
) -> EffectivePolicy:
    """Compose five policy layers while making expansion mathematically impossible."""
    effective: dict[str, Any] = {
        name: {"allow": [], "deny": []} for name in _CAPABILITY_AXES
    }
    violations: list[dict[str, str]] = []
    sources: list[str] = []
    for source, layer in zip(
        ("system", "organization", "project", "user", "session"),
        (system, organization, project, user, session),
    ):
        if not isinstance(layer, Mapping) or not layer:
            continue
        validation = validate_policy_narrowing(effective, layer)
        violations.extend({**item, "source": source} for item in validation.violations)
        effective = _merge_policy_pair(effective, layer)
        sources.append(source)
    return EffectivePolicy(
        policy=effective,
        violations=tuple(violations),
        sources=tuple(sources),
    )


def effective_policy_from_config(config: Mapping[str, Any] | None = None) -> EffectivePolicy:
    """Resolve the optional managed governance block from an effective config."""
    if config is None:
        try:
            from hairball_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            config = {}
    governance = (config or {}).get("governance")
    if not isinstance(governance, Mapping) or not governance.get("enabled"):
        return EffectivePolicy(policy={}, sources=())
    layers = governance.get("layers")
    if isinstance(layers, Mapping):
        return merge_policy_layers(
            layers.get("system"),
            layers.get("organization"),
            layers.get("project"),
            layers.get("user"),
            layers.get("session"),
        )
    policy = governance.get("policy")
    if not isinstance(policy, Mapping):
        policy = {}
    normalized = _merge_policy_pair({}, policy)
    return EffectivePolicy(policy=normalized, sources=("managed",))


def apply_runtime_policy(
    *,
    provider: str | None,
    enabled_toolsets: list[str] | None,
    disabled_toolsets: list[str] | None,
    visible_toolsets: list[str] | None = None,
    config: Mapping[str, Any] | None = None,
) -> RuntimePolicyScope:
    """Apply provider/tool policy at the common AIAgent construction seam."""
    if config is None:
        try:
            from hairball_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            config = {}

    effective = effective_policy_from_config(config)
    agent_config = (config or {}).get("agent")
    configured_disabled = _names(
        agent_config.get("disabled_toolsets")
        if isinstance(agent_config, Mapping)
        else None
    )
    if not effective.sources:
        enabled = (
            None
            if enabled_toolsets is None
            else sorted(_names(enabled_toolsets) - configured_disabled)
        )
        visible = (
            None
            if visible_toolsets is None
            else sorted(_names(visible_toolsets) - configured_disabled)
        )
        return RuntimePolicyScope(
            enabled_toolsets=enabled,
            disabled_toolsets=sorted(
                _names(disabled_toolsets) | configured_disabled
            )
            or None,
            visible_toolsets=visible,
            policy_active=False,
        )

    provider_name = str(provider or "auto").strip().lower() or "auto"
    provider_allow, provider_deny = _axis(effective.policy, "providers")
    provider_allow = {name.lower() for name in provider_allow}
    provider_deny = {name.lower() for name in provider_deny}
    if provider_name in provider_deny or (
        provider_allow and provider_name not in provider_allow
    ):
        audit_event_if_enabled(
            {
                "event": "policy.provider.denied",
                "subject": provider_name,
                "payload": {"reason": "explicit_deny" if provider_name in provider_deny else "not_in_allowlist"},
            },
            config=config,
        )
        raise PolicyDenied(
            "providers",
            provider_name,
            f"Provider '{provider_name}' is blocked by managed Hairball policy",
        )

    tool_allow, tool_deny = _axis(effective.policy, "toolsets")
    disabled = _names(disabled_toolsets) | configured_disabled | tool_deny
    if enabled_toolsets is None:
        enabled = set(tool_allow) if tool_allow else None
    else:
        enabled = _names(enabled_toolsets)
        if tool_allow:
            enabled &= tool_allow
        enabled -= disabled
    visible = None if visible_toolsets is None else (_names(visible_toolsets) - disabled)
    if visible is not None and tool_allow:
        visible &= tool_allow

    result = RuntimePolicyScope(
        enabled_toolsets=None if enabled is None else sorted(enabled),
        disabled_toolsets=sorted(disabled) or None,
        visible_toolsets=None if visible is None else sorted(visible),
        policy_active=True,
    )
    audit_event_if_enabled(
        {
            "event": "policy.runtime.applied",
            "subject": provider_name,
            "payload": {
                "enabled_toolsets": result.enabled_toolsets,
                "disabled_toolsets": result.disabled_toolsets,
                "visible_toolsets": result.visible_toolsets,
            },
        },
        config=config,
    )
    return result


def evaluate_capability(
    axis: str,
    value: str,
    *,
    aliases: tuple[str, ...] = (),
    config: Mapping[str, Any] | None = None,
) -> CapabilityDecision:
    """Evaluate one named skill/plugin/provider/toolset against effective policy."""
    if axis not in _CAPABILITY_AXES:
        raise ValueError(f"unsupported governance axis: {axis}")
    effective = effective_policy_from_config(config)
    primary = str(value or "").strip()
    if not effective.sources:
        return CapabilityDecision(axis, primary, True, "policy_inactive", False)
    allow, deny = _axis(effective.policy, axis)
    candidates = {primary, *(str(alias).strip() for alias in aliases if str(alias).strip())}
    if axis == "providers":
        allow = {item.lower() for item in allow}
        deny = {item.lower() for item in deny}
        candidates = {item.lower() for item in candidates}
    if candidates & deny:
        decision = CapabilityDecision(axis, primary, False, "explicit_deny", True)
        audit_event_if_enabled(
            {"event": f"policy.{axis}.denied", "subject": primary, "payload": {"reason": decision.reason}},
            config=config,
        )
        return decision
    if allow and not (candidates & allow):
        decision = CapabilityDecision(axis, primary, False, "not_in_allowlist", True)
        audit_event_if_enabled(
            {"event": f"policy.{axis}.denied", "subject": primary, "payload": {"reason": decision.reason}},
            config=config,
        )
        return decision
    return CapabilityDecision(axis, primary, True, "allowed", True)


def filter_provider_routes(
    routes: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Remove fallback/auxiliary routes that would bypass provider policy."""
    kept: list[dict[str, Any]] = []
    for route in routes or ():
        if not isinstance(route, Mapping):
            continue
        provider = str(route.get("provider") or "auto")
        decision = evaluate_capability("providers", provider, config=config)
        if decision.allowed:
            kept.append(dict(route))
    return kept


def enforce_managed_setting(key: str, value: Any, source: str) -> ManagedDecision:
    """Explain whether a requested value conflicts with an administrator pin."""
    from hairball_cli.managed_scope import is_key_managed, load_managed_config

    dotted = str(key or "").strip()
    if not dotted or not is_key_managed(dotted):
        return ManagedDecision(dotted, value, value, True, str(source), "unmanaged")
    current: Any = load_managed_config()
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            current = None
            break
        current = current[part]
    same = value == current
    return ManagedDecision(
        dotted,
        value,
        current,
        same,
        "managed",
        "matches_managed_value" if same else "administrator_pinned",
    )


def redact_audit_payload(
    event: Mapping[str, Any] | Any,
    policy: Mapping[str, Any] | None = None,
) -> Any:
    """Return a recursively redacted copy suitable for durable audit storage."""
    from agent.redact import redact_sensitive_value

    redacted = redact_sensitive_value(event, force=True)
    extra_keys = _names((policy or {}).get("redact_fields"))
    if not extra_keys:
        return redacted

    def _extra(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "«redacted-policy»" if str(key) in extra_keys else _extra(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [_extra(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_extra(item) for item in value)
        return value

    return _extra(redacted)


_AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS governance_audit (
    seq INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL DEFAULT '',
    subject TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_governance_audit_ts ON governance_audit(ts, seq);
CREATE INDEX IF NOT EXISTS idx_governance_audit_event ON governance_audit(event_type, seq);
CREATE TABLE IF NOT EXISTS governance_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""
_CHAIN_ROOT = "0" * 64


def _audit_db_path() -> Path:
    home = Path(os.environ.get("HAIRBALL_HOME") or (Path.home() / ".hairball"))
    return home / "logs" / "governance.db"


def _scope_id() -> str:
    home = Path(os.environ.get("HAIRBALL_HOME") or (Path.home() / ".hairball"))
    profile = str(os.environ.get("HAIRBALL_PROFILE") or "").strip()
    home_digest = hashlib.sha256(str(home.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"{profile or 'default'}:{home_digest}"


def _meta_key(name: str, scope_id: str | None = None) -> str:
    """Namespace mutable chain metadata by the current tenant/profile scope."""
    return f"{name}:{scope_id or _scope_id()}"


def _connect_audit(*, create: bool = True) -> sqlite3.Connection | None:
    path = _audit_db_path()
    if not create and not path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    conn.executescript(_AUDIT_SCHEMA)
    return conn


def _event_document(
    *,
    seq: int,
    ts: str,
    scope_id: str,
    event_type: str,
    actor: str,
    subject: str,
    payload: Any,
    prev_hash: str,
) -> dict[str, Any]:
    return {
        "seq": seq,
        "ts": ts,
        "scope_id": scope_id,
        "event_type": event_type,
        "actor": actor,
        "subject": subject,
        "payload": payload,
        "prev_hash": prev_hash,
    }


def _audit_signing_key() -> bytes | None:
    try:
        from hairball_cli.managed_scope import load_managed_env

        value = load_managed_env().get("HAIRBALL_GOVERNANCE_AUDIT_KEY")
    except Exception:
        value = None
    value = value or os.environ.get("HAIRBALL_GOVERNANCE_AUDIT_KEY")
    if not value:
        return None
    return str(value).encode("utf-8")


def _document_hash(document: Mapping[str, Any], *, signing_key: bytes | None = None) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if signing_key:
        return hmac.new(signing_key, encoded, hashlib.sha256).hexdigest()
    return hashlib.sha256(encoded).hexdigest()


def audit_hash_mode() -> str:
    """Return the current scope's durable authenticity mode without its key."""
    conn = _connect_audit(create=False)
    if conn is not None:
        try:
            row = conn.execute(
                "SELECT value FROM governance_meta WHERE key=?",
                (_meta_key("hash_mode"),),
            ).fetchone()
            if row is not None:
                return str(row["value"])
        finally:
            conn.close()
    return "hmac-sha256" if _audit_signing_key() else "sha256-chain"


def append_audit_event(
    event: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] | None = None,
) -> AuditCursor:
    """Atomically append one redacted, hash-chained event to the current profile."""
    event_type = str(event.get("event") or event.get("event_type") or "unknown").strip()
    actor = str(event.get("actor") or "")
    subject = str(event.get("subject") or "")
    ts = str(event.get("ts") or dt.datetime.now(dt.timezone.utc).isoformat())
    payload = event.get("payload")
    if payload is None:
        payload = {
            key: value
            for key, value in event.items()
            if key not in {"event", "event_type", "actor", "subject", "ts"}
        }
    payload = redact_audit_payload(payload, policy=policy)
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    scope_id = _scope_id()
    signing_key = _audit_signing_key()
    hash_mode = "hmac-sha256" if signing_key else "sha256-chain"

    conn = _connect_audit(create=True)
    assert conn is not None
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing_mode = conn.execute(
            "SELECT value FROM governance_meta WHERE key=?",
            (_meta_key("hash_mode", scope_id),),
        ).fetchone()
        if existing_mode is not None and str(existing_mode["value"]) != hash_mode:
            raise ManagedScopeError(
                "governance audit signing mode changed; export/rotate the existing ledger first"
            )
        last = conn.execute(
            "SELECT seq, event_hash FROM governance_audit "
            "WHERE scope_id=? ORDER BY seq DESC LIMIT 1",
            (scope_id,),
        ).fetchone()
        if last is not None:
            prev_hash = str(last["event_hash"])
        else:
            anchor = conn.execute(
                "SELECT value FROM governance_meta WHERE key=?",
                (_meta_key("anchor_hash", scope_id),),
            ).fetchone()
            prev_hash = str(anchor["value"]) if anchor is not None else _CHAIN_ROOT
        global_last = conn.execute(
            "SELECT MAX(seq) AS seq FROM governance_audit"
        ).fetchone()
        if global_last is not None and global_last["seq"] is not None:
            seq = int(global_last["seq"]) + 1
        else:
            seq_row = conn.execute(
                "SELECT value FROM governance_meta WHERE key=?",
                (_meta_key("last_sequence", scope_id),),
            ).fetchone()
            seq = int(seq_row["value"]) + 1 if seq_row is not None else 1
        document = _event_document(
            seq=seq,
            ts=ts,
            scope_id=scope_id,
            event_type=event_type,
            actor=actor,
            subject=subject,
            payload=payload,
            prev_hash=prev_hash,
        )
        event_hash = _document_hash(document, signing_key=signing_key)
        conn.execute(
            "INSERT INTO governance_audit"
            "(seq,ts,scope_id,event_type,actor,subject,payload_json,prev_hash,event_hash) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (seq, ts, scope_id, event_type, actor, subject, payload_json, prev_hash, event_hash),
        )
        conn.execute(
            "INSERT INTO governance_meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (_meta_key("last_sequence", scope_id), str(seq)),
        )
        conn.execute(
            "INSERT INTO governance_meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (_meta_key("hash_mode", scope_id), hash_mode),
        )
        conn.commit()
        return AuditCursor(sequence=seq, digest=event_hash)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def audit_event_if_enabled(
    event: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
) -> AuditCursor | None:
    """Append only when governance.audit.enabled; optionally fail closed."""
    if config is None:
        try:
            from hairball_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            config = {}
    governance = (config or {}).get("governance")
    audit = governance.get("audit") if isinstance(governance, Mapping) else None
    if not isinstance(audit, Mapping) or not audit.get("enabled"):
        return None
    if audit.get("require_signing") and _audit_signing_key() is None:
        raise ManagedScopeError(
            "governance audit signing key is required but HAIRBALL_GOVERNANCE_AUDIT_KEY is unavailable"
        )
    try:
        return append_audit_event(event, policy=audit)
    except Exception:
        if audit.get("required"):
            raise
        _log.warning("governance audit append failed", exc_info=True)
        return None


def _row_document(row: sqlite3.Row) -> dict[str, Any]:
    payload = json.loads(str(row["payload_json"]))
    return _event_document(
        seq=int(row["seq"]),
        ts=str(row["ts"]),
        scope_id=str(row["scope_id"]),
        event_type=str(row["event_type"]),
        actor=str(row["actor"]),
        subject=str(row["subject"]),
        payload=payload,
        prev_hash=str(row["prev_hash"]),
    )


def verify_audit_chain() -> AuditVerification:
    """Verify the current profile's complete retained chain."""
    conn = _connect_audit(create=False)
    if conn is None:
        return AuditVerification(valid=True, checked=0)
    try:
        scope_id = _scope_id()
        anchor = conn.execute(
            "SELECT value FROM governance_meta WHERE key=?",
            (_meta_key("anchor_hash", scope_id),),
        ).fetchone()
        expected_prev = str(anchor["value"]) if anchor is not None else _CHAIN_ROOT
        mode_row = conn.execute(
            "SELECT value FROM governance_meta WHERE key=?",
            (_meta_key("hash_mode", scope_id),),
        ).fetchone()
        mode = str(mode_row["value"]) if mode_row is not None else "sha256-chain"
        signing_key = _audit_signing_key() if mode == "hmac-sha256" else None
        if mode == "hmac-sha256" and signing_key is None:
            return AuditVerification(False, 0, 0)
        checked = 0
        for row in conn.execute(
            "SELECT * FROM governance_audit WHERE scope_id=? ORDER BY seq",
            (scope_id,),
        ):
            document = _row_document(row)
            seq = int(row["seq"])
            if str(row["prev_hash"]) != expected_prev:
                return AuditVerification(False, checked, seq)
            if _document_hash(document, signing_key=signing_key) != str(row["event_hash"]):
                return AuditVerification(False, checked, seq)
            expected_prev = str(row["event_hash"])
            checked += 1
        return AuditVerification(True, checked)
    finally:
        conn.close()


def query_audit_events(
    filters: Mapping[str, Any] | None,
    cursor: AuditCursor | int | None,
    *,
    limit: int = 100,
) -> AuditPage:
    """Cursor-page the current profile without accepting a cross-profile scope."""
    filters = filters or {}
    requested_scope = filters.get("scope_id")
    current_scope = _scope_id()
    if requested_scope and str(requested_scope) != current_scope:
        raise PolicyDenied("scope", str(requested_scope), "cross-profile audit query denied")
    after = cursor.sequence if isinstance(cursor, AuditCursor) else int(cursor or 0)
    limit = max(1, min(int(limit), 1000))
    conn = _connect_audit(create=False)
    if conn is None:
        return AuditPage((), None, False, True)
    try:
        clauses = ["scope_id = ?", "seq > ?"]
        params: list[Any] = [current_scope, after]
        if filters.get("event") or filters.get("event_type"):
            clauses.append("event_type = ?")
            params.append(str(filters.get("event") or filters.get("event_type")))
        if filters.get("actor"):
            clauses.append("actor = ?")
            params.append(str(filters["actor"]))
        if filters.get("since"):
            clauses.append("ts >= ?")
            params.append(str(filters["since"]))
        if filters.get("until"):
            clauses.append("ts <= ?")
            params.append(str(filters["until"]))
        rows = conn.execute(
            "SELECT * FROM governance_audit WHERE " + " AND ".join(clauses)
            + " ORDER BY seq LIMIT ?",
            (*params, limit + 1),
        ).fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        events = tuple(
            {**_row_document(row), "hash": str(row["event_hash"])} for row in rows
        )
        next_cursor = (
            AuditCursor(int(rows[-1]["seq"]), str(rows[-1]["event_hash"])) if rows else None
        )
        return AuditPage(events, next_cursor, has_more, verify_audit_chain().valid)
    finally:
        conn.close()


def _audit_policy_from_config() -> dict[str, Any]:
    try:
        from hairball_cli.config import load_config_readonly

        governance = (load_config_readonly() or {}).get("governance") or {}
        audit = governance.get("audit") if isinstance(governance, dict) else {}
        return dict(audit) if isinstance(audit, dict) else {}
    except Exception:
        return {}


def apply_retention_policy(
    now: dt.datetime,
    policy: Mapping[str, Any] | None = None,
) -> RetentionOutcome:
    """Delete expired current-profile events while preserving a verifiable anchor."""
    policy = dict(policy or _audit_policy_from_config())
    days = int(policy.get("retention_days") or 0)
    legal_hold = bool(policy.get("legal_hold"))
    conn = _connect_audit(create=False)
    if conn is None:
        return RetentionOutcome(0, 0, 0, legal_hold, None)
    try:
        total = int(
            conn.execute(
                "SELECT COUNT(*) FROM governance_audit WHERE scope_id=?", (_scope_id(),)
            ).fetchone()[0]
        )
    finally:
        conn.close()
    if legal_hold or days <= 0:
        return RetentionOutcome(total, 0, total, legal_hold, None)
    verification = verify_audit_chain()
    if not verification.valid:
        raise ManagedScopeError(
            f"governance audit chain is broken at sequence {verification.broken_sequence}; retention refused"
        )
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    cutoff = (now.astimezone(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()

    conn = _connect_audit(create=True)
    assert conn is not None
    try:
        conn.execute("BEGIN IMMEDIATE")
        doomed = conn.execute(
            "SELECT seq,event_hash FROM governance_audit "
            "WHERE scope_id=? AND ts < ? ORDER BY seq",
            (_scope_id(), cutoff),
        ).fetchall()
        if not doomed:
            conn.commit()
            return RetentionOutcome(total, 0, total, False, cutoff)
        last_deleted_seq = int(doomed[-1]["seq"])
        first_kept = conn.execute(
            "SELECT prev_hash FROM governance_audit WHERE scope_id=? AND seq>? ORDER BY seq LIMIT 1",
            (_scope_id(), last_deleted_seq),
        ).fetchone()
        anchor_hash = (
            str(first_kept["prev_hash"])
            if first_kept is not None
            else str(doomed[-1]["event_hash"])
        )
        conn.execute(
            "INSERT INTO governance_meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (_meta_key("anchor_hash"), anchor_hash),
        )
        conn.execute(
            "DELETE FROM governance_audit WHERE scope_id=? AND ts < ?",
            (_scope_id(), cutoff),
        )
        conn.commit()
        deleted = len(doomed)
        return RetentionOutcome(total, deleted, total - deleted, False, cutoff)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def export_audit_bundle(scope: Mapping[str, Any] | None = None) -> ArtifactRef:
    """Export current-profile filtered events and a verification manifest as ZIP."""
    filters = dict(scope or {})
    verification = verify_audit_chain()
    if not verification.valid:
        raise ManagedScopeError(
            f"governance audit chain is broken at sequence {verification.broken_sequence}; export refused"
        )
    events: list[dict[str, Any]] = []
    cursor: AuditCursor | None = None
    while True:
        page = query_audit_events(filters, cursor, limit=1000)
        events.extend(page.events)
        if not page.has_more or page.next_cursor is None:
            break
        cursor = page.next_cursor
    event_bytes = "".join(
        json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for event in events
    ).encode("utf-8")
    event_digest = hashlib.sha256(event_bytes).hexdigest()
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "created_at": created_at,
        "scope_id": _scope_id(),
        "filters": redact_audit_payload(filters, policy={}),
        "event_count": len(events),
        "events_sha256": event_digest,
        "hash_mode": audit_hash_mode(),
        "chain_valid": verification.valid,
        "chain_checked": verification.checked,
    }
    home = Path(os.environ.get("HAIRBALL_HOME") or (Path.home() / ".hairball"))
    export_dir = home / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = export_dir / f"governance-audit-{stamp}.zip"
    temp = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("events.jsonl", event_bytes)
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        )
    os.replace(temp, target)
    return ArtifactRef(
        path=target,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        event_count=len(events),
    )
