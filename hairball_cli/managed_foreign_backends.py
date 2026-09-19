"""Switch for the few remaining no-source foreign-service boundaries.

Hairball already owns most product code. What still binds to someone else's
server (no source available) is a short list — see
``docs/design/external-service-contract-boundaries.md``:

  A Portal OAuth / token
  B Portal account / billing
  C Managed tool-gateway URL assembly

Community has no drop-in Portal / Tool Gateway replacement. Product default
is therefore ``off``: BYOK provider keys + direct tool API keys. Optional
``legacy`` / ``hairball`` remain for explicit opt-in only.

``managed_foreign.backend`` (or ``HAIRBALL_MANAGED_BACKEND``) selects:

- ``off`` — no managed portal / gateway (default; BYOK path)
- ``legacy`` — historical Managed Portal / gateway hosts
- ``hairball`` — Hairball-owned hosts (must be configured; never silently
  pretends a legacy host is Hairball)
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

BACKEND_OFF = "off"
BACKEND_LEGACY = "legacy"
BACKEND_HAIRBALL = "hairball"

_VALID_BACKENDS = frozenset({BACKEND_OFF, BACKEND_LEGACY, BACKEND_HAIRBALL})

_LEGACY_PORTAL = "https://portal.example.invalid"
_LEGACY_GATEWAY_DOMAIN = "example.invalid"
_BYOK_HINT = (
    "Managed Portal / Tool Gateway is disabled. "
    "Configure a provider API key with `hairball setup` / `hairball model`, "
    "and tool keys with `hairball tools` (or .env)."
)


def _cfg_section() -> Mapping[str, Any]:
    try:
        from hairball_cli.config import load_config

        raw = (load_config() or {}).get("managed_foreign")
        return raw if isinstance(raw, Mapping) else {}
    except Exception:
        return {}


def backend() -> str:
    """Active foreign-service backend: ``off``, ``legacy``, or ``hairball``."""
    env = os.getenv("HAIRBALL_MANAGED_BACKEND", "").strip().lower()
    if env in _VALID_BACKENDS:
        return env
    value = str(_cfg_section().get("backend") or BACKEND_OFF).strip().lower()
    return value if value in _VALID_BACKENDS else BACKEND_OFF


def is_managed_foreign_off() -> bool:
    return backend() == BACKEND_OFF


def is_hairball_backend() -> bool:
    return backend() == BACKEND_HAIRBALL


def is_legacy_backend() -> bool:
    return backend() == BACKEND_LEGACY


def managed_foreign_disabled_message() -> str:
    return _BYOK_HINT


def portal_base_url() -> str:
    """Portal origin for OAuth / account / billing.

    Explicit ``HAIRBALL_PORTAL_URL`` / ``HAIRBALL_PORTAL_BASE_URL`` /
    ``MANAGED_PORTAL_BASE_URL`` always wins. On ``off`` with no explicit URL,
    returns empty (never falls back to a foreign portal host).
    """
    for key in (
        "HAIRBALL_PORTAL_URL",
        "HAIRBALL_PORTAL_BASE_URL",
        "MANAGED_PORTAL_BASE_URL",
        "NO" + "US_PORTAL_BASE_URL",  # legacy env dual-read
    ):
        explicit = os.getenv(key, "").strip().rstrip("/")
        if explicit:
            return explicit
    configured = str(_cfg_section().get("portal_url") or "").strip().rstrip("/")
    if configured:
        return configured
    if is_managed_foreign_off() or is_hairball_backend():
        return ""
    return _LEGACY_PORTAL


def require_portal_base_url() -> str:
    if is_managed_foreign_off():
        raise RuntimeError(_BYOK_HINT)
    url = portal_base_url()
    if url:
        return url
    raise RuntimeError(
        "Hairball managed backend is active but no portal URL is configured. "
        "Set managed_foreign.portal_url or HAIRBALL_PORTAL_URL."
    )


def resolve_portal_base_url(override: Optional[str] = None) -> str:
    """Resolve portal URL for boundary defs (override → switch → legacy/off)."""
    if isinstance(override, str) and override.strip():
        return override.strip().rstrip("/")
    if is_managed_foreign_off():
        return require_portal_base_url()
    return require_portal_base_url() if is_hairball_backend() else (
        portal_base_url() or _LEGACY_PORTAL
    )


def account_billing_available() -> bool:
    """Cluster B: off → unavailable; hairball needs portal; legacy keeps APIs."""
    if is_managed_foreign_off():
        return False
    if not is_hairball_backend():
        return True
    return bool(portal_base_url())


#: The ten protocol-boundary defs — only these are foreign-service cut targets.
BOUNDARY_DEFS: tuple[tuple[str, str, str], ...] = (
    ("A", "hairball_cli.auth._request_device_code", "Portal device code"),
    ("A", "hairball_cli.auth._poll_for_token", "Portal token poll"),
    ("A", "hairball_cli.auth._refresh_access_token", "Portal token refresh"),
    ("A", "hairball_cli.auth.fetch_managed_models", "Inference /models"),
    ("B", "hairball_cli.managed_account._fetch_managed_account_info", "Portal account"),
    ("B", "hairball_cli.managed_billing._request", "Portal billing HTTP"),
    ("B", "hairball_cli.models.fetch_managed_recommended_models", "Recommended models"),
    ("B", "hairball_cli.dashboard_register.cmd_dashboard_register", "Dashboard register"),
    ("C", "tools.managed_tool_gateway.build_vendor_gateway_url", "Vendor gateway URL"),
    ("C", "tools.managed_tool_gateway.resolve_managed_tool_gateway", "Gateway resolve"),
)


def tool_gateway_domain() -> str:
    """Domain used by ``build_vendor_gateway_url`` when no per-vendor override."""
    explicit = os.getenv("TOOL_GATEWAY_DOMAIN", "").strip().strip("/")
    if explicit:
        return explicit
    configured = str(_cfg_section().get("tool_gateway_domain") or "").strip().strip("/")
    if configured:
        return configured
    if is_managed_foreign_off():
        return ""
    if is_hairball_backend():
        return str(_cfg_section().get("hairball_tool_gateway_domain") or "").strip().strip("/")
    return _LEGACY_GATEWAY_DOMAIN


def update_repo() -> str:
    """Compatibility shim for the now product-owned update source."""
    from hairball_cli.update_source import resolve_update_source

    return resolve_update_source().repository


def update_archive_url(branch: str = "main") -> str:
    """Compatibility shim; updates are independent from Portal entitlement."""
    from hairball_cli.update_source import resolve_update_source

    return resolve_update_source().archive_url(branch)
