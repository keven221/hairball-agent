"""Portal device OAuth (RFC 8628) — OpenCode-shaped, Hairball-hosted.

Black-box of the no-source Managed Portal login surface. Structure follows
OpenCode's device-auth provider (``packages/core/src/plugin/provider/opencode.ts``,
MIT): request device code → poll token → refresh. Endpoint paths stay the
Hairball/Portal ``/api/oauth/...`` forms so a self-hosted portal can swap
hosts via ``managed_foreign.portal_url`` without rewriting callers.

Callers keep using the legacy names in ``hairball_cli.auth``; those are thin
wrappers around this module.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

# Match OpenCode's slow_down ceiling (30s) and Hairball's historical poll floor.
DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS = 1
_SLOW_DOWN_CAP_SECONDS = 30


def request_device_code(
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    scope: Optional[str],
) -> Dict[str, Any]:
    """POST ``/api/oauth/device/code`` — OpenCode ``authorize`` equivalent."""
    response = client.post(
        f"{portal_base_url.rstrip('/')}/api/oauth/device/code",
        data={
            "client_id": client_id,
            **({"scope": scope} if scope else {}),
        },
    )
    response.raise_for_status()
    data = response.json()

    required_fields = [
        "device_code",
        "user_code",
        "verification_uri",
        "verification_uri_complete",
        "expires_in",
        "interval",
    ]
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"Device code response missing fields: {', '.join(missing)}")
    return data


def poll_for_token(
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    device_code: str,
    expires_in: int,
    poll_interval: int,
) -> Dict[str, Any]:
    """Poll ``/api/oauth/token`` until approve / expire — OpenCode ``callback``."""
    deadline = time.monotonic() + max(1, expires_in)
    current_interval = max(
        1, min(poll_interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS)
    )
    token_url = f"{portal_base_url.rstrip('/')}/api/oauth/token"

    while time.monotonic() < deadline:
        response = client.post(
            token_url,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": client_id,
                "device_code": device_code,
            },
        )

        if response.status_code == 200:
            payload = response.json()
            if "access_token" not in payload:
                raise ValueError("Token response did not include access_token")
            return payload

        try:
            error_payload = response.json()
        except Exception:
            response.raise_for_status()
            raise RuntimeError("Token endpoint returned a non-JSON error response")

        error_code = error_payload.get("error", "")
        if error_code == "authorization_pending":
            time.sleep(current_interval)
            continue
        if error_code == "slow_down":
            current_interval = min(current_interval + 1, _SLOW_DOWN_CAP_SECONDS)
            time.sleep(current_interval)
            continue

        description = error_payload.get("error_description") or "Unknown authentication error"
        raise RuntimeError(f"{error_code}: {description}")

    raise TimeoutError("Timed out waiting for device authorization")


def refresh_access_token(
    *,
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    refresh_token: str,
) -> Dict[str, Any]:
    """POST refresh — OpenCode ``refresh`` equivalent (Hairball header kept)."""
    response = client.post(
        f"{portal_base_url.rstrip('/')}/api/oauth/token",
        headers={"x-managed-refresh-token": refresh_token},
        data={
            "grant_type": "refresh_token",
            "client_id": client_id,
        },
    )

    if response.status_code == 200:
        payload = response.json()
        if "access_token" not in payload:
            raise ValueError("Refresh response missing access_token")
        return payload

    try:
        error_payload = response.json()
    except Exception as exc:
        raise RuntimeError("Refresh token exchange failed") from exc

    code = str(error_payload.get("error", "invalid_grant"))
    description = str(
        error_payload.get("error_description") or "Refresh token exchange failed"
    )
    raise RuntimeError(f"{code}: {description}")


def fetch_models(
    *,
    inference_base_url: str,
    api_key: str,
    timeout_seconds: float = 15.0,
    verify: bool | str = True,
) -> List[str]:
    """GET ``/models`` on an OpenAI-compatible inference base."""
    timeout = httpx.Timeout(timeout_seconds)
    with httpx.Client(
        timeout=timeout, headers={"Accept": "application/json"}, verify=verify
    ) as client:
        response = client.get(
            f"{inference_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    names: List[str] = []
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            names.append(str(item["id"]))
        elif isinstance(item, str):
            names.append(item)
    return names
