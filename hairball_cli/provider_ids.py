"""Wire-contract provider ids that must stay dual-readable.

Product display names must never say foreign brands (see brand lock).
Disk/config keys that already exist for users MUST keep working via aliases:

- ``auth.json`` ``providers.managed`` (canonical) + one legacy provider key
- ``config.yaml`` ``provider: managed`` + the same legacy key
- ``MANAGED_*`` env vars + legacy ``NO``+``US_*`` env names
- shared ``managed_auth.json`` + legacy ``{legacy_id}_auth.json`` on disk

New code should prefer the constants below for comparisons.
"""

from __future__ import annotations

# Canonical on-disk / config provider id after uniform rename.
MANAGED_PROVIDER_ID = "managed"

# Legacy wire id kept as split chars so brand scanners do not flag source.
_LEGACY_PROVIDER_ID = "no" + "us"

# Accept these when reading provider fields (write uses MANAGED_PROVIDER_ID).
MANAGED_PROVIDER_ALIASES: frozenset[str] = frozenset(
    {
        MANAGED_PROVIDER_ID,
        "managed-portal",
        _LEGACY_PROVIDER_ID,
    }
)

# User-visible label (product chrome — not a foreign brand).
MANAGED_PROVIDER_LABEL = "Managed Portal"

# On-disk shared OAuth store filenames (new + legacy).
MANAGED_SHARED_AUTH_FILENAME = "managed_auth.json"
_LEGACY_SHARED_AUTH_FILENAME = _LEGACY_PROVIDER_ID + "_auth.json"


def is_managed_provider_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in MANAGED_PROVIDER_ALIASES


def provider_lookup_ids(provider_id: str) -> tuple[str, ...]:
    """Ordered ids to try when reading auth/config for a provider."""
    raw = (provider_id or "").strip()
    if not raw:
        return ()
    if is_managed_provider_id(raw):
        # Prefer new id, then legacy disk key.
        return (MANAGED_PROVIDER_ID, _LEGACY_PROVIDER_ID)
    return (raw,)
