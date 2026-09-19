"""The shipped Hairball UI Electron shell uses one reproducible version.

``apps/desktop`` is retired.  The authoritative native shell lives under
``hairball-ui/electron-app`` and is installed from its own lockfile.  Keep the
manifest, lockfile root request, and resolved package on the same exact version
so a fresh ``npm ci`` cannot silently change the runtime between releases.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
DESKTOP_DIR = REPO_ROOT / "hairball-ui" / "electron-app"
DESKTOP_PKG = DESKTOP_DIR / "package.json"
DESKTOP_LOCK = DESKTOP_DIR / "package-lock.json"

# An exact semver: digits.digits.digits with an optional prerelease/build tag,
# but NO range operators (^ ~ > < = * x || spaces || -range).
_EXACT_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def _desktop_pkg() -> dict:
    assert DESKTOP_PKG.is_file(), f"missing {DESKTOP_PKG}"
    return json.loads(DESKTOP_PKG.read_text(encoding="utf-8"))


def _electron_spec(pkg: dict) -> str:
    for section in ("dependencies", "devDependencies"):
        spec = pkg.get(section, {}).get("electron")
        if spec:
            return spec
    pytest.fail("electron is not listed in hairball-ui/electron-app dependencies")


def test_electron_dependency_is_exactly_pinned():
    """A loose range lets npm drift onto an Electron with a different installer."""
    spec = _electron_spec(_desktop_pkg())
    assert _EXACT_SEMVER.match(spec), (
        f"electron must be pinned to an exact version, got {spec!r}. "
        "A range (^/~) lets npm ci resolve a newer Electron whose postinstall "
        "may differ from the one the build was validated against."
    )


def test_lockfile_request_matches_dependency():
    """The lockfile's root request must preserve the exact manifest pin."""
    spec = _electron_spec(_desktop_pkg())
    lock = json.loads(DESKTOP_LOCK.read_text(encoding="utf-8"))
    root_request = lock.get("packages", {}).get("", {})
    locked_spec = root_request.get("dependencies", {}).get("electron") or root_request.get(
        "devDependencies", {}
    ).get("electron")
    assert locked_spec == spec, (
        f"electron manifest pin is {spec!r}, but the lockfile requests "
        f"{locked_spec!r}; refresh hairball-ui/electron-app/package-lock.json"
    )


def test_lockfile_resolves_the_pinned_electron():
    """npm ci installs from the lockfile, so it must agree with the pin."""
    if not DESKTOP_LOCK.is_file():
        pytest.skip("hairball-ui/electron-app/package-lock.json not present")
    spec = _electron_spec(_desktop_pkg())
    lock = json.loads(DESKTOP_LOCK.read_text(encoding="utf-8"))
    packages = lock.get("packages", {})
    resolved = [
        meta.get("version")
        for path, meta in packages.items()
        if path.endswith("node_modules/electron") and meta.get("version")
    ]
    assert resolved, "no electron entry found in Electron shell package-lock.json"
    assert all(v == spec for v in resolved), (
        f"Electron shell package-lock.json resolves to {sorted(set(resolved))}, "
        f"but the pin is {spec!r}; run `npm install --package-lock-only` so "
        "`npm ci` stays consistent."
    )
