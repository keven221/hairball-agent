"""Managed scope — IT-pushed, user-immutable config & env layer.

A system-level directory (default ``/etc/hairball``, root-owned and not
user-writable) supplies ``config.yaml`` and ``.env`` values that WIN over the
user's ``~/.hairball/config.yaml`` and ``~/.hairball/.env`` on a per-leaf-key basis.

This is DISTINCT from ``hairball_cli.config.is_managed()`` / ``HAIRBALL_MANAGED``,
which is a coarse package-manager write-lock (declarative-distro / formula
installs). That lock blocks all mutation; this layer injects specific immutable
values. The two are independent and may coexist.

v1 enforcement is filesystem permissions only — see
``docs/design/managed-scope.md`` §7. v1 is Linux/POSIX-first; ``get_managed_dir()``
is the single seam for adding macOS / Windows native locations later.

Attribution: do not reference any third-party product by name in this file.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_MANAGED_DIR = Path("/etc/hairball")

_CACHE_LOCK = threading.Lock()
# path_key -> (mtime_ns, size, parsed)
_CONFIG_CACHE: Dict[str, tuple] = {}
_ENV_CACHE: Dict[str, tuple] = {}


class ManagedScopeError(RuntimeError):
    """A mandatory administrator policy could not be loaded or verified."""


def _merge_mapping(base: dict, overlay: dict) -> dict:
    """Return a recursive mapping merge where *overlay* wins at each leaf."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_mapping(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        value = [value]
    try:
        return {str(item).strip() for item in value if str(item).strip()}
    except TypeError:
        return set()


def _merge_policy_axes(parent: dict, child: dict) -> dict:
    """Merge capability axes so later managed fragments can only narrow."""
    policy: dict[str, dict[str, list[str]]] = {}
    for axis in ("toolsets", "skills", "plugins", "providers"):
        p = parent.get(axis) if isinstance(parent.get(axis), dict) else {}
        c = child.get(axis) if isinstance(child.get(axis), dict) else {}
        if not p and not c:
            continue
        p_allow, c_allow = _string_set(p.get("allow")), _string_set(c.get("allow"))
        allow = (p_allow & c_allow) if p_allow and c_allow else (p_allow or c_allow)
        deny = _string_set(p.get("deny")) | _string_set(c.get("deny"))
        policy[axis] = {"allow": sorted(allow - deny), "deny": sorted(deny)}
    return policy


def _merge_governance_layer(parent: dict, child: dict) -> dict:
    """Merge an administrator drop-in without permitting policy expansion."""
    merged = _merge_mapping(parent, child)
    if "enabled" in parent or "enabled" in child:
        merged["enabled"] = bool(parent.get("enabled")) or bool(child.get("enabled"))

    parent_audit = parent.get("audit") if isinstance(parent.get("audit"), dict) else {}
    child_audit = child.get("audit") if isinstance(child.get("audit"), dict) else {}
    if parent_audit or child_audit:
        audit = _merge_mapping(parent_audit, child_audit)
        for key in ("enabled", "required", "legal_hold", "require_signing"):
            if key in parent_audit or key in child_audit:
                audit[key] = bool(parent_audit.get(key)) or bool(child_audit.get(key))
        redact_fields = _string_set(parent_audit.get("redact_fields")) | _string_set(
            child_audit.get("redact_fields")
        )
        if redact_fields:
            audit["redact_fields"] = sorted(redact_fields)
        parent_days = int(parent_audit.get("retention_days") or 0)
        child_days = int(child_audit.get("retention_days") or 0)
        positive_days = [days for days in (parent_days, child_days) if days > 0]
        if positive_days:
            audit["retention_days"] = min(positive_days)
        merged["audit"] = audit

    parent_policy = parent.get("policy") if isinstance(parent.get("policy"), dict) else {}
    child_policy = child.get("policy") if isinstance(child.get("policy"), dict) else {}
    if parent_policy or child_policy:
        merged["policy"] = _merge_policy_axes(parent_policy, child_policy)

    parent_layers = parent.get("layers") if isinstance(parent.get("layers"), dict) else {}
    child_layers = child.get("layers") if isinstance(child.get("layers"), dict) else {}
    if parent_layers or child_layers:
        layers = _merge_mapping(parent_layers, child_layers)
        for layer_name in ("system", "organization", "project", "user", "session"):
            p_layer = (
                parent_layers.get(layer_name)
                if isinstance(parent_layers.get(layer_name), dict)
                else {}
            )
            c_layer = (
                child_layers.get(layer_name)
                if isinstance(child_layers.get(layer_name), dict)
                else {}
            )
            if p_layer or c_layer:
                layers[layer_name] = _merge_policy_axes(p_layer, c_layer)
        merged["layers"] = layers
    return merged


def _managed_files(managed_dir: Path, base_name: str, fragment_dir: str, suffix: str) -> list[Path]:
    """Base file followed by lexically ordered administrator drop-ins."""
    paths = [managed_dir / base_name]
    dropin_root = managed_dir / fragment_dir
    if dropin_root.is_dir():
        paths.extend(sorted(p for p in dropin_root.glob(f"*{suffix}") if p.is_file()))
    return paths


def _verify_managed_integrity(managed_dir: Path) -> None:
    """Verify ``manifest.sha256`` without allowing paths outside managed scope."""
    manifest = managed_dir / "manifest.sha256"
    if not manifest.is_file():
        return
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ManagedScopeError(f"managed scope: cannot read integrity manifest: {exc}") from exc

    expected: dict[str, str] = {}
    for line_no, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ManagedScopeError(
                f"managed scope: invalid integrity manifest line {line_no}"
            )
        digest, rel_text = parts[0].lower(), parts[1].strip().lstrip("*")
        if any(ch not in "0123456789abcdef" for ch in digest):
            raise ManagedScopeError(
                f"managed scope: invalid integrity digest on line {line_no}"
            )
        rel = Path(rel_text)
        if rel.is_absolute() or ".." in rel.parts:
            raise ManagedScopeError(
                f"managed scope: unsafe integrity path on line {line_no}"
            )
        expected[rel.as_posix()] = digest

    policy_files = [
        path
        for path in (
            _managed_files(managed_dir, "config.yaml", "config.d", ".yaml")
            + _managed_files(managed_dir, ".env", ".env.d", ".env")
        )
        if path.is_file()
    ]
    for path in policy_files:
        rel = path.relative_to(managed_dir).as_posix()
        wanted = expected.get(rel)
        if wanted is None:
            raise ManagedScopeError(
                f"managed scope: integrity manifest does not cover {rel}"
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != wanted:
            raise ManagedScopeError(
                f"managed scope: integrity mismatch for {rel}"
            )


def _under_pytest() -> bool:
    """True when running inside the test suite.

    Used to ignore the system default ``/etc/hairball`` during tests so a real
    managed scope on a developer/CI box can't leak policy into the suite. Tests
    that exercise managed scope set ``HAIRBALL_MANAGED_DIR`` explicitly, which is
    still honored (the override path below runs before this guard takes effect).
    """
    return "PYTEST_CURRENT_TEST" in os.environ


def native_managed_directory_candidates(
    platform: str | None = None,
    environ: dict[str, str] | None = None,
) -> tuple[Path, ...]:
    """Return Hairball-owned machine policy locations for the host OS.

    This is intentionally a pure resolver so installers and tests can inspect
    the contract without mutating process environment or touching the disk.
    """
    platform = (platform or sys.platform).lower()
    environ = os.environ if environ is None else environ
    if platform.startswith("darwin"):
        return (Path("/Library/Application Support/Hairball"),)
    if platform.startswith("win"):
        program_data = str(environ.get("PROGRAMDATA") or r"C:\ProgramData")
        return (Path(program_data) / "Hairball",)
    return (_DEFAULT_MANAGED_DIR,)


def get_managed_dir() -> Optional[Path]:
    """Resolve the managed-scope directory, or None when no scope is present.

    Resolution (highest priority first):
      1. ``$HAIRBALL_MANAGED_DIR`` — deployment/bootstrap path override (IT-only;
         never persisted to any .env). Honored only when set to a non-empty value
         AND the directory exists.
      2. The native machine-policy directory: ``/etc/hairball`` on Linux,
         ``/Library/Application Support/Hairball`` on macOS, and
         ``%PROGRAMDATA%\\Hairball`` on Windows. Ignored under pytest so a real
         system managed scope can't leak into the test suite.

    A non-existent directory at either tier resolves to None (no managed scope),
    which is the common case and must be cheap + side-effect-free.
    """
    override = os.environ.get("HAIRBALL_MANAGED_DIR", "").strip()
    if override:
        p = Path(override)
        return p if p.is_dir() else None
    if _under_pytest():
        return None
    for candidate in native_managed_directory_candidates():
        if candidate.is_dir():
            return candidate
    return None


def invalidate_managed_cache() -> None:
    """Drop cached managed config/env. For tests and post-edit reloads."""
    with _CACHE_LOCK:
        _CONFIG_CACHE.clear()
        _ENV_CACHE.clear()


def managed_config_signature() -> tuple[int, int]:
    """Cheap aggregate signature for base, drop-ins, strict marker, and manifest."""
    managed_dir = get_managed_dir()
    if managed_dir is None:
        return (0, 0)
    paths = (
        _managed_files(managed_dir, "config.yaml", "config.d", ".yaml")
        + [managed_dir / ".fail-closed", managed_dir / "manifest.sha256"]
    )
    digest = hashlib.sha256()
    total_size = 0
    found = 0
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        found += 1
        total_size += int(stat.st_size)
        digest.update(path.relative_to(managed_dir).as_posix().encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(str(stat.st_size).encode("ascii"))
    if not found:
        return (0, 0)
    return (int.from_bytes(digest.digest()[:8], "big"), total_size)


def _cached_read(path: Path, cache: Dict[str, tuple], parse, *, fail_closed: bool = False):
    """Shared (mtime_ns, size)-keyed read. Returns a deepcopy of the parsed value.

    Returns ``None`` when the file is absent or fails to parse (fail-open). A
    parse failure is logged LOUDLY — the admin needs to know their policy isn't
    being applied — but never raises, so a malformed managed file can't brick
    startup.
    """
    try:
        st = path.stat()
    except OSError:
        return None  # absent
    key = (st.st_mtime_ns, st.st_size)
    path_key = str(path)
    with _CACHE_LOCK:
        hit = cache.get(path_key)
        if hit is not None and hit[:2] == key:
            return copy.deepcopy(hit[2])
    try:
        with open(path, encoding="utf-8") as f:
            parsed = parse(f)
    except Exception as exc:  # noqa: BLE001 — fail-open, but LOUD
        logger.warning(
            "managed scope: failed to parse %s: %s — IGNORING this managed file. "
            "Admin policy from this file is NOT being applied. Fix and restart.",
            path,
            exc,
        )
        if fail_closed:
            raise ManagedScopeError(f"managed scope: failed to parse {path}: {exc}") from exc
        return None
    with _CACHE_LOCK:
        cache[path_key] = (key[0], key[1], copy.deepcopy(parsed))
    return parsed


def load_managed_config() -> dict:
    """Merged managed config plus ``config.d/*.yaml`` drop-ins.

    The base file is applied first and fragments are applied in lexical order,
    so an administrator can keep machine, organization, and security policy in
    separate files without creating a second configuration mechanism.
    """
    managed_dir = get_managed_dir()
    if managed_dir is None:
        return {}
    fail_closed = (managed_dir / ".fail-closed").is_file()
    if fail_closed:
        _verify_managed_integrity(managed_dir)
    merged: dict[str, Any] = {}
    for path in _managed_files(managed_dir, "config.yaml", "config.d", ".yaml"):
        parsed = _cached_read(
            path,
            _CONFIG_CACHE,
            lambda f: yaml.safe_load(f) or {},
            fail_closed=fail_closed,
        )
        if isinstance(parsed, dict):
            parent_governance = (
                merged.get("governance") if isinstance(merged.get("governance"), dict) else {}
            )
            child_governance = (
                parsed.get("governance") if isinstance(parsed.get("governance"), dict) else {}
            )
            merged = _merge_mapping(merged, parsed)
            if parent_governance and child_governance:
                merged["governance"] = _merge_governance_layer(
                    parent_governance,
                    child_governance,
                )
    return merged


def load_managed_env() -> Dict[str, str]:
    """Merged managed ``.env`` plus lexical ``.env.d/*.env`` drop-ins."""
    managed_dir = get_managed_dir()
    if managed_dir is None:
        return {}
    fail_closed = (managed_dir / ".fail-closed").is_file()
    if fail_closed:
        _verify_managed_integrity(managed_dir)
    merged: Dict[str, str] = {}
    for path in _managed_files(managed_dir, ".env", ".env.d", ".env"):
        parsed = _cached_read(path, _ENV_CACHE, _parse_env, fail_closed=fail_closed)
        if isinstance(parsed, dict):
            merged.update({str(k): str(v) for k, v in parsed.items()})
    return merged


def apply_managed_overlay(config: dict) -> dict:
    """Overlay administrator-pinned config values on top of an already-built dict.

    The single, shared way for any config loader that builds its own dict
    (rather than going through hairball_cli.config.load_config) to honor managed
    scope. Mirrors hairball_cli.config._load_config_impl's managed merge exactly:

      * expand the managed config's ``${VAR}`` refs against the PROCESS env only
        (never user-config-defined refs), so a user cannot shadow a managed
        literal via a ${VAR} they control;
      * normalize the managed config's root ``model`` key (a bare ``model: x/y``
        string is promoted to ``model.default``) so it can't clobber the dict
        shape callers expect;
      * leaf-level deep-merge managed ON TOP, so managed wins per-leaf while
        sibling keys stay user-controlled.

    Fail-open: returns ``config`` unchanged if no managed scope is present or on
    any error — managed scope must never break a caller's startup. Mutates and
    returns ``config`` (callers pass a dict they own).
    """
    try:
        managed = load_managed_config()
        if not managed:
            return config
        # Imported lazily to avoid an import cycle (config imports managed_scope).
        from hairball_cli.config import _deep_merge, _expand_env_vars, _normalize_root_model_keys

        managed_expanded = _normalize_root_model_keys(_expand_env_vars(managed))
        # A bare ``model: x/y`` string in the managed file must merge as
        # ``model.default`` — otherwise _deep_merge would replace the caller's
        # ``model`` dict with a string and break every ``cfg["model"]["..."]``
        # read. _normalize_root_model_keys only promotes the string when there
        # are root provider/base_url keys to migrate, so handle the bare case
        # here (matches cli.py's own string-model handling).
        if isinstance(managed_expanded.get("model"), str):
            managed_expanded = dict(managed_expanded)
            managed_expanded["model"] = {"default": managed_expanded["model"]}
        return _deep_merge(config, managed_expanded)
    except ManagedScopeError:
        raise
    except Exception:  # noqa: BLE001 — overlay must never break a caller
        logger.warning("managed scope: failed to apply config overlay", exc_info=True)
        return config


def _parse_env(f) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip("\"'")
    return out


def _flatten_keys(d: dict, prefix: str = "") -> set:
    keys: set = set()
    for k, v in d.items():
        dotted = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            keys |= _flatten_keys(v, dotted)
        else:
            keys.add(dotted)
    return keys


def managed_config_keys() -> set:
    """Dotted leaf keys pinned by the managed config (e.g. {'model.default'})."""
    return _flatten_keys(load_managed_config())


def is_key_managed(dotted_key: str) -> bool:
    """True if the exact dotted config key is pinned by the managed layer."""
    return dotted_key in managed_config_keys()


def is_env_managed(name: str) -> bool:
    """True if the env var name is pinned by the managed .env layer."""
    return name in load_managed_env()
