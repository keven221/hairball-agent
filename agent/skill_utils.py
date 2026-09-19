"""Lightweight skill metadata utilities shared by prompt_builder and skills_tool.

This module intentionally avoids importing the tool registry, CLI config, or any
heavy dependency chain.  It is safe to import at module level without triggering
tool registration or provider resolution.
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hairball_constants import get_config_path, get_skills_dir, is_termux

logger = logging.getLogger(__name__)

# ── Platform mapping ──────────────────────────────────────────────────────

PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}

EXCLUDED_SKILL_DIRS = frozenset(
    (
        ".git",
        ".github",
        ".hub",
        ".archive",
        ".venv",
        "venv",
        "node_modules",
        "site-packages",
        "__pycache__",
        ".tox",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        # Legacy duplicate skill packages kept in-tree for a later delete pass.
        # Excluding them from sync/index avoids frontmatter-name collisions with
        # the canonical hairball-* counterparts (same `name:` in SKILL.md).
        "hairball-agent",
        "hairball-agent-skill-authoring",
        "hairball-s6-container-supervision",
    )
)

# Supporting files live inside a skill package and are loaded explicitly via
# skill_view(skill, file_path=...). They are not standalone skills and must not
# be scanned for active SKILL.md/DESCRIPTION.md entries, even if a Curator or
# archive workflow preserves a complete old skill package under references/.
SKILL_SUPPORT_DIRS = frozenset(("references", "templates", "assets", "scripts"))


def is_excluded_skill_path(path) -> bool:
    """True if *path* should be skipped by active skill scanners.

    Use this on every ``SKILL.md`` path produced by direct ``rglob`` scans to
    prune dependency, virtualenv, VCS, cache, and progressive-disclosure
    support-package paths. Centralising the check here keeps every
    skill-scanning site in sync with the shared exclusion set.

    Accepts a Path or string.
    """
    try:
        parts = path.parts  # Path
    except AttributeError:
        from pathlib import PurePath
        parts = PurePath(str(path)).parts
    return any(part in EXCLUDED_SKILL_DIRS for part in parts) or is_skill_support_path(
        path
    )


def is_skill_support_path(path) -> bool:
    """True if *path* is under a support dir of an actual skill root.

    ``references/``, ``templates/``, ``assets/``, and ``scripts/`` are
    progressive-disclosure support areas when they sit directly inside a skill
    directory containing ``SKILL.md``. They are not active discovery roots for
    standalone skills. A preserved package such as
    ``some-skill/references/old-skill-package/SKILL.md`` is documentation data
    unless the caller explicitly loads it via ``file_path``.

    Legitimate categories or skill names such as ``skills/scripts/foo`` remain
    discoverable because their ``scripts`` component is not directly under a
    directory that contains ``SKILL.md``.
    """
    path_obj = path if isinstance(path, Path) else Path(str(path))
    parts = path_obj.parts
    # Last component may be a file or candidate skill directory name. Only
    # components before the leaf can be containing support directories.
    for idx, part in enumerate(parts[:-1]):
        if part not in SKILL_SUPPORT_DIRS or idx == 0:
            continue
        skill_root = Path(*parts[:idx])
        if (skill_root / "SKILL.md").exists():
            return True
    return False


# ── Lazy YAML loader ─────────────────────────────────────────────────────

_yaml_load_fn = None


def yaml_load(content: str):
    """Parse YAML with lazy import and CSafeLoader preference."""
    global _yaml_load_fn
    if _yaml_load_fn is None:
        import yaml

        loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader

        def _load(value: str):
            return yaml.load(value, Loader=loader)

        _yaml_load_fn = _load
    return _yaml_load_fn(content)


# ── Frontmatter parsing ──────────────────────────────────────────────────


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from a markdown string.

    Uses yaml with CSafeLoader for full YAML support (nested metadata, lists)
    with a fallback to simple key:value splitting for robustness.

    Returns:
        (frontmatter_dict, remaining_body)
    """
    frontmatter: Dict[str, Any] = {}
    body = content

    if not content.startswith("---"):
        return frontmatter, body

    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return frontmatter, body

    yaml_content = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]

    try:
        parsed = yaml_load(yaml_content)
        if isinstance(parsed, dict):
            frontmatter = parsed
    except Exception:
        # Fallback: simple key:value parsing for malformed YAML
        for line in yaml_content.strip().split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()

    return frontmatter, body


# ── Platform matching ─────────────────────────────────────────────────────


def skill_matches_platform_list(platforms: Any) -> bool:
    """Return True when *platforms* is compatible with the current OS."""
    if not platforms:
        return True
    if not isinstance(platforms, list):
        platforms = [platforms]
    current = sys.platform
    running_in_termux = is_termux()
    for platform in platforms:
        normalized = str(platform).lower().strip()
        mapped = PLATFORM_MAP.get(normalized, normalized)
        if current.startswith(mapped):
            return True
        # Termux runs a Linux userland on Android. Accept linux-tagged
        # skills regardless of whether sys.platform is "linux" (pre-3.13
        # Termux) or "android" (Python 3.13+ Termux, and any other
        # Android runtime).
        if running_in_termux and mapped == "linux":
            return True
        # Explicit termux/android tags match a Termux session too.
        if running_in_termux and mapped in ("termux", "android"):
            return True
    return False


def skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    """Return True when the skill is compatible with the current OS.

    Skills declare platform requirements via a top-level ``platforms`` list
    in their YAML frontmatter::

        platforms: [macos]          # macOS only
        platforms: [macos, linux]   # macOS and Linux

    If the field is absent or empty the skill is compatible with **all**
    platforms (backward-compatible default).

    Termux note: on Termux/Android, ``sys.platform`` is ``"linux"`` on
    older Pythons but became ``"android"`` on Python 3.13+. Termux is a
    Linux userland riding on the Android kernel, so skills tagged
    ``linux`` are treated as compatible in Termux regardless of which
    ``sys.platform`` value Python reports. Individual Linux commands
    inside a skill may still misbehave (no systemd, BusyBox utils, no
    apt/dnf, etc.) but that is on the skill, not on platform gating.
    """
    return skill_matches_platform_list(frontmatter.get("platforms"))


# ── Environment matching ──────────────────────────────────────────────────

# Recognized environment tags and how each is detected. An environment tag is
# a *relevance* gate, not a hard-compatibility gate (that is what ``platforms:``
# is for). A skill tagged for an environment it isn't relevant to is hidden from
# the skills index / offer surfaces so it does not add noise for users who will
# never need it — but it can ALWAYS still be loaded explicitly (``skill_view``,
# ``--skills``), because an explicit request is explicit consent.
#
# Detection is cached for the process lifetime via ``_ENV_DETECT_CACHE``.
_KNOWN_ENVIRONMENTS = frozenset({"kanban", "docker", "s6"})

_ENV_DETECT_CACHE: Dict[str, bool] = {}


def _detect_environment(env: str) -> bool:
    """Return True when the named runtime environment is currently active.

    Cached per process. Unknown env names return True (fail-open: never hide a
    skill because of a tag we don't understand).
    """
    if env in _ENV_DETECT_CACHE:
        return _ENV_DETECT_CACHE[env]

    result = True
    if env == "kanban":
        # Kanban is "active" either as a dispatcher-spawned worker (the
        # dispatcher sets ``HAIRBALL_KANBAN_TASK`` / ``HAIRBALL_KANBAN_BOARD`` in the
        # worker env) or as an orchestrator profile that has opted into the
        # kanban toolset. Mirror the same signals the kanban tools themselves
        # gate on (``tools/kanban_tools.py``) so the offer filter agrees with
        # tool availability.
        if os.getenv("HAIRBALL_KANBAN_TASK") or os.getenv("HAIRBALL_KANBAN_BOARD"):
            result = True
        else:
            try:
                from tools.kanban_tools import _profile_has_kanban_toolset

                result = bool(_profile_has_kanban_toolset())
            except Exception:
                result = False
    elif env == "docker":
        try:
            from hairball_constants import is_container

            result = is_container()
        except Exception:
            result = False
    elif env == "s6":
        # The Hairball Docker image runs s6-overlay as PID 1 (/init). s6 plants
        # its runtime scaffolding under /run/s6 and ships its admin tree under
        # /package/admin/s6-overlay. Either marker means we're inside an
        # s6-supervised container.
        result = os.path.isdir("/run/s6") or os.path.isdir(
            "/package/admin/s6-overlay"
        )

    _ENV_DETECT_CACHE[env] = result
    return result


def skill_matches_environment(frontmatter: Dict[str, Any]) -> bool:
    """Return True when the skill is relevant to the current runtime environment.

    Skills may declare an ``environments`` list in their YAML frontmatter::

        environments: [kanban]        # only relevant when kanban is active
        environments: [s6]            # only relevant inside the s6 Docker image
        environments: [docker]        # only relevant inside any container

    If the field is absent or empty the skill is relevant in **all**
    environments (backward-compatible default).

    This is an OFFER-time filter: it controls whether a skill shows up in the
    skills index / autocomplete / slash-command list. It is intentionally NOT
    enforced by ``skill_view`` or ``--skills`` preloading — an explicit load is
    explicit consent, and load-bearing force-loads (e.g. a dispatcher pinning
    a task to a specialist skill via ``--skills``) must always succeed
    regardless of how the offer surfaces filter the skill.

    A skill matches when ANY of its declared environments is currently active
    (OR semantics, mirroring ``platforms``). Unknown env tags fail open.
    """
    environments = frontmatter.get("environments")
    if not environments:
        return True
    if not isinstance(environments, list):
        environments = [environments]
    for env in environments:
        normalized = str(env).lower().strip()
        if not normalized:
            continue
        if normalized not in _KNOWN_ENVIRONMENTS:
            # Tag we don't understand — don't hide the skill over it.
            return True
        if _detect_environment(normalized):
            return True
    return False


# ── Disabled skills ───────────────────────────────────────────────────────


_RAW_CONFIG_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def _raw_config_cache_clear() -> None:
    """Test hook — drop the shared raw config cache."""
    _RAW_CONFIG_CACHE.clear()


def _load_raw_config() -> Dict[str, Any]:
    """Read config.yaml with a shared mtime+size keyed cache.

    This module intentionally avoids importing ``hairball_cli.config`` on the
    skill prompt/build path. A tiny local cache gives the same repeated-read
    win without pulling the heavier CLI config stack into startup.
    """
    config_path = get_config_path()
    from hairball_cli import managed_scope

    managed_sig = managed_scope.managed_config_signature()
    try:
        stat = config_path.stat() if config_path.exists() else None
        cache_key = (
            str(config_path),
            stat.st_mtime_ns if stat else 0,
            stat.st_size if stat else 0,
            managed_sig[0],
            managed_sig[1],
        )
    except OSError:
        cache_key = None

    if cache_key is not None:
        cached = _RAW_CONFIG_CACHE.get(cache_key)
        if cached is not None:
            return cached

    try:
        parsed = (
            yaml_load(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
    except Exception as e:
        logger.debug("Could not read skill config %s: %s", config_path, e)
        return {}
    if not isinstance(parsed, dict):
        parsed = {}

    # Skills intentionally keep a light raw-config cache, but administrator
    # policy must still reach every prompt/slash/tool surface. Preserve a
    # user's additional disables when managed config pins its own deny list:
    # a lower layer may narrow access further, never re-enable an admin or user
    # deny by replacing a list wholesale.
    user_skills = parsed.get("skills") if isinstance(parsed.get("skills"), dict) else {}
    parsed = managed_scope.apply_managed_overlay(parsed)
    skills_cfg = parsed.get("skills") if isinstance(parsed.get("skills"), dict) else {}
    if user_skills or skills_cfg:
        merged_skills = dict(skills_cfg)
        merged_skills["disabled"] = sorted(
            _normalize_string_set(user_skills.get("disabled"))
            | _normalize_string_set(skills_cfg.get("disabled"))
        )
        user_platform = user_skills.get("platform_disabled") or {}
        managed_platform = skills_cfg.get("platform_disabled") or {}
        if isinstance(user_platform, dict) or isinstance(managed_platform, dict):
            platform_out: dict[str, list[str]] = {}
            for platform_name in set(
                (user_platform if isinstance(user_platform, dict) else {})
            ) | set((managed_platform if isinstance(managed_platform, dict) else {})):
                platform_out[str(platform_name)] = sorted(
                    _normalize_string_set(
                        (user_platform if isinstance(user_platform, dict) else {}).get(platform_name)
                    )
                    | _normalize_string_set(
                        (managed_platform if isinstance(managed_platform, dict) else {}).get(platform_name)
                    )
                )
            merged_skills["platform_disabled"] = platform_out
        parsed = dict(parsed)
        parsed["skills"] = merged_skills

    if cache_key is not None:
        _RAW_CONFIG_CACHE.clear()
        _RAW_CONFIG_CACHE[cache_key] = parsed
    return parsed


def get_disabled_skill_names(platform: str | None = None) -> Set[str]:
    """Read disabled skill names from config.yaml.

    Args:
        platform: Explicit platform name (e.g. ``"telegram"``).  When
            *None*, resolves from ``HAIRBALL_PLATFORM`` or
            ``HAIRBALL_SESSION_PLATFORM`` env vars.  Returns the global
            disabled list, unioned with the platform-specific list when a
            platform is resolved (a globally-disabled skill stays disabled
            on every platform).

    Reads the config file directly (no CLI config imports) to stay
    lightweight.
    """
    parsed = _load_raw_config()
    if not parsed:
        return set()

    skills_cfg = parsed.get("skills")
    if not isinstance(skills_cfg, dict):
        skills_cfg = {}

    from gateway.session_context import get_session_env
    resolved_platform = (
        platform
        or os.getenv("HAIRBALL_PLATFORM")
        or get_session_env("HAIRBALL_SESSION_PLATFORM")
    )
    global_disabled = _normalize_string_set(skills_cfg.get("disabled"))
    try:
        from hairball_cli.governance import effective_policy_from_config

        policy = effective_policy_from_config(parsed)
        governed_allow, governed_denies = _axis_for_skills(policy.policy)
        global_disabled |= governed_denies
        if governed_allow:
            global_disabled |= (
                _installed_skill_names_from_config(parsed) - governed_allow
            )
    except Exception:
        pass
    if resolved_platform:
        platform_disabled = (skills_cfg.get("platform_disabled") or {}).get(
            resolved_platform
        )
        if platform_disabled is not None:
            return global_disabled | _normalize_string_set(platform_disabled)
    return global_disabled


def _axis_for_skills(policy: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """Local lightweight view of governance.skills without importing internals."""
    raw = policy.get("skills") if isinstance(policy, dict) else None
    if not isinstance(raw, dict):
        return set(), set()
    return _normalize_string_set(raw.get("allow")), _normalize_string_set(raw.get("deny"))


def _installed_skill_names_from_config(config: Dict[str, Any]) -> Set[str]:
    """Discover names only when an enterprise allowlist needs a closed world."""
    roots = [get_skills_dir()]
    skills_cfg = config.get("skills") if isinstance(config.get("skills"), dict) else {}
    raw_dirs = skills_cfg.get("external_dirs") or []
    if isinstance(raw_dirs, str):
        raw_dirs = [raw_dirs]
    if isinstance(raw_dirs, list):
        from hairball_constants import get_hairball_home

        for raw in raw_dirs:
            expanded = Path(os.path.expanduser(os.path.expandvars(str(raw))))
            if not expanded.is_absolute():
                expanded = get_hairball_home() / expanded
            if expanded.is_dir():
                roots.append(expanded.resolve())
    names: Set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("SKILL.md"):
            if is_excluded_skill_path(path) or is_skill_support_path(path):
                continue
            names.add(path.parent.name)
    return names


def _normalize_string_set(values) -> Set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    return {str(v).strip() for v in values if str(v).strip()}


# ── External skills directories ──────────────────────────────────────────

# (config_path_str, mtime_ns) -> resolved external dirs list.  Keyed by
# mtime_ns so a config.yaml edit mid-run is picked up automatically;
# otherwise every call would re-read + re-YAML-parse the 15KB config,
# which becomes the dominant cost of ``hairball`` startup when ~120 skills
# each trigger a category lookup during banner construction (10+ seconds
# of pure waste).
_EXTERNAL_DIRS_CACHE: Dict[Tuple[Any, ...], List[Path]] = {}


def _external_dirs_cache_clear() -> None:
    """Test hook — drop the in-process cache."""
    _EXTERNAL_DIRS_CACHE.clear()
    _raw_config_cache_clear()


def get_external_skills_dirs() -> List[Path]:
    """Read ``skills.external_dirs`` from config.yaml and return validated paths.

    Each entry is expanded (``~`` and ``${VAR}``) and resolved to an absolute
    path.  Only directories that actually exist are returned.  Duplicates and
    paths that resolve to the local ``~/.hairball/skills/`` are silently skipped.

    Cached in-process, keyed on ``config.yaml`` mtime — the function is
    called once per skill during banner / tool-registry scans, and YAML
    parsing a non-trivial config dominates ``hairball`` cold-start time
    when the cache is absent.
    """
    config_path = get_config_path()
    from hairball_cli import managed_scope

    # Cache key covers both the optional user config and the full managed
    # config/drop-in signature.  A managed-only deployment may intentionally
    # omit the user's config.yaml, so absence is part of the key rather than an
    # early return.
    try:
        stat = config_path.stat() if config_path.exists() else None
        managed_sig = managed_scope.managed_config_signature()
        cache_key: Tuple[Any, ...] = (
            str(config_path),
            stat.st_mtime_ns if stat else 0,
            stat.st_size if stat else 0,
            managed_sig[0],
            managed_sig[1],
        )
    except OSError:
        cache_key = None  # type: ignore[assignment]

    if cache_key is not None:
        cached = _EXTERNAL_DIRS_CACHE.get(cache_key)
        if cached is not None:
            # Return a copy so callers can't mutate the cached list.
            return list(cached)

    parsed = _load_raw_config()
    if not parsed:
        return []

    skills_cfg = parsed.get("skills")
    if not isinstance(skills_cfg, dict):
        return []

    raw_dirs = skills_cfg.get("external_dirs")
    if not raw_dirs:
        result: List[Path] = []
        if cache_key is not None:
            _EXTERNAL_DIRS_CACHE[cache_key] = list(result)
        return result
    if isinstance(raw_dirs, str):
        raw_dirs = [raw_dirs]
    if not isinstance(raw_dirs, list):
        return []

    from hairball_constants import get_hairball_home

    hairball_home = get_hairball_home()
    local_skills = get_skills_dir().resolve()
    seen: Set[Path] = set()
    result = []

    for entry in raw_dirs:
        entry = str(entry).strip()
        if not entry:
            continue
        # Expand ~ and environment variables
        expanded = os.path.expanduser(os.path.expandvars(entry))
        p = Path(expanded)
        # Resolve relative paths against HAIRBALL_HOME, not cwd
        if not p.is_absolute():
            p = (hairball_home / p).resolve()
        else:
            p = p.resolve()
        if p == local_skills:
            continue
        if p in seen:
            continue
        if p.is_dir():
            seen.add(p)
            result.append(p)
        else:
            logger.debug("External skills dir does not exist, skipping: %s", p)

    if cache_key is not None:
        _EXTERNAL_DIRS_CACHE[cache_key] = list(result)
    return result


def get_all_skills_dirs() -> List[Path]:
    """Return all skill directories: local ``~/.hairball/skills/`` first, then external.

    The local dir is always first (and always included even if it doesn't exist
    yet — callers handle that).  External dirs follow in config order.
    """
    dirs = [get_skills_dir()]
    dirs.extend(get_external_skills_dirs())
    return dirs


def normalize_skill_lookup_name(identifier: str) -> str:
    """Normalize a skill identifier to a ``skill_view()``-safe relative path.

    Slash commands and cron jobs may store absolute paths to skills that live
    under ``~/.hairball/skills/`` (including via symlinks) or configured
    ``skills.external_dirs``. ``skill_view()`` rejects absolute names for
    security, so callers must translate trusted absolute paths to their
    relative form first.
    """
    raw_identifier = (identifier or "").strip()
    if not raw_identifier:
        return raw_identifier

    identifier_path = Path(raw_identifier).expanduser()
    if not identifier_path.is_absolute():
        return raw_identifier.lstrip("/")

    # Look the primary skills root up on tools.skills_tool at CALL time
    # (not via get_skills_dir()): callers and tests patch
    # ``tools.skills_tool.SKILLS_DIR`` and skill_view() itself resolves
    # against that module attribute, so normalization must agree with the
    # exact root skill_view() will enforce.  Import deferred to avoid a
    # module cycle (tools.skills_tool imports agent.skill_utils).
    try:
        from tools import skills_tool as _skills_tool
        primary_root = Path(_skills_tool.SKILLS_DIR)
    except Exception:
        primary_root = get_skills_dir()

    trusted_roots = [primary_root]
    try:
        trusted_roots.extend(get_external_skills_dirs())
    except Exception:
        pass

    # Prefer the lexical path under a trusted skill root before resolving
    # symlinks. Slash-command discovery can legitimately find a skill via
    # ~/.hairball/skills/<name> where <name> is a symlink to a checked-out
    # skill elsewhere. Resolving first turns that trusted visible path into
    # an arbitrary absolute path that skill_view() refuses to load.
    for root in trusted_roots:
        try:
            return str(identifier_path.relative_to(root))
        except ValueError:
            continue

    try:
        return str(identifier_path.resolve().relative_to(primary_root.resolve()))
    except Exception:
        logger.debug(
            "Skill identifier %r is an absolute path outside trusted skills "
            "roots — passing through unchanged (skill_view will reject it)",
            raw_identifier,
        )
        return raw_identifier


def _resolve_for_skill_ownership(path) -> Path:
    path_obj = path if isinstance(path, Path) else Path(str(path))
    try:
        return path_obj.expanduser().resolve()
    except (OSError, RuntimeError):
        return path_obj.expanduser().absolute()


def is_external_skill_path(path) -> bool:
    """Return True when ``path`` lives under a configured external skills dir.

    ``skills.external_dirs`` are externally owned: Hairball can discover and view
    their skills, and foreground user-directed tool calls may still edit them,
    but autonomous lifecycle maintenance must treat them as read-only. This
    helper centralizes the ownership boundary so curator/reporting/tool paths do
    not each need to re-interpret the config.
    """
    candidate = _resolve_for_skill_ownership(path)
    for root in get_external_skills_dirs():
        resolved_root = _resolve_for_skill_ownership(root)
        try:
            candidate.relative_to(resolved_root)
            return True
        except ValueError:
            continue
    return False


# ── Condition extraction ──────────────────────────────────────────────────


def extract_skill_conditions(frontmatter: Dict[str, Any]) -> Dict[str, List]:
    """Extract conditional activation fields from parsed frontmatter."""
    metadata = frontmatter.get("metadata")
    # Handle cases where metadata is not a dict (e.g., a string from malformed YAML)
    if not isinstance(metadata, dict):
        metadata = {}
    hairball = metadata.get("hairball") or {}
    if not isinstance(hairball, dict):
        hairball = {}
    return {
        "fallback_for_toolsets": hairball.get("fallback_for_toolsets", []),
        "requires_toolsets": hairball.get("requires_toolsets", []),
        "fallback_for_tools": hairball.get("fallback_for_tools", []),
        "requires_tools": hairball.get("requires_tools", []),
    }


# ── Skill config extraction ───────────────────────────────────────────────


def extract_skill_config_vars(frontmatter: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract config variable declarations from parsed frontmatter.

    Skills declare config.yaml settings they need via::

        metadata:
          hairball:
            config:
              - key: wiki.path
                description: Path to the LLM Wiki knowledge base directory
                default: "~/wiki"
                prompt: Wiki directory path

    Returns a list of dicts with keys: ``key``, ``description``, ``default``,
    ``prompt``.  Invalid or incomplete entries are silently skipped.
    """
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        return []
    hairball = metadata.get("hairball")
    if not isinstance(hairball, dict):
        return []
    raw = hairball.get("config")
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    result: List[Dict[str, Any]] = []
    seen: set = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        if not key or key in seen:
            continue
        # Must have at least key and description
        desc = str(item.get("description", "")).strip()
        if not desc:
            continue
        entry: Dict[str, Any] = {
            "key": key,
            "description": desc,
        }
        default = item.get("default")
        if default is not None:
            entry["default"] = default
        prompt_text = item.get("prompt")
        if isinstance(prompt_text, str) and prompt_text.strip():
            entry["prompt"] = prompt_text.strip()
        else:
            entry["prompt"] = desc
        seen.add(key)
        result.append(entry)
    return result


def discover_all_skill_config_vars() -> List[Dict[str, Any]]:
    """Scan all enabled skills and collect their config variable declarations.

    Walks every skills directory, parses each SKILL.md frontmatter, and returns
    a deduplicated list of config var dicts.  Each dict also includes a
    ``skill`` key with the skill name for attribution.

    Disabled and platform-incompatible skills are excluded.
    """
    all_vars: List[Dict[str, Any]] = []
    seen_keys: set = set()

    disabled = get_disabled_skill_names()
    for skills_dir in get_all_skills_dirs():
        if not skills_dir.is_dir():
            continue
        for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
            try:
                raw = skill_file.read_text(encoding="utf-8")
                frontmatter, _ = parse_frontmatter(raw)
            except Exception:
                continue

            skill_name = frontmatter.get("name") or skill_file.parent.name
            if str(skill_name) in disabled:
                continue
            if not skill_matches_platform(frontmatter):
                continue

            config_vars = extract_skill_config_vars(frontmatter)
            for var in config_vars:
                if var["key"] not in seen_keys:
                    var["skill"] = str(skill_name)
                    all_vars.append(var)
                    seen_keys.add(var["key"])

    return all_vars


# Storage prefix: all skill config vars are stored under skills.config.*
# in config.yaml.  Skill authors declare logical keys (e.g. "wiki.path");
# the system adds this prefix for storage and strips it for display.
SKILL_CONFIG_PREFIX = "skills.config"


def _resolve_dotpath(config: Dict[str, Any], dotted_key: str):
    """Walk a nested dict following a dotted key.  Returns None if any part is missing."""
    parts = dotted_key.split(".")
    current = config
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def resolve_skill_config_values(
    config_vars: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve current values for skill config vars from config.yaml.

    Skill config is stored under ``skills.config.<key>`` in config.yaml.
    Returns a dict mapping **logical** keys (as declared by skills) to their
    current values (or the declared default if the key isn't set).
    Path values are expanded via ``os.path.expanduser``.
    """
    config = _load_raw_config()

    resolved: Dict[str, Any] = {}
    for var in config_vars:
        logical_key = var["key"]
        storage_key = f"{SKILL_CONFIG_PREFIX}.{logical_key}"
        value = _resolve_dotpath(config, storage_key)

        if value is None or (isinstance(value, str) and not value.strip()):
            value = var.get("default", "")

        # Expand ~ in path-like values
        if isinstance(value, str) and ("~" in value or "${" in value):
            value = os.path.expanduser(os.path.expandvars(value))

        resolved[logical_key] = value

    return resolved


# ── Description extraction ────────────────────────────────────────────────


def extract_skill_description(frontmatter: Dict[str, Any]) -> str:
    """Extract a truncated description from parsed frontmatter."""
    raw_desc = frontmatter.get("description", "")
    if not raw_desc:
        return ""
    desc = str(raw_desc).strip().strip("'\"")
    if len(desc) > 60:
        return desc[:57] + "..."
    return desc


# ── File iteration ────────────────────────────────────────────────────────


def iter_skill_index_files(skills_dir: Path, filename: str):
    """Walk skills_dir yielding sorted paths matching *filename*.

    Excludes Hairball metadata, VCS, virtualenv/dependency, cache, and skill
    support directories. Support directories (references/templates/assets/
    scripts) can contain arbitrary markdown and even archived package
    ``SKILL.md`` files, but they are progressive-disclosure data loaded through
    ``skill_view(..., file_path=...)`` rather than active skill roots.
    """
    skills_dir_str = str(skills_dir)
    matches: list[str] = []
    for root, dirs, files in os.walk(skills_dir_str, followlinks=True):
        has_skill_md = "SKILL.md" in files
        dirs[:] = [
            d
            for d in dirs
            if d not in EXCLUDED_SKILL_DIRS
            and not (has_skill_md and d in SKILL_SUPPORT_DIRS)
        ]
        if filename in files:
            matches.append(os.path.join(root, filename))
    for path in sorted(matches):
        yield Path(path)


# ── Namespace helpers for plugin-provided skills ───────────────────────────

_NAMESPACE_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def parse_qualified_name(name: str) -> Tuple[Optional[str], str]:
    """Split ``'namespace:skill-name'`` into ``(namespace, bare_name)``.

    Returns ``(None, name)`` when there is no ``':'``.
    """
    if ":" not in name:
        return None, name
    return tuple(name.split(":", 1))  # type: ignore[return-value]


def is_valid_namespace(candidate: Optional[str]) -> bool:
    """Check whether *candidate* is a valid namespace (``[a-zA-Z0-9_-]+``)."""
    if not candidate:
        return False
    return bool(_NAMESPACE_RE.match(candidate))
