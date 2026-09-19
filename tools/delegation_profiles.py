"""Built-in agent profiles for ``delegate_task``.

Profiles narrow a child's toolsets / tool allow-deny list and append a short
system-prompt additive. They do not change the parent's conversation tools
or prompt cache.

Unknown profile names raise ``ValueError`` so the model can correct the call.
``None`` / empty / ``default`` / ``worker`` keep today's inherit-parent behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class DelegationProfile:
    """Resolved profile applied while constructing a child subagent."""

    name: str
    description: str
    # When set, passed as the child's requested toolsets (still intersected
    # with the parent's available toolsets inside ``_build_child_agent``).
    toolsets: Optional[List[str]] = None
    # Hard-drop these tool names after the child AIAgent is built.
    deny_tools: frozenset[str] = field(default_factory=frozenset)
    # When False, skip preserving parent MCP toolsets on a narrowed child.
    inherit_mcp: bool = True
    # Appended to the child ephemeral system prompt (after the base block).
    system_prompt_addendum: str = ""


_EXPLORE_ADDENDUM = """
## Profile: Explore (read-only)

You are a read-only search / reconnaissance subagent.

Hard rules:
- Do NOT create, edit, delete, move, or patch files.
- Do NOT commit, push, install packages, or mutate git state.
- Use `search_files` and `read_file` for code search/read. Use terminal only
  for read-only git/status/list commands (`git status`, `git log`,
  `git blame`, `ls`).
- Prefer excerpts and pointers (path:line) over dumping whole files.
- Your job is to locate and conclude — not to review style, audit security,
  or implement fixes. Leave remediation to the parent.

Return a tight summary: what you searched, what you found (with paths),
and any open questions. Do not claim to have changed anything.
""".strip()


BUILTIN_DELEGATION_PROFILES: Dict[str, DelegationProfile] = {
    "default": DelegationProfile(
        name="default",
        description=(
            "Inherit the parent's toolsets (minus always-blocked child tools). "
            "Same behavior as omitting profile."
        ),
    ),
    "worker": DelegationProfile(
        name="worker",
        description=(
            "Alias for default — general-purpose leaf worker that inherits "
            "parent toolsets."
        ),
    ),
    "explore": DelegationProfile(
        name="explore",
        description=(
            "Read-only fan-out searcher. Narrow toolsets; strips write_file/"
            "patch/ast_edit; no MCP inherit. Use for broad codebase sweeps when you "
            "need conclusions, not dumps."
        ),
        toolsets=["terminal", "file", "web", "session_search"],
        deny_tools=frozenset({"write_file", "patch", "skill_manage", "ast_edit", "debug"}),
        inherit_mcp=False,
        system_prompt_addendum=_EXPLORE_ADDENDUM,
    ),
}

# Names that mean "no profile override" (inherit parent).
_DEFAULT_ALIASES = frozenset({"", "default", "worker"})


def list_delegation_profiles() -> List[DelegationProfile]:
    """Return built-in profiles (stable order: default, worker, explore…)."""
    order = ["default", "worker", "explore"]
    seen = set(order)
    out = [BUILTIN_DELEGATION_PROFILES[n] for n in order if n in BUILTIN_DELEGATION_PROFILES]
    for name, profile in sorted(BUILTIN_DELEGATION_PROFILES.items()):
        if name not in seen:
            out.append(profile)
    return out


def resolve_delegation_profile(name: Optional[str]) -> Optional[DelegationProfile]:
    """Resolve a profile name.

    Returns ``None`` for default/worker/empty (caller keeps inherit-parent
    behavior). Raises ``ValueError`` for unknown non-empty names.
    """
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in _DEFAULT_ALIASES:
        return None
    profile = BUILTIN_DELEGATION_PROFILES.get(key)
    if profile is None:
        known = ", ".join(sorted(BUILTIN_DELEGATION_PROFILES))
        raise ValueError(
            f"Unknown delegation profile {name!r}. Known profiles: {known}."
        )
    # default/worker are in the map for discovery; resolve still yields None.
    if profile.name in _DEFAULT_ALIASES:
        return None
    return profile


def apply_profile_tool_denies(child, deny_tools: Sequence[str]) -> None:
    """Drop denied tools from a built child's schema + name set.

    Safe no-op when ``deny_tools`` is empty or the child has no tools attr.
    """
    if not deny_tools:
        return
    deny = {str(t) for t in deny_tools if t}
    if not deny:
        return
    tools = getattr(child, "tools", None)
    if not tools:
        return
    kept = []
    for tool in tools:
        try:
            tname = tool["function"]["name"]
        except (TypeError, KeyError, IndexError):
            kept.append(tool)
            continue
        if tname in deny:
            continue
        kept.append(tool)
    child.tools = kept
    names = getattr(child, "valid_tool_names", None)
    if isinstance(names, set):
        child.valid_tool_names = {n for n in names if n not in deny}
    else:
        child.valid_tool_names = {
            t["function"]["name"]
            for t in kept
            if isinstance(t, dict) and "function" in t and "name" in t["function"]
        }


def profile_schema_enum() -> List[str]:
    """Enum values exposed on the ``delegate_task`` schema."""
    return sorted(BUILTIN_DELEGATION_PROFILES.keys())


def build_profile_param_description() -> str:
    """Human-readable profile parameter help for the tool schema."""
    lines = [
        "Optional agent profile for the child subagent. "
        "Omit or use 'default'/'worker' to inherit the parent's toolsets "
        "(current behavior). Known profiles:"
    ]
    for profile in list_delegation_profiles():
        lines.append(f"- {profile.name}: {profile.description}")
    lines.append(
        "Per-task 'profile' overrides the top-level value. "
        "Unknown names are rejected with an error."
    )
    return "\n".join(lines)
