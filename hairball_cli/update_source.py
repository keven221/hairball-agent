"""Single source of truth for Hairball installation and update locations.

The public source repository is not a Managed Portal entitlement. Personal
and BYOK installs must be able to discover and download open-source releases
even when every managed service is disabled. A private mirror can be selected
with ``updates.repository`` in ``config.yaml``; the legacy environment/config
keys remain read-only compatibility inputs for existing installations.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote, urlparse

DEFAULT_UPDATE_REPOSITORY = "hairball-agent/hairball-agent"

_REPOSITORY_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?$"
)


def _config() -> Mapping[str, Any]:
    try:
        from hairball_cli.config import load_config

        loaded = load_config() or {}
        return loaded if isinstance(loaded, Mapping) else {}
    except Exception:
        return {}


def canonical_github_remote(url: str | None) -> str:
    """Return lower-case ``github.com/owner/repository`` for common URLs."""
    if not url:
        return ""
    value = str(url).strip()
    lowered = value.lower()
    if lowered.startswith("git@github.com:"):
        value = "github.com/" + value[len("git@github.com:") :]
    elif lowered.startswith("ssh://git@github.com/"):
        value = "github.com/" + value[len("ssh://git@github.com/") :]
    else:
        parsed = urlparse(value)
        if parsed.netloc and parsed.path:
            value = f"{parsed.netloc}{parsed.path}"
    value = value.strip().rstrip("/")
    if value.lower().endswith(".git"):
        value = value[:-4]
    return value.lower()


def _validate_repository(value: str) -> str:
    repository = value.strip().strip("/")
    if not _REPOSITORY_RE.fullmatch(repository):
        raise ValueError(
            "Hairball update repository must use owner/repository form "
            "with GitHub-safe characters"
        )
    return repository


def _safe_ref(value: str, *, label: str) -> str:
    ref = str(value or "").strip()
    if not ref or any(ch in ref for ch in ("\x00", "\n", "\r")) or ".." in ref:
        raise ValueError(f"Invalid update {label}")
    return quote(ref, safe="/-._")


@dataclass(frozen=True)
class UpdateSource:
    repository: str

    @property
    def canonical_remote(self) -> str:
        return f"github.com/{self.repository}".lower()

    @property
    def https_git_url(self) -> str:
        return f"https://github.com/{self.repository}.git"

    @property
    def ssh_git_url(self) -> str:
        return f"git@github.com:{self.repository}.git"

    @property
    def official_remote_urls(self) -> frozenset[str]:
        base = self.repository
        return frozenset(
            {
                f"https://github.com/{base}",
                f"https://github.com/{base}.git",
                f"git@github.com:{base}",
                f"git@github.com:{base}.git",
                f"ssh://git@github.com/{base}",
                f"ssh://git@github.com/{base}.git",
            }
        )

    def is_official_remote(self, url: str | None) -> bool:
        return canonical_github_remote(url) == self.canonical_remote

    def archive_url(self, branch: str = "main") -> str:
        safe_branch = _safe_ref(branch, label="branch")
        return (
            f"https://github.com/{self.repository}/archive/refs/heads/"
            f"{safe_branch}.zip"
        )

    def release_url(self, tag: str) -> str:
        return (
            f"https://github.com/{self.repository}/releases/tag/"
            f"{_safe_ref(tag, label='tag')}"
        )


def resolve_update_source(repository: str | None = None) -> UpdateSource:
    """Resolve the public source/mirror without consulting managed services.

    Precedence is explicit argument, compatibility environment variable,
    ``updates.repository``, legacy ``managed_foreign.update_repo``, then the
    Hairball-owned public repository.
    """
    if repository:
        return UpdateSource(_validate_repository(str(repository)))

    cfg = _config()
    updates = cfg.get("updates") if isinstance(cfg, Mapping) else None
    managed = cfg.get("managed_foreign") if isinstance(cfg, Mapping) else None
    updates = updates if isinstance(updates, Mapping) else {}
    managed = managed if isinstance(managed, Mapping) else {}
    selected = (
        os.getenv("HAIRBALL_UPDATE_REPO", "").strip()
        or str(updates.get("repository") or "").strip()
        or str(managed.get("update_repo") or "").strip()
        or DEFAULT_UPDATE_REPOSITORY
    )
    return UpdateSource(_validate_repository(str(selected)))
