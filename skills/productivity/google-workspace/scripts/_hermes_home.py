"""Resolve HAIRBALL_HOME for standalone skill scripts.

Skill scripts may run outside the Hairball process (e.g. system Python,
nix env, CI) where ``hairball_constants`` is not importable.  This module
provides the same ``get_hairball_home()`` and ``display_hairball_home()``
contracts as ``hairball_constants`` without requiring it on ``sys.path``.

When ``hairball_constants`` IS available it is used directly so that any
future enhancements (profile resolution, Docker detection, etc.) are
picked up automatically.  The fallback path replicates the core logic
from ``hairball_constants.py`` using only the stdlib.

All scripts under ``google-workspace/scripts/`` should import from here
instead of duplicating the ``HAIRBALL_HOME = Path(os.getenv(...))`` pattern.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from hairball_constants import display_hairball_home as display_hairball_home
    from hairball_constants import get_hairball_home as get_hairball_home
except (ModuleNotFoundError, ImportError):

    def get_hairball_home() -> Path:
        """Return the Hairball home directory (default: ~/.hairball).

        Mirrors ``hairball_constants.get_hairball_home()``."""
        val = os.environ.get("HAIRBALL_HOME", "").strip()
        return Path(val) if val else Path.home() / ".hairball"

    def display_hairball_home() -> str:
        """Return a user-friendly ``~/``-shortened display string.

        Mirrors ``hairball_constants.display_hairball_home()``."""
        home = get_hairball_home()
        try:
            return "~/" + str(home.relative_to(Path.home()))
        except ValueError:
            return str(home)
