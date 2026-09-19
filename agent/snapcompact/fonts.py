"""System font discovery for snapcompact frames (Silver / CJK approximation)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger("agent.snapcompact.fonts")

# Prefer monospace for Latin bitmap-like shapes; CJK/Silver needs a Unicode face.
_MONO_CANDIDATES = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/Library/Fonts/SF-Mono-Regular.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "C:\\Windows\\Fonts\\consola.ttf",
    "C:\\Windows\\Fonts\\cour.ttf",
)

_CJK_CANDIDATES = (
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "C:\\Windows\\Fonts\\msyh.ttc",
    "C:\\Windows\\Fonts\\simsun.ttc",
)


def _first_existing(paths: tuple[str, ...]) -> Optional[str]:
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return None


@lru_cache(maxsize=1)
def mono_font_path() -> Optional[str]:
    env = os.environ.get("HAIRBALL_SNAPCOMPACT_FONT")
    if env and os.path.isfile(env):
        return env
    return _first_existing(_MONO_CANDIDATES)


@lru_cache(maxsize=1)
def cjk_font_path() -> Optional[str]:
    env = os.environ.get("HAIRBALL_SNAPCOMPACT_CJK_FONT")
    if env and os.path.isfile(env):
        return env
    return _first_existing(_CJK_CANDIDATES) or mono_font_path()


def load_font(shape: dict[str, Any]):
    """Return a Pillow ImageFont for ``shape`` (falls back to default bitmap)."""
    from PIL import ImageFont

    cell_h = max(8, int(shape.get("cell_height") or 16))
    # Slightly under cell height so glyphs fit the pitch.
    size = max(8, cell_h - 2 if shape.get("font") == "silver" else max(8, cell_h - 4))
    path = cjk_font_path() if shape.get("font") == "silver" else mono_font_path()
    if path:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception as exc:  # noqa: BLE001
            logger.debug("truetype load failed (%s): %s", path, exc)
    return ImageFont.load_default()
