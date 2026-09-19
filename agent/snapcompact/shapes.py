"""Provider-aware frame shapes (OMP snapcompact contract, Hairball-owned).

Geometry and billing tables mirror oh-my-pi ``packages/snapcompact`` eval
winners. Rendering uses Pillow + system fonts — no Rust natives dependency.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# --- constants (oracle: snapcompact.ts) ---
MAX_FRAMES_DEFAULT = 80
FRAME_TOKEN_ESTIMATE = 5024
HQ_EDGE_FRAMES = 3
DOC_GUTTER = 3
CJK_HEAVY_MIN_WIDE_CHARS = 8
CJK_HEAVY_WIDE_RATIO = 0.25
DEFAULT_PROVIDER_IMAGE_BUDGET = 5
FRAME_DATA_BYTES_ESTIMATE = 170_000
FRAME_DATA_BYTES_BUDGET = 3_000_000

PROVIDER_IMAGE_BUDGETS: dict[str, int] = {
    "anthropic": 90,
    "amazon-bedrock": 90,
    "openrouter": 90,
    "openai": 200,
    "openai-codex": 200,
    "google": 200,
    "google-vertex": 200,
    "google-gemini-cli": 200,
    "umans": 10,
}

# Geometry half — production + research catalog (settings enum).
SHAPE_VARIANTS: dict[str, dict[str, Any]] = {
    "8x8r-bw": {"font": "8x8", "cell_width": 8, "cell_height": 8, "variant": "bw", "line_repeat": 2, "frame_size": 1568},
    "8x8r-sent": {"font": "8x8", "cell_width": 8, "cell_height": 8, "variant": "sent", "line_repeat": 2, "frame_size": 1568},
    "8x8u-bw": {"font": "8x8", "cell_width": 8, "cell_height": 8, "variant": "bw", "line_repeat": 1, "frame_size": 1568},
    "8x8u-sent": {"font": "8x8", "cell_width": 8, "cell_height": 8, "variant": "sent", "line_repeat": 1, "frame_size": 1568},
    "6x6u-bw": {"font": "8x8", "cell_width": 6, "cell_height": 6, "variant": "bw", "line_repeat": 1, "frame_size": 1568},
    "6x6u-sent": {"font": "8x8", "cell_width": 6, "cell_height": 6, "variant": "sent", "line_repeat": 1, "frame_size": 1568},
    "5x8-bw": {"font": "5x8", "cell_width": 5, "cell_height": 8, "variant": "bw", "line_repeat": 1, "frame_size": 2576},
    "5x8-sent": {"font": "5x8", "cell_width": 5, "cell_height": 8, "variant": "sent", "line_repeat": 1, "frame_size": 2576},
    "6x12-dim": {
        "font": "6x12",
        "cell_width": 6,
        "cell_height": 12,
        "variant": "bw",
        "stopword_dim": True,
        "line_repeat": 1,
        "frame_size": 1568,
    },
    "8x13-bw": {"font": "8x13", "cell_width": 8, "cell_height": 13, "variant": "bw", "line_repeat": 1, "frame_size": 1568},
    "8on16-bw": {
        "font": "8x13",
        "cell_width": 8,
        "cell_height": 16,
        "stretch": False,
        "variant": "bw",
        "line_repeat": 1,
        "frame_size": 1568,
    },
    "8on22-bw": {
        "font": "8x13",
        "cell_width": 8,
        "cell_height": 22,
        "stretch": False,
        "variant": "bw",
        "line_repeat": 1,
        "frame_size": 1568,
    },
    "11on16-bw": {
        "font": "8x13",
        "cell_width": 11,
        "cell_height": 16,
        "stretch": False,
        "variant": "bw",
        "line_repeat": 1,
        "frame_size": 1568,
    },
    "silver16-bw": {
        "font": "silver",
        "cell_width": 16,
        "cell_height": 16,
        "variant": "bw",
        "line_repeat": 1,
        "frame_size": 1568,
    },
    "doc-8on16-bw": {
        "font": "8x13",
        "cell_width": 8,
        "cell_height": 16,
        "stretch": False,
        "variant": "bw",
        "columns": 2,
        "line_repeat": 1,
        "frame_size": 1568,
    },
    "doc-8on16-sent": {
        "font": "8x13",
        "cell_width": 8,
        "cell_height": 16,
        "stretch": False,
        "variant": "sent",
        "columns": 2,
        "line_repeat": 1,
        "frame_size": 1568,
    },
    "doc-8on16-sent-dim": {
        "font": "8x13",
        "cell_width": 8,
        "cell_height": 16,
        "stretch": False,
        "variant": "sent",
        "stopword_dim": True,
        "columns": 2,
        "line_repeat": 1,
        "frame_size": 1568,
    },
}

SHAPE_VARIANT_NAMES = tuple(SHAPE_VARIANTS.keys())

FAMILY_VARIANT = {
    "anthropic": "11on16-bw",
    "google": "8on22-bw",
    "openai": "8on22-bw",
}

FAMILY_VARIANT_LOW = {
    "anthropic": "8on16-bw",
    "google": "8on16-bw",
    "openai": "8on16-bw",
}

_OPENAI_APIS = frozenset(
    {
        "openai-completions",
        "openai-responses",
        "openai-codex-responses",
        "azure-openai-responses",
        "chat_completions",
        "codex_responses",
    }
)
_GOOGLE_APIS = frozenset(
    {
        "google-generative-ai",
        "google-gemini-cli",
        "google-vertex",
        "gemini",
    }
)

_MODEL_VARIANTS: list[tuple[re.Pattern[str], dict[str, Any]]] = [
    (re.compile(r"claude.*(fable|mythos)", re.I), {"variant": "11on16-bw", "frame_size": 1932}),
    (re.compile(r"claude-?opus-?4[.-][7-9]", re.I), {"variant": "11on16-bw", "frame_size": 1932}),
    (re.compile(r"claude", re.I), {"variant": "11on16-bw"}),
    (re.compile(r"gemini", re.I), {"variant": "8on22-bw", "frame_size": 2048}),
    (re.compile(r"gpt|codex", re.I), {"variant": "8on22-bw"}),
    (re.compile(r"kimi", re.I), {"variant": "8on16-bw"}),
    (re.compile(r"glm", re.I), {"variant": "8on16-bw"}),
]


def is_shape_variant_name(value: object) -> bool:
    return isinstance(value, str) and value in SHAPE_VARIANTS


def billing_family(api: Optional[str] = None, provider: Optional[str] = None) -> str:
    api_l = (api or "").strip().lower().replace("_", "-")
    prov = (provider or "").strip().lower()
    if api_l in _OPENAI_APIS or prov in {"openai", "openai-codex"}:
        return "openai"
    if api_l in _GOOGLE_APIS or prov in {"google", "google-vertex", "google-gemini-cli", "gemini"}:
        return "google"
    if prov in {"anthropic", "amazon-bedrock"}:
        return "anthropic"
    # Unknown wire → Anthropic pixel-area ceiling (oracle default).
    return "anthropic"


def family_billing(family: str, frame_size: int) -> dict[str, Any]:
    if family == "google":
        return {"frame_token_estimate": 1120}
    if family == "openai":
        patches = min(int(-(-frame_size // 32)) ** 2, 10_000)
        return {"frame_token_estimate": int(-(-(patches * 12) // 10)), "image_detail": "original"}
    patches = min(int(-(-frame_size // 28)) ** 2, 4784)
    return {"frame_token_estimate": int(-(-(patches * 105) // 100))}


def price_shape(base: dict[str, Any], family: str) -> dict[str, Any]:
    out = dict(base)
    out.update(family_billing(family, int(base["frame_size"])))
    out["billing_family"] = family
    return out


def ideal_shape_variant(model_id: str) -> Optional[dict[str, Any]]:
    if not model_id:
        return None
    for pattern, ideal in _MODEL_VARIANTS:
        if pattern.search(model_id):
            return dict(ideal)
    return None


def provider_image_budget(provider: Optional[str]) -> int:
    if not provider:
        return DEFAULT_PROVIDER_IMAGE_BUDGET
    return PROVIDER_IMAGE_BUDGETS.get(provider.strip().lower(), DEFAULT_PROVIDER_IMAGE_BUDGET)


def max_frames_for_data_budget(max_bytes: int = FRAME_DATA_BYTES_BUDGET) -> int:
    return max(1, max_bytes // FRAME_DATA_BYTES_ESTIMATE)


def geometry(shape: dict[str, Any], size: Optional[int] = None) -> dict[str, int]:
    edge = int(size if size is not None else shape["frame_size"])
    cell_w = int(shape["cell_width"])
    cell_h = int(shape["cell_height"])
    line_repeat = max(1, int(shape.get("line_repeat") or 1))
    grid_cols = max(1, edge // cell_w)
    rows = max(1, edge // cell_h // line_repeat)
    columns = int(shape.get("columns") or 1)
    if columns == 2:
        cols = max(1, (grid_cols - DOC_GUTTER) // 2)
        capacity = 2 * cols * rows
    else:
        cols = grid_cols
        capacity = grid_cols * rows
    return {"cols": cols, "rows": rows, "capacity": capacity, "frame_size": edge}


def resolve_shape(
    *,
    model_id: Optional[str] = None,
    api: Optional[str] = None,
    provider: Optional[str] = None,
    variant: Optional[str] = None,
) -> dict[str, Any]:
    family = billing_family(api, provider)
    if variant and variant not in (None, "", "auto"):
        if not is_shape_variant_name(variant):
            variant = FAMILY_VARIANT[family]
        return price_shape(SHAPE_VARIANTS[variant], family)

    ideal = ideal_shape_variant(model_id or "")
    name = (ideal or {}).get("variant") or FAMILY_VARIANT[family]
    base = dict(SHAPE_VARIANTS[name])
    if ideal and ideal.get("frame_size") is not None:
        base["frame_size"] = int(ideal["frame_size"])
    out = price_shape(base, family)
    out["variant_name"] = name
    return out


def uses_wide_cells(shape: dict[str, Any]) -> bool:
    return shape.get("font") != "silver"


def is_wide_codepoint(cp: int) -> bool:
    # East Asian wide / fullwidth ranges (approx oracle isWideCodePoint).
    return (
        0x1100 <= cp <= 0x115F
        or 0x2329 <= cp <= 0x232A
        or 0x2E80 <= cp <= 0xA4CF
        or 0xAC00 <= cp <= 0xD7A3
        or 0xF900 <= cp <= 0xFAFF
        or 0xFE10 <= cp <= 0xFE19
        or 0xFE30 <= cp <= 0xFE6F
        or 0xFF00 <= cp <= 0xFF60
        or 0xFFE0 <= cp <= 0xFFE6
        or 0x1F300 <= cp <= 0x1F9FF
        or 0x20000 <= cp <= 0x2FA1F
    )


def is_cjk_heavy_text(text: str) -> bool:
    graphic = 0
    wide = 0
    for ch in text or "":
        if ch.isspace():
            continue
        cp = ord(ch)
        if cp < 32:
            continue
        graphic += 1
        if is_wide_codepoint(cp):
            wide += 1
    if graphic == 0:
        return False
    return wide >= CJK_HEAVY_MIN_WIDE_CHARS and (wide / graphic) >= CJK_HEAVY_WIDE_RATIO


def resolve_shape_for_text(
    text: str,
    *,
    model_id: Optional[str] = None,
    api: Optional[str] = None,
    provider: Optional[str] = None,
    variant: Optional[str] = None,
) -> dict[str, Any]:
    shape = resolve_shape(model_id=model_id, api=api, provider=provider, variant=variant)
    if variant and variant not in (None, "", "auto"):
        return shape
    if shape.get("font") != "silver" and is_cjk_heavy_text(text):
        silver = resolve_shape(
            model_id=model_id, api=api, provider=provider, variant="silver16-bw"
        )
        silver["variant_name"] = "silver16-bw"
        return silver
    return shape
