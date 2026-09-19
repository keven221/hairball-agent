"""Opt-in vision-frame compression (snapcompact-style, Hairball-owned).

Renders overflow text into dense PNG frames for vision-capable models.
Default **off** via ``compression.vision_frames``. Uses Pillow + system fonts;
provider shape tables / CJK fallback / savings journal mirror the OMP
snapcompact *contract* without a Rust natives dependency.

See ``docs/design/snapcompact-design.md``.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Optional

from agent.snapcompact.fonts import load_font
from agent.snapcompact.journal import append_savings, read_savings, savings_journal_path
from agent.snapcompact.normalize import normalize, wrap_lines
from agent.snapcompact.shapes import (
    FRAME_TOKEN_ESTIMATE,
    MAX_FRAMES_DEFAULT,
    SHAPE_VARIANT_NAMES,
    SHAPE_VARIANTS,
    geometry,
    is_shape_variant_name,
    is_wide_codepoint,
    provider_image_budget,
    resolve_shape,
    resolve_shape_for_text,
    uses_wide_cells,
)

logger = logging.getLogger("agent.snapcompact")

_PAD = 4

# Public re-exports
__all__ = [
    "FRAME_TOKEN_ESTIMATE",
    "MAX_FRAMES_DEFAULT",
    "SHAPE_VARIANT_NAMES",
    "SHAPE_VARIANTS",
    "append_savings",
    "frames_to_image_parts",
    "geometry",
    "inject_vision_frames_into_messages",
    "is_available",
    "is_shape_variant_name",
    "maybe_attach_vision_frames",
    "normalize",
    "pillow_available",
    "provider_image_budget",
    "read_savings",
    "render_frames",
    "render_text_png",
    "resolve_shape",
    "resolve_shape_for_text",
    "savings_journal_path",
    "vision_frames_enabled",
    "vision_frames_journal_enabled",
    "vision_frames_max",
    "vision_frames_shape",
]


def _load_config(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if config is not None:
        return config if isinstance(config, dict) else {}
    try:
        from hairball_cli.config import load_config

        return load_config() or {}
    except Exception:  # noqa: BLE001
        return {}


def _compression(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    cfg = _load_config(config)
    comp = cfg.get("compression") if isinstance(cfg, dict) else None
    return comp if isinstance(comp, dict) else {}


def vision_frames_enabled(config: Optional[dict[str, Any]] = None) -> bool:
    return bool(_compression(config).get("vision_frames", False))


def vision_frames_max(config: Optional[dict[str, Any]] = None) -> int:
    try:
        n = int(_compression(config).get("vision_frames_max", 4))
    except (TypeError, ValueError):
        n = 4
    return max(1, min(MAX_FRAMES_DEFAULT, n))


def vision_frames_shape(config: Optional[dict[str, Any]] = None) -> str:
    raw = str(_compression(config).get("vision_frames_shape") or "auto").strip()
    if raw in ("", "auto"):
        return "auto"
    return raw if is_shape_variant_name(raw) else "auto"


def vision_frames_journal_enabled(config: Optional[dict[str, Any]] = None) -> bool:
    return bool(_compression(config).get("vision_frames_journal", True))


def pillow_available() -> bool:
    try:
        import PIL.Image  # noqa: F401
        import PIL.ImageDraw  # noqa: F401
        import PIL.ImageFont  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def is_available(config: Optional[dict[str, Any]] = None) -> bool:
    return vision_frames_enabled(config) and pillow_available()


def render_text_png(
    text: str,
    *,
    shape: Optional[dict[str, Any]] = None,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
) -> bytes:
    """Render up to ``rows`` lines into a PNG sized to the shape frame edge."""
    from PIL import Image, ImageDraw

    shape = shape or resolve_shape()
    geo = geometry(shape)
    cols = int(cols if cols is not None else geo["cols"])
    rows = int(rows if rows is not None else geo["rows"])
    cell_w = int(shape["cell_width"])
    cell_h = int(shape["cell_height"])
    line_repeat = max(1, int(shape.get("line_repeat") or 1))
    columns = int(shape.get("columns") or 1)
    wide = uses_wide_cells(shape)

    normalized = normalize(text)
    lines = wrap_lines(normalized, cols, wide_cells=wide)[:rows]

    if columns == 2:
        # Newspaper: left then right column share the frame.
        mid = (len(lines) + 1) // 2
        left, right = lines[:mid], lines[mid:]
        while len(left) < rows:
            left.append("")
        while len(right) < rows:
            right.append("")
        display_rows = []
        gutter = " " * 3
        for a, b in zip(left[:rows], right[:rows]):
            display_rows.append(a.ljust(cols)[:cols] + gutter + b.ljust(cols)[:cols])
        lines = display_rows
        draw_cols = cols * 2 + 3
    else:
        while len(lines) < 1:
            lines.append("")
        draw_cols = cols

    edge = int(shape.get("frame_size") or geo["frame_size"])
    # Height hugs printed rows (oracle: never bill blank pixel rows).
    content_h = max(1, len(lines)) * cell_h * line_repeat
    width = min(edge, draw_cols * cell_w + 2 * _PAD)
    height = min(edge, content_h + 2 * _PAD)
    # Prefer square-ish frame when content fills; clamp to edge.
    if content_h + 2 * _PAD >= edge * 0.9:
        width = edge
        height = edge

    bg = (18, 18, 18) if shape.get("variant") == "bw" else (12, 12, 18)
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = load_font(shape)
    ink = (230, 230, 230)
    sent_hues = [(230, 230, 230), (200, 220, 255), (255, 210, 180), (200, 255, 210), (255, 200, 220), (220, 200, 255)]
    y = _PAD
    hue_i = 0
    for line in lines:
        color = ink
        if shape.get("variant") == "sent":
            color = sent_hues[hue_i % len(sent_hues)]
            if line.endswith((".", "!", "?", "。", "！", "？")):
                hue_i += 1
        for _ in range(line_repeat):
            # Character advance approximating cell pitch.
            x = _PAD
            for ch in line:
                draw.text((x, y), ch, fill=color, font=font)
                x += cell_w * (2 if (wide and len(ch) == 1 and is_wide_codepoint(ord(ch))) else 1)
                if x >= width - _PAD:
                    break
            y += cell_h
            if y >= height - _PAD:
                break
        if y >= height - _PAD:
            break

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_frames(
    text: str,
    *,
    max_frames: int = 4,
    shape: Optional[dict[str, Any]] = None,
    model_id: Optional[str] = None,
    api: Optional[str] = None,
    provider: Optional[str] = None,
    variant: Optional[str] = None,
) -> list[bytes]:
    """Split text into up to ``max_frames`` PNG pages for the resolved shape."""
    if not pillow_available():
        return []
    shape = shape or resolve_shape_for_text(
        text or "",
        model_id=model_id,
        api=api,
        provider=provider,
        variant=variant,
    )
    geo = geometry(shape)
    cols, rows = geo["cols"], geo["rows"]
    wide = uses_wide_cells(shape)
    lines = wrap_lines(normalize(text or ""), cols, wide_cells=wide)
    if not lines:
        return []
    cap = max(1, min(MAX_FRAMES_DEFAULT, int(max_frames)))
    # Also respect provider image-count budget and payload budget.
    if provider:
        cap = min(cap, provider_image_budget(provider))
    frames: list[bytes] = []
    step = rows if int(shape.get("columns") or 1) != 2 else rows
    for i in range(0, len(lines), step):
        if len(frames) >= cap:
            break
        chunk_text = "\n".join(lines[i : i + step])
        try:
            frames.append(render_text_png(chunk_text, shape=shape, cols=cols, rows=rows))
        except Exception as exc:  # noqa: BLE001
            logger.debug("snapcompact render failed: %s", exc)
            break
    return frames


def maybe_attach_vision_frames(
    summary_text: str,
    *,
    config: Optional[dict[str, Any]] = None,
    model_id: Optional[str] = None,
    api: Optional[str] = None,
    provider: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Return ``{enabled, frames, shape, reason, ...}`` for compressor integration."""
    if not vision_frames_enabled(config):
        return {"enabled": False, "frames": [], "reason": "vision_frames disabled"}
    if not pillow_available():
        return {"enabled": False, "frames": [], "reason": "Pillow not installed"}

    variant = vision_frames_shape(config)
    shape = resolve_shape_for_text(
        summary_text or "",
        model_id=model_id,
        api=api,
        provider=provider,
        variant=variant,
    )
    frames = render_frames(
        summary_text or "",
        max_frames=vision_frames_max(config),
        shape=shape,
        model_id=model_id,
        api=api,
        provider=provider,
        variant=variant,
    )
    estimate = int(shape.get("frame_token_estimate") or FRAME_TOKEN_ESTIMATE)
    # Rough text-token stand-in for journal (chars/4).
    text_tokens = max(1, len(summary_text or "") // 4)
    image_tokens = len(frames) * estimate
    saved = max(0, text_tokens - image_tokens)

    result: dict[str, Any] = {
        "enabled": True,
        "frames": frames,
        "count": len(frames),
        "shape": {
            "variant_name": shape.get("variant_name"),
            "font": shape.get("font"),
            "cell_width": shape.get("cell_width"),
            "cell_height": shape.get("cell_height"),
            "frame_size": shape.get("frame_size"),
            "frame_token_estimate": estimate,
            "billing_family": shape.get("billing_family"),
            "image_detail": shape.get("image_detail"),
        },
        "saved_tokens_estimate": saved,
        "reason": "ok" if frames else "empty text",
    }

    if frames and saved > 0 and vision_frames_journal_enabled(config):
        append_savings(
            [
                {
                    "session": session_id or "",
                    "provider": provider or "",
                    "model": model_id or "",
                    "toolCallId": "compress",
                    "savedTokens": saved,
                }
            ]
        )
        result["journaled"] = True
    return result


def frames_to_image_parts(
    frames: list[bytes],
    *,
    image_detail: Optional[str] = None,
) -> list[dict[str, Any]]:
    """OpenAI chat-completions multimodal parts for PNG frames."""
    import base64

    parts: list[dict[str, Any]] = []
    for png in frames:
        if not png:
            continue
        b64 = base64.b64encode(png).decode("ascii")
        image_url: dict[str, Any] = {"url": f"data:image/png;base64,{b64}"}
        if image_detail:
            image_url["detail"] = image_detail
        parts.append({"type": "image_url", "image_url": image_url})
    return parts


def inject_vision_frames_into_messages(
    messages: list[dict[str, Any]],
    frames: list[bytes],
    *,
    image_detail: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Attach PNG frames onto the first non-system message (OpenAI multimodal).

    Preserves role alternation by extending an existing user/assistant
    content list rather than inserting a new user turn. No-op when
    ``frames`` is empty. Returns a shallow-copied message list.
    """
    if not frames or not messages:
        return messages
    image_parts = frames_to_image_parts(frames, image_detail=image_detail)
    if not image_parts:
        return messages

    out = [dict(m) if isinstance(m, dict) else m for m in messages]
    idx = 0
    while idx < len(out) and isinstance(out[idx], dict) and out[idx].get("role") == "system":
        idx += 1
    if idx >= len(out) or not isinstance(out[idx], dict):
        out.insert(
            idx,
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "[vision compaction] Prior discarded context "
                            "rendered as glyph frames:"
                        ),
                    },
                    *image_parts,
                ],
            },
        )
        return out

    msg = dict(out[idx])
    content = msg.get("content")
    guide = {
        "type": "text",
        "text": "\n\n[vision compaction] Glyph frames of discarded context follow.",
    }
    if isinstance(content, str):
        msg["content"] = [{"type": "text", "text": content}, guide, *image_parts]
    elif isinstance(content, list):
        msg["content"] = list(content) + [guide, *image_parts]
    else:
        msg["content"] = [guide, *image_parts]
    out[idx] = msg
    return out
