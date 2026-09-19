"""Post-tool empty-response recovery (Path A2).

Hairball upgrade over legacy-identical wording: keep the same *mechanism*
(one synthetic user nudge after empty-after-tools, still reset on later tool
success so a later empty round can recover) but neutralize the copy so weak
models are not steered into re-doing work already completed by tools.

Mainstream posture (Anthropic/OpenAI tool loops; clawdbot incomplete-text
recovery comment in conversation_loop): tool results already carry the
continue signal — recovery text should ask for synthesis / next needed
action, not "continue with the task" (ambiguous = redo).
"""

from __future__ import annotations

# Stable constant for tests / persistence strip assertions.
POST_TOOL_EMPTY_NUDGE = (
    "You returned an empty response after tool results were delivered. "
    "Please read those tool results and produce the next assistant reply "
    "(a brief summary, the next required step, or a tool call that is still needed). "
    "Do not re-invoke a tool that already succeeded with the same arguments "
    "unless the tool result itself indicates failure or incomplete work."
)


def build_post_tool_empty_nudge() -> str:
    """User-role recovery hint injected after empty assistant post-tools."""
    return POST_TOOL_EMPTY_NUDGE


__all__ = ["POST_TOOL_EMPTY_NUDGE", "build_post_tool_empty_nudge"]
