#!/usr/bin/env python3
"""
Clarify Tool Module - Interactive Clarifying Questions

Allows the agent to present structured multiple-choice questions or open-ended
prompts to the user. In CLI mode, choices are navigable with arrow keys. On
messaging platforms, choices are rendered as a numbered list.

The actual user-interaction logic lives in the platform layer (cli.py for CLI,
gateway/run.py for messaging). This module defines the schema, validation, and
a thin dispatcher that delegates to a platform-provided callback.
"""

import json
from typing import Any, Callable, List, Optional


# Maximum number of predefined choices the agent can offer.
# A 5th "Other (type your answer)" option is always appended by the UI.
MAX_CHOICES = 4


def _flatten_choice(c) -> str:
    """Coerce a single choice into its user-facing display string.

    The schema declares choices as bare strings, but LLMs sometimes emit
    dict-shaped choices like ``[{"description": "..."}]``. A naive ``str(c)``
    turns the whole dict into its Python repr — ``{'description': '...'}`` —
    which then leaks onto every surface that renders the choice (CLI panel,
    Discord buttons, Telegram numbered list) AND is returned verbatim as the
    user's answer. Normalising here, at the one platform-agnostic entry point,
    fixes the whole class in one place instead of per-adapter.

    Dict unwrap order is the canonical LLM tool-call user-facing keys:
    ``label`` → ``description`` → ``text`` → ``title``. ``name`` and ``value``
    are deliberately excluded — they're component-shaped fields that could
    carry raw enum values or short identifiers, not human-readable labels. A
    dict with none of the canonical keys is dropped (returns ""), since a
    garbage label is worse than no choice at all.
    """
    if c is None:
        return ""
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, dict):
        for key in ("label", "description", "text", "title"):
            v = c.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if isinstance(c, (list, tuple)):
        return " ".join(_flatten_choice(x) for x in c).strip()
    return str(c).strip()


def _with_public_clarify_status(payload: dict[str, Any]) -> dict[str, Any]:
    """Adapt Plan's internal question state to Clarify's public wire contract.

    Plan persists ``pending``/``resolved`` because those are lifecycle states.
    The long-standing Clarify tool contract exposed ``answered`` and
    ``pending_partial`` to CLI, gateway, TUI, and desktop consumers.  Keep the
    two vocabularies isolated at this boundary instead of changing Plan's
    stored state or making every surface understand two status dialects.
    """

    result = dict(payload)
    unresolved = list(result.get("unresolved_question_ids") or [])
    if not unresolved:
        result["status"] = "answered"
    elif result.get("answers"):
        result["status"] = "pending_partial"
    else:
        result["status"] = "pending"
    return result


def clarify_tool(
    question: str,
    choices: Optional[List[str]] = None,
    callback: Optional[Callable] = None,
    questions: Optional[List[dict[str, Any]]] = None,
    session_id: str = "",
    surface: str = "agent",
) -> str:
    """
    Ask the user a question, optionally with multiple-choice options.

    Args:
        question: The question text to present.
        choices:  Up to 4 predefined answer choices. When omitted the
                  question is purely open-ended.
        callback: Platform-provided function that handles the actual UI
                  interaction. Signature: callback(question, choices) -> str.
                  Injected by the agent runner (cli.py / gateway).

    Returns:
        JSON string with the user's response.
    """
    if not question or not question.strip():
        return tool_error("Question text is required.")

    question = question.strip()

    if questions is not None:
        if not isinstance(questions, list):
            return tool_error("questions must be a list of question objects.")
        if not questions:
            return tool_error("questions must contain at least one question.")
        if len(questions) > 3:
            return tool_error("questions may contain at most 3 question objects.")
        normalized_questions: list[dict[str, Any]] = []
        for index, item in enumerate(questions, start=1):
            if not isinstance(item, dict):
                return tool_error(f"questions[{index - 1}] must be an object.")
            raw_choices = item.get("choices")
            if raw_choices is not None and not isinstance(raw_choices, list):
                return tool_error(
                    f"questions[{index - 1}].choices must be a list of strings."
                )
            clean_choices = None
            if isinstance(raw_choices, list):
                clean_choices = [
                    value
                    for value in (_flatten_choice(choice) for choice in raw_choices)
                    if value
                ][:MAX_CHOICES]
                if not clean_choices:
                    clean_choices = None
            normalized_questions.append(
                {
                    "id": item.get("id") or f"question-{index}",
                    "question": item.get("question") or item.get("text") or "",
                    "choices": clean_choices or [],
                    "required": bool(item.get("required", True)),
                    "allow_other": True,
                }
            )

        if callback is None:
            return json.dumps(
                {"error": "Clarify tool is not available in this execution context."},
                ensure_ascii=False,
            )
        try:
            from agent.plan_execution import (
                PlanContractError,
                ask_structured_questions,
                resolve_structured_answers,
            )

            request = ask_structured_questions(
                normalized_questions,
                session_id=str(session_id or ""),
                surface=str(surface or "agent"),
            )
        except (PlanContractError, ValueError, TypeError) as exc:
            return tool_error(str(exc))

        resolved = request
        for item in request["questions"]:
            try:
                answer = callback(
                    str(item.get("question") or ""),
                    list(item.get("choices") or []) or None,
                )
            except Exception as exc:
                partial = dict(resolved)
                partial["error"] = f"Failed to get user input: {exc}"
                return json.dumps(
                    _with_public_clarify_status(partial), ensure_ascii=False
                )
            try:
                resolved = resolve_structured_answers(
                    resolved,
                    {str(item.get("id") or ""): str(answer).strip()},
                    actor="user",
                )
            except PlanContractError as exc:
                partial = dict(resolved)
                partial["error"] = str(exc)
                return json.dumps(
                    _with_public_clarify_status(partial), ensure_ascii=False
                )
        resolved["prompt"] = question
        return json.dumps(
            _with_public_clarify_status(resolved), ensure_ascii=False
        )

    # Validate and trim choices
    if choices is not None:
        if not isinstance(choices, list):
            return tool_error("choices must be a list of strings.")
        # LLMs sometimes emit dict-shaped choices (e.g. [{"description": "..."}])
        # instead of bare strings. _flatten_choice unwraps them to their
        # user-facing text here — the single platform-agnostic entry point —
        # so the CLI panel, Discord buttons, and Telegram list all render clean
        # text and the resolved answer is never a raw Python dict repr.
        choices = [s for s in (_flatten_choice(c) for c in choices) if s]
        if len(choices) > MAX_CHOICES:
            choices = choices[:MAX_CHOICES]
        if not choices:
            choices = None  # empty list → open-ended

    if callback is None:
        return json.dumps(
            {"error": "Clarify tool is not available in this execution context."},
            ensure_ascii=False,
        )

    try:
        user_response = callback(question, choices)
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to get user input: {exc}"},
            ensure_ascii=False,
        )

    return json.dumps({
        "question": question,
        "choices_offered": choices,
        "user_response": str(user_response).strip(),
    }, ensure_ascii=False)


def check_clarify_requirements() -> bool:
    """Clarify tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

CLARIFY_SCHEMA = {
    "name": "clarify",
    "description": (
        "Ask the user a question when you need clarification, feedback, or a "
        "decision before proceeding. Supports two modes:\n\n"
        "1. **Multiple choice** — provide up to 4 choices. The user picks one "
        "or types their own answer via a 5th 'Other' option.\n"
        "2. **Open-ended** — omit choices entirely. The user types a free-form "
        "response.\n\n"
        "CRITICAL: when you are offering options, put each option ONLY in the "
        "`choices` array — NEVER enumerate the options inside the `question` "
        "text. The UI renders `choices` as selectable rows; options written "
        "into the question string render as dead prose the user can't pick. "
        "Right: question='Which deployment target?', choices=['staging', "
        "'prod']. Wrong: question='Which target? 1) staging 2) prod', choices=[].\n\n"
        "Use this tool when:\n"
        "- The task is ambiguous and you need the user to choose an approach\n"
        "- You want post-task feedback ('How did that work out?')\n"
        "- You want to offer to save a skill or update memory\n"
        "- A decision has meaningful trade-offs the user should weigh in on\n\n"
        "Do NOT use this tool for simple yes/no confirmation of dangerous "
        "commands (the terminal tool handles that). Prefer making a reasonable "
        "default choice yourself when the decision is low-stakes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "The question itself, and ONLY the question (e.g. 'Which "
                    "deployment target?'). Do NOT embed the answer options here "
                    "— pass them as separate elements in `choices`."
                ),
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_CHOICES,
                "description": (
                    "REQUIRED whenever you are presenting selectable options: "
                    "each distinct option is its own array element (up to 4). "
                    "The UI renders these as pickable rows and auto-appends an "
                    "'Other (type your answer)' option. Omit this parameter "
                    "entirely ONLY for a genuinely open-ended free-text question."
                ),
            },
            "questions": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "question": {"type": "string"},
                        "choices": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": MAX_CHOICES,
                        },
                        "required": {"type": "boolean"},
                    },
                    "required": ["question"],
                },
                "description": (
                    "Optional batch of up to 3 related questions. The legacy "
                    "top-level question remains a short heading; each batch item "
                    "is presented through the same platform-native clarify UI."
                ),
            },
        },
        "required": ["question"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="clarify",
    toolset="clarify",
    schema=CLARIFY_SCHEMA,
    handler=lambda args, **kw: clarify_tool(
        question=args.get("question", ""),
        choices=args.get("choices"),
        questions=args.get("questions"),
        callback=kw.get("callback"),
        session_id=str(kw.get("session_id") or kw.get("task_id") or ""),
        surface=str(kw.get("platform") or kw.get("surface") or "agent")),
    check_fn=check_clarify_requirements,
    emoji="❓",
)
