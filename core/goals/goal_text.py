from __future__ import annotations

import re
from typing import Any, Iterable

from core.autonomy.research_goal_filter import is_stale_or_prompt_scaffold_goal

_INTRINSIC_GOAL_TEXTS = frozenset(
    {
        "stabilize runtime load and preserve continuous cognition",
        "protect identity, memory integrity, and process continuity",
        "investigate the most novel unresolved pattern in the current context",
        "seek clearer social grounding and relational understanding",
        "consolidate learning into durable improvements",
        "maintain system stability",
        "expand knowledge base",
        "improve code quality",
        "serve the user",
        "protect continuity",
        "protect continuity and keep the timeline coherent",
    }
)

_INTRINSIC_GOAL_PREFIXES = (
    "protect identity",
    "protect continuity",
    "maintain system stability",
    "expand knowledge base",
    "improve code quality",
    "serve the user",
    "stabilize runtime load and preserve continuous cognition",
    "seek clearer social grounding and relational understanding",
    "consolidate learning into durable improvements",
)


#: Fields that carry the human-readable goal, in order of preference.
_GOAL_TEXT_KEYS = ("goal", "description", "title", "objective", "content", "name", "text")

#: ``{'key': ...`` or ``{"key": ...`` — the opening of a serialized mapping.
#: Distinguishes a clipped repr from prose that merely starts with a brace.
_LOOKS_LIKE_SERIALIZED_MAPPING_RE = re.compile(r"""^\{\s*['"][\w.-]+['"]\s*:""")


def _mapping_from_serialized(value: str) -> dict | None:
    """Recover a mapping that was stringified on its way in, or None.

    A goal recorded as ``str(some_dict)`` arrives as a plain string, so the
    dict branch below never sees it and the repr becomes the goal text. Live
    2026-08-10, three of five persisted active goals read like this:

        "{'id': 'db847edb9427', 'name': '[AUTONOMOUS INITIATIVE] ...', ...}"

    They passed every actionability check and counted as outstanding
    obligations, and the first of them was offered to the person as the reason
    she could not act. Parsing it back is more honest than reading it aloud.
    """
    text = value.strip()
    if not text.startswith("{"):
        return None
    if text.endswith("}"):
        for parse in (_literal_eval, _json_loads):
            parsed = parse(text)
            if isinstance(parsed, dict):
                return parsed
    # It opened like a mapping and could not be read back as one. The live
    # record held goals stringified AND then clipped mid-repr —
    # "{'id': 'db847edb9427', ... 'horizon': 'short_term', 's" — so nothing
    # can recover the text. An unreadable serialization is not a goal, and
    # reading the fragment aloud as one is worse than having none: it counted
    # as an outstanding obligation and was offered to the person as the reason
    # she could not act.
    if _LOOKS_LIKE_SERIALIZED_MAPPING_RE.search(text):
        return {}
    return None


def _literal_eval(text: str) -> Any:
    import ast

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _json_loads(text: str) -> Any:
    import json

    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def normalize_goal_text(value: Any) -> str:
    if isinstance(value, str):
        recovered = _mapping_from_serialized(value)
        if recovered is not None:
            value = recovered
    if isinstance(value, dict):
        for key in _GOAL_TEXT_KEYS:
            candidate = value.get(key)
            if candidate:
                return " ".join(str(candidate).split())
        return ""
    return " ".join(str(value or "").split())


def _goal_signature(value: Any) -> str:
    text = normalize_goal_text(value).strip().lower()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;!?")


def is_intrinsic_goal_text(value: Any) -> bool:
    signature = _goal_signature(value)
    if not signature:
        return False
    if signature in _INTRINSIC_GOAL_TEXTS:
        return True
    return any(signature.startswith(prefix) for prefix in _INTRINSIC_GOAL_PREFIXES)


def is_actionable_goal_text(value: Any) -> bool:
    text = normalize_goal_text(value)
    if not text or is_intrinsic_goal_text(text) or is_stale_or_prompt_scaffold_goal(text):
        return False
    # Standing-objective validity is the durable-goal ingress authority:
    # ephemeral chat turns, control-contract scaffolds, and non-linguistic
    # renders must never become actionable volitional state (live evidence:
    # a check-in question and a NetHack framebuffer both reached CURRENT
    # IMPERATIVE at urgency 0.98). Imported lazily — objective_lifecycle
    # imports this module at module scope.
    from core.goals.standing_objective import is_valid_standing_objective

    return is_valid_standing_objective(value)


def first_actionable_goal_text(values: Iterable[Any]) -> str:
    for value in values:
        text = normalize_goal_text(value)
        if text and is_actionable_goal_text(text):
            return text
    return ""
