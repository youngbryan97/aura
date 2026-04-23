from __future__ import annotations

import re
from typing import Any, Iterable

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


def normalize_goal_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("goal", "description", "title", "objective", "content", "name", "text"):
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
    return bool(text) and not is_intrinsic_goal_text(text)


def first_actionable_goal_text(values: Iterable[Any]) -> str:
    for value in values:
        text = normalize_goal_text(value)
        if text and is_actionable_goal_text(text):
            return text
    return ""
