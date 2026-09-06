"""The part of a result every skill actually keeps, and a check that it does.

CrewAI requires structured schemas both ways so a consumer knows the shape of
a result without running the tool. Aura declared the argument side and left
the result to whatever the caller happened to get: 82 tools, 78 of them
silent about what they give back.

Reading them, 72 of the 78 return a dict with an ``ok`` field, and on failure
an ``error``. That is a real shared contract rather than a convention someone
hoped for, so it is written down here once and pointed at, instead of 72
guesses at what each one returns in full.

``additionalProperties`` is true on purpose. This says what every skill keeps,
not everything any skill returns — a schema that claimed to be complete would
be wrong for every skill that adds a field, and a wrong schema is worse than
none because consumers act on it.

``how_a_result_differed`` is what stops this being decorative. A declaration
nothing checks is a comment; the runtime compares what came back against what
was declared and counts the differences.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.WhatEverySkillGivesBack")

__all__ = [
    "THE_SHARED_RESULT",
    "check_a_result",
    "how_results_have_differed",
]

#: What every skill keeps. Not everything any skill returns.
THE_SHARED_RESULT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {
            "type": "boolean",
            "description": "whether the skill did what it was asked",
        },
        "error": {
            "type": "string",
            "description": "why not, when ok is false",
        },
        "summary": {
            "type": "string",
            "description": "one line a person can read",
        },
    },
    "required": ["ok"],
    "additionalProperties": True,
}

_DIFFERED: dict[str, dict[str, int]] = {}

#: How many distinct complaints are kept per skill. Enough to see the shape.
_HOW_MANY = 6


def check_a_result(skill: str, declared: Any, result: Any) -> list[str]:
    """Compare a result against what the skill declared. Records, never raises.

    Returns what did not match. A skill returning the wrong shape is a defect
    in the skill, but a turn must not die because the shape was wrong — the
    caller already has an answer, and taking it away helps nobody.
    """
    complaints = _differences(declared, result)
    if complaints:
        held = _DIFFERED.setdefault(str(skill), {})
        for one in complaints:
            held[one] = held.get(one, 0) + 1
            if len(held) > _HOW_MANY:
                held.pop(next(iter(held)))
        logger.debug("%s gave back something it did not declare: %s", skill, complaints)
    return complaints


def _differences(declared: Any, result: Any) -> list[str]:
    if not isinstance(declared, dict) or not declared:
        return []
    complaints: list[str] = []
    if declared.get("type") == "object" and not isinstance(result, dict):
        return [f"declared an object and gave back {type(result).__name__}"]
    if not isinstance(result, dict):
        return complaints
    for name in declared.get("required") or ():
        if name not in result:
            complaints.append(f"declared {name} and did not give it")
    properties = declared.get("properties") or {}
    for name, shape in properties.items():
        if name not in result:
            continue
        wanted = shape.get("type")
        value = result[name]
        if wanted == "boolean" and not isinstance(value, bool):
            complaints.append(f"{name} was declared a boolean and is a {type(value).__name__}")
        elif wanted == "string" and not isinstance(value, str):
            complaints.append(f"{name} was declared a string and is a {type(value).__name__}")
        elif wanted == "object" and not isinstance(value, dict):
            complaints.append(f"{name} was declared an object and is a {type(value).__name__}")
        elif wanted == "array" and not isinstance(value, list):
            complaints.append(f"{name} was declared an array and is a {type(value).__name__}")
    if not declared.get("additionalProperties", True) and properties:
        extra = sorted(set(result) - set(properties))
        if extra:
            complaints.append(f"gave back fields it did not declare: {', '.join(extra)}")
    return complaints


def how_results_have_differed() -> dict[str, dict[str, int]]:
    """Every skill whose result did not match what it declared, and how.

    Empty is the goal. A declaration nothing checks is a comment.
    """
    return {name: dict(held) for name, held in sorted(_DIFFERED.items())}


def forget_everything() -> None:
    """For tests. The live runtime never calls this."""
    _DIFFERED.clear()
