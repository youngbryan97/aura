"""Turn a described game into a spec the solver can enumerate.

The language model's part is translation: reading someone's rules and saying
what the position is, what a move does, and who loses when nobody can move.
It does not work out who wins. The runtime does that by enumeration, so the
answer is checkable and the strategy it reports is the one that actually wins.

A plan that does not describe a solvable game produces nothing, and the turn
goes on as it would have.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.reasoning.finite_game import GameSpec, Move, Variable

__all__ = ["game_schema", "spec_from_plan", "plan_from_json", "describes_a_game"]

#: Words that make a message about a game with a winner, rather than about
#: anything else that has turns and players.
_ASKS_WHO_WINS = re.compile(
    r"\b(?:who\s+wins|who\s+would\s+win|winning\s+(?:move|strategy)|"
    r"best\s+(?:move|strategy|opening)|is\s+it\s+a\s+win|forced\s+win|"
    r"perfect\s+play|optimal\s+play|first\s+player|second\s+player)\b",
    re.IGNORECASE,
)
_HAS_RULES = re.compile(
    r"\b(?:your\s+turn|takes?\s+turns?|alternate|each\s+turn|per\s+turn|"
    r"you\s+lose|you\s+win|loses?\b|wins?\b|move[sd]?\b|players?\b)\b",
    re.IGNORECASE,
)


def describes_a_game(message: object) -> bool:
    """A loose gate. The solver refuses anything it cannot enumerate anyway."""
    text = str(message or "")
    return bool(_ASKS_WHO_WINS.search(text) and _HAS_RULES.search(text))


def game_schema() -> dict[str, Any]:
    """The shape of a game, as a JSON schema for a typed call."""
    return {
        "type": "object",
        "required": ["title", "variables", "moves"],
        "properties": {
            "title": {"type": "string"},
            "variables": {
                "type": "array",
                "description": (
                    "The numbers that describe a position. Prefer the quantity that "
                    "shrinks — the gap between two pieces, the size of a pile."
                ),
                "items": {
                    "type": "object",
                    "required": ["name", "initial", "low", "high"],
                    "properties": {
                        "name": {"type": "string"},
                        "initial": {"type": "integer"},
                        "low": {"type": "integer"},
                        "high": {"type": "integer"},
                    },
                },
            },
            "moves": {
                "type": "array",
                "description": "What a player may do, as a change to the variables.",
                "items": {
                    "type": "object",
                    "required": ["name", "deltas"],
                    "properties": {
                        "name": {"type": "string"},
                        "deltas": {
                            "type": "object",
                            "description": "Variable to change per step, e.g. {\"gap\": -1}.",
                            "additionalProperties": {"type": "integer"},
                        },
                        "min_step": {"type": "integer"},
                        "max_step": {"type": "integer"},
                    },
                },
            },
            "stuck_loses": {
                "type": "boolean",
                "description": "True when a player with no legal move loses.",
            },
        },
    }


def _identifier(value: object, fallback: str) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    return text or fallback


def _integer(value: object, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def spec_from_plan(plan: dict[str, Any]) -> GameSpec | None:
    """A solvable game, or None when the plan does not describe one."""
    raw = dict(plan or {})
    variables: list[Variable] = []
    for index, item in enumerate(list(raw.get("variables") or [])):
        item = dict(item or {})
        name = _identifier(item.get("name"), f"value {index + 1}")
        initial = _integer(item.get("initial"), 0)
        low = _integer(item.get("low"), 0)
        high = _integer(item.get("high"), max(initial, low))
        if high < low:
            low, high = high, low
        variables.append(Variable(name=name, initial=min(max(initial, low), high), low=low, high=high))

    known = {item.name for item in variables}
    moves: list[Move] = []
    for index, item in enumerate(list(raw.get("moves") or [])):
        item = dict(item or {})
        deltas = {
            _identifier(key, ""): _integer(value, 0)
            for key, value in dict(item.get("deltas") or {}).items()
        }
        deltas = {key: value for key, value in deltas.items() if key in known and value}
        if not deltas:
            continue
        low = max(1, _integer(item.get("min_step"), 1))
        high = max(low, _integer(item.get("max_step"), low))
        moves.append(Move(name=_identifier(item.get("name"), f"move {index + 1}"), deltas=deltas, steps=(low, high)))

    spec = GameSpec(
        title=str(raw.get("title") or "the game").strip()[:60],
        variables=tuple(variables),
        moves=tuple(moves),
        stuck_loses=bool(raw.get("stuck_loses", True)),
    )
    return None if spec.problems() else spec


def plan_from_json(text: str) -> GameSpec | None:
    """Read a game out of whatever came back, or None."""
    raw = str(text or "").strip()
    if not raw:
        return None
    for candidate in _objects(raw):
        try:
            loaded = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(loaded, dict) and loaded.get("variables"):
            spec = spec_from_plan(loaded)
            if spec is not None:
                return spec
    return None


def _objects(text: str) -> list[str]:
    found: list[str] = []
    for start, character in enumerate(text):
        if character != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    found.append(text[start : end + 1])
                    break
    return sorted(found, key=len, reverse=True)
