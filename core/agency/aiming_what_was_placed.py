"""A part put into a chain is turned until what it sends out points on toward where the chain must go.

A person building a chain of devices, pipes, mirrors or conveyors puts a part where the last one's output ends, then
turns it so its own output points on toward the goal before adding the next. A part pointing back the way the chain
came wastes the next part, and one pointing at a wall ends the chain. LIVE 2026-10-10 a player of a trap-building game
put each device at the end of the last one's arrow and turned it toward the cage with the game's TURN control; she put
parts down and never turned one.

So, once a carry has answered and the place has a control that turns things (TURN, Rotate: learned over her model's
representation of words, with a word floor), her eyes are asked where the place's words say the next part goes (the
output's end, by the place's own name for it) and where the chain must reach (by its own name). Where that end lies
on the far side of the part from the goal, the turn control comes first, at most as many times as a turn has sides;
each turn has her eyes look again. Nothing here knows a device, a pipe or a cage.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

__all__ = ["MOST_TURNS", "TURNS_A_THING", "a_turn_wanted", "points_on", "turns_a_thing"]

#: The most turns given one part: a quarter-turn control has come round by then.
MOST_TURNS = 4

_TURNS = re.compile(r"\b(?:turn|rotate|rotation|spin)\b|[↻↺⟳⟲]", re.IGNORECASE)


def _surface() -> Any:
    from core.language.learned_matcher import LearnedMatcher
    from core.language.model_features import model_hidden_features

    return LearnedMatcher(name="turns_a_thing",
                          positives=("TURN", "Rotate", "Rotate left", "Rotate right", "Spin", "Turn around", "↻"),
                          negatives=("DELETE", "TEST", "START", "Library", "Undo", "Close", "Play", "Next"),
                          features=model_hidden_features)


TURNS_A_THING = _surface()


def turns_a_thing(label: str) -> bool:
    """Whether a control turns the thing it is used on: as the learned surface decides, else by its words."""
    try:
        decided = TURNS_A_THING.decide_without_waiting(label)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
        decided = None
    return bool(_TURNS.search(str(label or ""))) if decided is None else bool(decided)


def points_on(part_at: tuple[float, float] | None, output_at: tuple[float, float] | None,
              goal_at: tuple[float, float] | None) -> bool | None:
    """Whether a part's output points on toward the goal: the output's end on the goal's side of the part; None where any
    of the three is not known."""
    if part_at is None or output_at is None or goal_at is None:
        return None
    out = (output_at[0] - part_at[0], output_at[1] - part_at[1])
    goal = (goal_at[0] - part_at[0], goal_at[1] - part_at[1])
    if out == (0.0, 0.0) or goal == (0.0, 0.0):
        return None
    return out[0] * goal[0] + out[1] * goal[1] >= 0.0


def _where_it_must_reach(guide: Any) -> str:
    """The place the chain must reach, by the place's own name for it: where what it says it is for is, else where the
    plan's last step goes."""
    rules, plan = getattr(guide, "rules", None), getattr(guide, "plan", None)
    said = next((f.where for f in (rules.in_order() if rules is not None else []) if f.is_what_it_is_for and f.where), "")
    planned = next((f.where for f in reversed(plan.steps()) if f.where), "") if plan is not None else ""
    return said or planned


def a_turn_wanted(can_do: Any, guide: Any, offered: Sequence[str]) -> str:
    """The turn control to press now, where the part put down last points away from where the chain must go; "" where
    there is none, nothing was put down, it points on, or it has been turned round already."""
    from core.agency.what_i_can_do_here import what_is_clicked
    from core.perception.where_the_words_point import PLACES

    part_at = getattr(can_do, "chain_ends_at", None)
    if part_at is None or getattr(can_do, "turned_since_placed", MOST_TURNS) >= MOST_TURNS:
        return ""
    turning = [move for move in offered if turns_a_thing(what_is_clicked(move) or "")]
    output, goal = str(getattr(can_do, "place_named", "") or ""), _where_it_must_reach(guide)
    if not turning or not output or not goal:
        return ""
    return turning[0] if points_on(part_at, PLACES.where(output), PLACES.where(goal)) is False else ""
