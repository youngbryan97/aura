"""Using words for one move still leaves the judgement behind the others.

Measured live on 2026-08-26: a cycle deciding without words committed to four
moves on one screen reading, and the same cycle with words committed to one —
so the run spent its budget re-reading the board between moves it had already
made up its mind about.
"""

from __future__ import annotations

import inspect

import pytest

from core.agency.deliberate_action import (
    ActionOption,
    Expectation,
    deliberate,
    plan_without_language,
)

MOVES = [
    ActionOption(
        name=name,
        detail=f"press {name}",
        expectation=Expectation(changed=True, describes=f"the view to be different after {name}"),
    )
    for name in ("up", "down", "left", "right")
]


class _Held:
    held = True
    observed_change = True

    def why(self):
        return "as expected"


class _Attempt:
    def __init__(self, option: str) -> None:
        self.option = option
        self.expected = "the view to be different"
        self.verdict = _Held()
        self.progressed = True

    def as_evidence(self) -> str:
        return f"{self.option} did what I expected"


async def _names_one(_objective, _evidence):
    return "I'll press up."


@pytest.mark.asyncio
async def test_a_spoken_move_still_carries_a_sequence():
    history = [_Attempt("up") for _ in range(4)]
    made = await deliberate(
        "get to 256",
        "2 4 8 64",
        MOVES,
        think=_names_one,
        history=history,
        announce=False,
        lived=False,
    )
    assert made.chosen is not None
    assert made.chosen.name == "up"
    assert made.then, "her words named one move and the ranking behind it was thrown away"


def test_the_ranking_is_worked_out_whether_or_not_she_spoke():
    source = inspect.getsource(deliberate)
    ranked_at = source.index("ranked=ranking")
    branch_at = source.index("if chosen is None:")
    assert ranked_at < branch_at, "the ranking is still only computed when she named nothing"


def test_carrying_a_ranking_forward_keeps_her_move_first():
    carried = plan_without_language([MOVES[0], MOVES[2], MOVES[1]], 3)
    assert [option.name for option in carried] == ["up", "left", "down"]
