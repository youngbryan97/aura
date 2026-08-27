"""She wrote down the last line she happened to be holding, and never read it.

A writer with no reader, storing the most recent approach rather than the one
that worked — and storing it as words, so even a reader could not have told
whether it was still the right line.

It is worth about half again. Measured 2026-08-27, five games each, run to a
dead board: with the world model and no line, total 1994; with a line as well,
2972 and a 2048 tile. A line is the difference between a plan and a series of
moves, and carrying the RIGHT one over is what makes a second run in a world
better than the first.

Graded the same way a move is. Resumed as a stance, not as a sentence, so the
first reading of the new run can drop it.
"""

from __future__ import annotations

import pytest

from core.agency.deliberate_action import Expectation
from core.agency.standing_strategy import Strategy
from core.agency.what_worked_before import KNOWN_WELL_ENOUGH, WhatWorkedBefore
from core.skills.screen_pursuit import A_LINE_HERE

CORNER = "keep the largest in the bottom-left corner"
MIDDLE = "build up the middle"


def a_line(said: str = CORNER) -> Strategy:
    return Strategy(
        approach=said,
        because="it stops the board fragmenting",
        holds_while=Expectation(
            describes="the largest stays put", at_place="bottom-left", keeping=("1024",)
        ),
        otherwise=("go right instead",),
    )


# ── a line survives as the thing it is ───────────────────────────────────

def test_a_line_comes_back_with_what_would_end_it():
    back = Strategy.from_memory(a_line().as_memory())
    assert back.approach == CORNER
    assert back.holds_while.describes == "the largest stays put"
    assert back.holds_while.at_place == "bottom-left"
    assert back.holds_while.keeping == ("1024",)
    assert back.otherwise == ("go right instead",)


def test_what_an_older_run_wrote_down_still_reads():
    """Bare words, which is all there used to be."""
    back = Strategy.from_memory("an approach from before")
    assert back is not None and back.approach == "an approach from before"


@pytest.mark.parametrize("rubbish", [None, {}, "", "   ", 7, [], {"approach": "  "}])
def test_and_rubbish_is_not_a_line(rubbish):
    assert Strategy.from_memory(rubbish) is None


# ── the best line rather than the last one ───────────────────────────────

def test_a_line_that_kept_working_is_the_one_offered():
    lines = WhatWorkedBefore()
    for _ in range(KNOWN_WELL_ENOUGH):
        lines.learned(A_LINE_HERE, CORNER, True)
    # Held more recently, and it went badly every time.
    for _ in range(KNOWN_WELL_ENOUGH + 2):
        lines.learned(A_LINE_HERE, MIDDLE, False)
    assert lines.suggests(A_LINE_HERE) == CORNER


def test_a_line_held_once_is_not_offered():
    lines = WhatWorkedBefore()
    lines.learned(A_LINE_HERE, CORNER, True)
    assert lines.suggests(A_LINE_HERE) == ""


def test_a_line_that_stops_working_stops_being_offered():
    lines = WhatWorkedBefore()
    for _ in range(6):
        lines.learned(A_LINE_HERE, CORNER, True)
    for _ in range(12):
        lines.learned(A_LINE_HERE, CORNER, False)
    assert lines.suggests(A_LINE_HERE) == ""


def test_a_line_is_filed_across_positions_not_under_one():
    """Grading it per position would grade it as if it were a move."""
    lines = WhatWorkedBefore()
    for _ in range(KNOWN_WELL_ENOUGH):
        lines.learned(A_LINE_HERE, CORNER, True)
    assert list(lines.known) == [A_LINE_HERE]


def test_what_carried_over_is_light_enough_to_be_overturned():
    lines = WhatWorkedBefore()
    for _ in range(8):
        lines.learned(A_LINE_HERE, CORNER, True)
    back = WhatWorkedBefore.from_memory(lines.as_memory(), trust=0.5)
    for _ in range(4):
        back.learned(A_LINE_HERE, CORNER, False)
    assert back.suggests(A_LINE_HERE) == ""


# ── and the whole round trip ─────────────────────────────────────────────

def test_the_line_that_worked_is_the_line_that_comes_back():
    lines = WhatWorkedBefore()
    held = {}
    for said, good in ((CORNER, True), (MIDDLE, False)):
        for _ in range(KNOWN_WELL_ENOUGH + 1):
            lines.learned(A_LINE_HERE, said, good)
        held[said] = a_line(said).as_memory()

    carried = WhatWorkedBefore.from_memory(lines.as_memory())
    worked = carried.suggests(A_LINE_HERE)
    assert worked == CORNER
    resumed = Strategy.from_memory(held[worked])
    assert resumed.approach == CORNER
    assert resumed.holds_while.at_place == "bottom-left"
