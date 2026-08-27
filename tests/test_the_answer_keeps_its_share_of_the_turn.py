"""Evidence nobody can say is worth nothing.

LIVE, 2026-08-27. A repository diagnosis ran in 389ms and came back complete —
the contradicting call, the line it is on, the project's own broken invariant.
The tool-calling pass had taken 65 seconds of a 148-second turn, and the pass
that writes the answer was refused before dispatch: "refused local generation
because the request budget was already spent". The finding was in hand and
there was no time left to say it, so the turn served "I lost the reply lane for
a moment" and then spent another 101 seconds failing to repair that.

The tool loop's job is to GET the evidence. The reply's job is to SAY it.
"""

from __future__ import annotations

import pytest

from core.brain.inference_gate import (
    _ANSWER_RESERVE_S,
    _TOOL_LOOP_FLOOR_S,
    _tool_loop_budget,
)


def test_a_long_turn_keeps_time_back_for_the_answer() -> None:
    assert _tool_loop_budget(148.0) == 148.0 - _ANSWER_RESERVE_S


def test_the_reserve_is_enough_to_write_over_an_evidence_block() -> None:
    """The live failure had 0 seconds left; anything under a few is the same bug."""
    assert _ANSWER_RESERVE_S >= 30.0


@pytest.mark.parametrize("budget", [60.0, 45.0, 20.0, 5.0, 0.0, -3.0])
def test_a_short_turn_still_gets_a_usable_tool_loop(budget: float) -> None:
    """Squeezing below the floor trades one failure for another.

    A tool loop that cannot complete a single call fails just as completely as
    a reply with no time, so the floor holds and the turn runs over instead.
    """
    assert _tool_loop_budget(budget) == _TOOL_LOOP_FLOOR_S


@pytest.mark.parametrize("bad", [None, "x", object()])
def test_an_unreadable_budget_falls_back_to_the_floor(bad: object) -> None:
    assert _tool_loop_budget(bad) == _TOOL_LOOP_FLOOR_S


def test_the_tool_loop_never_gets_the_whole_turn() -> None:
    """The property, not the arithmetic.

    Whatever the numbers become, a turn long enough to matter must not be able
    to spend all of itself on fetching.
    """
    for budget in (90.0, 120.0, 148.0, 300.0):
        assert _tool_loop_budget(budget) < budget
