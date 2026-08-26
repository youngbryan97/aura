"""Effort chosen by what a thought could change, not by a counter.

A counter cannot tell a forced move from the one that decides the shape of the
next thirty, so it spends the same on both and is wrong about both. Where she
can see where each move leads and one is plainly better, words will not change
the answer and buying them is buying nothing.
"""

from __future__ import annotations

import pytest

from core.agency.worth_thinking_about import TOO_CLOSE_TO_CALL, WORTH_A_PASS, worth_a_pass

CLEAR = {"left": (3.0, "clearly best"), "up": (1.0, ""), "down": (0.9, ""), "right": (0.5, "")}
CLOSE = {"left": (2.00, ""), "up": (1.99, ""), "down": (1.0, ""), "right": (0.5, "")}


def test_a_clear_best_move_does_not_need_words():
    asked, because = worth_a_pass(CLEAR, stakes=0.3)
    assert not asked
    assert "clear" in because


def test_two_moves_too_close_to_call_are_worth_a_thought():
    asked, because = worth_a_pass(CLOSE, stakes=0.3)
    assert asked
    assert "too close to call" in because


def test_seeing_nothing_ahead_is_always_worth_a_thought():
    asked, because = worth_a_pass({}, stakes=0.1)
    assert asked
    assert "cannot see" in because


def test_something_at_stake_buys_a_pass_whatever_the_arithmetic_says():
    asked, because = worth_a_pass(CLEAR, stakes=WORTH_A_PASS)
    assert asked
    assert "riding on this" in because


def test_a_run_that_never_uses_words_stops_being_hers():
    asked, because = worth_a_pass(CLEAR, stakes=0.1, since_words=5, horizon=5)
    assert asked
    assert "without saying anything" in because


def test_an_unusual_moment_is_worth_one_by_itself():
    asked, because = worth_a_pass(CLEAR, stakes=0.1, unusual=True)
    assert asked
    assert "routine" in because


def test_one_way_to_go_is_not_a_decision():
    asked, because = worth_a_pass({"left": (1.0, "")}, stakes=0.3)
    assert not asked
    assert "only one way" in because


@pytest.mark.parametrize("gap", [TOO_CLOSE_TO_CALL * 0.5, TOO_CLOSE_TO_CALL * 0.99])
def test_a_gap_smaller_than_the_weakest_reason_is_noise(gap):
    asked, _because = worth_a_pass({"a": (1.0 + gap, ""), "b": (1.0, "")}, stakes=0.3)
    assert asked


def test_the_reason_is_always_given():
    """A decision about how to decide is a decision, and one nobody can
    account for is indistinguishable from a habit."""
    for ahead in (CLEAR, CLOSE, {}, {"only": (1.0, "")}):
        _asked, because = worth_a_pass(ahead, stakes=0.3)
        assert because
