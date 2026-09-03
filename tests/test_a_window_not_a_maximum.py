"""Wanting something in a band, where more is worse past a point.

You have to hurt it, because a healthy one will not be caught, and you must
not hurt it too much, because a dead one cannot be caught at all.

The tests are a rollout: enough load to prove it works, not enough to fall
over.
"""

from __future__ import annotations

from core.cognition.a_window_not_a_maximum import (
    AWindow,
    how_close_to_it,
    which_act_lands_in_it,
)

ENOUGH_LOAD = AWindow(at_least=40.0, at_most=60.0)
MOVES = {"hold": 0.0, "a little more": 20.0, "double it": 45.0, "all of it": 95.0}


def test_more_stops_being_better_past_the_edge() -> None:
    assert how_close_to_it(50.0, ENOUGH_LOAD) == 1.0
    assert how_close_to_it(70.0, ENOUGH_LOAD) < 1.0
    assert how_close_to_it(90.0, ENOUGH_LOAD) == 0.0


def test_overshooting_and_falling_short_are_different_mistakes() -> None:
    """Falling short leaves the thing there to try again on."""
    assert ENOUGH_LOAD.short_of(20.0)
    assert not ENOUGH_LOAD.overshot(20.0)
    assert ENOUGH_LOAD.overshot(80.0)
    assert "still there" in ENOUGH_LOAD.describe(20.0)
    assert "gone rather than not got" in ENOUGH_LOAD.describe(80.0)


def test_it_picks_the_act_that_lands_in_the_window() -> None:
    ranked = which_act_lands_in_it(
        list(MOVES), now=25.0, what_it_moves=lambda one: MOVES[one], window=ENOUGH_LOAD
    )
    assert ranked[0][0] == "a little more", ranked


def test_the_biggest_act_is_not_the_best_act() -> None:
    ranked = which_act_lands_in_it(
        list(MOVES), now=25.0, what_it_moves=lambda one: MOVES[one], window=ENOUGH_LOAD
    )
    assert ranked[-1][0] == "all of it", ranked
    assert ranked[-1][2], "and it is flagged as going past, not merely as worse"


def test_going_past_is_kept_separate_so_it_can_be_refused_outright() -> None:
    ranked = which_act_lands_in_it(
        list(MOVES), now=55.0, what_it_moves=lambda one: MOVES[one], window=ENOUGH_LOAD
    )
    past = [one for one, _near, over in ranked if over]
    assert set(past) == {"a little more", "double it", "all of it"}
    assert ranked[0][0] == "hold", "the only one that stays in it"
