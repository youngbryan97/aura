"""A slow look reported as a screen she could not see.

A reading and a language pass want the same machine. A read that takes a
second and a half on its own takes many while a resident model is generating,
and bounded by a constant that difference reads as a wedged capture.

LIVE 2026-09-07, playing the real game: "no reading inside 8.0s", and a run
that ended saying it could not see — on a screen it had been reading perfectly
a moment earlier, at about one and a half seconds a look.

A busy machine and a broken one are not the same thing and do not have the
same answer. So the bound is what looking has actually cost here, which widens
when the machine gets busy: exactly when a read is slow, and exactly when
calling it broken is wrong.
"""

from __future__ import annotations

from core.skills.screen_pursuit import (
    LONGER_THAN_USUAL,
    OBSERVE_TIMEOUT_S,
    _how_long_a_look_takes,
)


def test_with_nothing_measured_the_standing_bound_applies():
    """What every caller assumed before there was anything to measure."""
    assert _how_long_a_look_takes([]) == OBSERVE_TIMEOUT_S
    assert _how_long_a_look_takes([1.5]) == OBSERVE_TIMEOUT_S
    assert _how_long_a_look_takes([1.5, 1.6]) == OBSERVE_TIMEOUT_S


def test_a_quick_machine_does_not_loosen_the_bound():
    """Reading fast is not a reason to wait longer."""
    assert _how_long_a_look_takes([1.5, 1.6, 1.4, 1.5, 1.55]) == OBSERVE_TIMEOUT_S


def test_a_busy_machine_is_given_the_time_it_has_been_taking():
    """The whole defect: this is where it used to say she could not see."""
    busy = _how_long_a_look_takes([1.5, 6.0, 7.0, 6.5, 7.2])
    assert busy > OBSERVE_TIMEOUT_S
    assert busy == 6.5 * LONGER_THAN_USUAL


def test_one_slow_look_does_not_move_it():
    """A single stall is not the machine changing."""
    assert _how_long_a_look_takes([1.5, 1.6, 30.0, 1.4, 1.5]) == OBSERVE_TIMEOUT_S


def test_a_look_that_never_returns_is_still_refused():
    """Widening the bound must not remove it."""
    assert _how_long_a_look_takes([6.0, 6.5, 7.0]) < float("inf")


def test_readings_that_took_no_time_are_not_counted():
    """A cached reading costs nothing and says nothing about the machine."""
    assert _how_long_a_look_takes([0.0, 0.0, 0.0, 0.0]) == OBSERVE_TIMEOUT_S
