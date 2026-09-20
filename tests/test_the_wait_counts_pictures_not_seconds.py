"""Whether the world was still moving is counted, not timed.

LIVE 2026-09-20: her moves drifted from 1.5 s to 11 s over one game. The
wait after acting grew whenever a look took longer than the shortest look
ever taken, and once a look's own cost began to vary — recognition getting a
tile on the second picture rather than the first — every look read as late
and the wait grew by that difference, every move, for ever.
"""
from __future__ import annotations

import math

import pytest

from core.skills import screen_pursuit_looking as looking


@pytest.fixture(autouse=True)
def _fresh_wait():
    looking._WAIT["seconds"] = 1.0
    looking._STILL_FLOOR["seconds"] = math.inf
    yield
    looking._WAIT["seconds"] = 0.0
    looking._STILL_FLOOR["seconds"] = math.inf


def test_a_world_already_at_rest_halves_the_wait():
    looking._it_was_ready(1.0, True, {"seconds_to_still": 0.25, "pictures_to_still": 2})
    assert looking._WAIT["seconds"] == 0.5


def test_a_slower_look_that_needed_no_more_pictures_does_not_lengthen_it():
    """The regression: the look cost more, the world did not move more."""
    looking._it_was_ready(1.0, True, {"seconds_to_still": 0.25, "pictures_to_still": 2})
    looking._it_was_ready(0.5, True, {"seconds_to_still": 0.54, "pictures_to_still": 2})
    assert looking._WAIT["seconds"] == 0.25


def test_a_world_still_moving_lengthens_it_by_what_it_was_short_by():
    looking._it_was_ready(1.0, True, {"seconds_to_still": 0.25, "pictures_to_still": 2})
    looking._it_was_ready(0.5, True, {"seconds_to_still": 0.85, "pictures_to_still": 5})
    assert looking._WAIT["seconds"] == pytest.approx(0.5 + 0.6)


def test_without_a_count_it_falls_back_to_the_clock():
    """An older reading carries no count; the previous rule still applies."""
    looking._it_was_ready(1.0, True, {"seconds_to_still": 0.25})
    assert looking._WAIT["seconds"] == 0.5
    looking._it_was_ready(0.5, True, {"seconds_to_still": 0.85})
    assert looking._WAIT["seconds"] == pytest.approx(0.5 + 0.6)
