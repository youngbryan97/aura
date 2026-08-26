"""A deadline set before this action began is the one that ends it.

Measured live on 2026-08-26: fifty narrated moves in a game of 2048,
cancelled from outside and reported as "Operation took too long", because the
time between planning the step and running it was free and the pursuit's own
budget started afterwards.
"""

from __future__ import annotations

import time

from core.skills.screen_pursuit import _time_left


def test_without_a_callers_clock_the_budget_is_its_own():
    began = time.monotonic()
    assert 9.0 < _time_left(began, 10.0, 0.0) <= 10.0


def test_a_caller_that_started_earlier_ends_it_earlier():
    began = time.monotonic()
    assert _time_left(began, 600.0, began + 30.0) <= 30.0


def test_a_caller_that_allows_more_does_not_extend_the_budget():
    began = time.monotonic()
    assert _time_left(began, 10.0, began + 600.0) <= 10.0


def test_a_deadline_already_past_still_leaves_room_to_finish_a_cycle():
    began = time.monotonic()
    assert _time_left(began, 600.0, began - 100.0) == 1.0
