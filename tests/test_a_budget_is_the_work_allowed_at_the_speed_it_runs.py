"""A watched goal asks for long enough to make the moves it is allowed.

Measured live on 2026-08-26: a cycle now reads the screen, grades the last
prediction and often thinks in words, and takes about fourteen seconds — so a
flat six hundred bought a sixth of the play it was written for, and a run
building a 128 into the corner was stopped at move 43 of a game that needs a
hundred and fifty.
"""

from __future__ import annotations

import pytest

from core.runtime import watched_goal
from core.runtime.watched_goal import (
    PURSUIT_CEILING_S,
    PURSUIT_CYCLES,
    PURSUIT_SECONDS,
    read_watched_goal,
    seconds_a_cycle,
    time_for,
)


@pytest.fixture(autouse=True)
def unmeasured():
    before = watched_goal._A_CYCLE["seconds"]
    watched_goal._A_CYCLE["seconds"] = 0.0
    yield
    watched_goal._A_CYCLE["seconds"] = before


def test_a_machine_that_has_never_been_measured_gets_the_declared_budget():
    assert time_for() == PURSUIT_SECONDS
    assert seconds_a_cycle() == 0.0


def test_a_measured_cycle_sizes_the_budget_to_the_work_allowed():
    watched_goal.a_cycle_took(14.0)
    assert time_for() == pytest.approx(PURSUIT_CYCLES * 14.0)


def test_a_fast_machine_never_gets_less_than_the_declared_budget():
    watched_goal.a_cycle_took(0.2)
    assert time_for() == PURSUIT_SECONDS


def test_nobody_is_asked_to_wait_past_the_ceiling():
    watched_goal.a_cycle_took(600.0)
    assert time_for() == PURSUIT_CEILING_S


def test_a_reading_of_nothing_is_not_a_reading():
    watched_goal.a_cycle_took(0.0)
    watched_goal.a_cycle_took(-3.0)
    assert seconds_a_cycle() == 0.0


def test_the_goal_a_person_asked_for_carries_the_sized_budget():
    watched_goal.a_cycle_took(14.0)
    watched = read_watched_goal("play 2048 until you get a 256 tile")
    assert watched is not None
    assert watched.max_seconds == pytest.approx(PURSUIT_CYCLES * 14.0)


def test_the_task_still_allows_more_than_the_pursuit_it_wraps():
    from core.skills.desktop_task import DesktopTaskSkill

    watched_goal.a_cycle_took(14.0)
    watched = read_watched_goal("play 2048 until you get a 256 tile")
    asked = DesktopTaskSkill.timeout_for({"objective": "play 2048 until you get a 256 tile"})
    assert asked > watched.max_seconds
