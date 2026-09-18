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
def unmeasured(tmp_path, monkeypatch):
    """A machine that has never run one, in memory and on disk."""
    monkeypatch.setattr(watched_goal, "_MEASURED_AT", tmp_path / "cycle.json")
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


def test_a_goal_with_no_end_carries_the_sized_budget():
    watched_goal.a_cycle_took(14.0)
    watched = read_watched_goal("keep playing 2048 and work out how it moves")
    assert watched is not None
    assert not watched.success_when
    assert watched.max_seconds == pytest.approx(PURSUIT_CYCLES * 14.0)
    assert watched.max_cycles == PURSUIT_CYCLES


def test_a_goal_that_names_its_end_is_kept_at_until_the_end():
    """"Until a 2048 tile" is about a thousand moves; two hundred stopped every
    such run a fifth of the way there."""
    watched_goal.a_cycle_took(1.0)
    watched = read_watched_goal("play 2048 until you get a 2048 tile")
    assert watched is not None
    assert watched.success_when
    assert watched.max_seconds == PURSUIT_CEILING_S
    assert watched.max_cycles == watched_goal.UNTIL_IT_IS_MET_CYCLES
    assert watched.as_target()["max_cycles"] == watched_goal.UNTIL_IT_IS_MET_CYCLES


def test_the_task_still_allows_more_than_the_pursuit_it_wraps():
    from core.skills.desktop_task import DesktopTaskSkill

    watched_goal.a_cycle_took(14.0)
    watched = read_watched_goal("play 2048 until you get a 256 tile")
    asked = DesktopTaskSkill.timeout_for({"objective": "play 2048 until you get a 256 tile"})
    assert asked > watched.max_seconds


def test_what_a_cycle_costs_survives_a_restart(tmp_path, monkeypatch):
    """The first watched goal after a restart is usually the one somebody is watching."""
    monkeypatch.setattr(watched_goal, "_MEASURED_AT", tmp_path / "cycle.json")
    watched_goal.a_cycle_took(14.0)
    # A fresh process: nothing in memory, everything on disk.
    watched_goal._A_CYCLE["seconds"] = 0.0
    assert seconds_a_cycle() == pytest.approx(14.0)


def test_a_machine_with_nothing_written_down_is_simply_unmeasured(tmp_path, monkeypatch):
    monkeypatch.setattr(watched_goal, "_MEASURED_AT", tmp_path / "missing.json")
    watched_goal._A_CYCLE["seconds"] = 0.0
    assert seconds_a_cycle() == 0.0
    assert time_for() == PURSUIT_SECONDS


def test_nonsense_written_down_is_ignored(tmp_path, monkeypatch):
    kept = tmp_path / "cycle.json"
    kept.write_text("not json at all")
    monkeypatch.setattr(watched_goal, "_MEASURED_AT", kept)
    watched_goal._A_CYCLE["seconds"] = 0.0
    assert seconds_a_cycle() == 0.0


def test_thinking_follows_the_reading_not_the_waiting():
    """The time between her act and her next look is mostly the world moving.

    Counted as the cost of looking, it made her think for two seconds a move
    on a game that answers in a fifth of one (live, 2026-09-17).
    """
    from screen_pursuit_support import pursuit_source

    # Asked as an ordering rather than as a distance. This read the 700
    # characters after the reading time was taken, and went red when
    # comments were written between the two — the derivation untouched,
    # nineteen lines apart instead of seven. A proximity window is a
    # measurement of formatting.
    from source_contract import in_order

    in_order(
        pursuit_source(),
        "a_read = float(observation.get(\"seconds_reading\"",
        "max(0.3, a_read)",
        "budget_s=thinking_for",
    )
