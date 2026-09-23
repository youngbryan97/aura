"""An executive-closure goal's urgency follows its need, not the turn that chose it.

A record kept the need pressure of the turn that chose it, and the goal list
keeps the five pressing hardest, so a goal chosen once at 0.99 held first place
for a whole run: D.goal_urgency read 0.9946 on every one of 24 offline turns on
22 September. These pin the refresh in `ExecutiveClosureEngine._sync_active_goals`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.consciousness.executive_closure import ExecutiveClosureEngine

pytestmark = pytest.mark.unit

TEXT = "Investigate the unresolved pattern in the new measurements"


def _state(goals: list) -> SimpleNamespace:
    return SimpleNamespace(cognition=SimpleNamespace(active_goals=goals))


def _sync(state, *, pressures, selected="", need="", pressure=0.0) -> None:
    ExecutiveClosureEngine()._sync_active_goals(
        state, selected, persist_selected=bool(selected), pressure=pressure, need=need, pressures=pressures
    )


def test_a_kept_goal_takes_its_needs_pressure_this_turn() -> None:
    state = _state([{"description": TEXT, "urgency": 0.99, "need": "curiosity", "source": "executive_closure"}])
    _sync(state, pressures={"curiosity": 0.41, "stability": 0.9})
    assert state.cognition.active_goals[0]["urgency"] == pytest.approx(0.41)


def test_this_turns_lifts_stay_on_top_of_the_new_floor() -> None:
    goal = {"description": TEXT, "urgency": 0.99, "need": "curiosity", "source": "executive_closure", "reminder_lift": 0.1}
    state = _state([goal])
    _sync(state, pressures={"curiosity": 0.4})
    assert state.cognition.active_goals[0]["urgency"] == pytest.approx(0.5)
    assert state.cognition.active_goals[0]["reminder_lift"] == 0.1


def test_goals_from_elsewhere_and_goals_without_a_need_are_left_alone() -> None:
    others = [
        {"description": TEXT + " again", "urgency": 0.7, "source": "goal_engine"},
        {"description": TEXT + " once more", "urgency": 0.6, "source": "executive_closure"},
    ]
    state = _state(others)
    _sync(state, pressures={"curiosity": 0.1})
    assert sorted(goal["urgency"] for goal in state.cognition.active_goals) == [0.6, 0.7]


def test_a_new_record_says_which_need_chose_it() -> None:
    state = _state([])
    _sync(state, pressures={"growth": 0.63}, selected=TEXT, need="growth", pressure=0.63)
    (record,) = state.cognition.active_goals
    assert record["need"] == "growth" and record["urgency"] == pytest.approx(0.63)
