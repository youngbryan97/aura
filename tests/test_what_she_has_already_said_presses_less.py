"""Catharsis reaches the intention, the state and the subject core.

The reading's own tests pin what counts as having said something. These pin
that it changes what an intention costs to leave unsaid, and that the columns
which read it are held by the lesion clamp.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.phases.motivation_update import MotivationUpdatePhase
from core.state.aura_state import AuraState
from core.subject.clamp import CLAMPED_FIELDS
from core.subject.state import _SCHEMAS

GOAL = "Working on the migration schedule"


def _asked_about_it(times: int) -> AuraState:
    state = AuraState.default()
    for _ in range(times):
        state.cognition.working_memory.append(
            {"role": "assistant", "content": "I am working on the migration schedule again"}
        )
    return state


def test_an_intention_she_has_never_raised_presses_with_its_whole_urgency() -> None:
    state = _asked_about_it(0)
    intention = MotivationUpdatePhase._drained(state, {"goal": GOAL, "urgency": 0.8})
    assert intention["urgency"] == pytest.approx(0.8)
    assert state.cognition.catharsis["times"] == 0


def test_saying_it_once_halves_what_is_left_of_it() -> None:
    state = _asked_about_it(1)
    intention = MotivationUpdatePhase._drained(state, {"goal": GOAL, "urgency": 0.8})
    assert intention["urgency"] == pytest.approx(0.4)
    assert intention["drained"] == 1
    assert state.cognition.catharsis["drain"] == pytest.approx(0.5)


def test_saying_it_three_times_leaves_a_quarter() -> None:
    state = _asked_about_it(3)
    intention = MotivationUpdatePhase._drained(state, {"goal": GOAL, "urgency": 0.8})
    assert intention["urgency"] == pytest.approx(0.2)


def test_the_pressure_never_reaches_zero() -> None:
    state = _asked_about_it(20)
    intention = MotivationUpdatePhase._drained(state, {"goal": GOAL, "urgency": 0.8})
    assert intention["urgency"] > 0.0


def test_an_intention_without_a_goal_is_left_alone() -> None:
    state = _asked_about_it(2)
    intention = MotivationUpdatePhase._drained(state, {"goal": "", "urgency": 0.6})
    assert intention["urgency"] == pytest.approx(0.6)


def test_it_survives_a_state_with_no_history_at_all() -> None:
    intention = MotivationUpdatePhase._drained(
        SimpleNamespace(cognition=SimpleNamespace()), {"goal": GOAL, "urgency": 0.5}
    )
    assert intention["urgency"] == pytest.approx(0.5)


def test_the_columns_that_read_it_are_held_by_the_clamp() -> None:
    sources = set(_SCHEMAS["D"].sources)
    assert "cognition.catharsis.times" in sources
    assert "cognition.catharsis.drain" in sources
    assert "cognition.catharsis" in CLAMPED_FIELDS["D"]
