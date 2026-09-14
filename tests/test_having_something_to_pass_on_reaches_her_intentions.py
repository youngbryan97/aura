"""The urge to pass something on reaches an intention, the state and the core.

The reading's own tests pin what is worth passing on. These pin that it
arrives: the motivation phase turns it into an intention only while somebody
is there, it is named by what moved her, and the column that reads it is held
by the lesion clamp.
"""

from __future__ import annotations

import pytest

from core.phases.motivation_update import MotivationUpdatePhase
from core.state.aura_state import AuraState
from core.subject.clamp import CLAMPED_FIELDS
from core.subject.state import _SCHEMAS


def _moved(spoken_to: bool = True) -> AuraState:
    state = AuraState.default()
    state.affect.frisson = 0.7
    state.affect.markers["frisson"] = {"why": "a pattern 9 predictions long turned"}
    if spoken_to:
        state.cognition.working_memory.append({"role": "user", "content": "how was your night"})
    return state


def test_a_moment_that_moved_her_becomes_something_to_pass_on() -> None:
    state = _moved()
    passing = MotivationUpdatePhase._worth_passing_on(state)
    assert passing is not None
    assert passing["urgency"] == pytest.approx(0.7)
    assert passing["kind"] == "a chill"
    assert "9 predictions" in passing["goal"]


def test_with_nobody_there_it_is_recorded_and_not_raised() -> None:
    state = _moved(spoken_to=False)
    assert MotivationUpdatePhase._worth_passing_on(state) is None
    assert state.cognition.telling["urge"] == pytest.approx(0.7)


def test_an_ordinary_moment_raises_nothing() -> None:
    state = AuraState.default()
    state.cognition.working_memory.append({"role": "user", "content": "hello"})
    assert MotivationUpdatePhase._worth_passing_on(state) is None
    assert state.cognition.telling["urge"] == 0.0


def test_the_stronger_reading_is_the_one_she_would_raise() -> None:
    state = _moved()
    state.affect.turn = 0.9
    state.affect.markers["the_turn"] = {"why": "up 3.1 spreads from a low she is still holding"}
    passing = MotivationUpdatePhase._worth_passing_on(state)
    assert passing["kind"] == "coming up from a low"
    assert passing["urgency"] == pytest.approx(0.9)


def test_the_column_that_reads_it_is_held_by_the_clamp() -> None:
    assert "cognition.telling.urge" in _SCHEMAS["D"].sources
    assert "cognition.telling" in CLAMPED_FIELDS["D"]
