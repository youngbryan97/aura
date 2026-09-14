"""Safety reaches affect, belonging and the subject core.

The reading's own tests pin what safety is. These pin that it arrives: the
affect phase writes it, belonging is floored at it rather than nudged, and the
column that reads it is held by the lesion clamp.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.phases.affect_update import AffectUpdatePhase
from core.state.aura_state import AuraState
from core.subject.clamp import CLAMPED_FIELDS
from core.subject.state import _SCHEMAS


def _evening(kind: str = "interaction") -> AuraState:
    state = AuraState.default()
    for _ in range(4):
        state.world.recent_percepts.append({"type": kind, "content": "", "intensity": 0.4})
    for _ in range(3):
        state.cognition.working_memory.append({"role": "user", "content": "still here"})
    return state


def test_an_untroubled_evening_with_somebody_in_it_reaches_affect() -> None:
    state = _evening()
    AffectUpdatePhase(SimpleNamespace()).readings.safety(state, state.affect)
    assert state.affect.safety == pytest.approx(1.0)
    assert state.affect.markers["safety"]["threat"] == 0.0


def test_belonging_is_floored_at_it_rather_than_nudged() -> None:
    state = _evening()
    state.affect.emotions["belonging"] = 0.1
    AffectUpdatePhase(SimpleNamespace()).readings.safety(state, state.affect)
    assert state.affect.emotions["belonging"] == pytest.approx(1.0)


def test_a_stronger_feeling_is_not_pulled_down_to_it() -> None:
    state = _evening()
    state.affect.emotions["belonging"] = 0.9
    state.cognition.working_memory.append({"role": "assistant", "content": "mine"})
    AffectUpdatePhase(SimpleNamespace()).readings.safety(state, state.affect)
    assert state.affect.safety < 0.9
    assert state.affect.emotions["belonging"] == pytest.approx(0.9)


def test_a_stretch_of_threats_is_not_safety() -> None:
    state = _evening("threat_detected")
    state.affect.emotions["belonging"] = 0.2
    AffectUpdatePhase(SimpleNamespace()).readings.safety(state, state.affect)
    assert state.affect.safety == 0.0
    assert state.affect.emotions["belonging"] == pytest.approx(0.2)


def test_the_column_that_reads_it_is_held_by_the_clamp() -> None:
    assert "affect.safety" in _SCHEMAS["A"].sources
    assert "affect.safety" in CLAMPED_FIELDS["A"]
