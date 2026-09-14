"""Frisson reaches affect, the emotion table, delivery and the subject core.

The ledger's own tests pin when a chill fires. These pin that a chill that
fires changes something: the affect phase writes it once, emits it as a
percept the emotion table carries, delivery counts it among the live feelings,
and the lesion clamp holds the column that reads it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.affect import frisson
from core.affect.confirmation import MIN_HISTORY, ExpectationLedger
from core.expression.delivery import live_readings
from core.phases.affect_update import AffectUpdatePhase
from core.state.aura_state import AuraState
from core.state.percepts import PERCEPT_EMOTIONS
from core.subject.clamp import CLAMPED_FIELDS
from core.subject.state import _SCHEMAS


@pytest.fixture(autouse=True)
def _fresh_ledger():
    frisson.reset_for_test()
    yield
    frisson.reset_for_test()


def _turn_a_pattern() -> None:
    expectations = ExpectationLedger()
    expectations.load([0.30, 0.35, 0.28, 0.40, 0.33, 0.31, 0.36, 0.29, 0.34, 0.32])
    chills = frisson.get_frisson_ledger()
    for error in [0.05] * MIN_HISTORY + [0.95]:
        chills.note(expectations.score_and_note(error))


def test_the_affect_phase_feels_a_chill_once() -> None:
    _turn_a_pattern()
    phase = AffectUpdatePhase(SimpleNamespace())
    state = AuraState.default()

    phase._read_frisson(state, state.affect)
    assert state.affect.frisson > 0.0
    assert state.affect.markers["frisson"]["fired"] is True
    assert "frisson" in str(state.world.recent_percepts)

    phase._read_frisson(state, state.affect)
    assert state.affect.frisson == 0.0
    assert state.affect.markers["frisson"]["fired"] is False


def test_no_pattern_no_chill() -> None:
    phase = AffectUpdatePhase(SimpleNamespace())
    state = AuraState.default()
    phase._read_frisson(state, state.affect)
    assert state.affect.frisson == 0.0


def test_the_emotions_a_chill_carries_are_ones_she_has() -> None:
    carried = set(AuraState.default().affect.emotions)
    assert set(PERCEPT_EMOTIONS["frisson"]) <= carried


def test_a_chill_competes_for_the_level_she_speaks_from() -> None:
    affect = AuraState.default().affect
    affect.frisson = 0.8
    readings = live_readings(affect)
    assert readings["frisson"] == pytest.approx(0.8)
    assert max(readings, key=readings.get) == "frisson"


def test_the_column_that_reads_frisson_is_held_by_the_clamp() -> None:
    assert "affect.frisson" in _SCHEMAS["A"].sources
    assert "affect.frisson" in CLAMPED_FIELDS["A"]
