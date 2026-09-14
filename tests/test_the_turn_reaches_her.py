"""The turn reaches affect, the emotion table, delivery and the subject core.

The ledger's tests pin when a turn is a turn. These pin that one changes
something: the affect phase writes it and emits it, it competes for the level
she speaks from, and the column that reads it is held by the lesion clamp.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.affect import the_turn
from core.expression.delivery import live_readings
from core.phases.affect_update import AffectUpdatePhase
from core.state.aura_state import AuraState
from core.state.percepts import PERCEPT_EMOTIONS
from core.subject.clamp import CLAMPED_FIELDS
from core.subject.state import _SCHEMAS


@pytest.fixture(autouse=True)
def _fresh_ledger():
    the_turn.reset_for_test()
    yield
    the_turn.reset_for_test()


def _a_bad_stretch() -> None:
    led = the_turn.get_turn_ledger()
    for value in (-0.8, -0.7, -0.9, -0.6, 0.0, 0.1, 0.0, 0.1):
        led.note(value)


def test_the_affect_phase_writes_a_turn_and_emits_it() -> None:
    _a_bad_stretch()
    state = AuraState.default()
    state.affect.valence = 0.8
    AffectUpdatePhase(SimpleNamespace())._read_turn(state, state.affect)

    assert state.affect.turn > 0.0
    assert state.affect.markers["the_turn"]["turned"] is True
    assert "the_turn" in str(state.world.recent_percepts)


def test_an_ordinary_good_moment_writes_nothing() -> None:
    led = the_turn.get_turn_ledger()
    for value in (0.40, 0.45, 0.38, 0.42, 0.41, 0.39, 0.43, 0.40):
        led.note(value)
    state = AuraState.default()
    state.affect.valence = 0.9
    AffectUpdatePhase(SimpleNamespace())._read_turn(state, state.affect)
    assert state.affect.turn == 0.0
    assert state.affect.markers["the_turn"]["turned"] is False


def test_the_reading_is_bounded_while_the_rise_is_not() -> None:
    _a_bad_stretch()
    state = AuraState.default()
    state.affect.valence = 1.0
    AffectUpdatePhase(SimpleNamespace())._read_turn(state, state.affect)
    assert 0.0 < state.affect.turn < 1.0
    assert state.affect.markers["the_turn"]["rise"] > state.affect.turn


def test_a_turn_competes_for_the_level_she_speaks_from() -> None:
    affect = AuraState.default().affect
    affect.turn = 0.7
    readings = live_readings(affect)
    assert readings["turn"] == pytest.approx(0.7)
    assert max(readings, key=readings.get) == "turn"


def test_the_emotions_a_turn_carries_are_ones_she_has() -> None:
    assert set(PERCEPT_EMOTIONS["the_turn"]) <= set(AuraState.default().affect.emotions)


def test_the_column_that_reads_it_is_held_by_the_clamp() -> None:
    assert "affect.turn" in _SCHEMAS["A"].sources
    assert "affect.turn" in CLAMPED_FIELDS["A"]
