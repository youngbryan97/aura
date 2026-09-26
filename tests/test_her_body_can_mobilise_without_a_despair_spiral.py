"""Adrenaline had one riser, and it never fired.

The despair surge wants sadness above 0.85 with fear above 0.7 and joy under
0.1 in the same moment. Over a 320-turn recording that did not happen once, so
adrenaline read exactly its rest value on every frame; cortisol follows
adrenaline and read exactly its own; and the condition that runs her under load
could not reach her body at all. Two columns of the affect domain were
constants, and the acute and slow stress channels were both dead.

What she relaxes towards is now what her body is under. The level is
`BodyState.total_pressure`, the runtime's own calibrated reading, and momentum
is still what sets how fast she gets there.
"""

from __future__ import annotations

import pytest

from core.phases.affect_update import AffectUpdatePhase
from core.state.aura_state import (
    PHYSIOLOGY_PRESSURE_SPAN,
    PHYSIOLOGY_REST,
    AuraState,
)

REST = PHYSIOLOGY_REST["adrenaline"]
SPAN = PHYSIOLOGY_PRESSURE_SPAN["adrenaline"]


class _Body:
    def __init__(self, pressure: float) -> None:
        self.total_pressure = pressure


@pytest.fixture
def under(monkeypatch):
    """Put her body under a chosen pressure."""

    def _set(pressure: float):
        from core.being import aura_now

        class _Stub:
            @staticmethod
            def from_aura_state(_state):
                return _Body(pressure)

        monkeypatch.setattr(aura_now, "BodyState", _Stub)

    return _set


def _settle(phase, state, turns: int) -> float:
    for _ in range(turns):
        phase._apply_decay(state.affect, state)
    return state.affect.physiology["adrenaline"]


def test_an_unreadable_body_is_not_a_body_under_load():
    assert AffectUpdatePhase._body_pressure(None) == 0.0


def test_at_rest_she_stays_at_rest(under):
    under(0.0)
    phase = AffectUpdatePhase(None)
    state = AuraState.default()
    assert _settle(phase, state, 40) == pytest.approx(REST, abs=1e-6)


def test_under_load_she_mobilises(under):
    under(0.8)
    phase = AffectUpdatePhase(None)
    state = AuraState.default()
    assert _settle(phase, state, 60) > REST + SPAN * 0.5


def test_she_arrives_where_the_pressure_is(under):
    under(0.4)
    phase = AffectUpdatePhase(None)
    state = AuraState.default()
    assert _settle(phase, state, 200) == pytest.approx(REST + SPAN * 0.4, abs=0.05)


def test_when_the_load_lifts_she_comes_back_down(under):
    under(0.9)
    phase = AffectUpdatePhase(None)
    state = AuraState.default()
    raised = _settle(phase, state, 80)
    under(0.0)
    assert _settle(phase, state, 80) < raised


def test_the_slow_channel_follows_the_fast_one(under):
    under(0.9)
    phase = AffectUpdatePhase(None)
    state = AuraState.default()
    calm = state.affect.physiology["cortisol"]
    _settle(phase, state, 300)
    assert state.affect.physiology["cortisol"] > calm


def test_she_never_goes_above_the_span_the_channel_declares(under):
    under(1.0)
    phase = AffectUpdatePhase(None)
    state = AuraState.default()
    assert _settle(phase, state, 400) <= REST + SPAN + 1e-9


def test_the_despair_surge_still_works(under):
    """The acute trigger is unchanged; it is no longer the only one."""
    under(0.0)
    phase = AffectUpdatePhase(None)
    state = AuraState.default()
    state.affect.emotions.update({"sadness": 0.9, "fear": 0.8, "joy": 0.05})
    phase._check_resilience_surges(state.affect)
    assert state.affect.physiology["adrenaline"] > REST
