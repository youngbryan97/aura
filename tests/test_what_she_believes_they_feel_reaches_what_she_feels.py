"""A borrowed feeling, wired: beliefs scored against statements, carried into affect, held by the clamp.

The measure is pinned in test_a_feeling_is_only_as_good_as_the_belief_it_rides_on.py.
These pin the wiring: a statement of how somebody feels scores the belief she
held before it, the affect phase lends joy or sadness by what she believes of
the person here, and the affect domain reads the column the clamp holds.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.social.borrowed_feeling as borrowed
from core.phases.affect_readings import AffectReadings
from core.social.other_agent_model import OtherAgentStateEstimator
from core.state.aura_state import AuraState


@pytest.fixture(autouse=True)
def fresh_ledger():
    borrowed.reset_for_test()
    yield
    borrowed.reset_for_test()


class _Signal:
    def __init__(self, value: float) -> None:
        self.value = value

    def decayed(self, _now: float) -> tuple[float, float]:
        return self.value, 0.5


def test_a_statement_scores_the_belief_held_before_it() -> None:
    before = SimpleNamespace(affect={"frustration": _Signal(0.3)})
    for _ in range(borrowed.MIN_REPORTS):
        OtherAgentStateEstimator._note_said(before, "frustration", 0.85, 10.0)
    assert borrowed.get_calibration_ledger().calibration() == pytest.approx(1.0 - 0.55)


def _estimate(satisfaction: float, frustration: float) -> SimpleNamespace:
    return SimpleNamespace(
        abstained=False,
        affect={"satisfaction": satisfaction, "frustration": frustration},
        affect_confidence={"satisfaction": 0.8, "frustration": 0.8},
    )


def _read(monkeypatch, estimate: SimpleNamespace) -> AuraState:
    import core.social.other_agent_model as other

    monkeypatch.setattr(other, "get_other_agent_model", lambda: SimpleNamespace(estimate=lambda _agent: estimate))
    ledger = borrowed.get_calibration_ledger()
    for _ in range(borrowed.MIN_REPORTS):
        ledger.said(believed=0.5, said=0.5)
    state = AuraState.default()
    state.cognition.current_partner = "sam"
    readings = AffectReadings.__new__(AffectReadings)
    readings.borrowed_feeling(state, state.affect)
    return state


def test_believing_they_are_pleased_lends_her_joy(monkeypatch) -> None:
    state = _read(monkeypatch, _estimate(satisfaction=0.95, frustration=0.10))
    assert state.affect.borrowed_feeling == pytest.approx(0.8 * 0.4)
    assert state.affect.emotions.get("joy", 0.0) >= state.affect.borrowed_feeling


def test_believing_they_are_troubled_lends_her_sadness(monkeypatch) -> None:
    state = _read(monkeypatch, _estimate(satisfaction=0.55, frustration=0.60))
    assert state.affect.borrowed_feeling == pytest.approx(-0.8 * 0.5)
    assert state.affect.emotions.get("sadness", 0.0) >= abs(state.affect.borrowed_feeling)


def test_the_affect_domain_reads_it_and_the_clamp_holds_it() -> None:
    from core.subject.clamp import CLAMPED_FIELDS
    from core.subject.state import feature_names

    assert "A.borrowed_feeling" in feature_names("A")
    assert "affect.borrowed_feeling" in CLAMPED_FIELDS["A"]
