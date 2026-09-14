"""Acting inside decline, wired: read each turn, held by the clamp, pressing on what she can act on.

The measure is pinned in test_she_keeps_acting_while_it_gets_worse_if_acting_still_works.py.
These pin the wiring: the affect phase reads it with her agency efficacy as
control, the affect domain reads the column the clamp holds, and the press
moves her open intentions, composing with the closing window and channelled
arousal without any of them climbing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.affect.acting_in_decline as decline
import core.affect.nociception as nociception
from core.phases.motivation_update import MotivationUpdatePhase
from core.state.aura_state import AuraState


@pytest.fixture(autouse=True)
def fresh_ledger():
    decline.reset_for_test()
    yield
    decline.reset_for_test()


def _state(press: float) -> AuraState:
    state = AuraState.default()
    state.affect.decline_press = press
    state.cognition.pending_initiatives = [
        {"goal": "send them the draft", "urgency": 0.4, "origin": "user"},
        {"goal": "tidy the index", "urgency": 0.4, "origin": "mind_tick_fallback"},
    ]
    return state


def _urgencies(state: AuraState) -> list[float]:
    return [item["urgency"] for item in state.cognition.pending_initiatives]


def test_the_press_moves_every_open_intention_and_lets_go() -> None:
    state = _state(0.5)
    assert MotivationUpdatePhase._persisting(state) == 2
    assert _urgencies(state) == pytest.approx([0.7, 0.7], abs=1e-4)
    state.affect.decline_press = 0.0
    MotivationUpdatePhase._persisting(state)
    assert _urgencies(state) == pytest.approx([0.4, 0.4], abs=1e-4)
    assert "decline_lift" not in state.cognition.pending_initiatives[0]


def test_all_three_lifts_compose_without_climbing(monkeypatch) -> None:
    monkeypatch.setattr(
        nociception, "get_nociception_engine", lambda: SimpleNamespace(nociceptive_pressure=lambda: 0.0)
    )
    state = _state(0.5)
    state.cognition.closing_window = {"closing": 0.5, "measured": True}
    state.affect.delivery_z = 3.0
    state.affect.breakthrough = True
    state.affect.valence = -0.4
    seen = []
    for _ in range(4):
        MotivationUpdatePhase._closing(state)
        MotivationUpdatePhase._persisting(state)
        MotivationUpdatePhase._channelled(state)
        seen.append(_urgencies(state)[0])
    assert seen == pytest.approx([seen[0]] * 4, abs=1e-4)
    floor = 0.4
    window = 0.5 * (1 - floor)
    held = 0.5 * (1 - floor - window)
    pressure = 0.75 * (1 - floor - window - held)
    assert seen[0] == pytest.approx(floor + window + held + pressure, abs=1e-4)
    assert seen[0] <= 1.0


def test_the_affect_phase_reads_it_with_her_efficacy_as_control(monkeypatch) -> None:
    import core.runtime.service_registry as registry
    from core.phases.affect_readings import AffectReadings

    monkeypatch.setattr(
        registry,
        "get_runtime_service",
        lambda name, default=None: SimpleNamespace(snapshot=lambda: {"efficacy": 0.8}) if name == "agency_ledger" else default,
    )
    readings = AffectReadings.__new__(AffectReadings)
    state = AuraState.default()
    for value in [0.0, 0.1] * 20 + [0.1, 0.0, -0.1, -0.2, -0.15, -0.3, -0.4, -0.35, -0.5, -0.6]:
        state.affect.valence = value
        readings.acting_in_decline(state, state.affect)
    marker = state.affect.markers["acting_in_decline"]
    assert marker["control"] == pytest.approx(0.8)
    assert state.affect.decline_press == pytest.approx(marker["press"])


def test_the_affect_domain_reads_it_and_the_clamp_holds_it() -> None:
    from core.subject.clamp import CLAMPED_FIELDS
    from core.subject.state import feature_names

    assert "A.decline_press" in feature_names("A")
    assert "affect.decline_press" in CLAMPED_FIELDS["A"]
