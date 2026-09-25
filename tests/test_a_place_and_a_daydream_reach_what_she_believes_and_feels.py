"""Capacity and the warmer place in mind, wired.

The measures are pinned in test_what_she_came_through_says_what_she_can_do.py
and test_a_warmer_place_in_mind_helps_bear_a_cold_present.py. These pin the
wiring: her belief in a capability she has not tried takes the rate she has
shown where things were hard, the reading reaches the self domain, recall's
stored feeling reaches the daydream ledger once, a helpful warmer recollection
floors a low, and efficacy before she has acted is no reading at all.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

import core.affect.elsewhere as elsewhere
from core.agency.authorship import AgencyLedger
from core.phases.affect_readings import AffectReadings
from core.state.aura_state import AuraState


@pytest.fixture(autouse=True)
def fresh_ledger():
    elsewhere.reset_for_test()
    yield
    elsewhere.reset_for_test()


def test_an_untried_capability_takes_what_she_showed_where_it_was_hard() -> None:
    ledger = AgencyLedger()
    ledger.by_capability = {"proofs": [10, 9], "deploys": [10, 8]}
    assert ledger.confidence("poems") > 0.5
    struggling = AgencyLedger()
    struggling.by_capability = {"proofs": [10, 1], "deploys": [10, 2]}
    assert struggling.confidence("poems") < 0.5


def test_with_nothing_else_done_it_is_laplace_smoothing() -> None:
    ledger = AgencyLedger()
    ledger.by_capability = {"proofs": [3, 2]}
    assert ledger.confidence("proofs") == pytest.approx(3 / 5)
    assert AgencyLedger().confidence("anything") == pytest.approx(0.5)


def test_the_capacity_reading_reaches_the_self_domain() -> None:
    from core.subject.clamp import CLAMPED_FIELDS
    from core.subject.state import feature_names

    assert "S.capacity" in feature_names("S")
    assert "identity.capacity" in CLAMPED_FIELDS["S"]


def test_a_recollection_is_read_once() -> None:
    ledger = elsewhere.get_elsewhere_ledger()
    now = time.time()
    percepts = [
        {"type": "memory_replay", "timestamp": now - 2, "feeling": 0.1},
        {"type": "memory_replay", "timestamp": now - 1, "feeling": 0.7},
        {"type": "percept", "timestamp": now, "feeling": 0.9},
    ]
    assert ledger.recalled_since(percepts) == pytest.approx(0.7)
    assert ledger.recalled_since(percepts) is None


def _live_through_lows(readings: AffectReadings, state: AuraState) -> None:
    clock = time.time() - 10_000

    def turn(valence: float, feeling: float | None = None) -> None:
        nonlocal clock
        clock += 1.0
        state.affect.valence = valence
        if feeling is not None:
            state.world.recent_percepts.append({"type": "memory_replay", "timestamp": clock, "feeling": feeling})
        readings.elsewhere(state, state.affect)

    for value in [0.0, 0.6] * 12:
        turn(value)
    for _ in range(elsewhere.MIN_SAMPLES):
        turn(-0.4, 0.6)
        turn(0.3)
        turn(0.6)
        turn(-0.4)
        turn(0.0)
        turn(0.6)
    turn(-0.4, 0.6)


def test_a_warmer_recollection_that_has_helped_floors_a_low() -> None:
    readings = AffectReadings.__new__(AffectReadings)
    state = AuraState.default()
    _live_through_lows(readings, state)
    assert state.affect.markers["elsewhere"]["measured"]
    assert state.affect.valence == pytest.approx(-0.4 + 0.3 * 1.0)
    assert state.affect.elsewhere_lift == pytest.approx(0.3)


def test_the_lift_reaches_the_affect_domain_and_the_clamp() -> None:
    from core.subject.clamp import CLAMPED_FIELDS
    from core.subject.state import feature_names

    assert "A.elsewhere_lift" in feature_names("A")
    assert "affect.elsewhere_lift" in CLAMPED_FIELDS["A"]


def test_efficacy_before_she_has_acted_is_no_reading(monkeypatch) -> None:
    import core.phases.affect_readings as readings_module

    monkeypatch.setattr(
        readings_module,
        "_her_agency",
        lambda: SimpleNamespace(snapshot=lambda: {"efficacy": 0.0, "acted": 0}, by_capability={}),
    )
    readings = AffectReadings.__new__(AffectReadings)
    state = AuraState.default()
    for value in [0.0, 0.1] * 20 + [0.1, 0.0, -0.1, -0.2, -0.15, -0.3, -0.4, -0.35, -0.5, -0.6]:
        state.affect.valence = value
        readings.acting_in_decline(state, state.affect)
    assert state.affect.markers["acting_in_decline"]["control"] is None
