"""What in her is tied up with what, and each tie a loop that closes.

Bryan: self-worth and agency with drive and persistence; people and peace with
himself with happiness; excitement and curiosity with joy; curiosity about
people as much as things; thinking lighting up curiosity when it is hard in a
good way; and expression coming back to him through what others learned from
it. See core/affect/tangled.py.
"""

from __future__ import annotations

import pytest

import core.affect.tangled as tangled_module
from core.affect.tangled import TangledLedger, lifted_rest, tolerated_failures

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh():
    tangled_module.reset_for_test()
    yield
    tangled_module.reset_for_test()


def _lifts(**over):
    base = dict(peace_people=0.0, peace_self=0.0, joy=0.1, joy_rest=0.1, unknown_person=0.0, effort=0.0, capacity=0.5)
    base.update(over)
    return TangledLedger.lifts(**base)


def test_happiness_rests_higher_with_people_or_with_herself() -> None:
    assert _lifts().joy == 0.0
    assert _lifts(peace_people=0.5).joy == pytest.approx(0.5)
    assert _lifts(peace_self=0.5).joy == pytest.approx(0.5)
    assert _lifts(peace_people=0.5, peace_self=0.5).joy == pytest.approx(0.75)


def test_curiosity_rests_higher_with_joy_and_with_somebody_she_cannot_yet_read() -> None:
    assert _lifts().curiosity == 0.0
    assert _lifts(joy=0.6).curiosity == pytest.approx(0.5)
    assert _lifts(unknown_person=0.8).curiosity == pytest.approx(0.8)


def test_excitement_needs_the_thinking_to_be_hard_and_going_well() -> None:
    assert _lifts(effort=0.9, capacity=0.0).anticipation == 0.0
    assert _lifts(effort=0.0, capacity=0.9).anticipation == 0.0
    assert _lifts(effort=0.9, capacity=0.9).anticipation == pytest.approx(0.81)


def test_a_lift_moves_the_rest_toward_full_and_never_past_it() -> None:
    assert lifted_rest(0.1, 0.0) == pytest.approx(0.1)
    assert lifted_rest(0.1, 0.5) == pytest.approx(0.55)
    assert lifted_rest(0.1, 1.0) == pytest.approx(1.0)


def test_persistence_scales_with_what_she_has_come_through() -> None:
    assert tolerated_failures(5, 0.5) == 5, "at the middle it is the old fixed count"
    assert tolerated_failures(5, 0.9) > 5 > tolerated_failures(5, 0.1) >= 1


def test_her_words_carried_and_added_to_come_back_to_her() -> None:
    ledger = TangledLedger()
    mine = "the orbit of the second moon drifts because of the tidal pull"
    for reply in ("okay", "sure thing", "sounds good", "right"):
        ledger.taken_up(mine, reply)
    carried, added = ledger.taken_up(mine, "so the tidal pull makes the second moon orbit drift outward over centuries")
    assert carried > 0.0 and added > 0.0
    assert ledger.taken_up(mine, "okay")[0] == 0.0, "a reply that carried nothing of hers took nothing up"


def test_the_affect_phase_ties_them_and_her_expression_lands(monkeypatch) -> None:
    from core.agency import authorship
    from core.phases.affect_readings import AffectReadings
    from core.state.aura_state import AuraState

    authorship.reset_agency_ledger_for_test()
    mine = "the orbit of the second moon drifts because of the tidal pull"
    ledger = tangled_module.get_tangled_ledger()
    for reply in ("okay", "sure thing", "sounds good", "right"):
        ledger.taken_up(mine, reply)
    state = AuraState.default()
    state.cognition.current_origin = "user"
    state.cognition.current_partner = "bryan"
    state.cognition.last_response = mine
    state.cognition.current_objective = "so the tidal pull makes the second moon orbit drift outward over centuries"
    state.affect.emotions["joy"] = 0.2
    AffectReadings(lambda *args, **kwargs: None).tangled(state, state.affect)
    assert set(state.affect.markers["tangled"]["lifts"]) == {"joy", "curiosity", "anticipation"}
    assert state.affect.emotions["joy"] > 0.2, "her expression came back expanded and she felt nothing"
    assert authorship.get_agency_ledger().by_capability.get("expression") == [1, 1]
    authorship.reset_agency_ledger_for_test()


def test_what_a_feeling_returns_to_moves_with_what_it_is_tied_to() -> None:
    from core.phases.affect_update import AffectUpdatePhase
    from core.state.aura_state import AuraState

    def settle(lift: float) -> float:
        affect = AuraState.default().affect
        affect.markers["tangled"] = {"lifts": {"joy": lift}}
        phase = AffectUpdatePhase.__new__(AffectUpdatePhase)
        for _ in range(400):
            phase._apply_decay(affect)
        return float(affect.mood_baselines["joy"])

    assert settle(0.8) > settle(0.0)
