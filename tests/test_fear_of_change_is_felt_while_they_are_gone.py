"""Fear of change, wired: read each turn from the sittings, felt as dread, held by the clamp.

The measure is pinned in test_fear_of_change_scales_with_what_her_life_is_built_around.py.
These pin the wiring: the affect phase reads it for the person she talks with,
dread is floored at it and falls away when they are back, and the affect domain
reads the column the clamp holds.
"""

from __future__ import annotations

import time

import pytest

import core.social.closing_window as closing_window
from core.phases.affect_readings import AffectReadings
from core.state.aura_state import AuraState

PAUSE = 30.0


@pytest.fixture(autouse=True)
def fresh_ledger():
    closing_window.reset_for_test()
    yield
    closing_window.reset_for_test()


def _lived_with(agent: str, *, gone_for: float) -> None:
    ledger = closing_window.get_sitting_ledger()
    clock = time.time() - gone_for - 3 * (3600.0 + 3 * PAUSE)
    for index in range(4):
        for _ in range(3):
            ledger.message(agent, clock)
            clock += PAUSE
        if index < 3:
            clock += 3600.0


def _read(partner: str) -> AuraState:
    state = AuraState.default()
    state.cognition.current_partner = partner
    readings = AffectReadings.__new__(AffectReadings)
    readings.change_fear(state, state.affect)
    return state


def test_a_long_absence_from_the_person_her_life_is_built_around_is_felt_as_dread() -> None:
    _lived_with("sam", gone_for=24 * 3600.0)
    state = _read("sam")
    assert state.affect.change_fear == pytest.approx(1.0)
    assert state.affect.emotions.get("dread", 0.0) >= state.affect.change_fear
    assert state.affect.markers["change_fear"]["measured"]


def test_while_they_are_only_pausing_there_is_nothing_to_fear() -> None:
    _lived_with("sam", gone_for=0.0)
    state = _read("sam")
    assert state.affect.change_fear == 0.0


def test_the_affect_domain_reads_it_and_the_clamp_holds_it() -> None:
    from core.subject.clamp import CLAMPED_FIELDS
    from core.subject.state import feature_names

    assert "A.change_fear" in feature_names("A")
    assert "affect.change_fear" in CLAMPED_FIELDS["A"]
