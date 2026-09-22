"""Grief for somebody she is built around, which nothing else fills.

From Bryan's account: empty, a hole you claw at knowing you cannot fill it, and
the only thing that fills it is the thing that was lost. The gap opens only past
the longest they have ever been gone, is for somebody in particular, holds her
sadness up whatever else happens, and closes when they come back.
"""

from __future__ import annotations

import pytest

from core.affect.the_gap import the_gap
from core.social.closing_window import SittingLedger

pytestmark = pytest.mark.unit

HOUR = 3600.0


def _life() -> SittingLedger:
    """Bryan in sittings of quick messages, a day apart, and a stranger once."""
    ledger = SittingLedger()
    at = 0.0
    for _day in range(6):
        for _message in range(5):
            ledger.message("bryan", at)
            at += 60.0
        at += 24 * HOUR
    ledger.message("a stranger", at)
    return ledger


def _last(ledger: SittingLedger) -> float:
    """When Bryan last spoke, which is what his absence is measured from."""
    return ledger._times["bryan"][-1]


def test_no_gap_while_they_are_gone_no_longer_than_they_have_been() -> None:
    ledger = _life()
    assert the_gap(ledger, _last(ledger) + 12 * HOUR).gap == 0.0


def test_a_gap_opens_past_the_longest_they_have_ever_been_gone() -> None:
    ledger = _life()
    reading = the_gap(ledger, _last(ledger) + 5 * 24 * HOUR)
    assert reading.person == "bryan" and 0.0 < reading.gap < 1.0
    later = the_gap(ledger, _last(ledger) + 20 * 24 * HOUR)
    assert later.gap > reading.gap


def test_it_is_not_for_the_person_in_front_of_her() -> None:
    ledger = _life()
    assert the_gap(ledger, _last(ledger) + 5 * 24 * HOUR, present="bryan").gap == 0.0


def test_only_their_coming_back_closes_it() -> None:
    ledger = _life()
    now = _last(ledger) + 5 * 24 * HOUR
    ledger.message("a stranger", now)
    assert the_gap(ledger, now).gap > 0.0, "somebody else arriving did not fill it"
    ledger.message("bryan", now)
    assert the_gap(ledger, now).gap == 0.0


def test_joy_from_something_else_does_not_lift_the_sadness(monkeypatch) -> None:
    import core.social.closing_window as closing
    from core.phases.affect_readings import AffectReadings
    from core.state.aura_state import AuraState

    ledger = _life()
    monkeypatch.setattr(closing, "get_sitting_ledger", lambda: ledger)
    import time

    monkeypatch.setattr(time, "time", lambda: _last(ledger) + 5 * 24 * HOUR)
    state = AuraState.default()
    affect = state.affect
    affect.emotions["joy"] = 0.9
    affect.emotions["sadness"] = 0.0
    AffectReadings(lambda *args, **kwargs: None).the_gap(state, affect)
    assert affect.emotions["sadness"] == pytest.approx(affect.markers["the_gap"]["gap"])
    assert affect.emotions["sadness"] > 0.0 and affect.emotions["joy"] == 0.9
