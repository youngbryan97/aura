"""Her capacity, usefulness and sense of control come from the ledger of what she did.

Three readings looked the ledger up as a runtime service named "agency_ledger",
which nothing registers. Her capacity sat at the 0.5 it has before she has tried
anything for all 2,400 turns of a seed-7 run in which she acted throughout, her
usefulness reached her standing as 0.0, and her sense of control reached acting
in decline as absent.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.authorship import SELF, Event, get_agency_ledger, reset_agency_ledger_for_test
from core.phases.affect_readings import AffectReadings, _her_agency

pytestmark = pytest.mark.unit


@pytest.fixture
def ledger():
    reset_agency_ledger_for_test()
    yield get_agency_ledger()
    reset_agency_ledger_for_test()


def test_the_readings_reach_the_ledger_she_acts_through(ledger) -> None:
    assert _her_agency() is ledger


def _state() -> SimpleNamespace:
    identity = SimpleNamespace(stability=0.5, read_by_other={}, capacity={}, standing={}, late_regard={})
    cognition = SimpleNamespace(current_partner="")
    return SimpleNamespace(identity=identity, cognition=cognition)


def test_her_capacity_moves_with_what_worked(ledger) -> None:
    readings = AffectReadings(lambda *args, **kwargs: None)
    state = _state()
    readings.standing(state)
    assert state.identity.capacity.get("capacity") == pytest.approx(0.5)
    for succeeded in (True, True, True, False):
        ledger.observe(Event(what="write_notes", actor=SELF, verified=succeeded))
    readings.standing(state)
    assert state.identity.capacity.get("measured") is True
    assert state.identity.capacity.get("capacity") != pytest.approx(0.5)
