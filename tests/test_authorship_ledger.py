"""Whether "I did this" has a consequence, or is only a way of speaking.

The distinction is worth nothing unless something downstream moves. So these
tests are all matched pairs: the same outcome, the same verification, the same
capability, arriving with a different cause. Anything that comes out the same
in both arms is not carrying authorship.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.authorship import (
    SELF,
    AgencyLedger,
    Event,
    get_agency_ledger,
    reset_agency_ledger_for_test,
)


@pytest.fixture(autouse=True)
def _fresh():
    reset_agency_ledger_for_test()
    yield
    reset_agency_ledger_for_test()


def _model():
    return SimpleNamespace(beliefs={}, version=0)


def test_the_same_outcome_moves_the_self_model_only_when_she_caused_it():
    mine, theirs = AgencyLedger(), AgencyLedger()
    my_model, their_model = _model(), _model()
    event = {"what": "write_file", "verified": True}
    mine.observe(Event(actor=SELF, **event), self_model=my_model)
    theirs.observe(Event(actor="external", **event), self_model=their_model)

    assert my_model.beliefs != their_model.beliefs
    assert my_model.version > their_model.version
    assert mine.snapshot() != theirs.snapshot()


def test_watching_something_succeed_is_not_evidence_about_her():
    ledger = AgencyLedger()
    for _ in range(20):
        ledger.observe(Event("write_file", "external", True))
    assert ledger.confidence("write_file") == pytest.approx(0.5)
    assert ledger.efficacy == 0.0
    assert ledger.authored_share == 0.0


def test_her_own_failures_lower_the_belief_and_watching_does_not_raise_it():
    ledger = AgencyLedger()
    for _ in range(10):
        ledger.observe(Event("write_file", SELF, False))
    low = ledger.confidence("write_file")
    for _ in range(10):
        ledger.observe(Event("write_file", "external", True))
    assert ledger.confidence("write_file") == pytest.approx(low)
    assert low < 0.2


def test_an_unattributed_outcome_is_not_hers():
    """Defaulting the other way is how a self-model inflates."""
    ledger = AgencyLedger()
    ledger.observe(Event("write_file", "unknown", True))
    assert ledger.acted == 0
    assert ledger.observed == 1


def test_the_ledger_survives_a_self_model_that_cannot_take_the_write():
    ledger = AgencyLedger()
    verdict = ledger.observe(Event("write_file", SELF, True), self_model=object())
    assert verdict.mine
    assert not verdict.self_model_updated
    assert ledger.acted == 1


def test_the_snapshot_is_readable_as_state():
    ledger = get_agency_ledger()
    ledger.observe(Event("a", SELF, True))
    ledger.observe(Event("b", "external", True))
    snapshot = ledger.snapshot()
    assert snapshot["acted"] == 1
    assert snapshot["observed"] == 1
    assert snapshot["authored_share"] == pytest.approx(0.5)


def test_the_intention_loop_reports_its_own_actions_to_the_ledger(tmp_path):
    """The live wiring: an observed intention is hers, and the ledger hears it."""
    from core.agency.intention_loop import IntentionLoop

    loop = IntentionLoop(db_path=str(tmp_path / "intentions.db"))
    identifier = loop.intend(
        intention="write the notes file",
        drive="curiosity",
        expected_outcome="notes.txt exists",
    )
    loop.record_action(
        identifier,
        tool_name="write_file",
        args={},
        result="ok",
        success=True,
        duration_ms=1.0,
    )
    loop.observe(identifier, observation="notes.txt exists", actual_outcome="notes.txt exists")

    ledger = get_agency_ledger()
    assert ledger.acted == 1
    assert ledger.by_capability.get("write_file", [0, 0])[0] == 1
    loop.close()
