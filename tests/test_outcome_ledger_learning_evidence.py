"""The production ledger distinguishes missing observations from measured failures."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.cognition.outcome_ledger import CreditSource, OutcomeLedger, _measured_learning_evidence


def test_registered_evidence_invariant():
    assert _measured_learning_evidence() == ()


@pytest.fixture
def feeds(tmp_path, monkeypatch):
    learner, mattering, credit = Mock(), Mock(), Mock()
    monkeypatch.setattr("core.cognition.outcome_learner.get_outcome_learner", lambda: learner)
    monkeypatch.setattr("core.cognition.mattering.get_mattering_model", lambda: mattering)
    monkeypatch.setattr("core.consciousness.credit_assignment.get_credit_assignment_system", lambda: credit)
    ledger = OutcomeLedger(db_path=str(tmp_path / "outcomes.db"))
    return ledger, learner, mattering, credit


def test_expiry_never_teaches_failure_or_mattering(feeds):
    ledger, learner, mattering, credit = feeds
    ledger.open("unobserved", 0.8, sources=[CreditSource("tool", "compute")],
                horizon_s=10, now=1000)
    expired = ledger.sweep(now=1011)
    assert len(expired) == 1 and not expired[0].is_evidence
    learner.record_outcome.assert_not_called()
    mattering.note_mattered.assert_not_called()
    credit.assign_credit.assert_called_once_with("tool:compute", -1.0, domain="tool")
    assert ledger.measured_action_stats() == {}


@pytest.mark.parametrize("observed,success", [(0.0, False), (1.0, True)])
def test_measured_success_and_failure_reach_both_learning_consumers(feeds, observed, success):
    ledger, learner, mattering, _ = feeds
    receipt_id = ledger.open("checked", 0.8, now=1000)
    receipt = ledger.resolve(receipt_id, observed, now=1001)
    assert receipt.is_evidence
    call = learner.record_outcome.call_args.kwargs
    assert call["success"] is success
    assert call["context"]["observation"] == "measured"
    assert call["context"]["receipt_id"] == receipt_id
    mattering.note_mattered.assert_called_once()
    assert ledger.resolve(receipt_id, observed, now=1002) is None
    assert learner.record_outcome.call_count == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "invalid"])
def test_bad_measurement_does_not_consume_pending_receipt(feeds, value):
    ledger, learner, mattering, _ = feeds
    receipt_id = ledger.open("retry observation", 0.8, now=1000)
    with pytest.raises(ValueError):
        ledger.resolve(receipt_id, value, now=1001)
    assert ledger.pending()[0]["receipt_id"] == receipt_id
    learner.record_outcome.assert_not_called()
    mattering.note_mattered.assert_not_called()
    assert ledger.resolve(receipt_id, 1.0, now=1002).is_evidence


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_expectations_are_rejected_before_open(feeds, value):
    ledger, *_ = feeds
    with pytest.raises(ValueError):
        ledger.open("bad forecast", value, now=1000)
    assert ledger.pending() == []


def test_mutated_or_unresolved_receipts_do_not_claim_measured_evidence(feeds):
    ledger, *_ = feeds
    receipt_id = ledger.open("checked", 0.8, now=1000)
    receipt = ledger.resolve(receipt_id, 1.0, now=1001)
    for changes in ({"observed": float("nan")}, {"status": "pending"},
                    {"observation": "unobserved"}, {"observed": 2.0}):
        assert not replace(receipt, **changes).is_evidence
