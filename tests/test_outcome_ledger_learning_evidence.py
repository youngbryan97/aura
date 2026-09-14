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


def test_collapsed_requests_are_one_observation_before_and_after_restart(feeds):
    ledger, *_ = feeds
    context = {"state": "one-context"}
    receipt_id = ledger.open("retry", 0.5, context=context, now=1000)
    for _ in range(100):
        assert ledger.open("retry", 0.5, context=context, now=1000) == receipt_id
    receipt = ledger.resolve(receipt_id, 1.0, now=1001)
    assert receipt.repeat_count == 101
    second = ledger.open("retry", 0.5, context=context, now=1002)
    ledger.resolve(second, 0.0, now=1003)
    restarted = OutcomeLedger(db_path=ledger._db_path)
    for source in (ledger, restarted):
        assert source.measured_action_stats()["retry"] == {"n": 2.0, "mean": 0.5, "m2": 0.5}
        assert source.measured_action_stats(by_state=True)["one-context|retry"] == {
            "n": 2.0, "mean": 0.5, "m2": 0.5,
        }


@pytest.mark.parametrize("changes", [
    {"status": "pending"}, {"observation": "unobserved"},
    {"observed": float("inf")}, {"observed": -0.1}, {"observed": 1.1}, {"observed": "invalid"},
])
def test_persisted_statistics_use_the_same_evidence_contract(feeds, changes):
    ledger, *_ = feeds
    receipt_id = ledger.open("invalid", 0.5, context={"state": "context"}, now=1000)
    receipt = ledger.resolve(receipt_id, 1.0, now=1001)
    ledger._persist(replace(receipt, **changes))
    assert ledger.measured_action_stats() == {}
    assert ledger.measured_action_stats(by_state=True) == {}


def test_retries_cannot_bias_the_shared_action_value_model(feeds):
    from core.reasoning.action_value import ActionValueModel

    ledger, *_ = feeds
    for label in ("once", "retried"):
        receipt_id = ledger.open(label, 0.5, now=1000)
        if label == "retried":
            for _ in range(100):
                assert ledger.open(label, 0.5, now=1000) == receipt_id
        ledger.resolve(receipt_id, 1.0, now=1001)
        receipt_id = ledger.open(label, 0.5, now=1002)
        ledger.resolve(receipt_id, 0.0, now=1003)
    model = ActionValueModel()
    model.refresh(ledger)
    once, retried = model.value_for("once"), model.value_for("retried")
    assert once.value == retried.value
    assert once.observations == retried.observations == 2
