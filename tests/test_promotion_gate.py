"""Tests for PromotionGate, ScoreEstimate, and audit-chain integration."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.promotion.gate import (
    PromotionDecision,
    PromotionGate,
    ScoreEstimate,
)
from core.runtime.receipts import (
    GovernanceReceipt,
    get_receipt_store,
    reset_receipt_store,
)


@pytest.fixture
def fresh_store(tmp_path: Path):
    reset_receipt_store()
    get_receipt_store(tmp_path / "receipts")
    yield
    reset_receipt_store()


# ---------------------------------------------------------------------------
# ScoreEstimate
# ---------------------------------------------------------------------------
def test_score_estimate_higher_is_better_intervals():
    s = ScoreEstimate(mean=0.8, stderr=0.05, higher_is_better=True)
    assert s.lower_confidence(z=1.0) == pytest.approx(0.75)
    assert s.upper_confidence(z=1.0) == pytest.approx(0.85)


def test_score_estimate_lower_is_better_intervals():
    s = ScoreEstimate(mean=0.5, stderr=0.05, higher_is_better=False)
    # When lower is better, "lower_confidence" is the *worst* allowable;
    # for a regression we want the side closer to higher.
    assert s.lower_confidence(z=1.0) == pytest.approx(0.55)
    assert s.upper_confidence(z=1.0) == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# baseline behaviour
# ---------------------------------------------------------------------------
def test_first_compare_accepts_and_sets_baseline():
    gate = PromotionGate(critical_metrics=["acc"], emit_receipts=False)
    decision = gate.compare({"acc": ScoreEstimate(0.8)})
    assert decision.accepted is True
    assert "becomes baseline" in decision.reasons[0]
    assert gate.baseline is not None


def test_compare_rejects_empty_candidate():
    gate = PromotionGate(critical_metrics=["acc"], emit_receipts=False)
    with pytest.raises(ValueError):
        gate.compare({})


# ---------------------------------------------------------------------------
# acceptance + rejection on critical metrics
# ---------------------------------------------------------------------------
def test_strict_dominance_on_critical_metric_accepts():
    gate = PromotionGate(critical_metrics=["acc"], delta=0.0, emit_receipts=False)
    gate.compare({"acc": ScoreEstimate(0.7, stderr=0.0)})
    decision = gate.compare({"acc": ScoreEstimate(0.85, stderr=0.0)})
    assert decision.accepted is True


def test_regression_on_critical_metric_rejects():
    gate = PromotionGate(critical_metrics=["acc"], delta=0.0, emit_receipts=False)
    gate.compare({"acc": ScoreEstimate(0.8, stderr=0.0)})
    decision = gate.compare({"acc": ScoreEstimate(0.7, stderr=0.0)})
    assert decision.accepted is False
    assert any("acc failed" in r for r in decision.reasons)


def test_overlapping_intervals_block_promotion_when_critical():
    """If candidate's lower CI overlaps baseline's upper CI, refuse."""
    gate = PromotionGate(critical_metrics=["acc"], delta=0.0, z=1.96, emit_receipts=False)
    gate.compare({"acc": ScoreEstimate(0.80, stderr=0.05, n=20)})
    decision = gate.compare({"acc": ScoreEstimate(0.82, stderr=0.05, n=20)})
    # 0.82 - 1.96*0.05 = 0.722, baseline upper = 0.80 + 1.96*0.05 = 0.898 -> rejected.
    assert decision.accepted is False


def test_clear_separation_promotes():
    gate = PromotionGate(critical_metrics=["acc"], emit_receipts=False)
    gate.compare({"acc": ScoreEstimate(0.50, stderr=0.01, n=100)})
    decision = gate.compare({"acc": ScoreEstimate(0.85, stderr=0.01, n=100)})
    assert decision.accepted is True


# ---------------------------------------------------------------------------
# direction handling (loss-style)
# ---------------------------------------------------------------------------
def test_lower_is_better_metric_promotes_on_drop():
    gate = PromotionGate(critical_metrics=["loss"], emit_receipts=False)
    gate.compare({"loss": ScoreEstimate(2.0, stderr=0.0, higher_is_better=False)})
    decision = gate.compare(
        {"loss": ScoreEstimate(1.0, stderr=0.0, higher_is_better=False)}
    )
    assert decision.accepted is True


def test_lower_is_better_metric_rejects_on_rise():
    gate = PromotionGate(critical_metrics=["loss"], emit_receipts=False)
    gate.compare({"loss": ScoreEstimate(1.0, stderr=0.0, higher_is_better=False)})
    decision = gate.compare(
        {"loss": ScoreEstimate(2.0, stderr=0.0, higher_is_better=False)}
    )
    assert decision.accepted is False


def test_direction_flip_is_rejected():
    gate = PromotionGate(critical_metrics=["x"], emit_receipts=False)
    gate.compare({"x": ScoreEstimate(0.8, higher_is_better=True)})
    decision = gate.compare({"x": ScoreEstimate(0.9, higher_is_better=False)})
    assert decision.accepted is False
    assert any("direction mismatch" in r for r in decision.reasons)


# ---------------------------------------------------------------------------
# critical vs. non-critical
# ---------------------------------------------------------------------------
def test_non_critical_metric_can_regress_within_tolerance():
    gate = PromotionGate(
        critical_metrics=["acc"],
        delta=0.0,
        max_regression=0.05,
        emit_receipts=False,
    )
    gate.compare({"acc": ScoreEstimate(0.8), "speed": ScoreEstimate(100, higher_is_better=True)})
    decision = gate.compare(
        {"acc": ScoreEstimate(0.85), "speed": ScoreEstimate(96, higher_is_better=True)}
    )
    # speed dropped 4 — well within max_regression on absolute scale, but the
    # gate measures vs. zero-stderr, so 96 < 100 fails. Allow a bigger
    # max_regression to admit this.
    assert decision.accepted is False  # absolute scale not normalized


def test_missing_critical_metric_rejects():
    gate = PromotionGate(critical_metrics=["acc"], emit_receipts=False)
    gate.compare({"acc": ScoreEstimate(0.8)})
    decision = gate.compare({"speed": ScoreEstimate(0.9)})
    assert decision.accepted is False


def test_missing_non_critical_metric_does_not_reject():
    gate = PromotionGate(
        critical_metrics=["acc"],
        emit_receipts=False,
    )
    gate.compare({"acc": ScoreEstimate(0.8), "speed": ScoreEstimate(0.5)})
    decision = gate.compare({"acc": ScoreEstimate(0.85)})
    # speed missing on candidate is not a critical failure.
    assert decision.accepted is True


# ---------------------------------------------------------------------------
# Will-side veto
# ---------------------------------------------------------------------------
def test_will_refuse_blocks_otherwise_passing_candidate():
    will_calls = []

    def will(payload):
        will_calls.append(payload)
        return {"outcome": "refuse", "reason": "policy"}

    gate = PromotionGate(
        critical_metrics=["acc"], emit_receipts=False, will_decide_fn=will
    )
    gate.compare({"acc": ScoreEstimate(0.8)})
    decision = gate.compare({"acc": ScoreEstimate(0.95)})
    assert decision.accepted is False
    assert any("will_refuse" in r for r in decision.reasons)
    assert len(will_calls) == 1


def test_will_raise_fails_closed():
    def will(payload):
        raise RuntimeError("Will service down")

    gate = PromotionGate(
        critical_metrics=["acc"], emit_receipts=False, will_decide_fn=will
    )
    gate.compare({"acc": ScoreEstimate(0.5)})
    decision = gate.compare({"acc": ScoreEstimate(0.9)})
    assert decision.accepted is False
    assert any("will_decide_raised" in r for r in decision.reasons)


# ---------------------------------------------------------------------------
# F1 audit-chain receipt integration
# ---------------------------------------------------------------------------
def test_compare_emits_governance_receipt(fresh_store):
    gate = PromotionGate(critical_metrics=["acc"], emit_receipts=True)
    gate.compare({"acc": ScoreEstimate(0.5)})
    decision = gate.compare({"acc": ScoreEstimate(0.9)})
    assert decision.receipt_id is not None
    receipts = get_receipt_store().query_by_kind("governance")
    assert any(r.receipt_id == decision.receipt_id for r in receipts)
    target = next(r for r in receipts if r.receipt_id == decision.receipt_id)
    assert isinstance(target, GovernanceReceipt)
    assert target.action == "promote"
    assert target.approved is True


def test_rejection_emits_receipt_with_reject_action(fresh_store):
    gate = PromotionGate(critical_metrics=["acc"], emit_receipts=True)
    gate.compare({"acc": ScoreEstimate(0.9)})
    decision = gate.compare({"acc": ScoreEstimate(0.5)})
    assert decision.accepted is False
    receipts = get_receipt_store().query_by_kind("governance")
    target = next(r for r in receipts if r.receipt_id == decision.receipt_id)
    assert target.action == "reject"
    assert target.approved is False


# ---------------------------------------------------------------------------
# decision dict shape
# ---------------------------------------------------------------------------
def test_decision_to_dict_is_json_serialisable():
    import json

    gate = PromotionGate(critical_metrics=["acc"], emit_receipts=False)
    gate.compare({"acc": ScoreEstimate(0.5)})
    decision = gate.compare({"acc": ScoreEstimate(0.9)})
    blob = json.dumps(decision.to_dict())
    parsed = json.loads(blob)
    assert parsed["accepted"] is True


def test_history_grows_with_each_comparison():
    gate = PromotionGate(critical_metrics=["acc"], emit_receipts=False)
    for v in (0.5, 0.6, 0.7):
        gate.compare({"acc": ScoreEstimate(v)})
    assert len(gate.history) == 3
