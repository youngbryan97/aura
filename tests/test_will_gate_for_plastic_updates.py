"""Tests for the runtime Will gate on plastic-adapter updates."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from core.grounding import GroundingService
from core.plasticity import GroundingPlasticAdapter, SemanticWeightGovernor
from core.runtime.receipts import (
    SemanticWeightUpdateReceipt,
    get_receipt_store,
    reset_receipt_store,
)


def _make_will_decider(outcome: str, reason: str = "", receipt_id: str = "will-001"):
    """Return a stub will_decide_fn that always returns the given outcome
    plus a list capturing every call (for assertion)."""
    calls: List[Dict[str, Any]] = []

    def will_decide(domain: str, module: str, reward: float) -> Dict[str, Any]:
        calls.append({"domain": domain, "module": module, "reward": reward})
        return {"outcome": outcome, "reason": reason, "receipt_id": receipt_id}

    return will_decide, calls


@pytest.fixture
def loop_factory(tmp_path: Path):
    """Factory that builds GroundingService with optional will_decide_fn."""
    reset_receipt_store()
    get_receipt_store(tmp_path / "receipts")

    def _build(will_decide_fn=None):
        adapter = GroundingPlasticAdapter(feature_dim=128)
        governor = SemanticWeightGovernor()
        svc = GroundingService(
            tmp_path / "g",
            plastic_adapter=adapter,
            governor=governor,
            will_decide_fn=will_decide_fn,
            emit_receipts=True,
        )
        svc.learn_from_example(symbol="x", raw="alpha", confirmed=True)
        pred = svc.predict_symbol_applies(symbol="x", raw="alpha beta")
        return svc, adapter, governor, pred["prediction_id"]

    yield _build
    reset_receipt_store()


# ---------------------------------------------------------------------------
# proceed / refuse / defer outcomes
# ---------------------------------------------------------------------------
def test_will_proceed_allows_plastic_update(loop_factory):
    will, calls = _make_will_decider("proceed", reason="ok")
    svc, adapter, _gov, pid = loop_factory(will_decide_fn=will)

    pre = adapter.snapshot()["total_updates"]
    out = svc.confirm_prediction(pid, applies=True)
    post = adapter.snapshot()["total_updates"]

    assert post == pre + 1
    assert out["weight_update"] is not None
    assert out["weight_update_reason"] == "applied"
    assert out["will_outcome"] == "proceed"
    assert len(calls) == 1
    assert calls[0]["domain"] == "semantic_weight_update"
    assert calls[0]["module"] == "grounding_plastic_adapter"


def test_will_refuse_blocks_plastic_update(loop_factory):
    will, _calls = _make_will_decider("refuse", reason="policy_violation")
    svc, adapter, _gov, pid = loop_factory(will_decide_fn=will)

    pre = adapter.snapshot()["total_updates"]
    out = svc.confirm_prediction(pid, applies=True)
    post = adapter.snapshot()["total_updates"]

    assert post == pre  # no update applied
    assert out["weight_update"] is None
    assert out["weight_update_reason"].startswith("will_refuse")
    assert "policy_violation" in out["weight_update_reason"]
    assert out["will_outcome"] == "refuse"


def test_will_defer_blocks_update(loop_factory):
    will, _calls = _make_will_decider("defer", reason="vitality_drop")
    svc, adapter, _gov, pid = loop_factory(will_decide_fn=will)

    pre = adapter.snapshot()["total_updates"]
    out = svc.confirm_prediction(pid, applies=True)
    post = adapter.snapshot()["total_updates"]

    assert post == pre
    assert out["weight_update"] is None
    assert out["weight_update_reason"].startswith("will_defer")


def test_will_constrain_treated_as_block(loop_factory):
    """CONSTRAIN means proceed-but-not-now-as-asked; for plastic
    updates that's the same as block — we don't have a partial-update
    mode, so any non-PROCEED is treated as no update."""
    will, _calls = _make_will_decider("constrain", reason="reduce_scope")
    svc, adapter, _gov, pid = loop_factory(will_decide_fn=will)

    pre = adapter.snapshot()["total_updates"]
    svc.confirm_prediction(pid, applies=True)
    post = adapter.snapshot()["total_updates"]
    assert post == pre


# ---------------------------------------------------------------------------
# defence in depth: governor independent of Will
# ---------------------------------------------------------------------------
def test_will_proceed_but_governor_blocks_low_vitality(loop_factory):
    will, _calls = _make_will_decider("proceed")
    svc, adapter, _gov, pid = loop_factory(will_decide_fn=will)

    pre = adapter.snapshot()["total_updates"]
    out = svc.confirm_prediction(pid, applies=True, vitality=0.05)
    post = adapter.snapshot()["total_updates"]

    assert post == pre
    assert out["weight_update_reason"] == "vitality_too_low"


# ---------------------------------------------------------------------------
# Will errors fail closed
# ---------------------------------------------------------------------------
def test_will_decide_raising_blocks_update(loop_factory):
    def will_raises(domain, module, reward):
        raise RuntimeError("Will service unreachable")

    svc, adapter, _gov, pid = loop_factory(will_decide_fn=will_raises)

    pre = adapter.snapshot()["total_updates"]
    out = svc.confirm_prediction(pid, applies=True)
    post = adapter.snapshot()["total_updates"]

    assert post == pre
    assert out["weight_update"] is None
    assert out["will_outcome"] == "refuse"
    assert "will_decide_raised" in out["weight_update_reason"]


# ---------------------------------------------------------------------------
# absence of Will falls back to existing behaviour
# ---------------------------------------------------------------------------
def test_no_will_decide_fn_uses_governor_only(loop_factory):
    svc, adapter, _gov, pid = loop_factory(will_decide_fn=None)

    pre = adapter.snapshot()["total_updates"]
    out = svc.confirm_prediction(pid, applies=True)
    post = adapter.snapshot()["total_updates"]

    assert post == pre + 1
    assert out["weight_update"] is not None
    assert out["weight_update_reason"] == "applied"
    assert out["will_outcome"] == "proceed"


# ---------------------------------------------------------------------------
# receipt records the will_receipt_id
# ---------------------------------------------------------------------------
def test_receipt_carries_will_receipt_id(loop_factory):
    will, _calls = _make_will_decider("proceed", receipt_id="will-xyz-42")
    svc, _adapter, _gov, pid = loop_factory(will_decide_fn=will)

    svc.confirm_prediction(pid, applies=True)

    store = get_receipt_store()
    receipts = store.query_by_kind("semantic_weight_update")
    assert receipts
    receipt = receipts[-1]
    assert isinstance(receipt, SemanticWeightUpdateReceipt)
    assert receipt.governance_receipt_id == "will-xyz-42"
    assert receipt.allowed is True


def test_receipt_records_blocked_state_when_will_refuses(loop_factory):
    will, _calls = _make_will_decider("refuse", reason="policy")
    svc, _adapter, _gov, pid = loop_factory(will_decide_fn=will)

    svc.confirm_prediction(pid, applies=True)

    store = get_receipt_store()
    receipts = store.query_by_kind("semantic_weight_update")
    assert receipts
    receipt = receipts[-1]
    assert receipt.allowed is False
    assert receipt.delta_norm == 0.0
    assert receipt.hebb_norm == 0.0
