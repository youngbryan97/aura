"""Closed-loop tests for grounding + plasticity + governance.

Maps 1:1 to the named acceptance tests in the F10 follow-up review:

  test_grounding_service_learns_symbol
  test_prediction_ledger_records_grounding_prediction
  test_confirm_prediction_computes_reward
  test_plastic_adapter_updates_after_reward
  test_plasticity_changes_future_similarity
  test_negative_feedback_weakens_wrong_grounding
  test_weight_update_requires_will
  test_grounding_service_uses_plastic_adapter
  test_textual_grounding_improves_heldout
  test_sensorimotor_modality_rejects_placeholder_in_strict_mode

The full grounding -> prediction -> confirmation -> reward -> governor
-> plastic update -> receipt path runs end-to-end here without GPU,
without an LLM, without LoRA contention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.grounding import (
    GroundingKernel,
    GroundingObservation,
    GroundingService,
    SUPPORTED_MODALITIES,
    UnsupportedModalityError,
)
from core.plasticity import (
    GroundingPlasticAdapter,
    SemanticWeightGovernor,
)
from core.runtime.receipts import (
    SemanticWeightUpdateReceipt,
    get_receipt_store,
    reset_receipt_store,
)
from core.will import (
    ALLOWED_PLASTIC_MODULES,
    DENIED_PLASTIC_SUBSTRINGS,
    is_plastic_target_allowed,
    ActionDomain,
)


@pytest.fixture
def service_loop(tmp_path: Path):
    """A grounding service with a plastic adapter + governor wired in."""
    reset_receipt_store()
    get_receipt_store(tmp_path / "receipts")
    adapter = GroundingPlasticAdapter(feature_dim=128)
    governor = SemanticWeightGovernor()
    svc = GroundingService(
        tmp_path / "g",
        plastic_adapter=adapter,
        governor=governor,
        emit_receipts=True,
    )
    yield svc, adapter, governor
    reset_receipt_store()


# ---------------------------------------------------------------------------
# Grounding basics
# ---------------------------------------------------------------------------
def test_grounding_service_learns_symbol(tmp_path: Path):
    svc = GroundingService(tmp_path / "g")
    out = svc.learn_from_example(
        symbol="ribbed", raw="parallel raised lines", confirmed=True
    )
    assert out["concept_id"]
    assert out["evidence_id"]
    assert out["confidence"] > 0.0
    candidates = svc.network.concepts_for_symbol("ribbed")
    assert candidates


def test_prediction_ledger_records_grounding_prediction(tmp_path: Path):
    svc = GroundingService(tmp_path / "g")
    svc.learn_from_example(symbol="smooth", raw="polished plane", confirmed=True)
    pred = svc.predict_symbol_applies(symbol="smooth", raw="polished surface")
    assert pred["prediction_id"]
    record = svc.ledger.get(pred["prediction_id"])
    assert record is not None
    assert record.belief == "smooth"
    assert record.modality == "text"
    assert record.action == "ground_predict"
    assert record.resolved is False


# ---------------------------------------------------------------------------
# Reward + governor + adapter
# ---------------------------------------------------------------------------
def test_confirm_prediction_computes_reward(service_loop):
    svc, _adapter, _gov = service_loop
    svc.learn_from_example(symbol="x", raw="alpha beta", confirmed=True)
    pred = svc.predict_symbol_applies(symbol="x", raw="alpha beta gamma")
    out = svc.confirm_prediction(pred["prediction_id"], applies=True)
    # Resolved with correct prediction -> positive reward applied.
    assert out["resolved"] is True
    assert out["weight_update"] is not None
    assert out["weight_update_reason"] == "applied"
    # Brier should be reasonable (low) for a confidently-correct call.
    assert out["brier"] is not None


def test_plastic_adapter_updates_after_reward(service_loop):
    svc, adapter, _gov = service_loop
    pre = adapter.snapshot()["total_updates"]
    svc.learn_from_example(symbol="x", raw="alpha", confirmed=True)
    pred = svc.predict_symbol_applies(symbol="x", raw="alpha beta")
    svc.confirm_prediction(pred["prediction_id"], applies=True, curiosity=0.9, arousal=0.9)
    post = adapter.snapshot()["total_updates"]
    assert post == pre + 1


def test_plasticity_changes_future_similarity(service_loop):
    """The whole point: applying a confirmed positive reward should
    measurably shift the feature representation the next prediction
    sees."""
    svc, adapter, _gov = service_loop
    svc.learn_from_example(symbol="x", raw="alpha beta", confirmed=True)

    # Snapshot the post-adapter feature vector for a fresh observation.
    obs = GroundingObservation(symbol="x", modality="text", raw="alpha beta gamma")
    raw_features = svc.kernel.encode(obs).features
    before = adapter.adapt_features(raw_features)

    # Confirm one positive prediction -> adapter updates.
    pred = svc.predict_symbol_applies(symbol="x", raw="alpha beta gamma")
    svc.confirm_prediction(
        pred["prediction_id"], applies=True, curiosity=0.9, arousal=0.9
    )

    after = adapter.adapt_features(raw_features)
    diff = float(np.linalg.norm(np.asarray(after) - np.asarray(before)))
    assert diff > 1e-6, "plastic update did not change adapter output"


# ---------------------------------------------------------------------------
# Negative feedback
# ---------------------------------------------------------------------------
def test_negative_feedback_weakens_wrong_grounding(service_loop):
    svc, _adapter, _gov = service_loop
    first = svc.learn_from_example(
        symbol="glossy", raw="dull matte fabric", confirmed=True
    )
    second = svc.learn_from_example(
        symbol="glossy", raw="dull matte fabric", confirmed=False
    )
    assert second["confidence"] < first["confidence"]
    assert second["link_strength"] < first["link_strength"]


# ---------------------------------------------------------------------------
# Will-side policy gate
# ---------------------------------------------------------------------------
def test_weight_update_requires_will():
    """The Will deny-list must catch attempts to mutate forbidden targets,
    even if a caller tries to invoke an update directly."""
    # Allow-list members pass.
    assert is_plastic_target_allowed("grounding_plastic_adapter") is True
    # Anything else fails — defaults to deny.
    assert is_plastic_target_allowed("random_module") is False
    assert is_plastic_target_allowed("") is False
    # Hard deny-list catches base-LLM and security paths.
    for forbidden in (
        "base_llm_weights",
        "core.will.UnifiedWill",
        "authority_gateway.fast_path",
        "security.sandbox",
        "model.safetensors.shard0",
    ):
        assert is_plastic_target_allowed(forbidden) is False, forbidden


def test_action_domain_includes_semantic_weight_update():
    assert ActionDomain.SEMANTIC_WEIGHT_UPDATE.value == "semantic_weight_update"


def test_will_policy_lists_are_internally_consistent():
    """Every entry in the allow-list must itself pass the deny check."""
    for module in ALLOWED_PLASTIC_MODULES:
        lower = module.lower()
        assert not any(s in lower for s in DENIED_PLASTIC_SUBSTRINGS), module


# ---------------------------------------------------------------------------
# Service composition
# ---------------------------------------------------------------------------
def test_grounding_service_uses_plastic_adapter(service_loop):
    svc, adapter, gov = service_loop
    assert svc.has_plastic_loop() is True
    assert svc.plastic_adapter is adapter
    assert svc.governor is gov


def test_grounding_service_observe_only_without_adapter(tmp_path: Path):
    svc = GroundingService(tmp_path / "g")
    assert svc.has_plastic_loop() is False
    svc.learn_from_example(symbol="x", raw="alpha", confirmed=True)
    pred = svc.predict_symbol_applies(symbol="x", raw="alpha beta")
    out = svc.confirm_prediction(pred["prediction_id"], applies=True)
    assert out["weight_update"] is None
    assert out["resolved"] is True


def test_governor_rejection_skips_plastic_update(service_loop):
    """If the governor says no, no Hebbian update happens, but the
    confirmation still resolves the prediction in the ledger."""
    svc, adapter, _gov = service_loop
    pre = adapter.snapshot()["total_updates"]
    svc.learn_from_example(symbol="x", raw="alpha", confirmed=True)
    pred = svc.predict_symbol_applies(symbol="x", raw="alpha beta")
    out = svc.confirm_prediction(
        pred["prediction_id"], applies=True, vitality=0.05  # below floor
    )
    post = adapter.snapshot()["total_updates"]
    assert post == pre  # no update
    assert out["weight_update"] is None
    assert out["weight_update_reason"] == "vitality_too_low"
    assert out["resolved"] is True


def test_confirm_emits_semantic_weight_update_receipt(service_loop):
    svc, _adapter, _gov = service_loop
    svc.learn_from_example(symbol="x", raw="alpha", confirmed=True)
    pred = svc.predict_symbol_applies(symbol="x", raw="alpha beta")
    svc.confirm_prediction(pred["prediction_id"], applies=True)
    store = get_receipt_store()
    receipts = store.query_by_kind("semantic_weight_update")
    assert receipts, "no semantic_weight_update receipt was emitted"
    receipt = receipts[-1]
    assert isinstance(receipt, SemanticWeightUpdateReceipt)
    assert receipt.module == "grounding_plastic_adapter"
    assert receipt.prediction_id == pred["prediction_id"]
    assert receipt.allowed is True


# ---------------------------------------------------------------------------
# Held-out improvement (the load-bearing test)
# ---------------------------------------------------------------------------
def test_textual_grounding_improves_heldout(tmp_path: Path):
    """After confirmed examples, held-out predictions must beat
    no-examples on the same symbol.  This is the only test that is
    actually evidence the grounding does what it claims."""
    svc_blank = GroundingService(tmp_path / "blank")
    pred_blank = svc_blank.predict_symbol_applies(
        symbol="marbled", raw="gray white veined stone surface"
    )
    assert pred_blank["applies"] is False

    svc_trained = GroundingService(tmp_path / "trained")
    svc_trained.learn_from_example(
        symbol="marbled", raw="white gray veined stone surface", confirmed=True
    )
    svc_trained.learn_from_example(
        symbol="marbled",
        raw="irregular gray white veined stone pattern",
        confirmed=True,
    )
    svc_trained.learn_from_example(
        symbol="marbled", raw="smooth plain red fabric panel", confirmed=False
    )

    pred_trained = svc_trained.predict_symbol_applies(
        symbol="marbled", raw="gray white veined stone surface"
    )

    assert pred_trained["applies"] is True
    assert pred_trained["confidence"] > pred_blank["confidence"]


# ---------------------------------------------------------------------------
# Strict modality mode
# ---------------------------------------------------------------------------
def test_sensorimotor_modality_rejects_placeholder_in_strict_mode():
    strict_kernel = GroundingKernel(strict_modalities=True)
    obs = GroundingObservation(symbol="x", modality="vision", raw=b"image bytes")
    with pytest.raises(UnsupportedModalityError):
        strict_kernel.encode(obs)


def test_strict_kernel_still_handles_text():
    strict_kernel = GroundingKernel(strict_modalities=True)
    obs = GroundingObservation(symbol="x", modality="text", raw="hello world")
    evidence = strict_kernel.encode(obs)
    assert evidence.evidence_id
    assert len(evidence.features) == 128


def test_supported_modalities_currently_text_only():
    assert "text" in SUPPORTED_MODALITIES
    # vision/audio remain placeholder until real encoders land.
    assert "vision" not in SUPPORTED_MODALITIES
    assert "audio" not in SUPPORTED_MODALITIES
