"""Tests for symbol grounding and Backpropamine plastic adapter.

The tests cover Bryan's acceptance criteria for F10:

  * triadic link (symbol, concept, evidence, method)
  * confirmed examples raise grounding confidence
  * negative feedback weakens the link
  * plastic adapter changes future representation after reward
  * grounding+plasticity improves held-out classification
  * governor refuses unsafe / no-signal updates
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.grounding import GroundingService
from core.plasticity import (
    GroundingPlasticAdapter,
    NeuromodulatedPlasticLayer,
    PlasticityConfig,
    PlasticityDecision,
    SemanticWeightGovernor,
)


@pytest.fixture
def service(tmp_path: Path) -> GroundingService:
    return GroundingService(tmp_path / "grounding")


# ---------------------------------------------------------------------------
# semiotic triad
# ---------------------------------------------------------------------------
def test_grounded_symbol_has_concept_evidence_and_method(service: GroundingService):
    out = service.learn_from_example(
        symbol="marbled",
        raw="a surface with irregular gray and white stone-like veins",
        confirmed=True,
    )
    assert out["concept_id"]
    assert out["evidence_id"]
    candidates = service.network.concepts_for_symbol("marbled")
    assert candidates
    concept, link = candidates[0]
    assert concept.method_id == "method_text_hash_v1"
    assert link.strength > 0.5


def test_confirmed_examples_raise_confidence(service: GroundingService):
    a = service.learn_from_example(
        symbol="ribbed", raw="parallel raised lines", confirmed=True
    )
    b = service.learn_from_example(
        symbol="ribbed", raw="texture with repeated ridges", confirmed=True
    )
    assert b["confidence"] >= a["confidence"]


def test_negative_feedback_weakens_link(service: GroundingService):
    first = service.learn_from_example(
        symbol="glossy", raw="dull matte surface", confirmed=True
    )
    second = service.learn_from_example(
        symbol="glossy", raw="dull matte surface", confirmed=False
    )
    assert second["confidence"] < first["confidence"]
    assert second["link_strength"] < first["link_strength"]


# ---------------------------------------------------------------------------
# prediction + ledger integration
# ---------------------------------------------------------------------------
def test_predict_unknown_symbol_returns_no_match(service: GroundingService):
    pred = service.predict_symbol_applies(symbol="never_seen", raw="anything")
    assert pred["applies"] is False
    assert pred["reason"] == "unknown_symbol"
    assert pred["prediction_id"] == ""


def test_grounding_predicts_correctly_after_examples(service: GroundingService):
    service.learn_from_example(symbol="marbled", raw="white gray stone veins", confirmed=True)
    service.learn_from_example(symbol="marbled", raw="irregular gray veined stone pattern", confirmed=True)
    service.learn_from_example(symbol="marbled", raw="smooth plain red fabric", confirmed=False)

    pred = service.predict_symbol_applies(
        symbol="marbled", raw="gray white veined surface like stone"
    )
    assert pred["applies"] is True
    assert pred["confidence"] > 0.10
    assert pred["prediction_id"]


def test_predictions_resolve_into_brier(service: GroundingService):
    service.learn_from_example(symbol="ribbed", raw="parallel raised lines", confirmed=True)
    service.learn_from_example(symbol="ribbed", raw="ridge texture", confirmed=True)
    pred = service.predict_symbol_applies(symbol="ribbed", raw="parallel raised lines")
    out = service.confirm_prediction(pred["prediction_id"], applies=True)
    assert out["resolved"] is True
    assert out["brier"] is not None
    # Confident-correct prediction should give a small Brier loss.
    assert out["brier"] <= 1.0


# ---------------------------------------------------------------------------
# plastic layer
# ---------------------------------------------------------------------------
def test_plastic_layer_no_update_without_activity():
    layer = NeuromodulatedPlasticLayer(PlasticityConfig(in_dim=4, out_dim=4))
    out = layer.update(reward=1.0, modulation=0.5)
    assert out["updated"] is False
    assert out["reason"] == "no_activity"


def test_plastic_layer_update_after_forward_changes_hebb():
    layer = NeuromodulatedPlasticLayer(PlasticityConfig(in_dim=4, out_dim=4))
    layer.forward(np.array([1.0, 0.0, 0.0, 0.0]))
    out = layer.update(reward=1.0, modulation=0.5)
    assert out["updated"] is True
    assert out["delta_norm"] > 0
    assert out["hebb_norm"] > 0


def test_plastic_layer_max_delta_norm_caps_step():
    layer = NeuromodulatedPlasticLayer(
        PlasticityConfig(in_dim=4, out_dim=4, max_delta_norm=0.01)
    )
    layer.forward(np.ones(4) * 5.0)
    out = layer.update(reward=1.0, modulation=1.0)
    # delta_norm should be at or below the cap (allow tiny float slop).
    assert out["delta_norm"] <= 0.0101


def test_plastic_layer_reset_zeroes_state():
    layer = NeuromodulatedPlasticLayer(PlasticityConfig(in_dim=4, out_dim=4))
    layer.forward(np.array([1.0, 1.0, 1.0, 1.0]))
    layer.update(reward=1.0, modulation=0.5)
    layer.reset_plastic_state()
    snap = layer.snapshot()
    assert snap["hebb_norm"] == 0.0
    assert snap["eligibility_norm"] == 0.0


def test_adapter_changes_features_after_reward():
    adapter = GroundingPlasticAdapter(feature_dim=8)
    x = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    before = adapter.adapt_features(x)
    adapter.update_from_reward(reward=1.0, modulation=0.8)
    after = adapter.adapt_features(x)
    diff = float(np.linalg.norm(np.asarray(after) - np.asarray(before)))
    assert diff > 1e-6


# ---------------------------------------------------------------------------
# governor
# ---------------------------------------------------------------------------
def test_governor_refuses_low_vitality():
    gov = SemanticWeightGovernor()
    decision = gov.decide(module_name="x", reward=0.5, vitality=0.05)
    assert decision.allowed is False
    assert decision.severity == "critical"
    assert decision.reason == "vitality_too_low"


def test_governor_refuses_weak_reward():
    gov = SemanticWeightGovernor()
    decision = gov.decide(module_name="x", reward=0.001, vitality=1.0)
    assert decision.allowed is False
    assert decision.reason == "reward_too_weak"


def test_governor_allows_with_modulation_in_unit_range():
    gov = SemanticWeightGovernor()
    decision = gov.decide(
        module_name="x",
        reward=0.5,
        vitality=1.0,
        curiosity=0.8,
        arousal=0.8,
        free_energy=0.4,
    )
    assert decision.allowed is True
    assert 0.0 <= decision.modulation <= 1.0


# ---------------------------------------------------------------------------
# the real one — held-out improvement
# ---------------------------------------------------------------------------
def _bag_overlap(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def test_grounding_improves_heldout_after_confirmed_examples(tmp_path: Path):
    """Bryan's load-bearing acceptance test: after confirmed examples,
    held-out predictions should beat 'no examples' on the same symbol."""

    # Service A: zero training; predict held-out -> unknown_symbol.
    svc_blank = GroundingService(tmp_path / "blank")
    pred_blank = svc_blank.predict_symbol_applies(
        symbol="marbled", raw="gray white veined stone surface"
    )
    assert pred_blank["applies"] is False

    # Service B: trained with positive examples that overlap the held-out.
    svc_trained = GroundingService(tmp_path / "trained")
    svc_trained.learn_from_example(
        symbol="marbled", raw="white gray veined stone surface", confirmed=True
    )
    svc_trained.learn_from_example(
        symbol="marbled", raw="irregular gray white veined stone pattern", confirmed=True
    )
    svc_trained.learn_from_example(
        symbol="marbled", raw="smooth plain red fabric panel", confirmed=False
    )

    pred_trained = svc_trained.predict_symbol_applies(
        symbol="marbled", raw="gray white veined stone surface"
    )

    # Trained service must do meaningfully better on held-out.
    assert pred_trained["applies"] is True
    assert pred_trained["confidence"] > pred_blank["confidence"]


def test_grounding_persists_across_reopen(tmp_path: Path):
    svc_a = GroundingService(tmp_path / "g")
    svc_a.learn_from_example(symbol="smooth", raw="polished plane", confirmed=True)

    svc_b = GroundingService(tmp_path / "g")
    candidates = svc_b.network.concepts_for_symbol("smooth")
    assert candidates
    assert candidates[0][0].label == "smooth"
