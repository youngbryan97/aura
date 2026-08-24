"""Steering vectors are regenerated, never carried, and never trusted early.

The 45 retained vectors are 5120-wide and so is the new residual stream, so all
of them load. They describe directions in a space that no longer exists, which
is the failure with no exception attached to it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.learning.hybrid_recurrence_geometry import LayerGeometry
from core.learning.steering_regeneration import (
    REQUIRED_EVIDENCE,
    SteeringRegenerationError,
    authority_errors,
    build_plan,
    may_serve,
    resolve_target_layers,
)

DENSE = LayerGeometry(num_hidden_layers=64)
HYBRID = LayerGeometry(num_hidden_layers=64, full_attention_interval=4)
INSTALL = Path("/Users/bryan/.aura/live-source")


def _plan(geometry=HYBRID):
    return build_plan(
        descriptor_fingerprint="f" * 64,
        model_path="/models/target",
        geometry=geometry,
        hidden_size=5120,
        dimensions=("valence_positive", "arousal"),
    )


def test_the_same_depth_band_is_all_attention_on_a_dense_model():
    targets = resolve_target_layers(DENSE)
    assert all(t.carries_attention for t in targets)


def test_the_same_depth_band_is_mostly_linear_on_the_hybrid_model():
    # 25..40 holds four attention layers; the other twelve advance a recurrent
    # state instead of being re-read from a K/V cache, so an injection there
    # propagates differently.
    targets = resolve_target_layers(HYBRID)
    attention = [t.index for t in targets if t.carries_attention]
    assert attention == [27, 31, 35, 39]
    assert len(targets) - len(attention) == 12


def test_every_target_records_which_kind_of_layer_it_is():
    for target in resolve_target_layers(HYBRID):
        assert target.kind in {"full_attention", "linear_attention"}


def test_a_nonsense_depth_band_is_refused():
    with pytest.raises(SteeringRegenerationError):
        resolve_target_layers(HYBRID, (0.7, 0.3))


def test_a_fresh_plan_never_reuses_previous_vectors():
    plan = _plan()
    assert plan.reuses_previous_vectors is False
    assert plan.serving_authority is False


def test_a_plan_must_name_its_checkpoint():
    with pytest.raises(SteeringRegenerationError):
        build_plan(
            descriptor_fingerprint="",
            model_path="/m",
            geometry=HYBRID,
            hidden_size=5120,
            dimensions=("valence",),
        )


def test_a_plan_must_name_dimensions():
    with pytest.raises(SteeringRegenerationError):
        build_plan(
            descriptor_fingerprint="f" * 64,
            model_path="/m",
            geometry=HYBRID,
            hidden_size=5120,
            dimensions=(),
        )


def test_capture_alone_grants_nothing():
    plan = _plan().as_dict()
    errors = authority_errors(plan)
    # Nothing has been shown, so nothing is granted: the full requirement list.
    assert len(errors) == len(REQUIRED_EVIDENCE)
    assert may_serve(plan) is False


def test_absent_evidence_is_never_a_pass():
    plan = _plan().as_dict()
    assert may_serve(plan, {}) is False
    assert may_serve(plan, None) is False


def test_partial_evidence_is_refused_and_names_what_is_missing():
    plan = _plan().as_dict()
    errors = authority_errors(
        plan,
        {
            "extraction_bound_to_active_descriptor": True,
            "causal_ab_vs_matched_noop": True,
        },
    )
    assert "missing_evidence:lesion_removes_the_effect" in errors
    assert "missing_evidence:causal_ab_vs_matched_noop" not in errors


def test_complete_evidence_on_this_checkpoint_grants_authority():
    plan = _plan().as_dict()
    evidence = dict.fromkeys(REQUIRED_EVIDENCE, True)
    evidence["descriptor_fingerprint"] = plan["descriptor_fingerprint"]
    assert may_serve(plan, evidence) is True


def test_complete_evidence_from_another_checkpoint_does_not():
    plan = _plan().as_dict()
    evidence = dict.fromkeys(REQUIRED_EVIDENCE, True)
    evidence["descriptor_fingerprint"] = "0" * 64
    errors = authority_errors(plan, evidence)
    assert "evidence_measured_on_a_different_checkpoint" in errors
    assert may_serve(plan, evidence) is False


def test_a_plan_that_reuses_old_vectors_can_never_serve():
    plan = _plan().as_dict()
    plan["reuses_previous_vectors"] = True
    evidence = dict.fromkeys(REQUIRED_EVIDENCE, True)
    evidence["descriptor_fingerprint"] = plan["descriptor_fingerprint"]
    assert "plan_reuses_vectors_from_another_checkpoint" in authority_errors(
        plan, evidence
    )
    assert may_serve(plan, evidence) is False


def test_the_lesion_requirement_is_not_optional():
    assert "lesion_removes_the_effect" in REQUIRED_EVIDENCE


def test_the_old_bundle_sites_are_mostly_linear_on_this_checkpoint():
    """The retained metadata's layers, read against the new topology."""
    meta = INSTALL / "training/vectors/caa_steering_meta.json"
    if not meta.exists():
        pytest.skip("no retained steering metadata")
    dimensions = json.loads(meta.read_text()).get("dimensions") or []
    layers = sorted({
        int(layer)
        for entry in dimensions
        for layer in (entry.get("layers_extracted") or [])
    })
    if not layers:
        pytest.skip("retained metadata lists no extraction layers")
    attention = [i for i in layers if HYBRID.carries_attention(i)]
    # Nine sites on the old model, four of which are attention layers here.
    # Most of where the retained vectors were captured is now a layer that
    # carries a recurrent state instead of a K/V cache.
    assert layers == [25, 27, 29, 31, 33, 35, 37, 39, 41]
    assert attention == [27, 31, 35, 39]
    assert 2 * len(attention) < len(layers)
