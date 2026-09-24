"""The source-trained triadic factor remains bound to the ordinary proposer."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.semantic_program_compositional_refits import (
    refit_compositional_triadic_bindings,
)
from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
    fit_compositional_semantic_program_transducer,
)
from core.learning.semantic_triadic_binding import (
    TriadicBindingHead,
    evaluate_triadic_gold_binding,
    fit_source_triadic_projection,
    fit_triadic_binding_heads,
    joint_representation_binding_feature,
    joint_source_binding_feature,
    projected_joint_binding_feature,
    triadic_binding_feature,
)
from tests.test_semantic_program_shared_transducer import _examples, _grounding


def test_feature_is_ordered_and_operation_conditioned():
    op = np.asarray([1, 2], dtype=np.float32)
    mention = np.asarray([2, 3], dtype=np.float32)
    definition = np.asarray([3, 5], dtype=np.float32)
    original = triadic_binding_feature(op, mention, definition)
    assert not np.array_equal(original, triadic_binding_feature(op, definition, mention))
    assert not np.array_equal(original, triadic_binding_feature(-op, mention, definition))
    with pytest.raises(ValueError, match="width"):
        triadic_binding_feature(op, mention, definition[:1])


def test_joint_feature_keeps_source_occurrences_and_schema():
    from core.learning.semantic_program_ir import TokenSpan

    op = np.asarray([1, 2], dtype=np.float32)
    mention = np.asarray([2, 3], dtype=np.float32)
    definition = np.asarray([3, 5], dtype=np.float32)
    spans = dict(operation_span=TokenSpan(0, 1), mention_span=TokenSpan(3, 4),
                 definition_span=TokenSpan(1, 2), token_count=5)
    feature = joint_source_binding_feature(op, mention, definition, **spans)
    assert feature.shape == (8 * len(op) + 8,)
    assert np.array_equal(feature[:-8], joint_representation_binding_feature(
        op, mention, definition))
    moved = {**spans, "definition_span": TokenSpan(2, 3)}
    assert not np.array_equal(feature, joint_source_binding_feature(op, mention, definition, **moved))
    weight = np.zeros_like(feature)
    weight[-7] = 1.0
    head = TriadicBindingHead(weight, 0.0, "joint_source_v2")
    assert head.channel_width == len(op)
    assert head.score(op, mention, definition, **spans) != head.score(op, mention, definition, **moved)
    assert head.to_dict()["feature_schema"] == "joint_source_v2"
    assert head.component_lesion("geometry").weight[:-8].sum() == 0
    assert head.component_lesion("representation").weight[-8:].sum() == 0
    with pytest.raises(ValueError, match="all source spans"):
        head.score(op, mention, definition)


def test_fit_requires_source_only_contrasts():
    examples = _examples()
    training = tuple(item for item in examples if item.split == "train")
    item = training[0]
    with pytest.raises(ValueError, match="source-only"):
        fit_triadic_binding_heads((replace(item, split="test"),), max_arity=2,
            hidden_channels=item.hidden_channels,
            hidden_channel_widths=item.hidden_channel_widths)
    heads, support = fit_triadic_binding_heads(training, max_arity=2,
        hidden_channels=item.hidden_channels,
        hidden_channel_widths=item.hidden_channel_widths)
    assert len(heads) == 2
    assert all(row["positive"] and row["negative"] for row in support["support"])
    assert all(np.isfinite(head.weight).all() for head in heads)


def test_projected_feature_preserves_roles_without_source_geometry():
    operation = np.asarray([1., 2.], dtype=np.float32)
    mention = np.asarray([3., 4.], dtype=np.float32)
    definition = np.asarray([5., 6.], dtype=np.float32)
    query = np.asarray([[1.], [0.]], dtype=np.float32)
    key = np.asarray([[0.], [1.]], dtype=np.float32)
    feature = projected_joint_binding_feature(operation, mention, definition, query, key)
    assert feature.shape == (8,)
    assert not np.array_equal(feature, projected_joint_binding_feature(
        operation, definition, mention, query, key))
    head = TriadicBindingHead(np.ones(8, dtype=np.float32), 0., "projected_joint_v4", query, key)
    assert head.channel_width == 2
    assert head.score(operation, mention, definition) == pytest.approx(feature.sum())
    assert head.score_lesion().score(operation, mention, definition) == 0.
    assert head.role_lesion("operation").score(operation, mention, definition) == pytest.approx(
        feature[[1, 2, 5, 7]].sum())
    assert head.role_lesion("mention").score(operation, mention, definition) == pytest.approx(
        feature[[0, 2, 4]].sum())
    assert head.role_lesion("definition").score(operation, mention, definition) == pytest.approx(
        feature[[0, 1, 3]].sum())
    with pytest.raises(ValueError, match="unknown triadic role"):
        head.role_lesion("source")
    with pytest.raises(ValueError, match="invalid"):
        TriadicBindingHead(np.ones(8), 0., "projected_joint_v4")


def test_projection_fit_refuses_source_overlap():
    examples = _examples()
    fit = tuple(item for item in examples if item.split == "train")
    with pytest.raises(ValueError, match="disjoint"):
        fit_source_triadic_projection(fit, tuple(replace(item, split="validation")
                                                 for item in fit),
            hidden_channels=fit[0].hidden_channels,
            hidden_channel_widths=fit[0].hidden_channel_widths)


def test_projection_fit_accepts_disjoint_training_labeled_calibration():
    examples = _examples()
    fit = tuple(item for item in examples if item.split == "train")
    calibration = tuple(replace(item, split="train") for item in examples
                        if item.split == "validation")
    basis, receipt = fit_source_triadic_projection(
        fit, calibration,
        hidden_channels=fit[0].hidden_channels,
        hidden_channel_widths=fit[0].hidden_channel_widths)
    assert basis[0].shape == basis[1].shape
    assert receipt["training_sources"] > 0
    assert receipt["calibration_sources"] > 0


@pytest.mark.parametrize("feature_schema", (
    "triple_product_v1", "joint_source_v2", "joint_representation_v3",
    "projected_joint_v4"))
def test_refit_round_trip_and_isolated_lesion(monkeypatch, feature_schema):
    examples = _examples()
    parent = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    training = tuple(item for item in examples if item.split == "train")
    with monkeypatch.context() as patch:
        patch.setattr(type(parent), "decode", lambda self, **kwargs: SimpleNamespace(ir=None, refusal="test"))
        candidate = refit_compositional_triadic_bindings(
            parent, examples, feature_schema=feature_schema)
    assert candidate.triadic_binding_heads is not None
    assert candidate.training_receipt["triadic_binding_fit"]["serving_authority"] is False
    assert candidate.training_receipt["triadic_binding_fit"]["fit"]["source_count"] == len(training)
    replay = compositional_semantic_program_transducer_from_dict(candidate.to_dict())
    assert replay.receipt_sha256 == candidate.receipt_sha256
    assert replay.triadic_binding_heads is not None
    assert all(head.feature_schema == feature_schema for head in replay.triadic_binding_heads)
    assert any(np.any(head.weight) for head in replay.triadic_binding_heads)
    if feature_schema == "projected_joint_v4":
        assert all(head.query_projection is not None for head in replay.triadic_binding_heads)
        assert replay.training_receipt["triadic_binding_fit"]["projection_fit"]["inherited_relation_score"] is False
    lesion = replay.triadic_binding_lesion()
    assert all(not np.any(head.weight) for head in lesion.triadic_binding_heads)
    assert all(not np.any(head.weight) for head in replay.coefficient_lesion().triadic_binding_heads)
    assert all(not np.any(head.weight) for head in replay.relation_lesion().triadic_binding_heads)
    assert lesion.receipt_sha256 != replay.receipt_sha256
    calls = []
    original_score = TriadicBindingHead.score

    def observed_score(self, operation, mention, definition, **kwargs):
        calls.append((operation, mention, definition))
        return original_score(self, operation, mention, definition, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(TriadicBindingHead, "score", observed_score)
        sample = next(item for item in examples if item.split == "test")
        replay.decode(
            source_token_ids=sample.ir.source_token_ids,
            hidden_states=sample.hidden_states,
            public_inputs=sample.public_inputs,
            source_text_sha256=sample.ir.source_text_sha256,
            model_basis_sha256=replay.model_basis_sha256,
        )
    assert calls
    with pytest.raises(ValueError, match="unfitted parent"):
        refit_compositional_triadic_bindings(candidate, examples)
    with pytest.raises(ValueError, match="source population"):
        refit_compositional_triadic_bindings(parent, (*examples, training[0]))


def test_head_rejects_nonfinite_coefficients():
    with pytest.raises(ValueError, match="invalid"):
        TriadicBindingHead(np.asarray([float("nan")] * 12), 0.0)


def test_gold_binding_probe_keeps_held_sources_and_counts_ties():
    examples = _examples()
    training = tuple(item for item in examples if item.split == "train")
    held = tuple(item for item in examples if item.split == "test")
    heads, _ = fit_triadic_binding_heads(training, max_arity=2,
        hidden_channels=training[0].hidden_channels,
        hidden_channel_widths=training[0].hidden_channel_widths)
    training_ids = frozenset(item.ir.source_text_sha256 for item in training)
    result = evaluate_triadic_gold_binding(heads, held,
        hidden_channels=held[0].hidden_channels,
        hidden_channel_widths=held[0].hidden_channel_widths,
        training_source_ids=training_ids)
    assert result["source_count"] == len(held)
    assert result["opposed_roles"] == result["correct"] + result["wrong"] + result["tied"]
    assert sum(row["correct"] for row in result["by_family"].values()) == result["correct"]
    assert sum(row["wrong"] for row in result["by_family"].values()) == result["wrong"]
    assert result["opposed_roles"] > 0
    assert result["gold_operation_and_mention_spans"] is True
    zero = tuple(TriadicBindingHead(np.zeros_like(head.weight), head.bias) for head in heads)
    lesion = evaluate_triadic_gold_binding(zero, held,
        hidden_channels=held[0].hidden_channels,
        hidden_channel_widths=held[0].hidden_channel_widths,
        training_source_ids=training_ids)
    assert lesion["correct"] == lesion["wrong"] == 0
    assert lesion["tied"] == lesion["opposed_roles"]
    with pytest.raises(ValueError, match="unseen source"):
        evaluate_triadic_gold_binding(heads, training[:1],
            hidden_channels=training[0].hidden_channels,
            hidden_channel_widths=training[0].hidden_channel_widths,
            training_source_ids=training_ids)
