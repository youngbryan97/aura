"""Paired semantic boundaries preserve old receipts and score the full chart."""

from dataclasses import replace

import numpy as np
import pytest

from core.learning import semantic_paired_pointer_refit as refit
from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
    fit_compositional_semantic_program_transducer,
)
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import LinearPointerHead, _pointer_head_from_dict
from tests.test_semantic_program_shared_transducer import _examples, _grounding


def test_paired_ranking_equals_exhaustive_span_scores():
    rng = np.random.default_rng(91)
    hidden = rng.normal(size=(13, 8)).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    head = LinearPointerHead(rng.normal(size=8), 0.3, rng.normal(size=8), -0.7, rng.normal(size=8))
    sequence = head.score_sequence(hidden)
    all_spans = [TokenSpan(start, end) for start in range(13) for end in range(start + 1, min(13, start + 5) + 1)]
    expected = sorted(((span, sequence.score_span(span)) for span in all_spans), key=lambda pair: (-pair[1], pair[0].start, pair[0].end))
    actual = head.decode_candidates(hidden, limit=len(all_spans), max_span_tokens=5)
    assert [span for span, _ in actual] == [span for span, _ in expected]
    np.testing.assert_allclose([score for _, score in actual], [score for _, score in expected], atol=1e-6)
    coefficients = np.concatenate((head.start_weight, head.end_weight, head.pair_weight))
    for span, score in actual:
        assert score == pytest.approx(refit.paired_boundary_feature(hidden, span) @ coefficients + head.start_bias + head.end_bias, abs=1e-6)


def test_pair_interaction_separates_crossed_boundaries_with_equal_endpoint_scores():
    hidden = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    old = LinearPointerHead(np.zeros(2), 0, np.zeros(2), 0)
    paired = replace(old, pair_weight=np.ones(2))
    correct, crossed = TokenSpan(0, 3), TokenSpan(0, 4)
    assert old.score_span(hidden, correct) == old.score_span(hidden, crossed)
    assert paired.score_span(hidden, correct) > paired.score_span(hidden, crossed)
    assert "pair_weight" not in old.to_dict()
    for head in (old, paired):
        assert _pointer_head_from_dict(head.to_dict()).to_dict() == head.to_dict()


@pytest.mark.parametrize("weight", [np.ones(3), np.asarray([float("nan"), 0])])
def test_bad_pair_geometry_is_rejected(weight):
    with pytest.raises(ValueError, match="pair geometry"):
        LinearPointerHead(np.zeros(2), 0, np.zeros(2), 0, weight)


@pytest.fixture(scope="module")
def parent():
    return fit_compositional_semantic_program_transducer(_examples(), input_grounding=_grounding())


def test_negative_mining_never_labels_another_positive_negative(parent):
    item = _examples()[0]
    positives = tuple(i.operation_span for i in item.ir.instructions)
    rows = refit.paired_boundary_training_spans(item, positives, parent.operation_pointer, parent.max_span_tokens)
    assert {span for span, label in rows if label} == set(positives)
    assert not {span for span, label in rows if not label} & set(positives)
    assert len({span for span, _ in rows}) == len(rows)


def test_fit_roundtrip_and_lesion_cover_pair_weights(parent, monkeypatch):
    original = refit.fit_paired_boundary_pointer

    def training_only(training, **kwargs):
        assert training and {item.split for item in training} == {"train"}
        return original(training, **kwargs)

    monkeypatch.setattr(refit, "fit_paired_boundary_pointer", training_only)
    model = refit.refit_compositional_paired_operation_pointer(parent, _examples())
    assert np.count_nonzero(model.operation_pointer.pair_weight)
    assert model.operation_pointer.width == parent.operation_pointer.width
    assert model.operation_head.to_dict() == parent.operation_head.to_dict()
    assert compositional_semantic_program_transducer_from_dict(model.to_dict()).to_dict() == model.to_dict()
    lesion = model.coefficient_lesion()
    assert not np.count_nonzero(lesion.operation_pointer.pair_weight)
    receipt = model.training_receipt["paired_operation_pointer_refit"]
    assert receipt["test_examples_used"] == 0
    assert receipt["validation_used_for_fit"] is False
    assert receipt["serving_authority"] is False


def test_duplicate_sources_fail_before_fit(parent, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("invalid data reached fitting")

    monkeypatch.setattr(refit, "fit_paired_boundary_pointer", forbid)
    examples = _examples()
    with pytest.raises(ValueError, match="unique disjoint"):
        refit.refit_compositional_paired_operation_pointer(parent, (*examples, examples[0]))


def test_ranked_pointer_fits_source_groups_and_preserves_nonpointer_modules(parent, monkeypatch):
    from core.learning import semantic_argument_ranking

    fit = semantic_argument_ranking.fit_pairwise_argument_weight
    captured = []
    def capture(features, labels, weights, **kwargs):
        starts = np.flatnonzero(labels == 1)
        assert starts[0] == 0
        stops = (*starts[1:], len(labels))
        assert all(stop - start >= 2 for start, stop in zip(starts, stops, strict=True))
        captured.append(len(starts))
        return fit(features, labels, weights, **kwargs)

    monkeypatch.setattr(semantic_argument_ranking, "fit_pairwise_argument_weight", capture)
    model = refit.refit_compositional_paired_operation_pointer(parent, _examples(), ranking=True)
    receipt = model.training_receipt["paired_operation_pointer_refit"]
    trained = [item for item in _examples() if item.split == "train"]
    assert captured == [sum(len(set(i.operation_span for i in item.ir.instructions)) for item in trained)]
    assert receipt["objective"] == "source_grouped_pairwise_logistic_v1"
    measured = receipt["supervision"]["ranking_fit"]
    assert measured["converged"] and measured["exported_loss"] < measured["initial_loss"]
    assert not receipt["validation_used_for_fit"]
    assert model.receipt_sha256 != parent.receipt_sha256
    assert model.operation_head.to_dict() == parent.operation_head.to_dict()
    assert model.argument_pointer.to_dict() == parent.argument_pointer.to_dict()
    assert model.operation_pointer.start_bias + model.operation_pointer.end_bias == pytest.approx(
        parent.operation_pointer.start_bias + parent.operation_pointer.end_bias)
    assert compositional_semantic_program_transducer_from_dict(model.to_dict()).to_dict() == model.to_dict()


@pytest.mark.parametrize("ranking", [1, "yes", None])
def test_ranking_policy_is_explicit(parent, ranking):
    with pytest.raises(ValueError, match="boolean"):
        refit.fit_paired_boundary_pointer(
            _examples(), spans=lambda item: tuple(i.operation_span for i in item.ir.instructions),
            pointer=parent.operation_pointer, max_span_tokens=parent.max_span_tokens, ranking=ranking,
        )
