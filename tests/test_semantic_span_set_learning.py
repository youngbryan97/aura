"""Exact span-set normalization and fitting agree with enumerated alternatives."""

from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest
from scipy.special import logsumexp

from core.learning import semantic_span_set_learning as learning
from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
    fit_compositional_semantic_program_transducer,
)
from core.learning.semantic_program_ir import TokenSpan
from tests.test_semantic_program_shared_transducer import _examples, _grounding


def _scores(n, length, seed=8):
    scores = np.random.default_rng(seed).normal(size=(n, length))
    scores[np.arange(n)[:, None] + np.arange(1, length + 1) > n] = -np.inf
    return scores


def _enumerate(scores, count):
    intervals = [(start, length) for start in range(len(scores))
                 for length in range(1, scores.shape[1] + 1) if start + length <= len(scores)]
    charts = [()]
    for k in range(1, count + 1):
        charts.extend(chart for chart in combinations(intervals, k)
                      if all(a + b <= c for (a, b), (c, _) in zip(chart, chart[1:])))
    values = np.asarray([sum(scores[s, length - 1] for s, length in chart) for chart in charts])
    partition = logsumexp(values)
    marginals = np.zeros_like(scores)
    for probability, chart in zip(np.exp(values - partition), charts, strict=True):
        for s, length in chart:
            marginals[s, length - 1] += probability
    return partition, marginals


@pytest.mark.parametrize("n,length,count", [(1, 1, 1), (4, 1, 4), (5, 3, 2), (5, 7, 9)])
def test_partition_and_marginals_equal_exhaustive_sets(n, length, count):
    scores = _scores(n, length)
    actual = learning.span_set_partition(scores, count)
    expected = _enumerate(scores, count)
    assert actual[0] == pytest.approx(expected[0], abs=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-12)
    assert actual[1].sum() <= count + 1e-12
    occupancy = np.zeros(n)
    for s in range(n):
        for length in range(1, min(scores.shape[1], n - s) + 1):
            occupancy[s:s + length] += actual[1][s, length - 1]
    assert np.max(occupancy) <= 1 + 1e-12


def test_background_has_one_path_and_empty_set_is_not_duplicated():
    scores = np.full((12, 1), -1000.)
    partition, marginal = learning.span_set_partition(scores, 8)
    assert partition == 0 and not np.any(marginal)
    scores = np.zeros((3, 1))
    assert learning.span_set_partition(scores, 3)[0] == pytest.approx(np.log(8))


def test_partition_gradient_matches_finite_differences_at_extreme_scores():
    scores = _scores(5, 3) * 500
    partition, marginal = learning.span_set_partition(scores, 3)
    assert np.isfinite(partition)
    for start, column in zip(*np.nonzero(np.isfinite(scores)), strict=True):
        plus, minus = scores.copy(), scores.copy()
        plus[start, column] += 1e-3
        minus[start, column] -= 1e-3
        numerical = (learning.span_set_partition(plus, 3)[0] - learning.span_set_partition(minus, 3)[0]) / 2e-3
        assert numerical == pytest.approx(marginal[start, column], abs=1e-7)


@pytest.mark.parametrize("scores,count", [(np.zeros((0, 2)), 2), (np.zeros((3, 2)), 2),
                                          (np.array([[np.nan]]), 1), (np.zeros((3, 1)), True),
                                          (np.zeros((3, 1)), 0)])
def test_invalid_partition_contract(scores, count):
    with pytest.raises(ValueError):
        learning.span_set_partition(scores, count)


def test_boundary_loss_gradient_includes_count_bias_and_fixed_scores():
    hidden = np.random.default_rng(30).normal(size=(5, 3))
    fixed = _scores(5, 3)
    starts, columns = np.nonzero(np.isfinite(fixed))
    rows = [(hidden, starts, starts + columns, fixed, ((0, 2), (3, 1)), 1.)]
    weight = np.random.default_rng(4).normal(size=7)
    center = weight / 2
    def evaluate(w):
        return learning._span_set_loss(w, rows, width=3, max_spans=3, regularization=.03, center=center)
    loss, gradient = evaluate(weight)
    assert np.isfinite(loss)
    for index in range(len(weight)):
        plus, minus = weight.copy(), weight.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        assert (evaluate(plus)[0] - evaluate(minus)[0]) / 2e-5 == pytest.approx(gradient[index], abs=1e-7)


@pytest.fixture(scope="module")
def parent():
    return fit_compositional_semantic_program_transducer(_examples(), input_grounding=_grounding())


def test_refit_exports_runtime_pointer_preserves_other_tissue_and_ignores_holdout(parent, monkeypatch):
    fit = learning.fit_span_set_pointer
    captured = []
    def inspect(training, **kwargs):
        captured.extend(training)
        assert {item.split for item in training} == {"train"}
        return fit(training, **kwargs)
    monkeypatch.setattr(learning, "fit_span_set_pointer", inspect)
    model = learning.refit_compositional_span_set_pointer(parent, _examples())
    assert len(captured) == len([item for item in _examples() if item.split == "train"])
    receipt = model.training_receipt["span_set_pointer_refit"]
    assert receipt["fit"]["converged"]
    assert receipt["fit"]["exported_loss"] < receipt["fit"]["initial_loss"]
    assert receipt["fit"]["test_examples_used"] == 0
    assert not receipt["fit"]["validation_used_for_fit"]
    assert not receipt["serving_authority"]
    assert model.operation_length_penalty == parent.operation_length_penalty
    assert model.operation_head.to_dict() == parent.operation_head.to_dict()
    assert model.argument_pointer.to_dict() == parent.argument_pointer.to_dict()
    assert compositional_semantic_program_transducer_from_dict(model.to_dict()).to_dict() == model.to_dict()
    assert not np.any(model.coefficient_lesion().operation_pointer.start_weight)


def test_training_refuses_duplicate_holdout_and_overlapping_targets(parent):
    item = next(item for item in _examples() if item.split == "train")
    kwargs = dict(spans=lambda item: (TokenSpan(0, 2),), pointer=parent.operation_pointer,
                  max_span_tokens=4, max_spans=3)
    with pytest.raises(ValueError, match="unique"):
        learning.fit_span_set_pointer((item, item), **kwargs)
    with pytest.raises(ValueError, match="source span-set"):
        learning.fit_span_set_pointer((replace(item, split="validation"),), **kwargs)
    kwargs["spans"] = lambda item: (TokenSpan(0, 2), TokenSpan(1, 3))
    with pytest.raises(ValueError, match="overlap"):
        learning.fit_span_set_pointer((item,), **kwargs)


def test_source_representation_mismatch_rejected_before_fit(parent, monkeypatch):
    from types import SimpleNamespace
    def forbidden(*args, **kwargs):
        raise AssertionError("mismatched source reached fitter")
    monkeypatch.setattr(learning, "fit_span_set_pointer", forbidden)
    with pytest.raises(ValueError, match="representation"):
        learning.refit_compositional_span_set_pointer(SimpleNamespace(
            model_basis_sha256="b" * 64, training_receipt={}), _examples())


def test_nonconverged_fit_does_not_export(parent, monkeypatch):
    from types import SimpleNamespace
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "minimize", lambda *args, **kwargs:
                        SimpleNamespace(success=False, status=1, message="iteration bound"))
    with pytest.raises(RuntimeError, match="incomplete"):
        learning.refit_compositional_span_set_pointer(parent, _examples())
