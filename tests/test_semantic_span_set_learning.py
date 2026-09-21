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
                      if all(a + b <= c for (a, b), (c, _) in zip(chart, chart[1:], strict=False)))
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


def test_training_span_inventory_matches_runtime_input_exclusions(parent, monkeypatch):
    from core.learning.semantic_program_transducer_fitting import _operation_nodes

    item = next(item for item in _examples() if item.split == "train")
    original = learning._span_set_loss
    inventories = []
    def inspect(weight, rows, **kwargs):
        inventory = {(int(start), int(end + 1)) for row in rows for start, end in zip(row[1], row[2], strict=True)}
        inventories.append(inventory)
        return original(weight, rows, **kwargs)
    monkeypatch.setattr(learning, "_span_set_loss", inspect)
    learning.fit_span_set_pointer((item,), spans=lambda item: tuple(ins.operation_span for ins in item.ir.instructions),
        pointer=parent.operation_pointer, max_span_tokens=parent.max_span_tokens, max_spans=parent.max_steps,
        excluded_spans=lambda item: item.ir.input_spans)
    nodes = _operation_nodes(pointer=parent.operation_pointer, classifier=parent.operation_head,
        hidden=item.hidden_states, input_spans=item.ir.input_spans, max_span_tokens=parent.max_span_tokens,
        hidden_channels=parent.hidden_channels, hidden_channel_widths=parent.hidden_channel_widths,
        complete_inventory=True)
    expected = {(node.span.start, node.span.end) for node in nodes}
    assert inventories and expected
    assert all(inventory == expected for inventory in inventories)


def test_excluded_target_is_rejected_before_optimization(parent):
    item = next(item for item in _examples() if item.split == "train")
    with pytest.raises(ValueError, match="target overlaps excluded"):
        learning.fit_span_set_pointer((item,), spans=lambda item: (TokenSpan(0, 2),),
            pointer=parent.operation_pointer, max_span_tokens=3, max_spans=2,
            excluded_spans=lambda item: (TokenSpan(1, 2),))


def test_conditional_label_partition_reduces_exactly_to_boundary_partition():
    from core.learning.semantic_labeled_span_learning import labeled_span_partition

    boundary = _scores(5, 3)
    boundary[1, 1] = -np.inf
    logits = np.random.default_rng(52).normal(size=(*boundary.shape, 4))
    log_probability = logits - logsumexp(logits, axis=-1, keepdims=True)
    actual_partition, actual_mass = labeled_span_partition(boundary[:, :, None] + log_probability, 3)
    expected_partition, expected_mass = learning.span_set_partition(boundary, 3)
    assert actual_partition == pytest.approx(expected_partition, abs=1e-12)
    np.testing.assert_allclose(actual_mass.sum(axis=-1), expected_mass, atol=1e-12)
    np.testing.assert_allclose(actual_mass, expected_mass[:, :, None] * np.exp(log_probability), atol=1e-12)


def test_paired_span_set_gradient_matches_numerical_derivative_with_exclusions():
    hidden = np.random.default_rng(83).normal(size=(5, 3))
    fixed = _scores(5, 3)
    fixed[1, 1] = -np.inf
    starts, columns = np.nonzero(np.isfinite(fixed))
    rows = [(hidden, starts, starts + columns, fixed, ((0, 2), (3, 1)), .7)]
    weight = np.random.default_rng(21).normal(size=10)
    def evaluate(value):
        return learning._span_set_loss(value, rows, width=3, max_spans=3,
            regularization=.03, center=weight / 2, learn_pair=True)
    loss, gradient = evaluate(weight)
    assert np.isfinite(loss)
    for index in range(len(weight)):
        plus, minus = weight.copy(), weight.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        assert (evaluate(plus)[0] - evaluate(minus)[0]) / 2e-5 == pytest.approx(gradient[index], abs=1e-7)


def test_pair_interaction_can_distinguish_crossed_boundaries_additive_scores_cannot():
    from core.learning.semantic_program_transducer import LinearPointerHead

    hidden = np.array([[-1.], [1.], [-1.], [1.]], dtype=np.float32)
    diagonal = (TokenSpan(0, 3), TokenSpan(1, 4))
    crossed = (TokenSpan(0, 4), TokenSpan(1, 3))
    additive = LinearPointerHead(np.array([2.]), .2, np.array([3.]), .4)
    scores = additive.score_sequence(hidden)
    assert sum(map(scores.score_span, diagonal)) == pytest.approx(sum(map(scores.score_span, crossed)))
    paired = replace(additive, pair_weight=np.array([1.])).score_sequence(hidden)
    assert min(map(paired.score_span, diagonal)) > min(map(scores.score_span, diagonal))
    assert sum(map(paired.score_span, diagonal)) > sum(map(paired.score_span, crossed))


def test_learned_pair_export_matches_runtime_and_retains_other_heads(parent):
    model = learning.refit_compositional_span_set_pointer(parent, _examples(), learn_pair=True)
    assert model.operation_pointer.pair_weight is not None
    fit = model.training_receipt["span_set_pointer_refit"]["fit"]
    assert fit["pair_interaction"] == "learned_sqrt_width_diagonal_product"
    assert fit["exported_loss"] < fit["initial_loss"]
    restored = compositional_semantic_program_transducer_from_dict(model.to_dict())
    assert restored.to_dict() == model.to_dict()
    assert model.operation_head.to_dict() == parent.operation_head.to_dict()
    assert model.argument_pointer.to_dict() == parent.argument_pointer.to_dict()
    item = next(item for item in _examples() if item.split == "train")
    from core.learning.semantic_paired_pointer_refit import paired_boundary_feature
    pointer = model.operation_pointer
    weight = np.concatenate((pointer.start_weight, pointer.end_weight, pointer.pair_weight))
    for ins in item.ir.instructions:
        expected = paired_boundary_feature(item.hidden_states, ins.operation_span) @ weight
        expected += pointer.start_bias + pointer.end_bias
        assert pointer.score_sequence(item.hidden_states).score_span(ins.operation_span) == pytest.approx(expected, abs=1e-5)


def test_invalid_pair_learning_switch_is_rejected(parent):
    with pytest.raises(ValueError, match="source span-set"):
        learning.refit_compositional_span_set_pointer(parent, _examples(), learn_pair=1)


def test_pair_objective_equals_enumerated_runtime_pointer_scores():
    from core.learning.semantic_program_transducer import LinearPointerHead

    hidden = np.random.default_rng(21).normal(size=(5, 3)).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    weight = np.random.default_rng(17).normal(size=10).astype(np.float32)
    pointer = LinearPointerHead(weight[:3], float(weight[-1]) / 2,
        weight[3:6], float(weight[-1]) / 2, weight[6:9])
    sequence = pointer.score_sequence(hidden)
    fixed = np.full((5, 3), -np.inf)
    runtime = fixed.copy()
    for start in range(5):
        for length in range(1, min(3, 5 - start) + 1):
            if (start, length) != (1, 2):
                fixed[start, length - 1] = -.7
                runtime[start, length - 1] = sequence.score_span(TokenSpan(start, start + length)) - .7
    starts, columns = np.nonzero(np.isfinite(fixed))
    target = ((0, 2), (3, 1))
    rows = [(hidden, starts, starts + columns, fixed, target, 1.)]
    loss, _ = learning._span_set_loss(weight.astype(np.float64), rows, width=3,
        max_spans=3, regularization=0., center=weight, learn_pair=True)
    expected = _enumerate(runtime, 3)[0] - sum(runtime[start, length - 1] for start, length in target)
    assert loss == pytest.approx(expected, abs=2e-6)


@pytest.mark.parametrize("objective,valid", [("span_set_pointer", True), ("operation_views", False)])
def test_standard_refit_routes_pair_learning_switch(parent, tmp_path, monkeypatch, capsys, objective, valid):
    import json
    import sys

    from tools import refit_semantic_argument_proposals as command

    model_path, report_path = tmp_path / "parent.json", tmp_path / "source.json"
    model_path.write_text(json.dumps(parent.to_dict()))
    report_path.write_text("{}")
    monkeypatch.setattr(command, "configure_refit_environment", lambda path: None)
    monkeypatch.setattr(command, "load_source_examples", lambda *args: _examples())
    def capture(model, examples, **options):
        assert options["learn_pair"] is True
        raise RuntimeError("pair fitter reached")
    monkeypatch.setattr(learning, "refit_compositional_span_set_pointer", capture)
    monkeypatch.setattr(sys, "argv", ["refit", "--transducer", str(model_path),
        "--source-report", str(report_path), "--bundle", "source=unused",
        "--output", str(tmp_path / "candidate.json"), "--objective", objective, "--learn-span-pairs"])
    if valid:
        with pytest.raises(RuntimeError, match="pair fitter reached"):
            command.main()
    else:
        with pytest.raises(SystemExit) as exc:
            command.main()
        assert exc.value.code == 2
        assert "span pair learning requires span_set_pointer" in capsys.readouterr().err
