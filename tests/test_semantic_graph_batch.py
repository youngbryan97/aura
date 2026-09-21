"""Batched graph calculations preserve runtime values and derivatives."""

from dataclasses import replace

import numpy as np
import pytest

from core.learning.semantic_argument_graph_learning import ArgumentScoreTerm
from core.learning.semantic_graph_batch import GraphConstraintBatch
from core.learning.semantic_operation_graph_learning import OperationEvidenceBank
from core.learning.semantic_relation_graph_learning import RelationGraphContrast, graph_margin, graph_margin_gradient
from tests.test_semantic_relation_graph_learning import fixture


def problem(normalizer=None):
    query, definition, relation = fixture()
    rng = np.random.default_rng(70)
    parameters = (query, definition, rng.normal(size=(3, 5)), rng.normal(size=3),
                  rng.normal(size=(3, 4)), rng.normal(size=3), rng.normal(size=6), np.array(.3))
    banks = [OperationEvidenceBank((rng.normal(size=5), rng.normal(size=4)), normalizer) for _ in range(5)]
    term = ArgumentScoreTerm(6, rng.normal(size=6), .7, "independent_positive_v1")
    rows = tuple(replace(relation, fixed_margin=i * .01,
                         positive_operations=((banks[i % 5], i % 3),),
                         negative_operations=((banks[(i + 1) % 5], (i + 1) % 3),),
                         argument_terms=((1. if i % 2 else -1., term),)) for i in range(20))
    return parameters, rows


@pytest.mark.parametrize("chunk_bytes", [1, 256, 1_000_000])
@pytest.mark.parametrize("normalizer", [None, 2])
def test_values_gradients_and_direction_agree_across_chunk_sizes(chunk_bytes, normalizer):
    parameters, rows = problem(normalizer)
    batch = GraphConstraintBatch(rows, scale=1.7, max_feature_bytes=chunk_bytes)
    coefficients = np.linspace(-1., 2., len(rows))
    direction = tuple(np.ones_like(value) * .07 for value in parameters)
    reference = [graph_margin_gradient(parameters, row, scale=1.7) for row in rows]
    np.testing.assert_allclose(batch.margins(parameters), [value for value, _ in reference], atol=1e-12)
    gradients = batch.weighted_gradient(parameters, coefficients)
    for index, value in enumerate(gradients):
        expected = sum(coefficients[row] * reference[row][1][index] for row in range(len(rows)))
        np.testing.assert_allclose(value, expected, atol=1e-12)
    expected = [sum(np.sum(a * b) for a, b in zip(gradient, direction, strict=True)) for _, gradient in reference]
    np.testing.assert_allclose(batch.directional_derivative(parameters, direction), expected, atol=1e-12)
    epsilon = 1e-5
    plus = tuple(a + epsilon * b for a, b in zip(parameters, direction, strict=True))
    minus = tuple(a - epsilon * b for a, b in zip(parameters, direction, strict=True))
    np.testing.assert_allclose(batch.directional_derivative(parameters, direction),
                               (batch.margins(plus) - batch.margins(minus)) / (2 * epsilon), atol=1e-7)


def test_probability_clamp_has_zero_derivative_and_uses_mean_of_probabilities():
    parameters = (np.zeros((1, 1)), np.zeros((1, 1)), np.array([[1000.], [-1000.]]), np.zeros(2),
                  np.array([[800.], [-800.]]), np.zeros(2))
    bank = OperationEvidenceBank((np.ones(1), np.ones(1)))
    rows = (RelationGraphContrast((), (), 0., positive_operations=((bank, 1),)),)
    batch = GraphConstraintBatch(rows)
    assert batch.margins(parameters)[0] == pytest.approx(np.log(1e-12))
    assert all(np.count_nonzero(value) == 0 for value in batch.weighted_gradient(parameters, [1.]))
    np.testing.assert_array_equal(batch.directional_derivative(parameters, parameters), [0.])
    parameters = (*parameters[:4], -parameters[4], parameters[5])
    assert batch.margins(parameters)[0] == pytest.approx(np.log(.5))


def test_empty_operations_and_zero_coefficients():
    parameters, rows = problem()
    rows = tuple(replace(row, positive_operations=(), negative_operations=()) for row in rows)
    batch = GraphConstraintBatch(rows)
    np.testing.assert_allclose(batch.margins(parameters), [graph_margin(parameters, row) for row in rows])
    assert all(np.count_nonzero(value) == 0 for value in batch.weighted_gradient(parameters, np.zeros(len(rows))))


@pytest.mark.parametrize("score", [100., -100., 1e16, -1e16])
@pytest.mark.parametrize("reverse", [False, True])
def test_shared_scores_cancel_without_erasing_the_retained_margin(score, reverse):
    parameters = (np.zeros((1, 1)), np.zeros((1, 1)), np.array([score]), np.array(0.))
    term = ArgumentScoreTerm(2, np.ones(1), 1., "conditional_log_odds_v1")
    terms = ((1., term), (-1., term))
    row = RelationGraphContrast((), (), .1, argument_terms=terms[::-1] if reverse else terms)
    assert graph_margin(parameters, row) == .1
    margin, gradient = graph_margin_gradient(parameters, row)
    assert margin == .1
    assert all(np.count_nonzero(part) == 0 for part in gradient)
    assert GraphConstraintBatch((row,)).margins(parameters)[0] == .1


@pytest.mark.parametrize("batched", [False, True])
def test_constant_cancelling_witness_does_not_block_an_independent_repair(batched):
    from core.learning.semantic_graph_constraints import _fit_graph_parameters

    parameters = (np.zeros((1, 1)), np.zeros((1, 1)), np.array([100., 0.]), np.array(0.))
    constant = ArgumentScoreTerm(2, np.array([1., 0.]), 1., "conditional_log_odds_v1")
    learned = ArgumentScoreTerm(2, np.array([0., 1.]), 1., "conditional_log_odds_v1")
    zero = ArgumentScoreTerm(2, np.zeros(2), 1., "conditional_log_odds_v1")
    rows = (RelationGraphContrast((), (), .1, argument_terms=((1., constant), (-1., constant))),
            RelationGraphContrast((), (), -.5, argument_terms=((1., learned), (-1., zero))))
    _, receipt = _fit_graph_parameters(parameters, rows, batched=batched,
        steps=2, update_rule="minimum_change")
    assert receipt["status"] == "retained_constraints_satisfied"
    assert receipt["initial_margins"][0] == receipt["stored_margins"][0] == .1
    assert receipt["stored_margins"][1] >= .1
    assert receipt["retained_positive_regressions"] == 0


def test_shared_bank_is_retained_once_and_invalid_configuration_rejected():
    parameters, rows = problem()
    batch = GraphConstraintBatch(rows, max_feature_bytes=1)
    assert len(batch.banks) == 5 and len(batch.chunks) == 5
    with pytest.raises(ValueError, match="coefficients"):
        batch.weighted_gradient(parameters, [1.])
    with pytest.raises(ValueError, match="direction"):
        batch.directional_derivative(parameters, parameters[:2])
    with pytest.raises(ValueError, match="configuration"):
        GraphConstraintBatch(rows, max_feature_bytes=0)


def test_relation_choices_share_projection_and_accumulate_before_gradient(monkeypatch):
    from core.learning.semantic_relation_graph_learning import RelationEvidenceBank

    rng = np.random.default_rng(291)
    bank = RelationEvidenceBank(rng.normal(size=5), rng.normal(size=(7, 5)), rng.normal(size=7))
    parameters = (rng.normal(size=(5, 3)), rng.normal(size=(5, 3)))
    rows = tuple(RelationGraphContrast(((bank, label),), ((bank, (label + 1) % 7),), .2)
                 for label in range(7))
    coefficients = rng.normal(size=7)
    expected = [graph_margin_gradient(parameters, row, scale=1.7) for row in rows]
    calls = []
    original = RelationEvidenceBank.scores

    def scores(self, *args):
        calls.append(self)
        return original(self, *args)

    monkeypatch.setattr(RelationEvidenceBank, 'scores', scores)
    batch = GraphConstraintBatch(rows, scale=1.7)
    assert len(batch.relations) == 1
    np.testing.assert_allclose(batch.margins(parameters), [value for value, _ in expected], atol=1e-12)
    assert len(calls) == 1
    for index, value in enumerate(batch.weighted_gradient(parameters, coefficients)):
        np.testing.assert_allclose(value, sum(weight * row[1][index]
            for weight, row in zip(coefficients, expected, strict=True)), atol=1e-12)
    direction = tuple(rng.normal(size=value.shape) for value in parameters)
    np.testing.assert_allclose(batch.directional_derivative(parameters, direction),
        [sum(np.sum(a * b) for a, b in zip(row[1], direction, strict=True)) for row in expected], atol=1e-12)


@pytest.mark.parametrize("adaptive", [False, True])
def test_optimizer_preserves_reference_objective_and_retention(adaptive):
    from core.learning.semantic_graph_constraints import fit_graph_constraints
    from tests.test_semantic_graph_constraints import simple_model, operation_constraint
    relation, operation = simple_model()
    rows = (operation_constraint([1., 0.]), operation_constraint([0., 1.]))
    options = dict(steps=25, learning_rate=.01, adaptive_step=adaptive)
    old = fit_graph_constraints(relation, operation, rows, batched=False, **options)
    new = fit_graph_constraints(relation, operation, rows, batched=True, **options)
    assert old[2]["status"] == new[2]["status"]
    assert new[2]["retained_positive_regressions"] == 0
    np.testing.assert_allclose(old[2]["stored_margins"], new[2]["stored_margins"], atol=1e-7)
