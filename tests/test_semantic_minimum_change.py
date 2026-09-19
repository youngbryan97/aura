"""The minimum-change solver updates the same coefficients used by decoding."""

import numpy as np
import pytest

from core.learning.semantic_argument_graph_learning import ArgumentScoreTerm
from core.learning.semantic_graph_constraints import _fit_graph_parameters, fit_graph_constraints
from core.learning.semantic_joint_graph_learning import refit_compositional_joint_graphs
from core.learning.semantic_relation_graph_learning import RelationGraphContrast
from tests.test_semantic_graph_constraints import operation_constraint, simple_model
from tests.test_semantic_relation_graph_learning import model_examples


def linear(features, fixed=0.):
    return RelationGraphContrast((), (), fixed, argument_terms=tuple(
        (sign, ArgumentScoreTerm(2, np.asarray(value, dtype=float), 1., "conditional_log_odds_v1"))
        for sign, value in ((1., features), (-1., np.zeros(len(features))))))


@pytest.mark.parametrize("batched", [False, True])
def test_affine_graph_repair_reaches_the_analytic_minimum(batched):
    parameters = (np.zeros((1, 1)), np.zeros((1, 1)), np.array([1., 0.]), np.array(0.))
    values, receipt = _fit_graph_parameters(parameters,
        (linear([0., 1.], -.4), linear([1., -3.], 0.)),
        steps=8, adaptive_step=True, batched=batched, update_rule="minimum_change")
    # y >= .5, x - 3y >= .1. The nearest point to (1,0) is (1.6,.5).
    np.testing.assert_allclose(values[2], [1.6, .5], atol=1e-6)
    assert receipt["stored_wrong_or_tied"] == 0
    assert receipt["retained_positive_regressions"] == 0
    assert receipt["displacement_from_anchor"] == pytest.approx(np.sqrt(.6 ** 2 + .5 ** 2), abs=1e-6)
    assert receipt["update_rule"] == "minimum_change"
    assert not receipt["global_minimum_change_proven"]
    assert all(row["local_affine_projection"]["status"] == "verified_numerically"
               for row in receipt["accepted_steps"])


def test_minimum_change_replays_real_multiclass_scores():
    head, operation = simple_model()
    _, _, receipt = fit_graph_constraints(head, operation,
        (operation_constraint([1., 0.]), operation_constraint([0., 1.])),
        steps=10, adaptive_step=True, update_rule="minimum_change")
    assert receipt["stored_wrong_or_tied"] == 0
    assert receipt["retained_positive_regressions"] == 0


def test_minimum_change_is_bound_into_durable_checkpoint_identity(tmp_path):
    path = tmp_path / "fit.npz"
    head, operation = simple_model()
    args = head, operation, (operation_constraint([0., 1.]),)
    _, _, first = fit_graph_constraints(*args, steps=4, adaptive_step=True,
        update_rule="minimum_change", checkpoint_path=path)
    _, _, resumed = fit_graph_constraints(*args, steps=4, adaptive_step=True,
        update_rule="minimum_change", checkpoint_path=path)
    assert first == resumed
    with pytest.raises(ValueError, match="identity"):
        fit_graph_constraints(*args, steps=4, adaptive_step=True, checkpoint_path=path)


def test_source_joint_trainer_records_the_opt_in_update_rule():
    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    fitted = refit_compositional_joint_graphs(model, examples, rounds=1, steps=2,
        retention_operation_charts=2, constraint_learning=True, learn_operation_pointer=True,
        update_rule="minimum_change")
    receipt = fitted.training_receipt["joint_graph_refit"]
    assert receipt["update_rule"] == "minimum_change"
    assert not receipt["validation_used_for_fit"]
    assert fitted.operation_pointer.pair_weight is None
    assert all(row["fit"]["retained_positive_regressions"] == 0 for row in receipt["rounds"])


@pytest.mark.parametrize("options", [dict(update_rule="unknown"),
    dict(update_rule="minimum_change", objective="pairwise_logistic")])
def test_unimplemented_objectives_are_not_silently_substituted(options):
    head, operation = simple_model()
    with pytest.raises(ValueError):
        fit_graph_constraints(head, operation, (operation_constraint([0., 1.]),), **options)
