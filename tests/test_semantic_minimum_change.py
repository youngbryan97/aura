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
@pytest.mark.parametrize("adaptive", [False, True])
def test_affine_graph_repair_reaches_the_analytic_minimum(batched, adaptive):
    parameters = (np.zeros((1, 1)), np.zeros((1, 1)), np.array([1., 0.]), np.array(0.))
    values, receipt = _fit_graph_parameters(parameters,
        (linear([0., 1.], -.4), linear([1., -3.], 0.)),
        steps=8, adaptive_step=adaptive, batched=batched, update_rule="minimum_change")
    # y >= .5, x - 3y >= .1. The nearest point to (1,0) is (1.6,.5).
    np.testing.assert_allclose(values[2], [1.6, .5], atol=1e-6)
    assert receipt["stored_wrong_or_tied"] == 0
    assert receipt["retained_positive_regressions"] == 0
    assert receipt["displacement_from_anchor"] == pytest.approx(np.sqrt(.6 ** 2 + .5 ** 2), abs=1e-6)
    assert receipt["update_rule"] == "minimum_change"
    assert receipt["step_policy"] == "minimum_change_stored_working_set_v2"
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


@pytest.mark.parametrize("batched", [False, True])
def test_unreachable_constant_target_does_not_block_independent_learning(batched):
    parameters = (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), np.array(0.))
    _, receipt = _fit_graph_parameters(parameters,
        (linear([0.], .05), linear([1.], -.5)), steps=3,
        update_rule="minimum_change", batched=batched)
    assert receipt["stored_wrong_or_tied"] == 0
    assert receipt["stored_margins"][0] == .05
    assert receipt["stored_loss"] < receipt["initial_loss"]
    assert receipt["retained_positive_regressions"] == 0
    assert receipt["status"] != "retained_constraints_satisfied"
    assert receipt["accepted_steps"][0]["proposal_rule"] == (
        "retained_loss_descent_after_unverified_projection")
    assert "local_affine_projection" not in receipt["accepted_steps"][0]


def test_descent_cannot_trade_a_retained_floor_for_an_unreachable_target():
    parameters = (np.zeros((1, 1)), np.zeros((1, 1)), np.array([.2]), np.array(0.))
    _, receipt = _fit_graph_parameters(parameters,
        (linear([1.]), linear([-1.], -.5)), steps=3, update_rule="minimum_change")
    assert receipt["stored_margins"][0] >= .1
    assert receipt["stored_wrong_or_tied"] == 1
    assert receipt["retained_positive_regressions"] == 0
    assert receipt["status"] != "retained_constraints_satisfied"


@pytest.mark.parametrize("batched", [False, True])
def test_many_retained_faces_do_not_discard_a_repair_due_to_storage_rounding(batched):
    rng = np.random.default_rng(912)
    initial = rng.normal(size=128).astype(np.float32).astype(float)
    features = rng.normal(size=(48, 128))
    parameters = (np.zeros((1, 1)), np.zeros((1, 1)), initial, np.array(0.))
    constraints = tuple(linear(f, .1 - f @ initial) for f in features)
    constraints += (linear(rng.normal(size=128), -25.),)
    _, receipt = _fit_graph_parameters(parameters, constraints, steps=8,
        batched=batched, adaptive_step=True, update_rule="minimum_change")
    assert receipt["stored_wrong_or_tied"] == 0
    assert receipt["accepted_steps"]
    before, after = np.asarray(receipt["initial_margins"]), np.asarray(receipt["stored_margins"])
    assert np.all(after[:-1] >= np.minimum(before[:-1], .1))
    assert all(row["stored_affine_projection"]["stored_primal_feasible"]
               for row in receipt["accepted_steps"])


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
