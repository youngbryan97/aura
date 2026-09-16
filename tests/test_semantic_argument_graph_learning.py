"""Argument mention scores receive the same derivatives as their runtime value."""

import numpy as np
import pytest

from core.learning.semantic_argument_graph_learning import ArgumentScoreTerm, argument_parameters
from core.learning.semantic_relation_graph_learning import RelationGraphContrast, graph_margin, graph_margin_gradient


@pytest.mark.parametrize("strategy", ["conditional_log_odds_v1", "independent_positive_v1"])
def test_argument_terms_have_exact_finite_difference_gradients(strategy):
    parameters = (np.zeros((2, 1)), np.zeros((2, 1)), np.array([.2, -.4]), np.array(.3))
    term = ArgumentScoreTerm(2, np.array([.7, -.2]), 1.3, strategy)
    row = RelationGraphContrast((), (), -.5, argument_terms=((1., term),))
    value, derivatives = graph_margin_gradient(parameters, row)
    assert value == pytest.approx(graph_margin(parameters, row))
    for index in (2, 3):
        for position in np.ndindex(parameters[index].shape):
            plus, minus = [x.copy() for x in parameters], [x.copy() for x in parameters]
            plus[index][position] += 1e-5
            minus[index][position] -= 1e-5
            numeric = (graph_margin(plus, row) - graph_margin(minus, row)) / 2e-5
            assert derivatives[index][position] == pytest.approx(numeric, abs=1e-7)


def test_runtime_argument_contrast_replays_with_trainable_mentions():
    from tests.test_semantic_relation_graph_learning import model_examples
    from core.learning.semantic_joint_graph_learning import mine_source_binding_constraint
    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    found = 0
    parameters = (model.definition_relation_head.query_projection.astype(np.float64),
                  model.definition_relation_head.definition_projection.astype(np.float64),
                  *(v.astype(np.float64) for h in model.operation_head.heads for v in (h.weight, h.bias)),
                  *argument_parameters(model))
    for item in examples:
        if item.split != "train":
            continue
        row, record = mine_source_binding_constraint(model, item, max_graphs=8, learn_arguments=True)
        if row is not None:
            found += 1
            assert row.argument_terms
            assert graph_margin(parameters, row, scale=model.definition_relation_scale) == pytest.approx(
                record["initial_margin"], abs=1e-5)
    assert found


def test_complete_constraint_fit_updates_existing_argument_heads():
    from tests.test_semantic_relation_graph_learning import model_examples
    from core.learning.semantic_graph_constraints import fit_complete_graph_constraints
    model, _examples = model_examples()
    offset = 2 + 2 * len(model.operation_head.heads)
    feature = np.zeros_like(model.argument_role_heads[0].weight, dtype=np.float64)
    feature[0] = 1.
    term = ArgumentScoreTerm(offset, feature, 1., "conditional_log_odds_v1")
    parameters = (model.definition_relation_head.query_projection, model.definition_relation_head.definition_projection,
                  *(v for h in model.operation_head.heads for v in (h.weight, h.bias)), *argument_parameters(model))
    row = RelationGraphContrast((), (), -.2 - term.score_gradient(parameters)[0], argument_terms=((1., term),))
    fitted, receipt = fit_complete_graph_constraints(model, (row,), steps=40, learning_rate=.02)
    assert receipt["stored_wrong_or_tied"] == 0
    assert not np.array_equal(fitted.argument_role_heads[0].weight, model.argument_role_heads[0].weight)
    assert fitted.input_grounding == model.input_grounding


def test_complete_source_fit_roundtrips_without_test_or_validation_training():
    from tests.test_semantic_relation_graph_learning import model_examples
    from core.learning.semantic_joint_graph_learning import refit_compositional_joint_graphs
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict
    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    fitted = refit_compositional_joint_graphs(model, examples, rounds=1, steps=2,
                                             constraint_learning=True, learn_arguments=True)
    receipt = fitted.training_receipt["joint_graph_refit"]
    assert receipt["argument_heads_trainable"]
    assert receipt["test_examples_used"] == 0 and not receipt["validation_used_for_fit"]
    assert all(row["fit"]["retained_positive_regressions"] == 0 for row in receipt["rounds"])
    assert compositional_semantic_program_transducer_from_dict(fitted.to_dict()).receipt_sha256 == fitted.receipt_sha256


@pytest.mark.parametrize("index,feature,scale,strategy", [
    (True, [1.], 1., "conditional_log_odds_v1"),
    (2, [float("nan")], 1., "conditional_log_odds_v1"),
    (2, [1.], -1., "conditional_log_odds_v1"),
    (2, [1.], 1., "unknown"),
])
def test_invalid_argument_evidence_is_rejected(index, feature, scale, strategy):
    with pytest.raises(ValueError):
        ArgumentScoreTerm(index, feature, scale, strategy)
