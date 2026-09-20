"""A hierarchical decoder cannot learn from a compensating summed margin."""

from dataclasses import replace

import numpy as np
import pytest

from core.learning.semantic_argument_graph_learning import argument_parameters
from core.learning.semantic_joint_graph_learning import (
    graph_selection_key, joint_graph_contrast, preferred_semantic_graph, refit_compositional_joint_graphs,
    score_annotated_graph, selection_graph_contrast, selection_score_margin,
)
from core.learning.semantic_operation_pointer_learning import operation_pointer_parameters
from core.learning.semantic_relation_graph_learning import graph_margin, graph_margin_gradient
from tests.test_semantic_relation_graph_learning import model_examples


@pytest.fixture(scope="module")
def evidence():
    model, examples = model_examples()
    item = next(item for item in examples if item.split == "train")
    graph = score_annotated_graph(model, item, item.ir.instructions, item.ir.input_spans,
                                 learn_arguments=True, learn_operation_pointer=True)
    assert graph is not None
    parameters = (model.definition_relation_head.query_projection.astype(np.float64),
                  model.definition_relation_head.definition_projection.astype(np.float64),
                  *(value.astype(np.float64) for head in model.operation_head.heads
                    for value in (head.weight, head.bias)),
                  *argument_parameters(model), *operation_pointer_parameters(model))
    return model, examples, graph, parameters


def test_argument_gain_cannot_hide_wrong_operation_order(evidence):
    model, _, graph, parameters = evidence
    positive = {**graph, "score": 99., "argument_score": 100., "operation_score": -1.}
    negative = {**graph, "score": -100., "argument_score": -100., "operation_score": 0.,
                "operation_signature": (("different", 1, 2),)}
    assert positive["score"] > negative["score"]
    assert graph_selection_key(model, positive) < graph_selection_key(model, negative)
    row = selection_graph_contrast(model, positive, negative)
    assert graph_margin(parameters, row, scale=model.definition_relation_scale) == pytest.approx(-1.)
    assert selection_score_margin(model, positive, negative) == -1.
    assert not row.positive and not row.negative
    assert all(any(term is pointer for pointer in graph["operation_pointer_terms"])
               for _, term in row.argument_terms)


def test_same_chart_learns_only_binding(evidence):
    model, _, graph, parameters = evidence
    other = {**graph, "argument_score": graph["argument_score"] + 2., "score": graph["score"] + 2.}
    row = selection_graph_contrast(model, graph, other)
    assert not row.positive_operations and not row.negative_operations
    assert len(row.argument_terms) == 2 * len(graph["binding_terms"])
    value, gradients = graph_margin_gradient(parameters, row, scale=model.definition_relation_scale)
    assert value == pytest.approx(-2., abs=1e-8)
    assert all(np.count_nonzero(part) == 0 for part in gradients[2:2 + 2 * len(model.operation_head.heads)])
    assert all(np.count_nonzero(part) == 0 for part in gradients[-2:])


def test_joint_policy_keeps_its_existing_objective(evidence):
    model, _, graph, parameters = evidence
    model = model.with_joint_operation_argument_scores()
    other = {**graph, "score": graph["score"] - .7}
    expected = joint_graph_contrast(model, graph, other)
    actual = selection_graph_contrast(model, graph, other)
    assert graph_margin(parameters, actual, scale=model.definition_relation_scale) == pytest.approx(
        graph_margin(parameters, expected, scale=model.definition_relation_scale))
    assert graph_selection_key(model, graph) == (graph["score"],)


def test_primary_gradient_does_not_update_secondary_heads(evidence):
    model, _, graph, parameters = evidence
    # Shift an operation feature to produce a nonzero primary derivative.
    bank, label = graph["operations"][0]
    changed = replace(bank, features=tuple(value * .7 for value in bank.features))
    other = {**graph, "operations": ((changed, label), *graph["operations"][1:]),
             "operation_signature": (("different", 1, 2),)}
    row = selection_graph_contrast(model, graph, other)
    _, gradients = graph_margin_gradient(parameters, row, scale=model.definition_relation_scale)
    assert np.count_nonzero(gradients[0]) == np.count_nonzero(gradients[1]) == 0
    end = 2 + 2 * len(model.operation_head.heads)
    assert any(np.count_nonzero(value) for value in gradients[2:end])
    assert all(np.count_nonzero(value) == 0 for value in gradients[end:-2])
    for index in (2, 3):
        position = tuple(0 for _ in parameters[index].shape)
        plus, minus = [value.copy() for value in parameters], [value.copy() for value in parameters]
        plus[index][position] += 1e-5
        minus[index][position] -= 1e-5
        numeric = (graph_margin(plus, row, scale=model.definition_relation_scale)
                   - graph_margin(minus, row, scale=model.definition_relation_scale)) / 2e-5
        assert numeric == pytest.approx(gradients[index][position], abs=1e-6)


def test_refit_preserves_incumbent_policy_and_excludes_test_rows(evidence):
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict

    model, examples, _, _ = evidence
    candidate = refit_compositional_joint_graphs(model, examples, rounds=1, steps=2,
        constraint_learning=True, learn_arguments=True, learn_operation_pointer=True,
        retention_operation_charts=2, solve_time_limit_s=2.)
    replay = refit_compositional_joint_graphs(model,
        tuple(item for item in examples if item.split != "test"), rounds=1, steps=2,
        constraint_learning=True, learn_arguments=True, learn_operation_pointer=True,
        retention_operation_charts=2, solve_time_limit_s=2.)
    receipt = candidate.training_receipt["joint_graph_refit"]
    assert receipt["selection_policy"] == "first_feasible_v1"
    assert receipt["selection_objective"] == "runtime_policy_aligned_v1"
    assert candidate.training_receipt.get("operation_assignment_policy", "first_feasible_v1") == "first_feasible_v1"
    assert receipt["test_examples_used"] == 0 and not receipt["validation_used_for_fit"]
    assert not receipt["serving_authority"]
    assert candidate.receipt_sha256 == replay.receipt_sha256
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).receipt_sha256 == candidate.receipt_sha256


def test_primary_order_does_not_subtract_large_secondary_scores(evidence):
    model, _, graph, _ = evidence
    positive = {**graph, "score": 1e20, "argument_score": 1e20, "operation_score": 1.}
    negative = {**graph, "score": 1e20, "argument_score": 1e20, "operation_score": 2.,
                "operation_signature": (("different", 1, 2),)}
    assert graph_selection_key(model, negative) > graph_selection_key(model, positive)
    assert selection_score_margin(model, positive, negative) == -1.


def test_equivalent_binding_alternative_uses_its_actual_secondary_score(evidence):
    model, _, graph, _ = evidence
    stronger = {**graph, "argument_score": graph["argument_score"] + 3.}
    assert graph_selection_key(model, stronger) == graph_selection_key(model, graph)
    assert preferred_semantic_graph(model, stronger, graph)
    assert not preferred_semantic_graph(model, graph, stronger)


def test_trial_can_mine_source_errors_without_switching_decoder_policy(evidence):
    from core.learning.semantic_graph_trial import run_semantic_graph_trial

    model, examples, _, _ = evidence
    result = run_semantic_graph_trial(model, examples, training_count=1, validation_count=1,
                                     training_pool_count=2, operation_retention_count=2,
                                     steps=1, max_charts=1)
    assert result["decoder_selection_policy"] == "first_feasible_v1"
    assert result["selection_objective"] == "runtime_policy_aligned_v1"
    assert not result["validation_used_for_fit"]
