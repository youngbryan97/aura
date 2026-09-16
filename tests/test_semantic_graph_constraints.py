"""Per-witness retention is distinct from reducing a mean training loss."""

from dataclasses import replace

import numpy as np
import pytest

from core.learning.semantic_graph_constraints import _project_direction, fit_graph_constraints
from core.learning.semantic_joint_graph_learning import (
    mine_source_binding_constraint, source_operation_constraints, source_operation_supervision,
)
from core.learning.semantic_operation_graph_learning import OperationEvidenceBank
from core.learning.semantic_program_transducer import LinearClassifierHead, MultiViewClassifierHead
from core.learning.semantic_relation_graph_learning import (
    RelationGraphContrast, graph_margin, graph_margin_gradient,
)
from core.learning.semantic_relation_tissue import DirectionalRelationHead
from tests.test_semantic_relation_graph_learning import model_examples


def simple_model():
    relation = DirectionalRelationHead(np.zeros(6), 0., 0., np.zeros((2, 1)), np.zeros((2, 1)))
    operation = MultiViewClassifierHead(("span_mean",), (
        LinearClassifierHead(("add", "sub"), np.array([[.2, -.2], [0., 0.]]), np.zeros(2)),))
    return relation, operation


def operation_constraint(features, positive=0, weight=1.):
    bank = OperationEvidenceBank((np.array(features),))
    return RelationGraphContrast((), (), 0., weight, ((bank, positive),), ((bank, 1 - positive),))


def test_a_wrong_binding_can_be_repaired_without_sacrificing_a_correct_one():
    head, operation = simple_model()
    constraints = (operation_constraint([1., 0.]), operation_constraint([0., 1.]))
    fitted, operations, receipt = fit_graph_constraints(head, operation, constraints,
        steps=80, learning_rate=.02, required_margin=.1)
    assert receipt["initial_wrong_or_tied"] == 1
    assert receipt["stored_wrong_or_tied"] == 0
    assert receipt["retained_positive_regressions"] == 0
    assert receipt["status"] == "retained_constraints_satisfied"
    parameters = (fitted.query_projection.astype(np.float64), fitted.definition_projection.astype(np.float64),
                  *(value.astype(np.float64) for part in operations.heads for value in (part.weight, part.bias)))
    assert [graph_margin(parameters, row) for row in constraints] == pytest.approx(receipt["stored_margins"])
    assert all(graph_margin(parameters, row) >= .1 for row in constraints)


@pytest.mark.parametrize("adaptive", [False, True])
def test_conflicting_objectives_cannot_buy_a_loss_gain_by_destroying_a_satisfied_witness(adaptive):
    head, operation = simple_model()
    constraints = (operation_constraint([1., 0.], weight=.001),
                   operation_constraint([1., 0.], positive=1, weight=1000.))
    _fitted, _operations, receipt = fit_graph_constraints(head, operation, constraints, steps=20, learning_rate=.1,
                                                        adaptive_step=adaptive)
    assert receipt["stored_margins"][0] > 0.
    assert receipt["stored_wrong_or_tied"] == 1
    assert receipt["retained_positive_regressions"] == 0
    assert receipt["status"] != "retained_constraints_satisfied"
    assert not receipt["infeasibility_proven"]


def test_projection_handles_multiple_binding_constraints():
    result = _project_direction(np.array([-2., -3., 5.]), [np.array([1., 0., 0.]), np.array([0., 1., 0.])])
    np.testing.assert_allclose(result, [0., 0., 5.], atol=1e-10)


def test_value_only_replay_agrees_with_gradient_path():
    from tests.test_semantic_joint_graph_learning import operation_fixture
    from tests.test_semantic_relation_graph_learning import fixture
    query, definition, relation = fixture()
    _head, bank, operations = operation_fixture()
    row = replace(relation, positive_operations=((bank, 0),), negative_operations=((bank, 1),))
    parameters = (query, definition, *operations)
    assert graph_margin(parameters, row, scale=1.7) == pytest.approx(
        graph_margin_gradient(parameters, row, scale=1.7)[0], abs=1e-12)


def test_every_source_label_competitor_is_retained():
    model, examples = model_examples()
    source = source_operation_supervision(model, tuple(item for item in examples if item.split == "train"))
    constraints = source_operation_constraints(model, source)
    assert len(constraints) == len(source.labels) * (len(model.operation_head.labels) - 1)
    assert all(row.positive_operations[0][1] != row.negative_operations[0][1] for row in constraints)


def test_correct_source_interpretations_also_supply_binding_constraints():
    model, examples = model_examples()
    found = []
    for item in examples:
        if item.split != "train":
            continue
        contrast, record = mine_source_binding_constraint(model, item, max_graphs=8)
        if contrast is not None and record["initial_margin"] > 0:
            found.append(record)
    assert found
    assert all(row["status"] == "counterexample" for row in found)


def test_validation_cannot_enter_constraint_mining():
    model, examples = model_examples()
    with pytest.raises(ValueError, match="source training"):
        mine_source_binding_constraint(model, replace(examples[0], split="validation"))


def test_constraints_follow_current_supervision_not_an_incumbent_prediction():
    model, examples = model_examples()
    source = source_operation_supervision(model, tuple(item for item in examples if item.split == "train"))
    revised = replace(source, labels=(source.labels + 1) % len(model.operation_head.labels))
    constraints = source_operation_constraints(model, revised)
    count = len(model.operation_head.labels) - 1
    assert all(row.positive_operations[0][1] == int(revised.labels[index // count])
               for index, row in enumerate(constraints))


def test_full_constraint_training_path_keeps_test_rows_out_and_roundtrips():
    from core.learning.semantic_joint_graph_learning import refit_compositional_joint_graphs
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict
    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    candidate = refit_compositional_joint_graphs(model, examples, rounds=1, steps=2, constraint_learning=True)
    receipt = candidate.training_receipt["joint_graph_refit"]
    assert receipt["constraint_learning"] and receipt["already_correct_binding_competitors_retained"]
    assert not receipt["validation_used_for_fit"] and receipt["test_examples_used"] == 0
    assert all(row["fit"]["retained_positive_regressions"] == 0 for row in receipt["rounds"])
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).receipt_sha256 == candidate.receipt_sha256


@pytest.mark.parametrize("options", [{"required_margin": 0}, {"learning_rate": float("nan")},
                                     {"max_active": 0}, {"steps": False}])
def test_invalid_search_configuration_is_not_a_result(options):
    head, operation = simple_model()
    with pytest.raises(ValueError):
        fit_graph_constraints(head, operation, (operation_constraint([1., 0.]),), **options)
