"""Auxiliary annotation retention must not masquerade as a runtime correction."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.semantic_joint_graph_learning import source_operation_constraints
from core.learning.semantic_operation_graph_learning import OperationSourceSupervision
from core.learning.semantic_program_transducer import LinearClassifierHead, MultiViewClassifierHead
from core.learning.semantic_relation_graph_learning import graph_margin


def fixture(label):
    head = LinearClassifierHead(('add', 'sub', 'mul'), np.array([[0.], [.05], [1.]]), np.zeros(3))
    model = SimpleNamespace(operation_head=MultiViewClassifierHead(('span_mean',), (head,)))
    supervision = OperationSourceSupervision((np.ones((1, 1)),), np.array([label]), np.ones(1))
    parameters = (np.zeros((1, 1)), np.zeros((1, 1)), head.weight.astype(float), head.bias.astype(float))
    return model, supervision, parameters


@pytest.mark.parametrize('label', [0, 1, 2])
def test_retention_preserves_each_original_floor_without_auxiliary_deficit(label):
    model, supervision, parameters = fixture(label)
    supervised = source_operation_constraints(model, supervision)
    retained = source_operation_constraints(model, supervision, policy='retain_existing')
    for old, new in zip(supervised, retained, strict=True):
        old_margin = graph_margin(parameters, old)
        new_margin = graph_margin(parameters, new)
        assert new_margin >= .1 - 1e-14
        assert .1 - new.fixed_margin == pytest.approx(min(old_margin, .1))
        assert old.positive_operations[0][1] == new.positive_operations[0][1]
        assert old.negative_operations[0][1] == new.negative_operations[0][1]


def test_retention_still_detects_a_worsened_auxiliary_comparison():
    model, supervision, parameters = fixture(0)
    row = source_operation_constraints(model, supervision, policy='retain_existing')[0]
    changed = (*parameters[:2], parameters[2].copy(), parameters[3])
    changed[2][0, 0] -= .2
    assert graph_margin(changed, row) < .1


def test_supervised_policy_keeps_the_original_training_objective():
    model, supervision, parameters = fixture(0)
    rows = source_operation_constraints(model, supervision, policy='supervised')
    assert all(row.fixed_margin == 0 for row in rows)
    assert all(graph_margin(parameters, row) < 0 for row in rows)


@pytest.mark.parametrize('options', [dict(policy='unknown'), dict(required_margin=0),
    dict(required_margin=float('nan')), dict(weight=-1)])
def test_invalid_retention_policy_fails(options):
    model, supervision, _ = fixture(0)
    with pytest.raises(ValueError, match='configuration'):
        source_operation_constraints(model, supervision, **options)


def test_joint_fit_records_retention_policy():
    from core.learning.semantic_joint_graph_learning import refit_compositional_joint_graphs
    from tests.test_semantic_relation_graph_learning import model_examples

    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    result = refit_compositional_joint_graphs(model, examples, constraint_learning=True,
        operation_policy='retain_existing', rounds=1, steps=1)
    assert result.training_receipt['joint_graph_refit']['operation_policy'] == 'retain_existing'
    assert not result.training_receipt['joint_graph_refit']['validation_used_for_fit']
