"""An explicit training subspace must survive storage, restoration and resume."""

import numpy as np
import pytest

from core.learning.semantic_argument_graph_learning import ArgumentScoreTerm
from core.learning.semantic_graph_constraints import _fit_graph_parameters, fit_graph_constraints
from core.learning.semantic_relation_graph_learning import RelationGraphContrast
from tests.test_semantic_graph_constraints import operation_constraint, simple_model


def linear_problem():
    parameters = (np.zeros((1, 1)), np.zeros((1, 1)),
                  np.array([.123456789]), np.array(0.), np.array([0.]), np.array(0.))
    terms = tuple((sign, ArgumentScoreTerm(block, np.array([feature]), 1., "conditional_log_odds_v1"))
                  for block in (2, 4) for sign, feature in ((1., 1.), (-1., 0.)))
    return parameters, (RelationGraphContrast((), (), -.5, argument_terms=terms),)


@pytest.mark.parametrize("update_rule", ["working_face", "minimum_change"])
@pytest.mark.parametrize("batched", [False, True])
def test_fit_changes_only_declared_blocks_at_storage_precision(update_rule, batched):
    parameters, rows = linear_problem()
    values, receipt = _fit_graph_parameters(parameters, rows, steps=8, adaptive_step=True,
        batched=batched, update_rule=update_rule,
        trainable_parameters=(False, False, False, False, True, True))
    assert receipt["stored_wrong_or_tied"] == 0
    assert receipt["frozen_parameters_unchanged"]
    assert receipt["parameter_displacements"][:4] == [0.] * 4
    for before, after in zip(parameters[:4], values[:4], strict=True):
        np.testing.assert_array_equal(before, after)
    assert values[4] > parameters[4]


@pytest.mark.parametrize("update_rule", ["working_face", "minimum_change"])
def test_frozen_only_error_is_not_reported_as_repaired(update_rule):
    relation, operation = simple_model()
    _relation, fitted, receipt = fit_graph_constraints(relation, operation,
        (operation_constraint([0., 1.]),), steps=2,
        learn_operations=False, update_rule=update_rule)
    assert receipt["stored_wrong_or_tied"] == 1
    assert receipt["status"] != "retained_constraints_satisfied"
    assert not receipt["infeasibility_proven"]
    assert receipt["frozen_parameters_unchanged"]
    for original, candidate in zip(operation.heads, fitted.heads, strict=True):
        np.testing.assert_array_equal(original.weight, candidate.weight)
        np.testing.assert_array_equal(original.bias, candidate.bias)


def test_subspace_is_bound_into_checkpoint_identity(tmp_path):
    parameters, rows = linear_problem()
    options = dict(steps=2, update_rule="minimum_change", checkpoint_path=tmp_path / "fit.npz",
                   trainable_parameters=(False, False, False, False, True, True))
    values, receipt = _fit_graph_parameters(parameters, rows, **options)
    resumed, resumed_receipt = _fit_graph_parameters(parameters, rows, **options)
    assert receipt == resumed_receipt
    for first, second in zip(values, resumed, strict=True):
        np.testing.assert_array_equal(first, second)
    options["trainable_parameters"] = (True,) * len(parameters)
    with pytest.raises(ValueError):
        _fit_graph_parameters(parameters, rows, **options)


@pytest.mark.parametrize("mask", [(), (True,), (False,) * 6, (1,) * 6])
def test_invalid_parameter_mask_is_rejected(mask):
    parameters, rows = linear_problem()
    with pytest.raises(ValueError, match="trainable parameter blocks"):
        _fit_graph_parameters(parameters, rows, trainable_parameters=mask)
