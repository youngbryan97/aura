"""A negative local relation score is not proof that a dependency is invalid."""

import json
from pathlib import Path

import numpy as np
import pytest

from core.learning import semantic_program_transducer_fitting as fitting
from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
)
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import LinearPointerSequenceScores


def parent():
    path = Path(__file__).parents[1] / "artifacts/rlc/semantic_program_27b_frozen_path_v1/transducer.json"
    return compositional_semantic_program_transducer_from_dict(json.loads(path.read_text()))


def test_order_invariant_policy_is_receipt_bound_and_requires_complete_graph_search():
    original = parent()
    candidate = original.with_order_invariant_argument_graph()
    assert candidate._coefficient_body() == original._coefficient_body()
    assert candidate.receipt_sha256 != original.receipt_sha256
    assert candidate.training_receipt["forward_reference_policy"] == "joint_graph_v1"
    restored = compositional_semantic_program_transducer_from_dict(candidate.to_dict())
    assert restored.receipt_sha256 == candidate.receipt_sha256
    assert restored.to_dict() == candidate.to_dict()
    with pytest.raises(ValueError):
        candidate.with_prefix_feasible_arguments()


def test_joint_search_can_use_negative_forward_evidence_without_accepting_a_cycle(monkeypatch):
    original = parent().with_global_constraint_arguments()
    inputs = (TokenSpan(0, 1), TokenSpan(1, 2), TokenSpan(2, 3))
    reference = TokenSpan(6, 7)
    nodes = (
        fitting._OperationNode(TokenSpan(4, 5), "add", 1., 1., 1.),
        fitting._OperationNode(TokenSpan(8, 9), "at", 1., 1., 1.),
    )
    monkeypatch.setattr(fitting, "_argument_proposals_by_operation", lambda *a, **kw: (
        ((inputs[2], 0.), (reference, 0.)), ((inputs[0], 0.), (inputs[1], 0.)),
    ))
    monkeypatch.setattr(fitting, "_definition_relation_score_banks", lambda head, refs, defs, scores: (
        {span: (-2.,) * len(defs) for span in refs},
        {span: (-2.,) * len(defs) for span in refs},
    ))
    kwargs = dict(
        hidden=np.full((10, original.hidden_size), 1. / np.sqrt(original.hidden_size), dtype=np.float32),
        inputs=((4, 6), 0, 5), input_spans=inputs, operation_nodes=nodes,
        argument_pointer_scores=LinearPointerSequenceScores(np.zeros(10), np.zeros(10)),
    )
    assert fitting._assign_typed_arguments(model=original, **kwargs) is None
    result = fitting._assign_typed_arguments(
        model=original.with_order_invariant_argument_graph(), **kwargs
    )
    assert result is not None
    assert tuple(n.operation for n in result.operation_nodes) == ("at", "add")
    assert result.arguments[0] == (0, 1)
    assert set(result.arguments[1]) == {2, 3}
    assert fitting._operation_order(((1,), (0,)), nodes, require_connected=True) is None


def test_dependency_lesion_still_prevents_computed_references():
    model = parent().with_order_invariant_argument_graph().dependency_lesion()
    assert model.allow_computed_dependencies is False
    assert fitting._assign_typed_arguments(
        model=model, hidden=np.zeros((10, model.hidden_size), dtype=np.float32),
        inputs=(1, 2, 3), input_spans=(TokenSpan(0, 1), TokenSpan(1, 2), TokenSpan(2, 3)),
        operation_nodes=(
            fitting._OperationNode(TokenSpan(4, 5), "add", 1., 1., 1.),
            fitting._OperationNode(TokenSpan(8, 9), "mul", 1., 1., 1.),
        ),
        argument_pointer_scores=LinearPointerSequenceScores(np.zeros(10), np.zeros(10)),
    ) is None
