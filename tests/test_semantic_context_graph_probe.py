"""Predicted operations, not diagnostic targets, reach argument resolution."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.semantic_context_graph_probe import resolve_operation_set


def model():
    return SimpleNamespace(hidden_size=2, inference_step_limit=lambda _: 2,
                           _runtime_input_grounding=lambda *args: ((), (), "pointer"))


def test_only_supplied_operation_and_public_evidence_reach_assignment(monkeypatch):
    import core.learning.semantic_context_graph_probe as module

    calls = []

    def assign(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(operation_nodes=kwargs["operation_nodes"], arguments=((1, 0),))

    monkeypatch.setattr(module, "_assign_typed_arguments", assign)
    program, refusal = resolve_operation_set(
        model(), source_token_ids=(1, 2), hidden_states=np.eye(2),
        public_inputs=(7, 3), operations=((0, 1, "sub"),),
    )
    assert not refusal and program.run((7, 3)) == -4
    assert calls[0]["source_token_ids"] == (1, 2)
    assert calls[0]["inputs"] == (7, 3)
    assert calls[0]["operation_nodes"][0].operation == "sub"


def test_empty_prediction_is_not_replaced_with_a_parent_answer():
    program, reason = resolve_operation_set(
        model(), source_token_ids=(1, 2), hidden_states=np.eye(2),
        public_inputs=(7, 3), operations=(),
    )
    assert program is None and reason == "empty_predicted_operation_set"


def test_overlapping_prediction_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        resolve_operation_set(model(), source_token_ids=(1, 2), hidden_states=np.eye(2),
                              public_inputs=(7, 3), operations=((0, 2, "sub"), (1, 2, "add")))
