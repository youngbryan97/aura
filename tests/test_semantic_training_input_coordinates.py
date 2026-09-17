"""Equal public values retain distinct source identities in graph supervision."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.semantic_joint_graph_learning import (
    align_source_input_registers,
    mine_runtime_graph_contrast,
)
from core.learning.semantic_program_ir import SemanticIRInstruction, TokenSpan


def example():
    spans = (TokenSpan(23, 51), TokenSpan(54, 55), TokenSpan(10, 11))
    instructions = (
        SemanticIRInstruction("at", (0, 1), TokenSpan(18, 22),
                              (spans[0], spans[1]), ()),
        SemanticIRInstruction("mul", (3, 2), TokenSpan(7, 9),
                              (TokenSpan(4, 7), spans[2]), (0,)),
    )
    return SimpleNamespace(
        split="train", public_inputs=((12, 16, 18, 3, 3, 12, 4, 9), 3, 3),
        hidden_states=np.zeros((56, 2)),
        ir=SimpleNamespace(input_spans=spans, instructions=instructions,
                           source_token_ids=tuple(range(56)), source_text_sha256="a" * 64),
    )


def test_equal_literals_are_remapped_by_source_position_not_value():
    item = example()
    spans = (item.ir.input_spans[0], item.ir.input_spans[2], item.ir.input_spans[1])
    instructions, mapping = align_source_input_registers(item, spans)
    assert mapping == (0, 2, 1)
    assert tuple(ins.args for ins in instructions) == ((0, 2), (3, 1))
    assert instructions[1].depends_on == (0,)
    assert instructions[0].argument_spans == item.ir.instructions[0].argument_spans
    assert item.ir.instructions[0].args == (0, 1)


def test_grounding_mismatch_is_not_silently_relabelled():
    item = example()
    with pytest.raises(ValueError, match="anchors differ"):
        align_source_input_registers(item, (TokenSpan(1, 2), *item.ir.input_spans[1:]))
    item.public_inputs = (item.public_inputs[0], 3, 5)
    with pytest.raises(ValueError, match="public values"):
        align_source_input_registers(item, (item.ir.input_spans[0],
                                           item.ir.input_spans[2], item.ir.input_spans[1]))


def test_correct_runtime_binding_does_not_become_a_counterexample():
    item = example()
    spans = (item.ir.input_spans[0], item.ir.input_spans[2], item.ir.input_spans[1])
    instructions, _ = align_source_input_registers(item, spans)
    outcome = SimpleNamespace(ir=SimpleNamespace(instructions=instructions, input_spans=spans))
    model = SimpleNamespace(model_basis_sha256="b" * 64, decode=lambda **_: outcome)
    contrast, record = mine_runtime_graph_contrast(model, item)
    assert contrast is None
    assert record["status"] == "equivalent"
    assert record["source_to_runtime_input_registers"] == [0, 2, 1]


def test_wrong_binding_is_still_distinguished_after_coordinate_alignment(monkeypatch):
    from core.learning import semantic_joint_graph_learning as learning

    item = example()
    spans = (item.ir.input_spans[0], item.ir.input_spans[2], item.ir.input_spans[1])
    # Retaining the old indices while exchanging the input anchors is wrong,
    # even though the two literals happen to have the same observed value.
    outcome = SimpleNamespace(ir=SimpleNamespace(instructions=item.ir.instructions, input_spans=spans))
    model = SimpleNamespace(model_basis_sha256="b" * 64, decode=lambda **_: outcome)
    scored = []
    def score(_model, _item, instructions, _spans, **kwargs):
        scored.append(instructions)
        return None
    monkeypatch.setattr(learning, "score_annotated_graph", score)
    contrast, record = mine_runtime_graph_contrast(model, item)
    assert contrast is None
    assert record["comparison"]["status"] == "different"
    assert tuple(ins.args for ins in scored[1]) == ((0, 2), (3, 1))
    assert record["status"] == "target_unreachable"
