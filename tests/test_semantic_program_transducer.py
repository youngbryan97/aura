from __future__ import annotations

import hashlib

import numpy as np
import pytest

from core.learning.procedure_induction import Instruction
from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    TokenSpan,
)
from core.learning.semantic_program_transducer import (
    SemanticTransducerTrainingExample,
    fit_semantic_program_transducer,
    semantic_program_transducer_from_dict,
)

_MODEL_BASIS = "a" * 64
_ROLE_NAMES = (
    "input:0",
    "input:1",
    "input:2",
    "operation:0",
    "operation:1",
    "argument:0:0",
    "argument:0:1",
    "argument:1:0",
    "argument:1:1",
)
_OPERATIONS = ("add", "sub", "mul", "idiv")
_TOPOLOGIES = (
    ((0, 1), (3, 2), "left-01-2"),
    ((1, 2), (3, 0), "left-12-0"),
    ((0, 2), (1, 3), "right-02-1"),
    ((1, 2), (0, 3), "right-12-0"),
)


def _example(
    first_op: str,
    second_op: str,
    topology_index: int,
    *,
    split: str = "train",
    order: tuple[int, ...] | None = None,
    model_basis: str = _MODEL_BASIS,
) -> SemanticTransducerTrainingExample:
    first_args, second_args, topology_id = _TOPOLOGIES[topology_index]
    token_count = 13
    positions = order or tuple(range(len(_ROLE_NAMES)))
    if len(positions) != len(_ROLE_NAMES) or len(set(positions)) != len(positions):
        raise ValueError("test role positions are invalid")
    spans = {
        role: TokenSpan(positions[index], positions[index] + 1)
        for index, role in enumerate(_ROLE_NAMES)
    }
    instructions = (
        SemanticIRInstruction(
            op=first_op,
            args=first_args,
            operation_span=spans["operation:0"],
            argument_spans=(spans["argument:0:0"], spans["argument:0:1"]),
            depends_on=(),
        ),
        SemanticIRInstruction(
            op=second_op,
            args=second_args,
            operation_span=spans["operation:1"],
            argument_spans=(spans["argument:1:0"], spans["argument:1:1"]),
            depends_on=(0,),
        ),
    )
    ir = SemanticProgramIR(
        source_token_ids=tuple(range(100, 100 + token_count)),
        source_text_sha256=hashlib.sha256(
            f"{first_op}:{second_op}:{topology_index}:{split}".encode()
        ).hexdigest(),
        input_spans=(spans["input:0"], spans["input:1"], spans["input:2"]),
        instructions=instructions,
        report_value=4,
        model_basis_receipt_sha256=model_basis,
        transducer_receipt_sha256="b" * 64,
    )

    width = len(_ROLE_NAMES) + len(_OPERATIONS) + 4 + 1
    hidden = np.zeros((token_count, width), dtype=np.float32)
    hidden[:, -1] = 1.0
    for role_index, role in enumerate(_ROLE_NAMES):
        position = spans[role].start
        hidden[position, -1] = 0.0
        hidden[position, role_index] = 3.0
    for step, operation in enumerate((first_op, second_op)):
        position = spans[f"operation:{step}"].start
        hidden[position, len(_ROLE_NAMES) + _OPERATIONS.index(operation)] = 3.0
    for step, args in enumerate((first_args, second_args)):
        for argument_position, register in enumerate(args):
            position = spans[f"argument:{step}:{argument_position}"].start
            hidden[position, len(_ROLE_NAMES) + len(_OPERATIONS) + register] = 3.0
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    return SemanticTransducerTrainingExample(
        ir=ir,
        hidden_states=hidden,
        split=split,
        construction_id="synthetic-sequential" if split == "train" else "held-out-clause",
        topology_id=topology_id,
    )


def _training() -> list[SemanticTransducerTrainingExample]:
    return [
        _example(first, second, topology)
        for topology in range(len(_TOPOLOGIES))
        for first in _OPERATIONS
        for second in _OPERATIONS
    ]


def test_learned_heads_decode_operations_pointers_and_register_graph() -> None:
    model = fit_semantic_program_transducer(_training())
    held_out = _example(
        "sub",
        "idiv",
        2,
        split="test",
        order=(8, 6, 4, 2, 0, 7, 5, 3, 1),
    )

    outcome = model.decode(
        source_token_ids=held_out.ir.source_token_ids,
        hidden_states=held_out.hidden_states,
        source_text_sha256=held_out.ir.source_text_sha256,
        model_basis_sha256=_MODEL_BASIS,
    )

    assert outcome.accepted
    assert outcome.ir is not None
    assert outcome.ir.to_program() == held_out.ir.to_program()
    assert outcome.ir.input_spans == held_out.ir.input_spans
    assert tuple(item.operation_span for item in outcome.ir.instructions) == tuple(
        item.operation_span for item in held_out.ir.instructions
    )
    assert len(outcome.pointer_scores) == 9
    assert len(outcome.classification_confidences) == 6


def test_training_receipt_denies_answer_and_verifier_access() -> None:
    model = fit_semantic_program_transducer(_training())
    receipt = model.training_receipt

    assert receipt["training_example_count"] == 64
    assert receipt["primitive_support"] == ["add", "idiv", "mul", "sub"]
    assert receipt["expected_answers_available"] is False
    assert receipt["verifier_traces_available"] is False
    assert receipt["generated_compiler_text_available"] is False
    assert receipt["correctness_authority"] is False
    assert len(model.receipt_sha256) == 64


def test_serialized_model_round_trips_and_refuses_coefficient_drift() -> None:
    model = fit_semantic_program_transducer(_training())
    payload = model.to_dict()
    replay = semantic_program_transducer_from_dict(payload)

    assert replay.receipt_sha256 == model.receipt_sha256
    assert replay.to_dict() == payload

    payload["pointer_heads"]["input:0"]["start_weight"][0] += 0.5
    with pytest.raises(ValueError, match="receipt does not match"):
        semantic_program_transducer_from_dict(payload)


def test_decode_refuses_wrong_model_basis_or_hidden_geometry() -> None:
    model = fit_semantic_program_transducer(_training())
    example = _example("add", "mul", 0, split="test")

    wrong_basis = model.decode(
        source_token_ids=example.ir.source_token_ids,
        hidden_states=example.hidden_states,
        source_text_sha256=example.ir.source_text_sha256,
        model_basis_sha256="f" * 64,
    )
    wrong_width = model.decode(
        source_token_ids=example.ir.source_token_ids,
        hidden_states=example.hidden_states[:, :-1],
        source_text_sha256=example.ir.source_text_sha256,
        model_basis_sha256=_MODEL_BASIS,
    )

    assert wrong_basis.refusal == "model_basis_mismatch"
    assert "hidden width" in wrong_width.refusal


def test_training_refuses_mixed_model_bases_and_nonunit_evidence() -> None:
    mixed = _training()
    mixed.append(_example("add", "sub", 0, model_basis="f" * 64))
    with pytest.raises(ValueError, match="model bases"):
        fit_semantic_program_transducer(mixed)

    example = _example("add", "sub", 0)
    broken = np.array(example.hidden_states, copy=True)
    broken[0] *= 2.0
    with pytest.raises(ValueError, match="unit normalized"):
        SemanticTransducerTrainingExample(
            ir=example.ir,
            hidden_states=broken,
            split="train",
            construction_id="broken",
            topology_id="broken",
        )


def test_gold_programs_are_two_real_instructions() -> None:
    example = _example("sub", "idiv", 1)

    assert example.ir.to_program().instructions == (
        Instruction("sub", (1, 2)),
        Instruction("idiv", (3, 0)),
    )
