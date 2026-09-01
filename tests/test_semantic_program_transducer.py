from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from core.learning.procedure_induction import Instruction
from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    TokenSpan,
)
from core.learning.semantic_program_transducer import (
    LinearClassifierHead,
    LinearPointerHead,
    SemanticTransducerTrainingExample,
    _joint_pointer_assignment,
    _resolve_prior_result_register,
    _structured_argument_assignment,
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
        public_inputs=(12, 6, 3),
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
            public_inputs=example.public_inputs,
        )


def test_gold_programs_are_two_real_instructions() -> None:
    example = _example("sub", "idiv", 1)

    assert example.ir.to_program().instructions == (
        Instruction("sub", (1, 2)),
        Instruction("idiv", (3, 0)),
    )


_SEQUENCE_FIRST_OPS = ("unique", "sorted_up", "reversed_", "tail")
_SEQUENCE_SECOND_OPS = ("length", "total", "largest", "smallest")


def _unary_sequence_example(
    first_op: str,
    second_op: str,
    *,
    split: str = "train",
    order: tuple[int, ...] | None = None,
) -> SemanticTransducerTrainingExample:
    roles = (
        "input:0",
        "operation:0",
        "operation:1",
        "argument:0:0",
        "argument:1:0",
    )
    positions = order or tuple(range(len(roles)))
    spans = {
        role: TokenSpan(positions[index], positions[index] + 1) for index, role in enumerate(roles)
    }
    operations = (*_SEQUENCE_FIRST_OPS, *_SEQUENCE_SECOND_OPS)
    ir = SemanticProgramIR(
        source_token_ids=tuple(range(300, 310)),
        source_text_sha256=hashlib.sha256(
            f"{first_op}:{second_op}:{split}:{positions}".encode()
        ).hexdigest(),
        input_spans=(spans["input:0"],),
        instructions=(
            SemanticIRInstruction(
                first_op,
                (0,),
                spans["operation:0"],
                (spans["argument:0:0"],),
                (),
            ),
            SemanticIRInstruction(
                second_op,
                (1,),
                spans["operation:1"],
                (spans["argument:1:0"],),
                (0,),
            ),
        ),
        report_value=2,
        model_basis_receipt_sha256=_MODEL_BASIS,
        transducer_receipt_sha256="b" * 64,
    )
    width = len(roles) + len(operations) + 2 + 1
    hidden = np.zeros((10, width), dtype=np.float32)
    hidden[:, -1] = 1.0
    for role_index, role in enumerate(roles):
        position = spans[role].start
        hidden[position, -1] = 0.0
        hidden[position, role_index] = 3.0
    for step, operation in enumerate((first_op, second_op)):
        position = spans[f"operation:{step}"].start
        hidden[position, len(roles) + operations.index(operation)] = 3.0
    for step, register in enumerate((0, 1)):
        position = spans[f"argument:{step}:0"].start
        hidden[position, len(roles) + len(operations) + register] = 3.0
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    return SemanticTransducerTrainingExample(
        ir=ir,
        hidden_states=hidden,
        split=split,
        construction_id="sequence-train" if split == "train" else "sequence-held-out",
        topology_id="unary-sequence-chain",
        public_inputs=((3, 1, 3, 2),),
    )


def test_typed_transducer_learns_unary_sequence_programs() -> None:
    training = [
        _unary_sequence_example(first, second)
        for first in _SEQUENCE_FIRST_OPS
        for second in _SEQUENCE_SECOND_OPS
    ]
    model = fit_semantic_program_transducer(training)
    held_out = _unary_sequence_example(
        "unique",
        "total",
        split="test",
        order=(4, 2, 0, 3, 1),
    )

    outcome = model.decode(
        source_token_ids=held_out.ir.source_token_ids,
        hidden_states=held_out.hidden_states,
        source_text_sha256=held_out.ir.source_text_sha256,
        model_basis_sha256=_MODEL_BASIS,
    )
    replay = semantic_program_transducer_from_dict(model.to_dict())

    assert model.schema == "aura.semantic_program_transducer.v4"
    assert model.argument_arities == (1, 1)
    assert model.training_receipt["argument_arities"] == [1, 1]
    assert model.training_receipt["classifier_sharing"] == (
        "by_operation_support_and_argument_slot"
    )
    assert model.training_receipt["operation_support_by_step"] == [
        sorted(_SEQUENCE_FIRST_OPS),
        sorted(_SEQUENCE_SECOND_OPS),
    ]
    assert model.operation_heads[0].labels == tuple(sorted(_SEQUENCE_FIRST_OPS))
    assert model.operation_heads[1].labels == tuple(sorted(_SEQUENCE_SECOND_OPS))
    assert outcome.accepted
    assert outcome.ir is not None
    assert outcome.ir.to_program() == held_out.ir.to_program()
    assert replay.to_dict() == model.to_dict()


def _lexical_contextual_example(
    first_op: str,
    second_op: str,
    *,
    construction: int,
    split: str = "train",
) -> SemanticTransducerTrainingExample:
    base = _unary_sequence_example(first_op, second_op, split=split)
    width = base.hidden_states.shape[1]
    packed = np.concatenate((base.hidden_states, base.hidden_states), axis=1)
    packed /= np.sqrt(2.0)
    return replace(
        base,
        hidden_states=packed,
        construction_id=f"lexical-contextual-{construction}",
        hidden_channels=("input_token_embedding", "final_causal_hidden"),
        hidden_channel_widths=(width, width),
    )


def test_multiview_typed_transducer_learns_and_binds_its_selected_views() -> None:
    training = [
        _lexical_contextual_example(first, second, construction=construction)
        for construction in range(3)
        for first in _SEQUENCE_FIRST_OPS
        for second in _SEQUENCE_SECOND_OPS
    ]
    model = fit_semantic_program_transducer(training)
    held_out = _lexical_contextual_example(
        "unique",
        "total",
        construction=9,
        split="test",
    )
    outcome = model.decode(
        source_token_ids=held_out.ir.source_token_ids,
        hidden_states=held_out.hidden_states,
        source_text_sha256=held_out.ir.source_text_sha256,
        model_basis_sha256=_MODEL_BASIS,
    )
    payload = model.to_dict()
    replay = semantic_program_transducer_from_dict(payload)

    assert model.schema == "aura.semantic_program_transducer.v5"
    assert model.hidden_channel_widths == (training[0].hidden_states.shape[1] // 2,) * 2
    assert all(head.to_dict()["schema"].endswith(".v1") for head in model.operation_heads)
    assert all(
        item["leave_one_construction_out_total"] == len(training)
        for item in model.training_receipt["operation_feature_selection_by_step"]
    )
    assert outcome.accepted
    assert outcome.ir is not None
    assert outcome.ir.to_program() == held_out.ir.to_program()
    assert replay.to_dict() == model.to_dict()

    payload["training_receipt"]["operation_feature_selection_by_step"][0]["modes"] = ["span_mean"]
    with pytest.raises(ValueError, match="receipt does not match"):
        semantic_program_transducer_from_dict(payload)


def _three_step_example(
    operations: tuple[str, str, str],
    topology_index: int,
    *,
    split: str = "train",
    order: tuple[int, ...] | None = None,
) -> SemanticTransducerTrainingExample:
    topologies = (
        ((0, 1), (4, 2), (5, 3)),
        ((1, 2), (0, 4), (5, 3)),
        ((2, 3), (4, 1), (0, 5)),
        ((0, 3), (2, 4), (1, 5)),
    )
    arguments = topologies[topology_index]
    roles = (
        *(f"input:{index}" for index in range(4)),
        *(f"operation:{step}" for step in range(3)),
        *(f"argument:{step}:{position}" for step in range(3) for position in range(2)),
    )
    positions = order or tuple(range(len(roles)))
    if len(positions) != len(roles) or len(set(positions)) != len(positions):
        raise ValueError("test role positions are invalid")
    spans = {
        role: TokenSpan(positions[index], positions[index] + 1) for index, role in enumerate(roles)
    }
    instructions = tuple(
        SemanticIRInstruction(
            op=operations[step],
            args=arguments[step],
            operation_span=spans[f"operation:{step}"],
            argument_spans=(
                spans[f"argument:{step}:0"],
                spans[f"argument:{step}:1"],
            ),
            depends_on=tuple(
                sorted(register - 4 for register in set(arguments[step]) if register >= 4)
            ),
        )
        for step in range(3)
    )
    token_count = 18
    ir = SemanticProgramIR(
        source_token_ids=tuple(range(200, 200 + token_count)),
        source_text_sha256=hashlib.sha256(
            f"{operations}:{topology_index}:{split}".encode()
        ).hexdigest(),
        input_spans=tuple(spans[f"input:{index}"] for index in range(4)),
        instructions=instructions,
        report_value=6,
        model_basis_receipt_sha256=_MODEL_BASIS,
        transducer_receipt_sha256="b" * 64,
    )

    width = len(roles) + len(_OPERATIONS) + 7 + 1
    hidden = np.zeros((token_count, width), dtype=np.float32)
    hidden[:, -1] = 1.0
    for role_index, role in enumerate(roles):
        position = spans[role].start
        hidden[position, -1] = 0.0
        hidden[position, role_index] = 3.0
    for step, operation in enumerate(operations):
        position = spans[f"operation:{step}"].start
        hidden[position, len(roles) + _OPERATIONS.index(operation)] = 3.0
    for step, registers in enumerate(arguments):
        for position_index, register in enumerate(registers):
            position = spans[f"argument:{step}:{position_index}"].start
            hidden[position, len(roles) + len(_OPERATIONS) + register] = 3.0
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    return SemanticTransducerTrainingExample(
        ir=ir,
        hidden_states=hidden,
        split=split,
        construction_id="three-step-train" if split == "train" else "held-out",
        topology_id=f"topology-{topology_index}",
        public_inputs=(24, 6, 3, 2),
    )


def test_geometry_is_learned_for_four_inputs_and_three_steps() -> None:
    training = [
        _three_step_example((first, second, third), topology)
        for topology in range(4)
        for first, second, third in zip(
            _OPERATIONS,
            _OPERATIONS[1:] + _OPERATIONS[:1],
            _OPERATIONS[2:] + _OPERATIONS[:2],
            strict=True,
        )
    ]
    model = fit_semantic_program_transducer(training)
    held_out = _three_step_example(
        ("idiv", "sub", "mul"),
        2,
        split="test",
        order=(12, 10, 8, 6, 4, 2, 0, 11, 9, 7, 5, 3, 1),
    )

    outcome = model.decode(
        source_token_ids=held_out.ir.source_token_ids,
        hidden_states=held_out.hidden_states,
        source_text_sha256=held_out.ir.source_text_sha256,
        model_basis_sha256=_MODEL_BASIS,
    )

    assert model.schema == "aura.semantic_program_transducer.v2"
    assert (model.input_count, model.step_count) == (4, 3)
    assert model.training_receipt["input_count"] == 4
    assert model.training_receipt["step_count"] == 3
    assert model.training_receipt["classifier_sharing"] == "across_step_slots"
    assert model.training_receipt["shared_argument_support"] == [
        [0, 1, 2, 4, 5],
        [1, 2, 3, 4, 5],
    ]
    assert len(model.pointer_heads) == 13
    assert all(
        head.to_dict() == model.operation_heads[0].to_dict() for head in model.operation_heads
    )
    assert all(
        heads[position].to_dict() == model.argument_heads[0][position].to_dict()
        for heads in model.argument_heads
        for position in range(2)
    )
    assert outcome.ir is not None
    assert outcome.ir.to_program() == held_out.ir.to_program()
    assert semantic_program_transducer_from_dict(model.to_dict()).to_dict() == model.to_dict()


def test_training_refuses_mixed_program_geometry() -> None:
    mixed = _training()
    mixed.append(_three_step_example(("add", "sub", "mul"), 0))

    with pytest.raises(ValueError, match="geometries differ"):
        fit_semantic_program_transducer(mixed)


def test_direct_input_arguments_resolve_from_grounded_span_identity() -> None:
    training = [
        _three_step_example((first, second, third), topology)
        for topology in range(4)
        for first, second, third in zip(
            _OPERATIONS,
            _OPERATIONS[1:] + _OPERATIONS[:1],
            _OPERATIONS[2:] + _OPERATIONS[:2],
            strict=True,
        )
    ]
    model = fit_semantic_program_transducer(training)
    held_out = _three_step_example(("add", "sub", "mul"), 0, split="test")
    input_spans = held_out.ir.input_spans
    instructions = list(held_out.ir.instructions)
    instructions[0] = SemanticIRInstruction(
        op=instructions[0].op,
        args=(0, 1),
        operation_span=instructions[0].operation_span,
        argument_spans=(input_spans[0], input_spans[1]),
        depends_on=(),
    )
    grounded = replace(
        held_out,
        ir=SemanticProgramIR(
            source_token_ids=held_out.ir.source_token_ids,
            source_text_sha256=held_out.ir.source_text_sha256,
            input_spans=input_spans,
            instructions=tuple(instructions),
            report_value=held_out.ir.report_value,
            model_basis_receipt_sha256=held_out.ir.model_basis_receipt_sha256,
            transducer_receipt_sha256=held_out.ir.transducer_receipt_sha256,
        ),
    )

    outcome = model.decode(
        source_token_ids=grounded.ir.source_token_ids,
        hidden_states=grounded.hidden_states,
        source_text_sha256=grounded.ir.source_text_sha256,
        model_basis_sha256=_MODEL_BASIS,
    )

    assert outcome.ir is not None
    assert outcome.ir.instructions[0].args == (0, 1)


def test_prior_result_reference_resolves_from_unique_causal_definition_window() -> None:
    tokens = (10, 11, 700, 701, 20, 21, 800, 801, 30, 800, 801, 700, 701)
    operation_spans = (TokenSpan(0, 1), TokenSpan(4, 5), TokenSpan(8, 9))

    assert (
        _resolve_prior_result_register(
            token_ids=tokens,
            reference_span=TokenSpan(9, 11),
            operation_spans=operation_spans,
            current_step=2,
            input_count=4,
        )
        == 5
    )
    assert (
        _resolve_prior_result_register(
            token_ids=tokens,
            reference_span=TokenSpan(11, 13),
            operation_spans=operation_spans,
            current_step=2,
            input_count=4,
        )
        == 4
    )


def test_prior_result_reference_falls_back_when_antecedent_is_ambiguous() -> None:
    tokens = (10, 700, 20, 700, 30, 700)

    assert (
        _resolve_prior_result_register(
            token_ids=tokens,
            reference_span=TokenSpan(5, 6),
            operation_spans=(TokenSpan(0, 1), TokenSpan(2, 3), TokenSpan(4, 5)),
            current_step=2,
            input_count=4,
        )
        is None
    )


def test_pointer_candidates_preserve_the_original_best_span() -> None:
    hidden = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2**-0.5, 2**-0.5],
        ],
        dtype=np.float32,
    )
    head = LinearPointerHead(
        start_weight=np.asarray([2.0, 1.0], dtype=np.float32),
        start_bias=0.0,
        end_weight=np.asarray([1.0, 2.0], dtype=np.float32),
        end_bias=0.0,
    )

    candidates = head.decode_candidates(hidden, limit=4)

    assert candidates[0] == head.decode(hidden)
    assert len({span for span, _score in candidates}) == 4
    assert [score for _span, score in candidates] == sorted(
        (score for _span, score in candidates),
        reverse=True,
    )


def test_joint_pointer_assignment_replaces_overlapping_local_maxima() -> None:
    selected = _joint_pointer_assignment(
        (
            (
                (TokenSpan(0, 2), 10.0),
                (TokenSpan(0, 1), 9.0),
            ),
            (
                (TokenSpan(1, 3), 10.0),
                (TokenSpan(2, 3), 8.0),
            ),
        ),
        ordered=False,
    )

    assert selected == ((TokenSpan(0, 1), TokenSpan(1, 3)), (9.0, 10.0))


def test_structured_arguments_keep_every_intermediate_result_causal() -> None:
    hidden = np.zeros((15, 2), dtype=np.float32)
    hidden[:, 0] = 1.0
    classifier = LinearClassifierHead(
        labels=("0", "1", "2", "3", "4", "5"),
        weight=np.zeros((6, 2), dtype=np.float32),
        bias=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    input_spans = tuple(TokenSpan(index, index + 1) for index in range(4))
    operation_spans = (TokenSpan(4, 5), TokenSpan(8, 9), TokenSpan(12, 13))
    tokens = (10, 11, 12, 13, 20, 700, 21, 22, 30, 800, 31, 32, 40, 700, 800)
    pointer_candidates = {
        "argument:0:0": ((input_spans[0], 10.0),),
        "argument:0:1": ((input_spans[1], 10.0),),
        "argument:1:0": ((input_spans[2], 10.0),),
        "argument:1:1": ((input_spans[3], 10.0),),
        "argument:2:0": (
            (input_spans[0], 100.0),
            (TokenSpan(13, 14), 90.0),
        ),
        "argument:2:1": (
            (input_spans[1], 100.0),
            (TokenSpan(14, 15), 90.0),
        ),
    }
    pointer_head = LinearPointerHead(
        start_weight=np.asarray([1.0, 0.0], dtype=np.float32),
        start_bias=0.0,
        end_weight=np.asarray([1.0, 0.0], dtype=np.float32),
        end_bias=0.0,
    )

    assignment = _structured_argument_assignment(
        pointer_candidates=pointer_candidates,
        pointer_heads={role: pointer_head for role in pointer_candidates},
        argument_heads=((classifier, classifier),) * 3,
        hidden=hidden,
        tokens=tokens,
        input_spans=input_spans,
        operation_spans=operation_spans,
        input_count=4,
        step_count=3,
    )

    assert assignment is not None
    arguments, spans, _scores, _confidences = assignment
    assert arguments == ((0, 1), (2, 3), (4, 5))
    assert spans[2] == (TokenSpan(13, 14), TokenSpan(14, 15))
