from __future__ import annotations

import json

import pytest

from core.learning.semantic_program_execution import (
    SEMANTIC_PROGRAM_ANSWER_SCHEMA,
    SEMANTIC_PROGRAM_SEQUENCE_ANSWER_SCHEMA,
    execute_semantic_program,
    render_semantic_program_answer,
)
from core.learning.semantic_program_ir import SemanticIRInstruction, SemanticProgramIR, TokenSpan
from tests.test_semantic_program_transducer import _example


def test_validated_semantics_execute_and_emit_without_model_generation() -> None:
    example = _example("mul", "sub", 0, split="test")

    execution = execute_semantic_program(example.ir, example.public_inputs)
    answer = render_semantic_program_answer(execution)
    payload = json.loads(answer.removeprefix("FINAL_ANSWER:"))

    assert execution.result == example.ir.to_program().run(example.public_inputs)
    assert payload == {
        "schema": SEMANTIC_PROGRAM_ANSWER_SCHEMA,
        "result": execution.result,
        "execution_receipt_sha256": execution.receipt["receipt_sha256"],
    }
    assert execution.receipt["expected_answer_available"] is False
    assert execution.receipt["generated_text_available"] is False
    assert execution.receipt["execution_engine"] == "closed_exact_objective_program"


def test_execution_refuses_unbound_inputs_and_nonprogram_evidence() -> None:
    example = _example("add", "idiv", 0, split="test")

    with pytest.raises(ValueError, match="public inputs"):
        execute_semantic_program(example.ir, (1, 2))
    with pytest.raises(TypeError, match="validated program IR"):
        execute_semantic_program(object(), example.public_inputs)  # type: ignore[arg-type]


def test_sequence_semantics_stay_exact_through_execution_and_emission() -> None:
    ir = SemanticProgramIR(
        source_token_ids=tuple(range(12)),
        source_text_sha256="c" * 64,
        input_spans=(TokenSpan(1, 4),),
        instructions=(
            SemanticIRInstruction(
                "unique",
                (0,),
                TokenSpan(5, 6),
                (TokenSpan(1, 4),),
                (),
            ),
        ),
        report_value=1,
        model_basis_receipt_sha256="a" * 64,
        transducer_receipt_sha256="b" * 64,
    )

    execution = execute_semantic_program(ir, ((3, 1, 3, 2),))
    payload = json.loads(render_semantic_program_answer(execution).removeprefix("FINAL_ANSWER:"))

    assert execution.result == (1, 2, 3)
    assert payload["schema"] == SEMANTIC_PROGRAM_SEQUENCE_ANSWER_SCHEMA
    assert payload["result"] == [1, 2, 3]


def test_sequence_semantics_can_emit_an_exact_scalar_aggregation() -> None:
    ir = SemanticProgramIR(
        source_token_ids=tuple(range(12)),
        source_text_sha256="d" * 64,
        input_spans=(TokenSpan(1, 4),),
        instructions=(
            SemanticIRInstruction("unique", (0,), TokenSpan(5, 6), (TokenSpan(1, 4),), ()),
            SemanticIRInstruction("total", (1,), TokenSpan(8, 9), (TokenSpan(6, 8),), (0,)),
        ),
        report_value=2,
        model_basis_receipt_sha256="a" * 64,
        transducer_receipt_sha256="b" * 64,
    )

    execution = execute_semantic_program(ir, ((3, 1, 3, 2),))

    assert execution.result == 6
    assert execution.receipt["schema"] == "aura.semantic_program_execution.v1"
