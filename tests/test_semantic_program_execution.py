from __future__ import annotations

import json

import pytest

from core.learning.semantic_program_execution import (
    SEMANTIC_PROGRAM_ANSWER_SCHEMA,
    execute_semantic_program,
    render_semantic_program_answer,
)
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
