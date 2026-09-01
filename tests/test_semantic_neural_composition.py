from __future__ import annotations

import json

import pytest

from core.learning.semantic_neural_composition import (
    PUBLIC_TYPED_WORKFLOW_SCHEMA,
    compile_public_typed_workflow,
    execute_public_typed_workflow,
    render_public_typed_workflow,
)
from core.learning.semantic_neural_machine import SemanticNeuralMachine


def _document() -> dict[str, object]:
    return {
        "schema": PUBLIC_TYPED_WORKFLOW_SCHEMA,
        "initial": {"r0": 12, "r1": 5, "r2": 7, "r3": 9},
        "steps": [
            {"op": "copy", "dst": "r3", "src": "r0"},
            {"op": "mul", "dst": "r3", "factor": 4},
            {"op": "add", "dst": "r3", "left": "r3", "right": "r1"},
            {"op": "sub", "dst": "r3", "amount": 3},
            {"op": "set", "dst": "r2", "value": 5},
            {"op": "div_exact", "dst": "r0", "numerator": "r3", "denominator": "r2"},
            {"op": "set", "dst": "r1", "value": 16},
            {"op": "ratio_choice", "dst": "s0", "numerator": "r0", "denominator": "r1"},
            {"op": "ratio_band", "dst": "s0", "numerator": "r0", "denominator": "r1"},
            {"op": "euclid_step", "left": "r3", "right": "r1"},
            {"op": "euclid_step", "left": "r3", "right": "r1"},
        ],
        "report": ["r0", "r1", "r2", "r3", "s0"],
    }


def _lesion(operation: int, coefficient: int) -> SemanticNeuralMachine:
    tissue = SemanticNeuralMachine().tissue
    original = tissue.raw_coefficients[operation, coefficient]
    tissue.raw_coefficients = tissue.raw_coefficients.at[operation, coefficient].add(-original)
    return SemanticNeuralMachine(tissue)


def test_typed_workflow_recombines_existing_operations_without_an_answer_field() -> None:
    prompt = render_public_typed_workflow(_document())
    program = compile_public_typed_workflow(prompt)
    state = execute_public_typed_workflow(prompt)

    assert state.semantic_result == {"r0": 10, "r1": 0, "r2": 5, "r3": 2, "s0": 2}
    assert len(program.values) == 15
    assert program.receipt()["derived_answer_fields_present"] is False
    assert program.receipt()["correctness_authority"] is False
    assert "semantic_result" not in prompt
    assert all(receipt["teacher_available"] is False for receipt in state.transition_receipts)
    assert sum(row["learned_operation_count"] for row in state.transition_receipts) > 50


@pytest.mark.parametrize(("operation", "coefficient"), [(0, 1), (1, 2)])
def test_typed_workflow_depends_on_learned_arithmetic_tissue(
    operation: int,
    coefficient: int,
) -> None:
    prompt = render_public_typed_workflow(_document())
    with pytest.raises((RuntimeError, ValueError)):
        execute_public_typed_workflow(
            prompt,
            machine=_lesion(operation, coefficient),
        )


def test_typed_workflow_rejects_noncanonical_and_duplicate_input() -> None:
    document = _document()
    pretty = "TYPED_WORKFLOW_V1 " + json.dumps(document, indent=2)
    with pytest.raises(ValueError, match="not canonical"):
        compile_public_typed_workflow(pretty)

    duplicate = (
        'TYPED_WORKFLOW_V1 {"initial":{"r0":1,"r1":2,"r2":3,"r3":4},'
        '"report":["r0"],"schema":"aura.public_typed_workflow.v1",'
        '"steps":[{"amount":1,"amount":2,"dst":"r0","op":"sub"}]}'
    )
    with pytest.raises(ValueError, match="JSON is invalid"):
        compile_public_typed_workflow(duplicate)


def test_typed_workflow_refuses_invalid_registers_and_nonexact_division() -> None:
    document = _document()
    document["steps"] = [{"op": "copy", "dst": "r4", "src": "r0"}]
    with pytest.raises(ValueError, match="destination register"):
        render_public_typed_workflow(document)

    document = _document()
    document["steps"] = [
        {"op": "set", "dst": "r0", "value": 5},
        {"op": "set", "dst": "r1", "value": 2},
        {"op": "div_exact", "dst": "r2", "numerator": "r0", "denominator": "r1"},
    ]
    prompt = render_public_typed_workflow(document)
    with pytest.raises(ValueError, match="division is not exact"):
        execute_public_typed_workflow(prompt)
