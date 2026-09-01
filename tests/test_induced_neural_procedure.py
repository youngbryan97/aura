from __future__ import annotations

import inspect
import random

import pytest

from core.learning.induced_neural_procedure import (
    execute_induced_program,
    lower_induced_program,
)
from core.learning.procedure_induction import (
    Instruction,
    ProcedureInducer,
    Program,
    TaskFamily,
    TaskInstance,
)
from core.learning.semantic_neural_machine import SemanticNeuralMachine

pytestmark = pytest.mark.unit


def _exact_ratio_family() -> TaskFamily:
    def generate(rng: random.Random) -> TaskInstance:
        denominator = rng.randint(2, 9)
        quotient = rng.randint(3, 40)
        left = rng.randint(1, quotient * denominator - 1)
        right = quotient * denominator - left
        return TaskInstance((left, right, denominator), quotient)

    return TaskFamily("test_only_exact_ratio", generate)


def _induced_program() -> Program:
    support = _exact_ratio_family().sample(16, seed=20260831)
    outcome = ProcedureInducer(max_depth=3).induce(support)
    assert outcome.program is not None
    return outcome.program


def _additive_lesion() -> SemanticNeuralMachine:
    tissue = SemanticNeuralMachine().tissue
    original = tissue.raw_coefficients[0, 1]
    tissue.raw_coefficients = tissue.raw_coefficients.at[0, 1].add(-original)
    return SemanticNeuralMachine(tissue)


def test_family_blind_induction_executes_through_learned_tissue() -> None:
    program = _induced_program()
    assert program.describe() == "idiv(add(in0, in1), in2)"
    assert ProcedureInducer(max_depth=1).induce(
        _exact_ratio_family().sample(16, seed=20260831)
    ).program is None

    fresh = _exact_ratio_family().sample(96, seed=20260901)
    for task in fresh:
        execution = execute_induced_program(program, task.inputs)
        result = execution.composition.semantic_result
        assert result == {execution.lowered.output_register: task.output}
        assert execution.lowered.receipt["family_label_available"] is False
        assert execution.lowered.receipt["expected_output_available"] is False
        assert execution.lowered.receipt["correctness_authority"] is False


def test_lowering_is_a_generic_program_interface() -> None:
    parameters = set(inspect.signature(lower_induced_program).parameters)
    assert parameters == {"program", "inputs"}
    receipt = lower_induced_program(_induced_program(), (66, 9, 5)).receipt
    encoded = repr(receipt).lower()
    assert "test_only_exact_ratio" not in encoded
    assert "expected" in encoded and receipt["expected_output_available"] is False


def test_liveness_allocator_reuses_dead_input_registers() -> None:
    program = Program(
        4,
        (
            Instruction("add", (0, 1)),
            Instruction("add", (4, 2)),
            Instruction("add", (5, 3)),
        ),
    )
    execution = execute_induced_program(program, (3, 5, 7, 11))
    result = execution.composition.semantic_result
    assert result == {execution.lowered.output_register: 26}
    assert len(execution.lowered.receipt["allocation"]) == 3


def test_unsupported_induced_primitive_refuses_instead_of_falling_back() -> None:
    program = Program(1, (Instruction("largest", (0,)),))
    with pytest.raises(ValueError, match="unsupported neural primitives: largest"):
        lower_induced_program(program, (5,))


def test_neural_execution_depends_on_learned_addition() -> None:
    program = _induced_program()
    task = _exact_ratio_family().sample(1, seed=20260902)[0]
    with pytest.raises((RuntimeError, ValueError)):
        execute_induced_program(program, task.inputs, machine=_additive_lesion())
