"""Learned semantic programs and the universal floor have one meaning."""

from __future__ import annotations

import hashlib

import pytest

from core.cognition.the_floor_she_stands_on import Stuck
from core.learning.procedure_induction import PRIMITIVES_BY_NAME, Primitive
from core.learning.semantic_program_execution import execute_semantic_program
from core.learning.semantic_program_floor import (
    SemanticFloorExecution,
    SemanticFloorProgram,
    compile_semantic_program_to_floor,
    execute_semantic_floor_program,
)
from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    TokenSpan,
)

SHA = "1" * 64


def _ir(
    input_count: int,
    instructions: tuple[tuple[str, tuple[int, ...]], ...],
) -> SemanticProgramIR:
    token_count = input_count + len(instructions) + 2
    return SemanticProgramIR(
        source_token_ids=tuple(range(1, token_count + 1)),
        source_text_sha256=hashlib.sha256(b"semantic floor test").hexdigest(),
        input_spans=tuple(TokenSpan(index, index + 1) for index in range(input_count)),
        instructions=tuple(
            SemanticIRInstruction(
                op=op,
                args=args,
                operation_span=TokenSpan(input_count + ordinal, input_count + ordinal + 1),
                argument_spans=tuple(TokenSpan(argument, argument + 1) for argument in args),
                depends_on=tuple(
                    sorted({argument - input_count for argument in args if argument >= input_count})
                ),
            )
            for ordinal, (op, args) in enumerate(instructions)
        ),
        report_value=input_count + len(instructions) - 1,
        model_basis_receipt_sha256=SHA,
        transducer_receipt_sha256="2" * 64,
    )


@pytest.mark.parametrize(
    ("op", "inputs"),
    [
        ("add", (7, -3)),
        ("sub", (7, -3)),
        ("mul", (-4, 6)),
        ("idiv", (-17, 5)),
        ("mod", (-17, 5)),
        ("neg", (-9,)),
        ("absv", (-9,)),
        ("length", ((4, 1, 4, -2),)),
        ("total", ((4, 1, 4, -2),)),
        ("largest", ((4, 1, 4, -2),)),
        ("smallest", ((4, 1, 4, -2),)),
        ("sorted_up", ((4, 1, 4, -2),)),
        ("reversed_", ((4, 1, 4, -2),)),
        ("head", ((4, 1, 4, -2),)),
        ("last", ((4, 1, 4, -2),)),
        ("tail", ((4, 1, 4, -2),)),
        ("front", ((4, 1, 4, -2),)),
        ("unique", ((4, 1, 4, -2),)),
        ("at", ((4, 1, 4, -2), 2)),
        ("at", ((4, 1, 4, -2), -1)),
        ("count_of", ((4, 1, 4, -2), 4)),
    ],
)
def test_every_declared_primitive_agrees_with_the_exact_executor(op, inputs) -> None:
    ir = _ir(len(inputs), ((op, tuple(range(len(inputs)))),))
    exact = execute_semantic_program(ir, inputs)
    compiled = compile_semantic_program_to_floor(ir, inputs)
    floor = execute_semantic_floor_program(compiled)

    assert floor.result == exact.result
    assert floor.receipt["execution_engine"] == "universal_metered_floor"
    assert floor.receipt["expected_answer_available"] is False


def test_multi_step_definition_references_execute_on_the_floor() -> None:
    ir = _ir(
        4,
        (
            ("mul", (0, 1)),
            ("sub", (2, 3)),
            ("add", (4, 5)),
        ),
    )
    inputs = (7, 8, 20, 3)

    exact = execute_semantic_program(ir, inputs)
    compiled = compile_semantic_program_to_floor(ir, inputs)
    floor = execute_semantic_floor_program(compiled)

    assert exact.result == 73
    assert floor.result == exact.result
    assert compiled.receipt["family_router_present"] is False
    assert compiled.receipt["alpha_normalized_sha256"] == ir.alpha_normalized_sha256


def test_sequence_to_scalar_continuation_executes_on_the_floor() -> None:
    ir = _ir(
        3,
        (
            ("at", (0, 1)),
            ("mul", (3, 2)),
        ),
    )
    inputs = ((3, 5, 8, 13), -2, 4)

    exact = execute_semantic_program(ir, inputs)
    floor = execute_semantic_floor_program(
        compile_semantic_program_to_floor(ir, inputs)
    )

    assert exact.result == 32
    assert floor.result == exact.result


def test_an_ill_typed_intermediate_is_rejected_before_floor_execution() -> None:
    ir = _ir(
        3,
        (
            ("count_of", (0, 1)),
            ("count_of", (3, 2)),
        ),
    )
    inputs = ((3, 5, 3), 3, 2)

    with pytest.raises(RuntimeError):
        execute_semantic_program(ir, inputs)
    with pytest.raises(ValueError, match="primitive signature"):
        compile_semantic_program_to_floor(ir, inputs)


@pytest.mark.parametrize(
    ("op", "inputs"),
    [
        ("idiv", (8, 0)),
        ("head", ((),)),
        ("last", ((),)),
        ("largest", ((),)),
        ("smallest", ((),)),
        ("at", ((1, 2), 5)),
        ("at", ((1, 2), -3)),
    ],
)
def test_undefined_programs_refuse_in_both_engines(op, inputs) -> None:
    ir = _ir(len(inputs), ((op, tuple(range(len(inputs)))),))

    with pytest.raises(RuntimeError):
        execute_semantic_program(ir, inputs)
    with pytest.raises((Stuck, ArithmeticError, ValueError)):
        execute_semantic_floor_program(compile_semantic_program_to_floor(ir, inputs))


def test_a_new_primitive_cannot_silently_bypass_the_floor(monkeypatch) -> None:
    monkeypatch.setitem(
        PRIMITIVES_BY_NAME,
        "some_new_primitive",
        Primitive("some_new_primitive", 1, lambda value: value),
    )
    ir = _ir(1, (("some_new_primitive", (0,)),))

    with pytest.raises(RuntimeError, match="floor coverage is incomplete"):
        compile_semantic_program_to_floor(ir, (3,))


def test_floor_receipts_reject_tampering() -> None:
    ir = _ir(2, (("add", (0, 1)),))
    program = compile_semantic_program_to_floor(ir, (2, 3))
    execution = execute_semantic_floor_program(program)

    with pytest.raises(ValueError, match="program receipt"):
        SemanticFloorProgram(
            code=program.code,
            receipt={**program.receipt, "family_router_present": True},
        )
    with pytest.raises(ValueError, match="execution receipt"):
        SemanticFloorExecution(
            result=execution.result,
            receipt={**execution.receipt, "fuel_limit": 1},
        )
