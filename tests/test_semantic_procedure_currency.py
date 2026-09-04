"""Learned semantic programs become executable, typed common procedures."""

from __future__ import annotations

import hashlib

import pytest

from core.cognition.procedure import (
    Backend,
    Effect,
    Precondition,
    Signature,
    compose,
    reset_procedure_registry_for_test,
)
from core.learning.semantic_procedure_currency import (
    SemanticProcedureProgram,
    execute_semantic_procedure,
    from_semantic_program,
)
from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    TokenSpan,
)


def _ir(
    input_count: int,
    instructions: tuple[tuple[str, tuple[int, ...]], ...],
    *,
    source: str = "one wording",
) -> SemanticProgramIR:
    token_count = input_count + len(instructions) + 3
    return SemanticProgramIR(
        source_token_ids=tuple(range(1, token_count + 1)),
        source_text_sha256=hashlib.sha256(source.encode("ascii")).hexdigest(),
        input_spans=tuple(TokenSpan(index, index + 1) for index in range(input_count)),
        instructions=tuple(
            SemanticIRInstruction(
                op=op,
                args=args,
                operation_span=TokenSpan(
                    input_count + ordinal,
                    input_count + ordinal + 1,
                ),
                argument_spans=tuple(TokenSpan(argument, argument + 1) for argument in args),
                depends_on=tuple(
                    sorted({argument - input_count for argument in args if argument >= input_count})
                ),
            )
            for ordinal, (op, args) in enumerate(instructions)
        ),
        report_value=input_count + len(instructions) - 1,
        model_basis_receipt_sha256="1" * 64,
        transducer_receipt_sha256="2" * 64,
    )


def test_rlc_adapter_registers_a_typed_source_independent_program() -> None:
    registry = reset_procedure_registry_for_test()
    procedure = from_semantic_program(
        _ir(3, (("add", (0, 2)),)),
        input_keys=("left", "incidental", "right"),
        output_key="sum",
        observed_successes=79,
        observed_trials=96,
        registry=registry,
    )

    assert procedure.backend is Backend.RLC
    assert [(item.key, item.kind) for item in procedure.signature.preconditions] == [
        ("left", "integer"),
        ("right", "integer"),
    ]
    assert [(item.key, item.kind) for item in procedure.signature.effects] == [("sum", "integer")]
    assert procedure.origin is not None
    assert procedure.origin.rejected_conditions == ("incidental",)
    assert procedure.value.p_success == pytest.approx(79 / 96)
    assert procedure.evidence is not None and procedure.evidence.mass == 96

    stored = procedure.program
    assert isinstance(stored, SemanticProcedureProgram)
    assert stored.source_input_positions == (0, 2)
    assert stored.program.n_inputs == 2
    assert stored.program.instructions[0].args == (0, 1)
    assert stored.receipt()["source_tokens_retained"] is False
    assert stored.receipt()["family_label_present"] is False


def test_rlc_procedure_executes_on_the_universal_floor_without_old_source() -> None:
    procedure = from_semantic_program(
        _ir(
            3,
            (
                ("at", (0, 1)),
                ("mul", (3, 2)),
            ),
        ),
        input_keys=("numbers", "position", "scale"),
        output_key="answer",
        registry=reset_procedure_registry_for_test(),
    )

    execution = execute_semantic_procedure(
        procedure,
        {"numbers": (3, 5, 8, 13), "position": -2, "scale": 4},
    )

    assert execution.result == 32
    assert execution.resulting_state["answer"] == 32
    assert execution.floor_execution.receipt["execution_engine"] == "universal_metered_floor"
    assert execution.receipt["source_tokens_available"] is False


def test_two_wordings_lower_to_the_same_computation_identity() -> None:
    registry = reset_procedure_registry_for_test()
    first = from_semantic_program(
        _ir(2, (("mul", (0, 1)),), source="multiply these values"),
        registry=registry,
    )
    second = from_semantic_program(
        _ir(2, (("mul", (0, 1)),), source="find their product"),
        registry=registry,
    )

    assert first.procedure_id == second.procedure_id
    assert first.name == second.name
    assert first.program.program_sha256 == second.program.program_sha256
    assert second.evidence is not None
    assert second.evidence.independent_sources == 2
    assert registry.report()["procedures"] == 1


def test_structural_kinds_reject_present_but_wrongly_typed_state() -> None:
    procedure = from_semantic_program(
        _ir(2, (("add", (0, 1)),)),
        registry=reset_procedure_registry_for_test(),
    )

    assert procedure.signature.matches({"semantic:argument:0": 2, "semantic:argument:1": 3})
    assert not procedure.signature.matches({"semantic:argument:0": (2,), "semantic:argument:1": 3})
    assert not procedure.signature.matches({"semantic:argument:0": True, "semantic:argument:1": 3})


def test_composition_refuses_a_same_named_type_mismatch() -> None:
    registry = reset_procedure_registry_for_test()
    producer = registry.register(
        "produce sequence",
        Backend.TOOL,
        Signature(effects=(Effect("value", kind="integer_sequence"),)),
    )
    consumer = registry.register(
        "consume integer",
        Backend.TOOL,
        Signature(preconditions=(Precondition("value", kind="integer"),)),
    )

    with pytest.raises(ValueError, match="writes 'value'.*before it is read"):
        compose(registry, (producer, consumer))
    assert not consumer.signature.follows(producer.signature)


def test_an_ir_with_incompatible_internal_types_never_enters_the_registry() -> None:
    ir = _ir(
        3,
        (
            ("count_of", (0, 1)),
            ("count_of", (3, 2)),
        ),
    )

    with pytest.raises(ValueError, match="intermediate has incompatible type"):
        from_semantic_program(ir, registry=reset_procedure_registry_for_test())


def test_execution_refuses_a_state_that_does_not_match_the_typed_signature() -> None:
    procedure = from_semantic_program(
        _ir(2, (("add", (0, 1)),)),
        registry=reset_procedure_registry_for_test(),
    )

    with pytest.raises(ValueError, match="preconditions do not match"):
        execute_semantic_procedure(
            procedure,
            {"semantic:argument:0": 2, "semantic:argument:1": (3,)},
        )
