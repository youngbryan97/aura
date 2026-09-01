from __future__ import annotations

import hashlib

import pytest

from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    TokenSpan,
    semantic_program_ir_from_dict,
)


def _ir(*, shifted: int = 0) -> SemanticProgramIR:
    tokens = tuple(range(10 + shifted, 22 + shifted))
    return SemanticProgramIR(
        source_token_ids=tokens,
        source_text_sha256=hashlib.sha256(f"source-{shifted}".encode()).hexdigest(),
        input_spans=(TokenSpan(1, 2), TokenSpan(3, 4), TokenSpan(8, 9)),
        instructions=(
            SemanticIRInstruction(
                op="add",
                args=(0, 1),
                operation_span=TokenSpan(2, 3),
                argument_spans=(TokenSpan(1, 2), TokenSpan(3, 4)),
                depends_on=(),
            ),
            SemanticIRInstruction(
                op="idiv",
                args=(3, 2),
                operation_span=TokenSpan(6, 8),
                argument_spans=(TokenSpan(4, 6), TokenSpan(8, 9)),
                depends_on=(0,),
            ),
        ),
        report_value=4,
        model_basis_receipt_sha256="a" * 64,
        transducer_receipt_sha256="b" * 64,
    )


def test_ir_round_trips_and_alpha_normalizes_across_wordings() -> None:
    first = _ir()
    second = _ir(shifted=100)

    assert semantic_program_ir_from_dict(first.to_dict()) == first
    assert first.alpha_normalized_sha256 == second.alpha_normalized_sha256
    assert first.receipt()["expected_answer_available"] is False
    assert first.receipt()["generated_compiler_text_available"] is False


def test_ir_lowers_into_the_existing_neural_workflow() -> None:
    lowered = _ir().lower((12, 18, 3))

    assert lowered.program_sha == _ir().to_program().sha()
    assert lowered.output_register in {"r0", "r1", "r2", "r3"}
    assert "TYPED_WORKFLOW_V1" in lowered.public_workflow


@pytest.mark.parametrize(
    ("instructions", "message"),
    [
        (
            (
                SemanticIRInstruction(
                    "add",
                    (0, 3),
                    TokenSpan(2, 3),
                    (TokenSpan(1, 2), TokenSpan(3, 4)),
                    (),
                ),
            ),
            "forward SSA",
        ),
        (
            (
                SemanticIRInstruction(
                    "add",
                    (0, 1),
                    TokenSpan(2, 3),
                    (TokenSpan(1, 2), TokenSpan(3, 4)),
                    (),
                ),
                SemanticIRInstruction(
                    "idiv",
                    (3, 2),
                    TokenSpan(6, 8),
                    (TokenSpan(4, 6), TokenSpan(8, 9)),
                    (),
                ),
            ),
            "dependency receipt",
        ),
        (
            (
                SemanticIRInstruction(
                    "add",
                    (0, 1),
                    TokenSpan(2, 3),
                    (TokenSpan(1, 2), TokenSpan(3, 4)),
                    (),
                ),
                SemanticIRInstruction(
                    "idiv",
                    (0, 2),
                    TokenSpan(6, 8),
                    (TokenSpan(1, 2), TokenSpan(8, 9)),
                    (),
                ),
            ),
            "decorative",
        ),
    ],
)
def test_ir_rejects_invalid_causal_structure(
    instructions: tuple[SemanticIRInstruction, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SemanticProgramIR(
            source_token_ids=tuple(range(12)),
            source_text_sha256="c" * 64,
            input_spans=(TokenSpan(1, 2), TokenSpan(3, 4), TokenSpan(8, 9)),
            instructions=instructions,
            report_value=3 + len(instructions) - 1,
            model_basis_receipt_sha256="a" * 64,
            transducer_receipt_sha256="b" * 64,
        )


def test_ir_rejects_overlapping_or_out_of_range_source_pointers() -> None:
    with pytest.raises(ValueError, match="overlap"):
        SemanticProgramIR(
            source_token_ids=tuple(range(12)),
            source_text_sha256="c" * 64,
            input_spans=(TokenSpan(1, 3), TokenSpan(2, 4)),
            instructions=(
                SemanticIRInstruction(
                    "add",
                    (0, 1),
                    TokenSpan(4, 5),
                    (TokenSpan(1, 3), TokenSpan(2, 4)),
                    (),
                ),
            ),
            report_value=2,
            model_basis_receipt_sha256="a" * 64,
            transducer_receipt_sha256="b" * 64,
        )

    payload = _ir().to_dict()
    payload["instructions"][0]["operation_span"] = {"start": 20, "end": 21}
    with pytest.raises(ValueError, match="exceeds"):
        semantic_program_ir_from_dict(payload)


def test_untrusted_ir_refuses_extra_answer_bearing_fields() -> None:
    payload = _ir().to_dict()
    payload["expected_answer"] = 10

    with pytest.raises(ValueError, match="fields"):
        semantic_program_ir_from_dict(payload)


def test_ir_refuses_non_hex_receipt_identities() -> None:
    payload = _ir().to_dict()
    payload["transducer_receipt_sha256"] = "z" * 64

    with pytest.raises(ValueError, match="envelope"):
        semantic_program_ir_from_dict(payload)
