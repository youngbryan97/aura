from __future__ import annotations

import pytest

from core.learning.semantic_program_verification import (
    recount_semantic_arm,
    recount_semantic_pair,
    semantic_program_claim_boundary,
)


def _row(identity: str, *, program: bool, answer: bool) -> dict[str, object]:
    return {
        "source_text_sha256": identity * 64,
        "accepted": program or answer,
        "program_exact": program,
        "operation_exact": program,
        "argument_exact": program,
        "input_span_exact": program,
        "attribution_exact": program,
        "full_ir_exact": program,
        "answer_emitted": program or answer,
        "answer_exact": answer,
    }


def test_recount_arm_derives_aggregates_only_from_task_rows() -> None:
    rows = [
        _row("a", program=True, answer=True),
        _row("b", program=False, answer=True),
    ]
    arm = {
        "rows": rows,
        "total": 2,
        "accepted": 2,
        "program_exact": 1,
        "operation_exact": 1,
        "argument_exact": 1,
        "input_span_exact": 1,
        "attribution_exact": 1,
        "full_ir_exact": 1,
        "answer_emitted": 2,
        "answer_exact": 2,
    }

    assert recount_semantic_arm(arm)["answer_exact"] == 2
    arm["answer_exact"] = 1
    with pytest.raises(ValueError, match="aggregate differs"):
        recount_semantic_arm(arm)


def test_recount_pair_keeps_program_and_answer_outcomes_separate() -> None:
    treatment = [
        _row("a", program=True, answer=True),
        _row("b", program=False, answer=True),
    ]
    control = [
        _row("a", program=False, answer=False),
        _row("b", program=True, answer=True),
    ]

    assert (
        recount_semantic_pair(
            treatment,
            control,
            metric="program_exact",
        )["control_only"]
        == 1
    )
    assert (
        recount_semantic_pair(
            treatment,
            control,
            metric="answer_exact",
        )["treatment_only"]
        == 1
    )


def test_recount_rejects_duplicate_task_identity() -> None:
    row = _row("a", program=True, answer=True)
    arm = {
        "rows": [row, row],
        "total": 2,
        **{
            field: 2
            for field in (
                "accepted",
                "program_exact",
                "operation_exact",
                "argument_exact",
                "input_span_exact",
                "attribution_exact",
                "full_ir_exact",
                "answer_emitted",
                "answer_exact",
            )
        },
    }

    with pytest.raises(ValueError, match="identity"):
        recount_semantic_arm(arm)


def test_claim_boundary_tracks_the_declared_semantic_family() -> None:
    arithmetic = semantic_program_claim_boundary("fork_join_4x3_factorial16")
    sequence = semantic_program_claim_boundary("sequence_chain_1x2_factorial")

    assert "synthetic arithmetic language" in arithmetic
    assert "typed sequence transformation" in sequence
    assert "not broad-domain" in sequence
    with pytest.raises(ValueError, match="unsupported"):
        semantic_program_claim_boundary("undeclared-family")
