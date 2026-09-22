"""Mixed semantic methods keep source positions and original alternatives."""

from types import SimpleNamespace

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_candidate_union import (
    GroundedProgram,
    UnalignableProposalError,
    unify_semantic_candidates,
)
from core.learning.semantic_program_ir import TokenSpan

SOURCE = "a" * 64
PROVENANCE = "b" * 64
LEFT = TokenSpan(1, 2)
RIGHT = TokenSpan(4, 5)


def _bank(program):
    return SimpleNamespace(
        candidates=(SimpleNamespace(program=program),),
        input_spans=(LEFT, RIGHT),
        receipt={"source_text_sha256": SOURCE, "receipt_sha256": PROVENANCE},
        validate=lambda: None,
    )


def test_union_keeps_swapped_equal_literal_roles_distinct():
    program = Program(2, (Instruction("sub", (0, 1)),))
    result = unify_semantic_candidates(
        banks={"grammar": _bank(program)},
        additional={"context": GroundedProgram(
            program, (RIGHT, LEFT), SOURCE, PROVENANCE)},
        public_inputs=(5, 5), source_sha256=SOURCE,
    )
    result.validate()
    assert len(result.candidates) == 2
    assert {row.program.instructions[0].args for row in result.candidates} == {(0, 1), (1, 0)}
    assert result.receipt["target_available"] is False
    assert result.receipt["serving_authority"] is False


def test_identical_programs_keep_both_method_origins():
    program = Program(2, (Instruction("add", (0, 1)),))
    result = unify_semantic_candidates(
        banks={"grammar": _bank(program)},
        additional={"context": GroundedProgram(
            program, (LEFT, RIGHT), SOURCE, PROVENANCE)},
        public_inputs=(2, 3), source_sha256=SOURCE,
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].origins == ("grammar:0", "context")


def test_union_rejects_changed_request_and_unmatched_grounding():
    program = Program(2, (Instruction("sub", (0, 1)),))
    args = dict(banks={"grammar": _bank(program)}, public_inputs=(2, 3),
                source_sha256=SOURCE)
    with pytest.raises(ValueError, match="another source request"):
        unify_semantic_candidates(**args, additional={"other": GroundedProgram(
            program, (LEFT, RIGHT), "c" * 64, PROVENANCE)})
    with pytest.raises(UnalignableProposalError, match="source anchors"):
        unify_semantic_candidates(**args, additional={"other": GroundedProgram(
            program, (LEFT, TokenSpan(8, 9)), SOURCE, PROVENANCE)})
