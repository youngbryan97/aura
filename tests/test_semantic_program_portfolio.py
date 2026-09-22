"""Candidate retention and selection never consult the evaluation answer key."""

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_portfolio import select_semantic_program_portfolio


def choose(proposals, incumbent="baseline"):
    return select_semantic_program_portfolio(
        proposals=proposals, incumbent=incumbent,
        provenance={name: str(index + 1) * 64 for index, name in enumerate(proposals)},
        public_inputs=(7, 3), observation_sha256="a" * 64)


def program(op):
    return Program(2, (Instruction(op, (0, 1)),))


def test_all_candidates_survive_and_executable_baseline_is_preserved():
    proposals = {"baseline": program("sub"), "context": program("add"), "other": program("mul")}
    result = choose(proposals)
    assert dict(result.proposals) == proposals
    assert result.selected_program == proposals["baseline"]
    assert result.decision.candidate_order == tuple(proposals)
    assert len(result.decision.comparisons) == 2
    assert dict(result.executions)["context"]["result"] == 10


def test_complementary_method_repairs_absent_baseline_without_oracle():
    result = choose({"baseline": None, "context": None, "other": program("mul")})
    assert result.decision.selected == "other"
    assert dict(result.executions)["other"]["result"] == 21
    assert len(result.proposals) == 3


def test_real_execution_failure_is_measured_not_assumed_from_program_presence():
    invalid = Program(2, (Instruction("length", (0,)),))
    result = choose({"baseline": invalid, "context": program("sub")})
    assert result.decision.selected == "context"
    assert not dict(result.executions)["baseline"]["completed"]


def test_successful_execution_does_not_prove_a_different_meaning_wrong():
    result = choose({"baseline": program("add"), "context": program("sub")})
    # There is deliberately no target argument: both are executable and the
    # selector cannot know that a user intended subtraction.
    assert result.decision.selected == "baseline"
    assert dict(result.executions)["baseline"]["result"] != dict(result.executions)["context"]["result"]


def test_no_program_is_not_labeled_a_completed_answer():
    result = choose({"baseline": None, "context": None})
    assert result.selected_program is None
    assert all(not row["completed"] for _, row in result.executions)


def test_invalid_identity_rejected():
    with pytest.raises(ValueError, match="immutable"):
        select_semantic_program_portfolio(proposals={"a": program("add")},
            provenance={"a": "unknown"}, public_inputs=(7, 3), observation_sha256="a" * 64, incumbent="a")


def test_execution_and_program_identity_are_bound_even_when_both_complete():
    first = choose({"baseline": program("add"), "context": program("sub")})
    second = choose({"baseline": program("mul"), "context": program("sub")})
    assert (first.decision.comparisons[0].receipt["evidence_sources"]
            != second.decision.comparisons[0].receipt["evidence_sources"])


def test_different_code_can_have_provably_the_same_meaning():
    reversed_add = Program(2, (Instruction("add", (1, 0)),))
    result = choose({"baseline": program("add"), "context": reversed_add})
    assert len(result.proposals) == 2
    assert result.relations[0][2]["status"] == "equivalent"


def test_same_current_answer_does_not_establish_same_meaning():
    result = select_semantic_program_portfolio(
        proposals={"baseline": program("add"), "context": program("mul")},
        incumbent="baseline", provenance={"baseline": "a" * 64, "context": "b" * 64},
        public_inputs=(2, 2), observation_sha256="c" * 64)
    assert dict(result.executions)["baseline"]["result"] == dict(result.executions)["context"]["result"]
    assert result.relations[0][2]["status"] == "different"
    assert result.decision.selected == "baseline"
