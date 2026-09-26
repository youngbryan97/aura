"""Typed native search preserves arbitrary roles, branches, and termination."""

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_native_grammar import (
    NativeGrammarIncompleteError,
    decode_native_grammar,
)


def scripted(values):
    pending = iter(values)

    def score(choices):
        wanted = next(pending)
        assert wanted in [choice.value for choice in choices]
        return tuple(0. if choice.value == wanted else -10. for choice in choices)

    return score


def test_three_step_graph_is_built_without_a_candidate_bank():
    result = decode_native_grammar(("integer",) * 4,
        scripted(["add", 0, 1, "continue", "mul", 4, 2, "continue",
                  "sub", 5, 3, "finish"]), max_steps=8)
    assert result.program == Program(4, (Instruction("add", (0, 1)),
        Instruction("mul", (4, 2)), Instruction("sub", (5, 3))))
    assert not result.bound_forced_completion
    assert len(result.trace) == 12


def test_independent_branches_can_join_but_cannot_finish_disconnected():
    result = decode_native_grammar(("integer",) * 4,
        scripted(["sub", 0, 1, "continue", "mul", 2, 3, "continue",
                  "add", 4, 5, "finish"]), max_steps=8)
    assert result.program.instructions[-1] == Instruction("add", (4, 5))
    second_stop = [row for row in result.trace if row["kind"] == "termination"][1]
    assert second_stop["choices"] == ["continue"]


def test_sequence_roles_are_type_constrained_and_unary_primitives_supported():
    result = decode_native_grammar(("integer_sequence", "integer"),
        scripted(["at", 0, 1, "continue", "neg", 2, "finish"]), max_steps=8)
    assert result.program.run(((4, 8), 1)) == -8
    refs = [row for row in result.trace if row["kind"] == "reference"]
    assert refs[0]["choices"] == [0]
    assert refs[1]["choices"] == [1]


def test_bound_forced_completion_is_not_reported_as_model_halt():
    result = decode_native_grammar(("integer",) * 2,
                                   scripted(["add", 0, 1]), max_steps=1)
    assert result.bound_forced_completion
    assert not any(row["kind"] == "termination" for row in result.trace)


def test_bad_scores_have_no_execution_authority():
    with pytest.raises(ValueError, match="finite"):
        decode_native_grammar(("integer",), lambda choices: (float("nan"),) * len(choices))
    with pytest.raises(ValueError, match="input types"):
        decode_native_grammar(("unknown",), lambda choices: ())


def test_incomplete_graph_retains_decisions_but_does_not_return_an_executable_result():
    with pytest.raises(NativeGrammarIncompleteError) as caught:
        decode_native_grammar(("integer",) * 4,
            scripted(["sub", 0, 1, "continue", "mul", 2, 3]), max_steps=2)
    assert caught.value.program.depth == 2
    assert len(caught.value.trace) == 7
