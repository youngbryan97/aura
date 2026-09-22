"""Input renaming preserves meaning without collapsing equal literals."""

import itertools

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_graph_coordinates import reanchor_program_inputs
from core.learning.semantic_graph_counterexamples import compare_program_meanings, counterfactual_inputs
from core.learning.semantic_program_ir import TokenSpan


def test_equal_literals_need_source_coordinates_before_counterfactual_grading():
    anchors = (TokenSpan(0, 1), TokenSpan(2, 3))
    target = Program(2, (Instruction("sub", (0, 1)),))
    decoded = Program(2, (Instruction("sub", (1, 0)),))
    inputs = (3, 3)
    probes = counterfactual_inputs(inputs)
    assert compare_program_meanings(target, decoded, probes)["status"] == "different"
    normalized = reanchor_program_inputs(decoded, from_spans=anchors[::-1],
        to_spans=anchors, from_inputs=inputs, to_inputs=inputs)
    assert normalized == target
    assert compare_program_meanings(target, normalized, probes)["status"] == "equivalent"


def test_all_input_permutations_preserve_composed_program_on_independent_values():
    anchors = tuple(TokenSpan(i, i + 1) for i in range(4))
    program = Program(4, (Instruction("sub", (0, 1)), Instruction("mul", (4, 2)),
                          Instruction("add", (5, 3))))
    original = (11, 11, 11, 11)
    for permutation in itertools.permutations(range(4)):
        destination = tuple(anchors[i] for i in permutation)
        renamed = reanchor_program_inputs(program, from_spans=anchors, to_spans=destination,
            from_inputs=original, to_inputs=original)
        for values in ((1, 4, -2, 9), (0, 1, 2, 3), (-5, -3, 7, 2)):
            assert renamed.run(tuple(values[i] for i in permutation)) == program.run(values)
        assert reanchor_program_inputs(renamed, from_spans=destination, to_spans=anchors,
            from_inputs=original, to_inputs=original) == program


@pytest.mark.parametrize("target,values", [((0, 0), (3, 3)), ((0, 2), (3, 3)),
                                         ((1, 0), (3, 4)), ((0,), (3,))])
def test_invalid_or_value_changing_alignment_is_not_scored_as_equivalent(target, values):
    anchors = tuple(TokenSpan(i, i + 1) for i in range(3))
    with pytest.raises(ValueError):
        reanchor_program_inputs(Program(2, (Instruction("sub", (0, 1)),)),
            from_spans=anchors[:2], to_spans=tuple(anchors[i] for i in target),
            from_inputs=(3, 3), to_inputs=values)


@pytest.mark.parametrize("register", [-1, 2, True])
def test_bad_register_is_not_hidden_by_renaming(register):
    anchors = (TokenSpan(0, 1), TokenSpan(2, 3))
    with pytest.raises(ValueError, match="invalid register"):
        reanchor_program_inputs(Program(2, (Instruction("sub", (0, register)),)),
            from_spans=anchors, to_spans=anchors, from_inputs=(3, 3), to_inputs=(3, 3))
