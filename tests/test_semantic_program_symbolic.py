"""Unknown quantities admit algebra; undefined operations retain their domain."""

import itertools

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_graph_counterexamples import compare_program_meanings
from core.learning.semantic_program_symbolic import semantic_program_symbolic_key as key

I = Instruction


def distributed(operation="count_of"):
    # (f(sequence, value) + n) * m versus f(sequence, value)*m + n*m.
    return (Program(4, (I(operation, (0, 1)), I("add", (4, 2)), I("mul", (5, 3)))),
            Program(4, (I(operation, (0, 1)), I("mul", (4, 3)),
                        I("mul", (2, 3)), I("add", (5, 6)))))


@pytest.mark.parametrize("operation", ["count_of", "at", "idiv", "mod"])
def test_unknown_typed_computations_support_distributivity_without_values(operation):
    left, right = distributed(operation)
    assert key(left) is not None and key(left) == key(right)
    proof = compare_program_meanings(left, right, ())
    assert proof["status"] == "equivalent"
    assert proof["defined_domain_preserved"]
    assert proof["method"] == "floor_typed_partial_ring_v1"


def test_symbolic_proof_agrees_with_concrete_values_and_undefined_cases():
    for operation in ("idiv", "mod"):
        left, right = distributed(operation)
        for values in itertools.product(range(-2, 3), repeat=4):
            assert left.run(values) == right.run(values)
    for operation in ("count_of", "at"):
        left, right = distributed(operation)
        for seq in ((), (0,), (1, -1, 1)):
            for rest in itertools.product(range(-2, 3), repeat=3):
                assert left.run((seq, *rest)) == right.run((seq, *rest))


def test_cancelled_partial_call_is_not_promoted_to_total_zero():
    partial = Program(2, (I("idiv", (0, 1)), I("sub", (2, 2))))
    zero = Program(2, (I("sub", (0, 1)), I("sub", (2, 2))))
    assert key(partial) != key(zero)
    assert compare_program_meanings(partial, zero, ())['status'] == 'unknown'
    assert compare_program_meanings(partial, zero, ((3, 0),))['distinction'] == 'domain'


def test_congruence_handles_nested_sequence_calls_and_equal_integer_arguments():
    left = Program(3, (I("add", (1, 2)), I("at", (0, 3))))
    right = Program(3, (I("neg", (2,)), I("sub", (1, 3)), I("at", (0, 4))))
    assert key(left) == key(right)
    sequence_left = Program(2, (I("tail", (0,)), I("head", (2,)), I("add", (3, 1))))
    sequence_right = Program(2, (I("tail", (0,)), I("head", (2,)),
                                 I("neg", (1,)), I("sub", (3, 4))))
    assert key(sequence_left) == key(sequence_right)


def test_duplicate_partial_calls_have_the_same_domain_but_distinct_operators_do_not():
    left = Program(2, (I("idiv", (0, 1)), I("add", (2, 2))))
    right = Program(2, (I("idiv", (0, 1)), I("idiv", (0, 1)), I("add", (2, 3))))
    other = Program(2, (I("mod", (0, 1)), I("add", (2, 2))))
    assert key(left) == key(right)
    assert key(left) != key(other)


def test_argument_identity_and_direction_are_not_erased_by_symbol_names():
    left = Program(2, (I("idiv", (0, 1)), I("sub", (2, 2))))
    right = Program(2, (I("idiv", (1, 0)), I("sub", (2, 2))))
    assert key(left) != key(right)
    assert compare_program_meanings(left, right, ((3, 0),))["status"] == "different"


def test_sort_conflicts_and_unsupported_primitives_are_not_proofs():
    assert key(Program(1, (I("length", (0,)), I("add", (0, 1))))) is None
    assert key(Program(1, (I("uninterpreted_magic", (0,)),))) is None


def test_limits_return_unknown_not_a_false_equivalence():
    left, _right = distributed()
    assert key(left, max_terms=1) is None
    assert key(left, max_expression_chars=10) is None
    with pytest.raises(ValueError):
        key(left, max_terms=0)
