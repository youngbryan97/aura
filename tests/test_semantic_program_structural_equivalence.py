"""Safe graph symmetries must not relabel coincidentally equal answers."""

import itertools

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_floor import (
    semantic_program_structural_key,
    semantic_programs_structurally_equivalent,
)


def program(*instructions, inputs=2):
    return Program(inputs, tuple(Instruction(op, args) for op, args in instructions))


@pytest.mark.parametrize("op", ["add", "mul"])
def test_integer_symmetry_holds_for_all_test_inputs(op):
    a, b = program((op, (0, 1))), program((op, (1, 0)))
    assert a != b
    assert semantic_programs_structurally_equivalent(a, b)
    for inputs in itertools.product(range(-5, 6), repeat=2):
        assert a.run(inputs) == b.run(inputs)


@pytest.mark.parametrize("op", ["sub", "idiv", "mod", "at", "count_of"])
def test_directional_primitives_never_acquire_argument_symmetry(op):
    assert not semantic_programs_structurally_equivalent(
        program((op, (0, 1))), program((op, (1, 0)))
    )


def test_independent_schedule_and_register_renaming_preserve_computation():
    a = program(("add", (0, 1)), ("sub", (2, 3)), ("mul", (4, 5)), inputs=4)
    b = program(("sub", (2, 3)), ("add", (1, 0)), ("mul", (4, 5)), inputs=4)
    assert semantic_programs_structurally_equivalent(a, b)
    for inputs in itertools.product(range(-2, 3), repeat=4):
        assert a.run(inputs) == b.run(inputs)


def test_changed_input_binding_is_not_hidden_by_equal_example_values():
    a = program(("sub", (0, 1)))
    b = program(("sub", (1, 0)))
    assert a.run((3, 3)) == b.run((3, 3))
    assert not semantic_programs_structurally_equivalent(a, b)


def test_dead_computation_and_invalid_references_are_not_certified():
    dead = program(("idiv", (0, 1)), ("add", (0, 1)))
    assert semantic_program_structural_key(dead) is None
    assert semantic_program_structural_key(program(("add", (0, 2)))) is None
    assert semantic_program_structural_key(program(("unknown", (0, 1)))) is None


def test_dependency_topology_is_not_replaced_by_a_bag_of_operations():
    a = program(("add", (0, 1)), ("sub", (2, 1)))
    b = program(("sub", (0, 1)), ("add", (2, 1)))
    assert not semantic_programs_structurally_equivalent(a, b)
