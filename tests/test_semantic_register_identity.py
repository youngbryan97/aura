"""Register meaning is independent of packed input/result storage geometry."""

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_register_identity import (
    RegisterIdentity,
    program_from_register_identities,
    program_register_identities,
)


def test_result_identity_survives_changes_in_input_arity():
    for arity in (1, 3, 4, 8, 31):
        for ordinal in (0, 1, 9):
            ref = RegisterIdentity.from_absolute(arity + ordinal, input_count=arity)
            assert ref == RegisterIdentity("result", ordinal)
            assert ref.to_absolute(input_count=arity, result_count=ordinal + 1) == arity + ordinal


def test_graph_view_round_trips_and_preserves_arbitrary_branch_bindings():
    program = Program(4, (Instruction("sub", (0, 1)), Instruction("mul", (2, 3)),
                          Instruction("add", (4, 5))))
    view = program_register_identities(program)
    assert view[-1][1] == (RegisterIdentity("result", 0), RegisterIdentity("result", 1))
    assert program_from_register_identities(4, view) == program
    padded = program_from_register_identities(7, view)
    assert padded.instructions[-1].args == (7, 8)
    assert padded.run((10, 2, 3, 4, 100, 200, 300)) == program.run((10, 2, 3, 4))


@pytest.mark.parametrize("text", ["input:-1", "input:00", "result:01", "result:1.0", "step:0", "input: 1"])
def test_noncanonical_or_unknown_identity_cannot_be_admitted(text):
    with pytest.raises(ValueError, match="canonical"):
        RegisterIdentity.parse(text)


def test_identity_encoding_matches_existing_shared_transducer():
    from core.learning.semantic_program_shared_transducer import _relative_register

    for arity in (1, 3, 4, 8):
        for register in range(arity + 16):
            ref = RegisterIdentity.from_absolute(register, input_count=arity)
            assert ref.encode() == _relative_register(register, input_count=arity)
            assert RegisterIdentity.parse(ref.encode()) == ref


def test_forward_self_and_missing_input_references_are_not_repaired():
    for ref in (RegisterIdentity("result", 0), RegisterIdentity("input", 2)):
        with pytest.raises(ValueError, match="available definition"):
            program_from_register_identities(2, (("neg", (ref,)),))
    for value in (True, -1, 1.5):
        with pytest.raises(ValueError):
            RegisterIdentity("result", value)


def test_disconnected_program_is_not_granted_execution_authority():
    steps = (("add", (RegisterIdentity("input", 0), RegisterIdentity("input", 1))),
             ("mul", (RegisterIdentity("input", 0), RegisterIdentity("input", 1))))
    with pytest.raises(ValueError, match="admitted graph"):
        program_from_register_identities(2, steps)
