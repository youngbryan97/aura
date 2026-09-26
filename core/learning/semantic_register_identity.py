"""Role-relative coordinates for the existing semantic program registers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_floor import semantic_program_structural_key


@dataclass(frozen=True, slots=True)
class RegisterIdentity:
    """Separate an input's identity from a computed result's storage offset."""

    kind: str
    index: int

    def __post_init__(self):
        if self.kind not in {"input", "result"} or type(self.index) is not int or self.index < 0:
            raise ValueError("semantic register identity is invalid")

    @classmethod
    def from_absolute(cls, register: int, *, input_count: int) -> RegisterIdentity:
        if (type(input_count) is not int or input_count < 1
                or type(register) is not int or register < 0):
            raise ValueError("semantic register storage coordinate is invalid")
        return cls("input", register) if register < input_count else cls("result", register - input_count)

    def to_absolute(self, *, input_count: int, result_count: int) -> int:
        if (type(input_count) is not int or input_count < 1
                or type(result_count) is not int or result_count < 0):
            raise ValueError("semantic register environment is invalid")
        bound = input_count if self.kind == "input" else result_count
        if self.index >= bound:
            raise ValueError("semantic register identity has no available definition")
        return self.index if self.kind == "input" else input_count + self.index

    def encode(self) -> str:
        return f"{self.kind}:{self.index}"

    @classmethod
    def parse(cls, text: str) -> RegisterIdentity:
        if not isinstance(text, str) or re.fullmatch(r"(input|result):(0|[1-9][0-9]*)", text) is None:
            raise ValueError("semantic register identity differs from its canonical encoding")
        kind, index = text.split(":")
        return cls(kind, int(index))


def program_register_identities(program: Program) -> tuple[tuple[str, tuple[RegisterIdentity, ...]], ...]:
    """Expose a reversible view, retaining the same graph and execution floor."""
    if not isinstance(program, Program) or semantic_program_structural_key(program) is None:
        raise ValueError("semantic register view requires an admitted program")
    return tuple((step.op, tuple(RegisterIdentity.from_absolute(ref, input_count=program.n_inputs)
                                for ref in step.args)) for step in program.instructions)


def program_from_register_identities(
    input_count: int, steps: tuple[tuple[str, tuple[RegisterIdentity, ...]], ...],
) -> Program:
    """Resolve only prior definitions, then admit through the shared graph check."""
    if not isinstance(steps, tuple) or not steps:
        raise ValueError("semantic register view requires a nonempty instruction tuple")
    instructions = []
    for ordinal, step in enumerate(steps):
        if (not isinstance(step, tuple) or len(step) != 2 or not isinstance(step[0], str)
                or not isinstance(step[1], tuple)
                or any(not isinstance(ref, RegisterIdentity) for ref in step[1])):
            raise ValueError("semantic register instruction is invalid")
        instructions.append(Instruction(step[0], tuple(
            ref.to_absolute(input_count=input_count, result_count=ordinal) for ref in step[1])))
    program = Program(input_count, tuple(instructions))
    if semantic_program_structural_key(program) is None:
        raise ValueError("semantic register view does not define an admitted graph")
    return program
