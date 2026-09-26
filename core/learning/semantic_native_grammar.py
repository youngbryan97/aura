"""Target-blind native decisions inside the execution floor's typed SSA grammar."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

from core.learning.procedure_induction import PRIMITIVES_BY_NAME, Instruction, Program
from core.learning.semantic_native_program import parse_native_program
from core.learning.semantic_native_relative_program import (
    REGISTER_ENCODING,
    parse_relative_native_program,
)
from core.learning.semantic_program_floor import (
    semantic_primitive_type_signature,
    semantic_program_structural_key,
)
from core.learning.semantic_register_identity import RegisterIdentity


@dataclass(frozen=True)
class NativeGrammarDecision:
    text: str
    span: tuple[int, int]
    value: str | int


@dataclass(frozen=True)
class NativeGrammarResult:
    program: Program
    trace: tuple[dict, ...]
    bound_forced_completion: bool


class NativeGrammarIncompleteError(ValueError):
    """A finite search ended without a connected executable graph."""

    def __init__(self, program: Program, trace: tuple[dict, ...]):
        super().__init__("native grammar depth bound leaves disconnected branches")
        self.program = program
        self.trace = trace


def decode_native_grammar(input_types: tuple[str, ...],
                          scorer: Callable[[tuple[NativeGrammarDecision, ...]], tuple[float, ...]],
                          *, max_steps: int = 8,
                          register_encoding: str = "absolute_v1") -> NativeGrammarResult:
    """Choose semantic atoms; scaffolding never supplies meaning or a target.

    The scorer receives alternative text prefixes and their decision spans.
    It chooses operation, every role's register, and continue/finish. Type
    constraints are shared with the execution floor, not source templates.
    Disconnected intermediate branches are legal; completed graphs must be
    connected and pass the existing native parser. The finite step bound is
    reported separately from a model-chosen finish.
    """
    if (not isinstance(input_types, tuple) or not input_types
            or any(kind not in {"integer", "integer_sequence"} for kind in input_types)
            or type(max_steps) is not int or not 1 <= max_steps <= 128):
        raise ValueError("native grammar needs declared input types and finite depth")
    if register_encoding not in {"absolute_v1", REGISTER_ENCODING}:
        raise ValueError("native grammar register encoding is not declared")
    relative = register_encoding == REGISTER_ENCODING
    types = list(input_types)
    instructions = []
    trace = []
    text = ('{"inputs":' + str(len(types))
            + (',"registers":"' + REGISTER_ENCODING + '"' if relative else "")
            + ',"steps":[')

    def choose(choices: tuple[NativeGrammarDecision, ...], kind: str):
        if not choices:
            raise ValueError("native grammar has no admitted continuation")
        scores = scorer(choices)
        if (not isinstance(scores, tuple) or len(scores) != len(choices)
                or any(not math.isfinite(score) for score in scores)):
            raise ValueError("native grammar requires every finite decision score")
        index = max(range(len(scores)), key=scores.__getitem__)
        choice = choices[index]
        trace.append({"kind": kind, "choices": [item.value for item in choices],
                      "scores": scores, "chosen": choice.value})
        return choice

    def extend(prefix: str, value: str | int, *, string: bool = False):
        atom = str(value)
        rendered = ('"' + atom + '"') if string else atom
        left = len(prefix) + (1 if string else 0)
        return NativeGrammarDecision(prefix + rendered, (left, left + len(atom)), value)

    forced = False
    for ordinal in range(max_steps):
        operations = []
        for name in sorted(PRIMITIVES_BY_NAME):
            signature = semantic_primitive_type_signature(name)
            if signature is not None and all(kind in types for kind in signature[0]):
                operations.append(name)
        operation = choose(tuple(extend(text + "[", name, string=True)
                                 for name in operations), "operation")
        text = operation.text + ",["
        signature = semantic_primitive_type_signature(str(operation.value))
        refs = []
        for role, kind in enumerate(signature[0]):
            if role:
                text += ","
            reference = choose(tuple(extend(text,
                RegisterIdentity.from_absolute(index, input_count=len(input_types)).encode()
                if relative else index, string=relative)
                for index, actual in enumerate(types) if actual == kind), "reference")
            text = reference.text
            refs.append(RegisterIdentity.parse(reference.value).to_absolute(
                input_count=len(input_types), result_count=ordinal) if relative else reference.value)
        text += "]]"
        instructions.append(Instruction(str(operation.value), tuple(refs)))
        types.append(signature[1])
        program = Program(len(input_types), tuple(instructions))
        connected = semantic_program_structural_key(program) is not None
        if ordinal + 1 == max_steps:
            if not connected:
                raise NativeGrammarIncompleteError(program, tuple(trace))
            text += "]}"
            forced = True
            break
        endings = []
        if connected:
            endings.append(NativeGrammarDecision(text + "]}", (len(text), len(text) + 1), "finish"))
        endings.append(NativeGrammarDecision(text + ",", (len(text), len(text) + 1), "continue"))
        ending = choose(tuple(endings), "termination")
        text = ending.text
        if ending.value == "finish":
            break
    parsed = parse_relative_native_program(text) if relative else parse_native_program(text)
    if parsed != program:
        raise ValueError("native grammar and parser disagree")
    return NativeGrammarResult(parsed, tuple(trace), forced)


def typed_instruction_choices(input_types: tuple[str, ...]) -> tuple[Instruction, ...]:
    """Expose all type-admitted primitive/reference choices for control probes."""
    return tuple(Instruction(name, tuple(refs)) for name in sorted(PRIMITIVES_BY_NAME)
                 if (signature := semantic_primitive_type_signature(name)) is not None
                 for refs in product(*(tuple(index for index, actual in enumerate(input_types)
                                             if actual == kind) for kind in signature[0])))
