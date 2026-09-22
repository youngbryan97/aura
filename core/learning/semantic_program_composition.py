"""Compose retained semantic programs without treating recombination as evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_floor import (
    semantic_primitive_type_signature,
    semantic_program_structural_key,
)
from core.learning.semantic_program_ir import normalize_semantic_value


@dataclass(frozen=True)
class ComposedProgram:
    name: str
    program: Program
    instruction_sources: tuple[tuple[str, ...], ...]
    provenance_sha256: str


@dataclass(frozen=True)
class ProgramComposition:
    candidates: tuple[ComposedProgram, ...]
    examined: int
    search_exhausted: bool


def _typed_connected(program: Program, input_types: tuple[str, ...]) -> bool:
    if program.n_inputs != len(input_types) or semantic_program_structural_key(program) is None:
        return False
    types = list(input_types)
    for instruction in program.instructions:
        signature = semantic_primitive_type_signature(instruction.op)
        if signature is None or tuple(types[index] for index in instruction.args) != signature[0]:
            return False
        types.append(signature[1])
    return True


def compose_semantic_programs(
    proposals: Mapping[str, Program | None],
    public_inputs: tuple,
    *,
    max_candidates: int,
    max_examined: int,
) -> ProgramComposition:
    """Retain novel typed SSA crossovers from complete, independently named methods.

    Instruction positions and input register identities stay fixed. This is a
    finite proposal generator, not a search-completeness or correctness claim.
    All allowed combinations are counted, including malformed ones.
    """
    if (not isinstance(public_inputs, tuple) or type(max_candidates) is not int
            or type(max_examined) is not int or max_candidates < 1 or max_examined < 1):
        raise ValueError("semantic composition needs positive finite allowances")
    inputs = tuple(normalize_semantic_value(value) for value in public_inputs)
    input_types = tuple("integer_sequence" if isinstance(value, tuple) else "integer"
                        for value in inputs)
    parents = [(name, program) for name, program in proposals.items() if program is not None]
    if any(not isinstance(name, str) or not name or not isinstance(program, Program)
           for name, program in parents):
        raise ValueError("semantic composition requires named complete programs")
    parents.sort(key=lambda row: row[0])
    originals = {program.sha() for _, program in parents}
    groups: dict[tuple[int, int], list[tuple[str, Program]]] = {}
    for name, program in parents:
        if program.n_inputs != len(inputs) or not _typed_connected(program, input_types):
            continue
        groups.setdefault((program.n_inputs, program.depth), []).append((name, program))

    candidates, seen, examined = [], set(originals), 0
    for (n_inputs, depth), group in sorted(groups.items()):
        if len(group) < 2:
            continue
        positions = []
        for ordinal in range(depth):
            choices: dict[Instruction, list[str]] = {}
            for name, program in group:
                choices.setdefault(program.instructions[ordinal], []).append(name)
            positions.append(tuple(sorted(
                ((instruction, tuple(names)) for instruction, names in choices.items()),
                key=lambda row: (row[0].op, row[0].args),
            )))
        for choices in product(*positions):
            if examined == max_examined or len(candidates) == max_candidates:
                return ProgramComposition(tuple(candidates), examined, False)
            examined += 1
            program = Program(n_inputs, tuple(instruction for instruction, _ in choices))
            identity = program.sha()
            if identity in seen or not _typed_connected(program, input_types):
                continue
            seen.add(identity)
            sources = tuple(names for _, names in choices)
            body = {"program": identity, "sources": sources,
                    "parents": [(name, parent.sha()) for name, parent in group]}
            provenance = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            candidates.append(ComposedProgram(
                "composition:" + identity[7:], program, sources, provenance))
    return ProgramComposition(tuple(candidates), examined, True)
