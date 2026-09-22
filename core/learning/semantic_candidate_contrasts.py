"""Source-only typed contrasts for learning complete-program selection."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

from core.learning.procedure_induction import PRIMITIVES_BY_NAME, Instruction, Program
from core.learning.semantic_graph_counterexamples import counterfactual_inputs
from core.learning.semantic_program_floor import (
    semantic_primitive_type_signature,
    semantic_program_structural_key,
)


def _register_types(program: Program, input_kinds: Sequence[str]) -> tuple[str, ...] | None:
    if program.n_inputs != len(input_kinds) or semantic_program_structural_key(program) is None:
        return None
    types = list(input_kinds)
    for instruction in program.instructions:
        signature = semantic_primitive_type_signature(instruction.op)
        if signature is None or tuple(types[index] for index in instruction.args) != signature[0]:
            return None
        types.append(signature[1])
    return tuple(types)


def _witnessed_difference(target: Program, candidate: Program, probes: Sequence[tuple]) -> bool:
    for values in probes:
        try:
            left, right = target.run(values), candidate.run(values)
        except (ValueError, TypeError, RuntimeError, ArithmeticError, IndexError):
            continue
        if (type(left) is type(right) and left != right) or type(left) is not type(right):
            return True
    return False


def source_program_contrasts(
    target: Program, public_inputs: Sequence, peer_programs: Sequence[Program],
    *, source_sha256: str, limit: int = 32,
) -> tuple[Program, ...]:
    """Retain the positive and witnessed, type-correct source-training negatives.

    Target and peer programs are source-training annotations. They are never
    consulted when scoring a new request. A different program is a negative
    only after its value differs on an independently generated input probe.
    """
    if (type(limit) is not int or limit < 2 or len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)):
        raise ValueError("contrast construction needs a source identity and finite capacity")
    inputs = tuple(tuple(value) if isinstance(value, list) else value for value in public_inputs)
    input_kinds = tuple("integer_sequence" if isinstance(value, tuple) else "integer"
                        for value in inputs)
    types = _register_types(target, input_kinds)
    if types is None:
        raise ValueError("source target violates the floor type contract")
    probes = counterfactual_inputs(inputs, count=16, seed=int(source_sha256[:8], 16))
    variants: list[Program] = []
    for index, instruction in enumerate(target.instructions):
        signature = semantic_primitive_type_signature(instruction.op)
        for name in sorted(PRIMITIVES_BY_NAME):
            if name != instruction.op and semantic_primitive_type_signature(name) == signature:
                changed = list(target.instructions)
                changed[index] = Instruction(name, instruction.args)
                variants.append(Program(target.n_inputs, tuple(changed)))
        for argument_index, argument in enumerate(instruction.args):
            for replacement in range(target.n_inputs + index):
                if replacement == argument or types[replacement] != types[argument]:
                    continue
                changed = list(target.instructions)
                args = list(instruction.args)
                args[argument_index] = replacement
                changed[index] = Instruction(instruction.op, tuple(args))
                variants.append(Program(target.n_inputs, tuple(changed)))
    variants.extend(peer for peer in peer_programs if peer.depth == target.depth)
    seed = int(hashlib.sha256(source_sha256.encode("ascii")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    rng.shuffle(variants)
    candidates = [target]
    seen = {target.sha()}
    for candidate in variants:
        if len(candidates) >= limit:
            break
        key = candidate.sha()
        candidate_types = _register_types(candidate, input_kinds)
        if (key in seen or candidate_types is None or candidate_types[-1] != types[-1]
                or not _witnessed_difference(target, candidate, probes)):
            continue
        seen.add(key)
        candidates.append(candidate)
    if len(candidates) == 1:
        raise ValueError("source target produced no witnessed contrast")
    rng.shuffle(candidates)
    return tuple(candidates)
