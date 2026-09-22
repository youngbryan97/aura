"""Put independently grounded programs into one source-coordinate system."""

from dataclasses import replace

from core.learning.procedure_induction import Program
from core.learning.semantic_program_ir import normalize_semantic_value


def reanchor_program_inputs(
    program: Program, *, from_spans, to_spans, from_inputs, to_inputs
) -> Program:
    """Rename input registers by source anchors, never by expected answers.

    Equal-valued occurrences remain distinct variables in counterfactual
    comparisons. Intermediate registers and instruction order are unchanged.
    """
    old, new = tuple(from_spans), tuple(to_spans)
    before = tuple(normalize_semantic_value(x) for x in from_inputs)
    after = tuple(normalize_semantic_value(x) for x in to_inputs)
    count = program.n_inputs
    if (len(old) != count or len(new) != count or len(before) != count
            or len(after) != count or len(set(old)) != count
            or len(set(new)) != count or set(old) != set(new)):
        raise ValueError("program input anchors are not a bijection")
    positions = {span: index for index, span in enumerate(new)}
    mapping = tuple(positions[span] for span in old)
    if any(before[index] != after[target] for index, target in enumerate(mapping)):
        raise ValueError("program anchor permutation changes public values")
    for index, instruction in enumerate(program.instructions):
        if any(type(r) is not int or not 0 <= r < count + index for r in instruction.args):
            raise ValueError("program has an invalid register reference")
    return Program(count, tuple(replace(instruction, args=tuple(
        mapping[r] if r < count else r for r in instruction.args
    )) for instruction in program.instructions))
