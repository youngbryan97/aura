"""Exact ring reasoning around opaque typed computations on the existing floor.

An opaque call has no guessed value. Its identity and arguments are retained,
along with its definedness obligation, even when its output cancels. Equal
keys prove output and domain equality under shared pure primitive semantics.
Unequal keys establish neither inequality nor a natural-language meaning.
"""

from core.learning.procedure_induction import Program
from core.learning.semantic_program_floor import (
    semantic_primitive_type_signature,
    semantic_program_structural_key,
)
from core.verify.invariants import invariant
from typing import Any


def semantic_program_symbolic_key(
    program: Program,
    *,
    max_terms: int=4096,
    max_expression_chars: int=65536,
) -> Any:
    """Normalize a typed program without evaluating any public inputs.

    All non-ring primitives are treated as possibly partial uninterpreted
    functions. This conservative abstraction uses only congruence and integer
    ring identities, never sampled agreement or a rule invented for a task.
    """
    if (type(max_terms) is not int or max_terms < 1
            or type(max_expression_chars) is not int or max_expression_chars < 1):
        raise ValueError("symbolic proof allowances must be positive integers")
    if (not isinstance(program, Program) or type(program.n_inputs) is not int
            or not 1 <= program.n_inputs <= 64 or not 1 <= len(program.instructions) <= 128
            or semantic_program_structural_key(program) is None):
        return None
    types = [None] * program.n_inputs
    for instruction in program.instructions:
        arguments, result = semantic_primitive_type_signature(instruction.op)
        for register, expected in zip(instruction.args, arguments, strict=True):
            if types[register] is not None and types[register] != expected:
                return None
            types[register] = expected
        types.append(result)

    import sympy as sp

    symbols = sp.symbols(f"input_0:{program.n_inputs}")
    values = [sp.Poly(symbol, symbol, domain=sp.ZZ) if kind == "integer" else symbol
              for symbol, kind in zip(symbols, types[:program.n_inputs], strict=True)]
    obligations = set()
    obligation_chars = 0
    for index, instruction in enumerate(program.instructions):
        args = [values[register] for register in instruction.args]
        if instruction.op in {"add", "sub", "mul", "neg"}:
            sizes = [len(value.terms()) for value in args]
            if ((instruction.op == "mul" and sizes[0] * sizes[1] > max_terms)
                    or (instruction.op in {"add", "sub"} and sum(sizes) > max_terms)):
                return None
            if instruction.op == "add":
                value = args[0] + args[1]
            elif instruction.op == "sub":
                value = args[0] - args[1]
            elif instruction.op == "mul":
                value = args[0] * args[1]
            else:
                value = -args[0]
        else:
            expression = sp.Function("primitive_" + instruction.op)(
                *(arg.as_expr() if isinstance(arg, sp.Poly) else arg for arg in args))
            obligation = sp.srepr(expression)
            if obligation not in obligations:
                obligations.add(obligation)
                obligation_chars += len(obligation)
            if obligation_chars > max_expression_chars:
                return None
            value = (sp.Poly(expression, expression, domain=sp.ZZ)
                     if types[program.n_inputs + index] == "integer" else expression)
        if isinstance(value, sp.Poly) and any(abs(int(c)).bit_length() > 4096 for c in value.coeffs()):
            return None
        normal = sp.srepr(value.as_expr() if isinstance(value, sp.Poly) else value)
        if len(normal) + obligation_chars > max_expression_chars:
            return None
        values.append(value)
    return ("typed_partial_ring_v1", tuple(types[:program.n_inputs]),
            tuple(sorted(obligations)), types[-1], normal)


@invariant("learning.symbolic_cancellation_retains_definedness", scope="learning",
           owner="core/learning/semantic_program_symbolic.py", observational=False)
def _cancellation_keeps_domain() -> tuple[()]:
    from core.learning.procedure_induction import Instruction

    partial = Program(2, (Instruction("idiv", (0, 1)), Instruction("sub", (2, 2))))
    total = Program(2, (Instruction("sub", (0, 0)),))
    assert semantic_program_symbolic_key(partial) != semantic_program_symbolic_key(total)
    return ()
