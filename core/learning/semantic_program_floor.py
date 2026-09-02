"""Compile learned typed semantic programs into Aura's universal metered floor.

The semantic transducer learns a validated SSA graph. The old execution path
then called a Python table of primitives. This module gives that graph the same
semantics as Aura's endogenous language: every operation becomes a term on the
floor, public inputs become numbers or pair lists, and the result is recovered
without a family router or generated code.

The compiler is deliberately total over the declared primitive vocabulary.
Adding a primitive to that vocabulary without defining its floor semantics is
therefore a refused migration, not an implicit fallback to Python.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.cognition.the_floor_she_stands_on import (
    BELOW,
    FST,
    IF,
    ISPAIR,
    LEFTOVER,
    LET,
    MINUS,
    NIL,
    OVER,
    PAIR,
    PLUS,
    SAME,
    SND,
    TIMES,
    A,
    Code,
    L,
    N,
    Nothing,
    Pair,
    V,
    Y,
    build,
    run,
)
from core.learning.procedure_induction import PRIMITIVES_BY_NAME
from core.learning.semantic_program_ir import (
    SemanticProgramIR,
    SemanticValue,
    normalize_semantic_value,
)

SEMANTIC_FLOOR_PROGRAM_SCHEMA: Final = "aura.semantic_floor_program.v1"
SEMANTIC_FLOOR_EXECUTION_SCHEMA: Final = "aura.semantic_floor_execution.v1"
DEFAULT_SEMANTIC_FLOOR_FUEL: Final = 2_000_000
_INT: Final = "integer"
_SEQUENCE: Final = "integer_sequence"
_TYPE_SIGNATURES: Final = {
    "add": ((_INT, _INT), _INT),
    "sub": ((_INT, _INT), _INT),
    "mul": ((_INT, _INT), _INT),
    "idiv": ((_INT, _INT), _INT),
    "mod": ((_INT, _INT), _INT),
    "neg": ((_INT,), _INT),
    "absv": ((_INT,), _INT),
    "length": ((_SEQUENCE,), _INT),
    "total": ((_SEQUENCE,), _INT),
    "largest": ((_SEQUENCE,), _INT),
    "smallest": ((_SEQUENCE,), _INT),
    "sorted_up": ((_SEQUENCE,), _SEQUENCE),
    "reversed_": ((_SEQUENCE,), _SEQUENCE),
    "head": ((_SEQUENCE,), _INT),
    "last": ((_SEQUENCE,), _INT),
    "tail": ((_SEQUENCE,), _SEQUENCE),
    "front": ((_SEQUENCE,), _SEQUENCE),
    "unique": ((_SEQUENCE,), _SEQUENCE),
    "at": ((_SEQUENCE, _INT), _INT),
    "count_of": ((_SEQUENCE, _INT), _INT),
}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _stuck() -> Any:
    """A floor expression that always refuses."""

    return FST(N(0))


def _not(value: Any) -> Any:
    return IF(value, N(0), N(1))


def _not_below(one: Any, other: Any) -> Any:
    return _not(BELOW(one, other))


def _list_length(values: Any) -> Any:
    return A(
        Y(
            "length_again",
            L(
                "values",
                IF(
                    ISPAIR(V("values")),
                    PLUS(N(1), A(V("length_again"), SND(V("values")))),
                    N(0),
                ),
            ),
        ),
        values,
    )


def _list_total(values: Any) -> Any:
    return A(
        Y(
            "total_again",
            L(
                "values",
                IF(
                    ISPAIR(V("values")),
                    PLUS(FST(V("values")), A(V("total_again"), SND(V("values")))),
                    N(0),
                ),
            ),
        ),
        values,
    )


def _list_extreme(values: Any, *, largest: bool) -> Any:
    choose = (
        IF(BELOW(V("best"), FST(V("rest"))), FST(V("rest")), V("best"))
        if largest
        else IF(BELOW(FST(V("rest")), V("best")), FST(V("rest")), V("best"))
    )
    return IF(
        ISPAIR(values),
        A(
            Y(
                "extreme_again",
                L(
                    "best",
                    L(
                        "rest",
                        IF(
                            ISPAIR(V("rest")),
                            A(
                                V("extreme_again"),
                                choose,
                                SND(V("rest")),
                            ),
                            V("best"),
                        ),
                    ),
                ),
            ),
            FST(values),
            SND(values),
        ),
        _stuck(),
    )


def _reverse(values: Any) -> Any:
    return A(
        Y(
            "reverse_again",
            L(
                "left",
                L(
                    "made",
                    IF(
                        ISPAIR(V("left")),
                        A(
                            V("reverse_again"),
                            SND(V("left")),
                            PAIR(FST(V("left")), V("made")),
                        ),
                        V("made"),
                    ),
                ),
            ),
        ),
        values,
        NIL,
    )


def _insert_sorted(value: Any, values: Any) -> Any:
    return A(
        Y(
            "insert_again",
            L(
                "value",
                L(
                    "values",
                    IF(
                        ISPAIR(V("values")),
                        IF(
                            BELOW(V("value"), FST(V("values"))),
                            PAIR(V("value"), V("values")),
                            PAIR(
                                FST(V("values")),
                                A(
                                    V("insert_again"),
                                    V("value"),
                                    SND(V("values")),
                                ),
                            ),
                        ),
                        PAIR(V("value"), NIL),
                    ),
                ),
            ),
        ),
        value,
        values,
    )


def _sort(values: Any) -> Any:
    return A(
        Y(
            "sort_again",
            L(
                "values",
                IF(
                    ISPAIR(V("values")),
                    _insert_sorted(
                        FST(V("values")),
                        A(V("sort_again"), SND(V("values"))),
                    ),
                    NIL,
                ),
            ),
        ),
        values,
    )


def _unique_sorted(values: Any) -> Any:
    sorted_values = _sort(values)
    return A(
        Y(
            "unique_again",
            L(
                "values",
                IF(
                    ISPAIR(V("values")),
                    PAIR(
                        FST(V("values")),
                        A(
                            Y(
                                "drop_same",
                                L(
                                    "previous",
                                    L(
                                        "rest",
                                        IF(
                                            ISPAIR(V("rest")),
                                            IF(
                                                SAME(V("previous"), FST(V("rest"))),
                                                A(
                                                    V("drop_same"),
                                                    V("previous"),
                                                    SND(V("rest")),
                                                ),
                                                A(V("unique_again"), V("rest")),
                                            ),
                                            NIL,
                                        ),
                                    ),
                                ),
                            ),
                            FST(V("values")),
                            SND(V("values")),
                        ),
                    ),
                    NIL,
                ),
            ),
        ),
        sorted_values,
    )


def _last(values: Any) -> Any:
    return IF(
        ISPAIR(values),
        A(
            Y(
                "last_again",
                L(
                    "values",
                    IF(
                        ISPAIR(SND(V("values"))),
                        A(V("last_again"), SND(V("values"))),
                        FST(V("values")),
                    ),
                ),
            ),
            values,
        ),
        _stuck(),
    )


def _front(values: Any) -> Any:
    return A(
        Y(
            "front_again",
            L(
                "values",
                IF(
                    ISPAIR(V("values")),
                    IF(
                        ISPAIR(SND(V("values"))),
                        PAIR(FST(V("values")), A(V("front_again"), SND(V("values")))),
                        NIL,
                    ),
                    NIL,
                ),
            ),
        ),
        values,
    )


def _at(values: Any, index: Any) -> Any:
    normalized = IF(BELOW(index, N(0)), PLUS(_list_length(values), index), index)
    return LET(
        "index",
        normalized,
        IF(
            BELOW(V("index"), N(0)),
            _stuck(),
            A(
                Y(
                    "at_again",
                    L(
                        "values",
                        L(
                            "index",
                            IF(
                                ISPAIR(V("values")),
                                IF(
                                    SAME(V("index"), N(0)),
                                    FST(V("values")),
                                    A(
                                        V("at_again"),
                                        SND(V("values")),
                                        MINUS(V("index"), N(1)),
                                    ),
                                ),
                                _stuck(),
                            ),
                        ),
                    ),
                ),
                values,
                V("index"),
            ),
        ),
    )


def _count_of(values: Any, wanted: Any) -> Any:
    return A(
        Y(
            "count_again",
            L(
                "values",
                IF(
                    ISPAIR(V("values")),
                    PLUS(
                        SAME(FST(V("values")), wanted),
                        A(V("count_again"), SND(V("values"))),
                    ),
                    N(0),
                ),
            ),
        ),
        values,
    )


def _primitive(op: str, args: Sequence[Any]) -> Any:
    if op == "add":
        return PLUS(args[0], args[1])
    if op == "sub":
        return MINUS(args[0], args[1])
    if op == "mul":
        return TIMES(args[0], args[1])
    if op == "idiv":
        return OVER(args[0], args[1])
    if op == "mod":
        return LEFTOVER(args[0], args[1])
    if op == "neg":
        return MINUS(N(0), args[0])
    if op == "absv":
        return IF(BELOW(args[0], N(0)), MINUS(N(0), args[0]), args[0])
    if op == "length":
        return _list_length(args[0])
    if op == "total":
        return _list_total(args[0])
    if op == "largest":
        return _list_extreme(args[0], largest=True)
    if op == "smallest":
        return _list_extreme(args[0], largest=False)
    if op == "sorted_up":
        return _sort(args[0])
    if op == "reversed_":
        return _reverse(args[0])
    if op == "head":
        return IF(ISPAIR(args[0]), FST(args[0]), _stuck())
    if op == "last":
        return _last(args[0])
    if op == "tail":
        return IF(ISPAIR(args[0]), SND(args[0]), NIL)
    if op == "front":
        return _front(args[0])
    if op == "unique":
        return _unique_sorted(args[0])
    if op == "at":
        return _at(args[0], args[1])
    if op == "count_of":
        return _count_of(args[0], args[1])
    raise ValueError(f"semantic primitive {op!r} has no floor semantics")


def _floor_value(value: SemanticValue) -> Any:
    normalized = normalize_semantic_value(value)
    if isinstance(normalized, int):
        return N(normalized)
    made: Any = NIL
    for item in reversed(normalized):
        made = PAIR(N(item), made)
    return made


def _semantic_value(value: Any) -> SemanticValue:
    if type(value) is int:
        return value
    if isinstance(value, Nothing):
        return ()
    if isinstance(value, Pair):
        out: list[int] = []
        here: Any = value
        while isinstance(here, Pair):
            if type(here.first) is not int:
                raise ValueError("semantic floor emitted a nested or non-integer sequence")
            out.append(here.first)
            here = here.second
        if not isinstance(here, Nothing):
            raise ValueError("semantic floor emitted an improper sequence")
        return tuple(out)
    raise ValueError("semantic floor emitted a value outside the exact algebra")


def _compile_body(ir: SemanticProgramIR, public_inputs: tuple[SemanticValue, ...]) -> Any:
    registers = [f"register_{index}" for index in range(ir.n_inputs)]
    register_types = [
        _INT if isinstance(value, int) else _SEQUENCE for value in public_inputs
    ]
    body: Any = None
    bindings: list[tuple[str, Any]] = [
        (registers[index], _floor_value(public_inputs[index]))
        for index in range(ir.n_inputs)
    ]
    for ordinal, instruction in enumerate(ir.instructions):
        primitive = PRIMITIVES_BY_NAME.get(instruction.op)
        signature = _TYPE_SIGNATURES.get(instruction.op)
        if (
            primitive is None
            or signature is None
            or len(instruction.args) != primitive.arity
            or tuple(register_types[argument] for argument in instruction.args)
            != signature[0]
        ):
            raise ValueError("semantic IR primitive signature has no floor meaning")
        name = f"register_{ir.n_inputs + ordinal}"
        expression = _primitive(
            instruction.op,
            tuple(V(registers[argument]) for argument in instruction.args),
        )
        registers.append(name)
        register_types.append(signature[1])
        bindings.append((name, expression))
    body = V(registers[ir.report_value])
    for name, value in reversed(bindings):
        body = LET(name, value, body)
    return body


@dataclass(frozen=True, slots=True)
class SemanticFloorProgram:
    """One closed floor term bound to the learned IR and its public inputs."""

    code: Code
    receipt: dict[str, Any]

    def __post_init__(self) -> None:
        body = {key: value for key, value in self.receipt.items() if key != "receipt_sha256"}
        if (
            not isinstance(self.code, Code)
            or body.get("schema") != SEMANTIC_FLOOR_PROGRAM_SCHEMA
            or self.receipt.get("receipt_sha256") != _sha(body)
        ):
            raise ValueError("semantic floor program receipt is invalid")


@dataclass(frozen=True, slots=True)
class SemanticFloorExecution:
    """A metered floor result with no expected-answer authority."""

    result: SemanticValue
    receipt: dict[str, Any]

    def __post_init__(self) -> None:
        normalized = normalize_semantic_value(self.result)
        body = {key: value for key, value in self.receipt.items() if key != "receipt_sha256"}
        if (
            normalized != self.result
            or body.get("schema") != SEMANTIC_FLOOR_EXECUTION_SCHEMA
            or self.receipt.get("receipt_sha256") != _sha(body)
            or body.get("result_sha256") != _sha(
                list(normalized) if isinstance(normalized, tuple) else normalized
            )
        ):
            raise ValueError("semantic floor execution receipt is invalid")


def compile_semantic_program_to_floor(
    ir: SemanticProgramIR,
    public_inputs: tuple[Any, ...],
) -> SemanticFloorProgram:
    """Compile one validated learned program into a closed floor term."""

    if not isinstance(ir, SemanticProgramIR):
        raise TypeError("semantic floor compilation requires validated program IR")
    if not isinstance(public_inputs, tuple) or len(public_inputs) != ir.n_inputs:
        raise ValueError("semantic floor public inputs are invalid")
    inputs = tuple(normalize_semantic_value(value) for value in public_inputs)
    missing = set(PRIMITIVES_BY_NAME) - _FLOOR_PRIMITIVES
    missing_types = set(PRIMITIVES_BY_NAME) - set(_TYPE_SIGNATURES)
    if missing or missing_types:
        uncovered = missing | missing_types
        raise RuntimeError(
            f"semantic primitive floor coverage is incomplete: {sorted(uncovered)}"
        )
    code = build(_compile_body(ir, inputs))
    body = {
        "schema": SEMANTIC_FLOOR_PROGRAM_SCHEMA,
        "ir_receipt_sha256": ir.receipt()["receipt_sha256"],
        "alpha_normalized_sha256": ir.alpha_normalized_sha256,
        "public_inputs_sha256": _sha(
            [list(value) if isinstance(value, tuple) else value for value in inputs]
        ),
        "floor_semantics": "core.cognition.the_floor_she_stands_on.v1",
        "expected_answer_available": False,
        "verifier_trace_available": False,
        "generated_code_available": False,
        "family_router_present": False,
        "correctness_authority": False,
    }
    return SemanticFloorProgram(code=code, receipt={**body, "receipt_sha256": _sha(body)})


def execute_semantic_floor_program(
    program: SemanticFloorProgram,
    *,
    fuel: int = DEFAULT_SEMANTIC_FLOOR_FUEL,
) -> SemanticFloorExecution:
    """Execute a compiled semantic program under the floor's explicit meter."""

    if not isinstance(program, SemanticFloorProgram):
        raise TypeError("semantic floor execution requires a compiled program")
    if type(fuel) is not int or fuel < 1:
        raise ValueError("semantic floor fuel must be a positive integer")
    result = normalize_semantic_value(_semantic_value(run(program.code, fuel=fuel)))
    body = {
        "schema": SEMANTIC_FLOOR_EXECUTION_SCHEMA,
        "program_receipt_sha256": program.receipt["receipt_sha256"],
        "result_sha256": _sha(list(result) if isinstance(result, tuple) else result),
        "fuel_limit": fuel,
        "execution_engine": "universal_metered_floor",
        "expected_answer_available": False,
        "verifier_trace_available": False,
        "generated_code_available": False,
        "correctness_authority": False,
    }
    return SemanticFloorExecution(
        result=result,
        receipt={**body, "receipt_sha256": _sha(body)},
    )


def semantic_floor_primitive_coverage() -> dict[str, Any]:
    """Name any declared primitive lacking floor code or a type signature."""

    declared = set(PRIMITIVES_BY_NAME)
    missing_semantics = sorted(declared - _FLOOR_PRIMITIVES)
    missing_types = sorted(declared - set(_TYPE_SIGNATURES))
    extra_semantics = sorted(_FLOOR_PRIMITIVES - declared)
    extra_types = sorted(set(_TYPE_SIGNATURES) - declared)
    return {
        "declared": sorted(declared),
        "missing_semantics": missing_semantics,
        "missing_types": missing_types,
        "extra_semantics": extra_semantics,
        "extra_types": extra_types,
        "complete": not any(
            (missing_semantics, missing_types, extra_semantics, extra_types)
        ),
    }


_FLOOR_PRIMITIVES: Final = frozenset(
    {
        "add",
        "sub",
        "mul",
        "idiv",
        "mod",
        "neg",
        "absv",
        "length",
        "total",
        "largest",
        "smallest",
        "sorted_up",
        "reversed_",
        "head",
        "last",
        "tail",
        "front",
        "unique",
        "at",
        "count_of",
    }
)

__all__ = [
    "DEFAULT_SEMANTIC_FLOOR_FUEL",
    "SEMANTIC_FLOOR_EXECUTION_SCHEMA",
    "SEMANTIC_FLOOR_PROGRAM_SCHEMA",
    "SemanticFloorExecution",
    "SemanticFloorProgram",
    "compile_semantic_program_to_floor",
    "execute_semantic_floor_program",
    "semantic_floor_primitive_coverage",
]
