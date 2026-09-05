"""core/learning/procedure_induction.py — inducing a procedure without a compiler.

The frontier the external review put its flag on: Aura can learn *within*
representations and procedures the architecture supplies, but the strongest
controlled evidence still runs through a hand-written compiler that recognises
the domain and supplies the program structure. Written as the shape of the
system, that is

    x --> P_k(x) --> N_theta

where `k` is a recognised family and `P_k` is engineered for it. What would be
needed instead is a `P` that learns what program a family requires from
experience.

This module is that `P`, for a bounded but honest case. It receives input and
output pairs and nothing else — no family label, no hint, no per-family code —
and searches compositions of a fixed, domain-general primitive set for a
program that reproduces the outputs. What it finds is a *new procedure*: a
composition that did not exist before, that can be frozen, applied to instances
it never saw, and removed again to show the gain came from it.

The algorithm is bottom-up enumerative synthesis with observational
equivalence pruning, which is the standard method for this and is what makes
depth-4 compositions tractable: two programs that produce identical values on
every support input are interchangeable for every purpose the search has, so
only the first is kept. Without it the bank grows as the product of the
primitive count and the bank size at every level; with it, it grows as the
number of behaviours actually reachable.

What would make this dishonest, and what stops it
------------------------------------------------
The failure mode is a primitive set built after seeing the held-out family, so
that one primitive *is* the answer and "composition" is a formality. Three
things guard it, all mechanical:

1.  `PRIMITIVE_SET_SHA` pins the primitive set. Adding, removing or renaming a
    primitive changes the hash, and the experiment records it, so a set edited
    to fit a family is visible in the artifact.
2.  `single_primitive_shortcut` re-runs the search restricted to depth 1. A
    family solvable that way cannot support the claim, and the experiment
    refuses rather than reporting a win.
3.  A shuffled-output control runs the whole search on the same inputs with the
    outputs permuted. It must find nothing. A searcher that finds a program for
    noise is fitting the support set, and its successes mean nothing either.

None of this establishes open-ended representation discovery. The primitives
are given, the value types are given, and inventing the vocabulary itself is a
further step. What it does establish is the part directly under the flag: no
`P_k` anywhere in the path, and a procedure that was induced, frozen,
transferred, and lesioned.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("Aura.ProcedureInduction")


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Primitive:
    """One domain-general operation. Nothing here names a task family."""

    name: str
    arity: int
    fn: Callable[..., Any]


def _safe(fn: Callable[..., Any]) -> Callable[..., Any]:
    """A primitive that raises on some input is undefined there, not fatal.

    Enumerative synthesis applies every operation to every value in the bank,
    so type errors and empty sequences are the normal case rather than an
    exceptional one. Returning a sentinel keeps them out of the bank without
    aborting the search.
    """

    def wrapped(*args: Any) -> Any:
        try:
            return fn(*args)
        except (TypeError, ValueError, ZeroDivisionError, IndexError, KeyError, OverflowError):
            return _UNDEFINED

    return wrapped


class _Undefined:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<undefined>"


_UNDEFINED = _Undefined()


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _seq(value: Any) -> Sequence[Any]:
    if isinstance(value, (list, tuple)):
        return value
    raise TypeError("not a sequence")


def _nums(value: Any) -> list[float]:
    items = list(_seq(value))
    if not items or not all(_is_num(v) for v in items):
        raise TypeError("not a numeric sequence")
    return items


#: The primitive set. Fixed before any task family in this module was chosen,
#: and pinned by PRIMITIVE_SET_SHA below. Every entry is an ordinary operation
#: on numbers or sequences; none of them is about a family.
PRIMITIVES: tuple[Primitive, ...] = (
    Primitive("add", 2, _safe(lambda a, b: a + b if _is_num(a) and _is_num(b) else _UNDEFINED)),
    Primitive("sub", 2, _safe(lambda a, b: a - b if _is_num(a) and _is_num(b) else _UNDEFINED)),
    Primitive("mul", 2, _safe(lambda a, b: a * b if _is_num(a) and _is_num(b) else _UNDEFINED)),
    Primitive("idiv", 2, _safe(lambda a, b: a // b if _is_num(a) and _is_num(b) and b else _UNDEFINED)),
    Primitive("mod", 2, _safe(lambda a, b: a % b if _is_num(a) and _is_num(b) and b else _UNDEFINED)),
    Primitive("neg", 1, _safe(lambda a: -a if _is_num(a) else _UNDEFINED)),
    Primitive("absv", 1, _safe(lambda a: abs(a) if _is_num(a) else _UNDEFINED)),
    Primitive("length", 1, _safe(lambda a: len(_seq(a)))),
    Primitive("total", 1, _safe(lambda a: sum(_nums(a)))),
    Primitive("largest", 1, _safe(lambda a: max(_nums(a)))),
    Primitive("smallest", 1, _safe(lambda a: min(_nums(a)))),
    Primitive("sorted_up", 1, _safe(lambda a: tuple(sorted(_nums(a))))),
    Primitive("reversed_", 1, _safe(lambda a: tuple(reversed(_seq(a))))),
    Primitive("head", 1, _safe(lambda a: _seq(a)[0])),
    Primitive("last", 1, _safe(lambda a: _seq(a)[-1])),
    Primitive("tail", 1, _safe(lambda a: tuple(_seq(a)[1:]))),
    Primitive("front", 1, _safe(lambda a: tuple(_seq(a)[:-1]))),
    Primitive("unique", 1, _safe(lambda a: tuple(sorted(set(_nums(a)))))),
    Primitive("at", 2, _safe(lambda a, i: _seq(a)[int(i)] if _is_num(i) else _UNDEFINED)),
    Primitive("count_of", 2, _safe(lambda a, v: sum(1 for x in _seq(a) if x == v))),
)

PRIMITIVES_BY_NAME = {p.name: p for p in PRIMITIVES}


def _primitive_set_sha() -> str:
    payload = json.dumps([[p.name, p.arity] for p in PRIMITIVES], sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: Identity of the primitive set. Recorded in every result so a set edited to
#: fit a family cannot pass unnoticed.
PRIMITIVE_SET_SHA = _primitive_set_sha()


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instruction:
    op: str
    args: tuple[int, ...]


@dataclass(frozen=True)
class Program:
    """A straight-line composition. Register i < n_inputs is an input slot."""

    n_inputs: int
    instructions: tuple[Instruction, ...]

    def run(self, inputs: Sequence[Any]) -> Any:
        registers: list[Any] = list(inputs)
        for instruction in self.instructions:
            primitive = PRIMITIVES_BY_NAME[instruction.op]
            args = [registers[i] for i in instruction.args]
            if any(isinstance(a, _Undefined) for a in args):
                return _UNDEFINED
            registers.append(primitive.fn(*args))
        return registers[-1]

    @property
    def depth(self) -> int:
        return len(self.instructions)

    def describe(self) -> str:
        names = [f"in{i}" for i in range(self.n_inputs)]
        for instruction in self.instructions:
            names.append(
                f"{instruction.op}({', '.join(names[i] for i in instruction.args)})"
            )
        return names[-1]

    def sha(self) -> str:
        payload = json.dumps(
            {
                "n_inputs": self.n_inputs,
                "instructions": [[i.op, list(i.args)] for i in self.instructions],
            },
            sort_keys=True,
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.describe(),
            "depth": self.depth,
            "sha": self.sha(),
            "instructions": [[i.op, list(i.args)] for i in self.instructions],
        }


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskInstance:
    inputs: tuple[Any, ...]
    output: Any


@dataclass(frozen=True)
class TaskFamily:
    """A family is a *generator*, never a solver.

    Nothing in this module maps a family to the program that solves it. The
    generator produces instances; the inducer sees the instances.
    """

    family_id: str
    generate: Callable[[random.Random], TaskInstance]

    def sample(self, count: int, *, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        return [self.generate(rng) for _ in range(count)]


# ---------------------------------------------------------------------------
# The inducer
# ---------------------------------------------------------------------------


def _behaviour_key(values: Sequence[Any]) -> str:
    return json.dumps([repr(v) for v in values])


@dataclass(frozen=True)
class InductionOutcome:
    program: Program | None
    refusal: str = ""
    programs_considered: int = 0
    bank_size: int = 0

    @property
    def found(self) -> bool:
        return self.program is not None


class ProcedureInducer:
    """Bottom-up enumerative synthesis over the fixed primitive set."""

    def __init__(self, *, max_depth: int = 4, max_bank: int = 40_000) -> None:
        self.max_depth = max(1, int(max_depth))
        self.max_bank = max(64, int(max_bank))

    def induce(
        self, support: Sequence[TaskInstance], *, max_depth: int | None = None
    ) -> InductionOutcome:
        if not support:
            return InductionOutcome(None, refusal="no support instances")
        n_inputs = len(support[0].inputs)
        if any(len(item.inputs) != n_inputs for item in support):
            return InductionOutcome(None, refusal="support instances disagree on arity")

        target = [item.output for item in support]
        depth_limit = self.max_depth if max_depth is None else max(1, int(max_depth))

        # Level 0: the inputs themselves. A program is a value vector over the
        # support set plus the instructions that produced it.
        bank: list[tuple[list[Any], Program]] = []
        seen: set[str] = set()
        for slot in range(n_inputs):
            values = [item.inputs[slot] for item in support]
            program = Program(n_inputs, ())
            program = Program(n_inputs, (Instruction("head", (slot,)),)) if False else program
            key = _behaviour_key(values)
            if key in seen:
                continue
            seen.add(key)
            bank.append((values, _IdentityProgram(n_inputs, slot)))
            if values == target:
                return InductionOutcome(
                    _IdentityProgram(n_inputs, slot),
                    programs_considered=len(bank),
                    bank_size=len(bank),
                )

        considered = len(bank)
        level_start = 0
        for _ in range(depth_limit):
            level_end = len(bank)
            new_entries: list[tuple[list[Any], Program]] = []
            for primitive in PRIMITIVES:
                # At least one argument must come from the newest level, or the
                # same program is rebuilt at every depth.
                index_pool = range(level_end)
                for arg_indices in itertools.product(index_pool, repeat=primitive.arity):
                    if all(i < level_start for i in arg_indices):
                        continue
                    considered += 1
                    values = []
                    usable = True
                    for row in range(len(support)):
                        args = [bank[i][0][row] for i in arg_indices]
                        if any(isinstance(a, _Undefined) for a in args):
                            usable = False
                            break
                        result = primitive.fn(*args)
                        if isinstance(result, _Undefined):
                            usable = False
                            break
                        values.append(result)
                    if not usable:
                        continue
                    key = _behaviour_key(values)
                    if key in seen:
                        continue  # observational equivalence
                    seen.add(key)
                    program = _compose(bank, arg_indices, primitive, n_inputs)
                    if values == target:
                        return InductionOutcome(
                            program,
                            programs_considered=considered,
                            bank_size=len(bank) + len(new_entries),
                        )
                    new_entries.append((values, program))
                    if len(bank) + len(new_entries) >= self.max_bank:
                        return InductionOutcome(
                            None,
                            refusal=f"bank limit {self.max_bank} reached before a fit",
                            programs_considered=considered,
                            bank_size=len(bank) + len(new_entries),
                        )
            if not new_entries:
                break
            level_start = level_end
            bank.extend(new_entries)

        return InductionOutcome(
            None,
            refusal=f"no composition up to depth {depth_limit} reproduced the outputs",
            programs_considered=considered,
            bank_size=len(bank),
        )


class _IdentityProgram(Program):
    """`in_k` — the input itself, with no instructions."""

    def __new__(cls, n_inputs: int, slot: int) -> "_IdentityProgram":
        obj = object.__new__(cls)
        object.__setattr__(obj, "n_inputs", n_inputs)
        object.__setattr__(obj, "instructions", ())
        object.__setattr__(obj, "_slot", slot)
        return obj

    def run(self, inputs: Sequence[Any]) -> Any:
        return inputs[self._slot]  # type: ignore[attr-defined]

    def describe(self) -> str:
        return f"in{self._slot}"  # type: ignore[attr-defined]


def _compose(
    bank: list[tuple[list[Any], Program]],
    arg_indices: tuple[int, ...],
    primitive: Primitive,
    n_inputs: int,
) -> Program:
    """Flatten the argument programs into one straight-line instruction list."""
    instructions: list[Instruction] = []
    arg_registers: list[int] = []
    for index in arg_indices:
        sub = bank[index][1]
        if isinstance(sub, _IdentityProgram):
            arg_registers.append(sub._slot)  # type: ignore[attr-defined]
            continue
        offset = n_inputs + len(instructions)
        for instruction in sub.instructions:
            instructions.append(
                Instruction(
                    instruction.op,
                    tuple(
                        a if a < n_inputs else a + (offset - n_inputs)
                        for a in instruction.args
                    ),
                )
            )
        arg_registers.append(n_inputs + len(instructions) - 1)
    instructions.append(Instruction(primitive.name, tuple(arg_registers)))
    return Program(n_inputs, tuple(instructions))


def accuracy(program: Program, instances: Sequence[TaskInstance]) -> float:
    if not instances:
        return 0.0
    hits = 0
    for item in instances:
        try:
            predicted = program.run(item.inputs)
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            continue
        if predicted == item.output:
            hits += 1
    return hits / len(instances)


__all__ = [
    "PRIMITIVES",
    "PRIMITIVES_BY_NAME",
    "PRIMITIVE_SET_SHA",
    "InductionOutcome",
    "Instruction",
    "Primitive",
    "ProcedureInducer",
    "Program",
    "TaskFamily",
    "TaskInstance",
    "accuracy",
]
