"""The floor: one language small enough to author once and wide enough to stop.

`core/cognition/one_algebra.py` collapsed the tower of makers. A way of
building words became a term with a hole in it, so there was no list of
constructors to be at the end of. What it did not collapse is the grammar those
terms are written in. `HEADS` is seven arithmetic operations, and `run` has an
if-chain for `where`, `many`, `fixed`, `hole`, `through`, `undo`, `over again`
and `if`. A family needing anything else waits for somebody to edit `run`.

That is the regress, and it is not an aesthetic complaint. Every term in that
algebra halts, and the terms are recursively enumerable, so what the algebra
expresses is a recursively enumerable class of total functions — and a class
like that always leaves something computable outside it. Diagonalise the
enumeration and the witness is explicit:
:func:`core.cognition.what_the_old_language_cannot_say.the_one_it_cannot_say`
builds it. Her own growth mechanisms only ever add terms over the existing
heads, and a term over existing heads can be substituted away, so no amount of
inventing moves that witness inside. Only a person editing `run` does.

So the regress ends in exactly one place, and there is a theorem saying which:

    A bedrock that is NOT universal leaves a computable behaviour outside it
    forever, and only authoring can put that behaviour in. A bedrock that IS
    universal leaves nothing outside it, so authoring is never again required
    — and never again possible either, since there is nothing left to add.

Universality is therefore not a preference among substrates. It is the
necessary and sufficient condition for the tower to have a top. The price is
that expressiveness stops growing on the day it is reached, which is why
everything after this is measured in reach rather than in meanings.

What is here
------------
Eighteen heads. Numbers, pairs, a variable, a function, an application, a
branch, seven arithmetic operations, and quotation. Application and function
with no type discipline give unbounded recursion by self-application, and that
is what makes the set universal; the certificate is in
:mod:`core.cognition.what_the_floor_can_say`, which exhibits composition,
primitive recursion and unbounded search as terms and leans on Kleene for the
rest.

Quotation is what makes the floor its own object. ``as it is written`` turns a
term into a value built of numbers and pairs, so a term can take a term apart,
build another, and hand it back. :data:`THE_INTERPRETER` is the floor's
evaluator written as a term in the floor, which is what turns "the mechanism is
an object of the hypothesis space" from a slogan into a value you can print.

Every run is metered. A universal language has programs that do not stop, so
the machine counts steps and refuses rather than waiting; the language is
universal in the limit of fuel and every single evaluation is bounded. That is
also what makes reach a measurement: a behaviour is in reach at a budget, or it
is not.

What is deliberately not here: types, effects, and anything that can touch the
world. A term computes over numbers and pairs. It cannot open a file, call a
tool, or reach the governor, so admitting a term is never a route to a
privilege she did not have.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

__all__ = [
    "ARITHMETIC",
    "A",
    "Closure",
    "Code",
    "FST",
    "IF",
    "ISPAIR",
    "L",
    "LET",
    "N",
    "NIL",
    "NOTHING",
    "Nothing",
    "BELOW",
    "ENOUGH_STEPS",
    "HOW_MANY_PARTS",
    "LEFTOVER",
    "MINUS",
    "OVER",
    "OutOfFuel",
    "PAIR",
    "PLUS",
    "SAME",
    "TIMES",
    "every_code",
    "steps_taken",
    "QUOTE",
    "SIGNATURE",
    "SND",
    "Stuck",
    "V",
    "Y",
    "Pair",
    "as_list",
    "build",
    "decode",
    "encode",
    "from_list",
    "how_long",
    "read_back",
    "run",
    "written_down",
]

logger = logging.getLogger("Aura.TheFloor")

#: What one evaluation may spend. Not a limit on what the language can say —
#: a limit on what one question costs to ask. Raise it and more is reachable;
#: there is no setting at which everything is.
ENOUGH_STEPS = 200_000


class OutOfFuel(RuntimeError):
    """The step budget ran out. A fact about the budget, not about the term."""


class Stuck(RuntimeError):
    """The term asked for something the floor does not do."""


# ── values ────────────────────────────────────────────────────────────────


class Nothing:
    """The empty thing. One of it, so equality is identity."""

    _only: "Nothing | None" = None

    def __new__(cls) -> "Nothing":
        if cls._only is None:
            cls._only = super().__new__(cls)
        return cls._only

    def __repr__(self) -> str:
        return "nothing"


NOTHING = Nothing()


@dataclass(frozen=True, slots=True)
class Pair:
    """Two things held together. Every structure here is made of these."""

    first: Any
    second: Any


@dataclass(frozen=True, slots=True)
class Closure:
    """A function that remembers where it was written."""

    body: "Code"
    env: tuple[Any, ...]


# ── terms ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Code:
    """A piece of the floor.

    The same three fields a positional term has, because the positional
    language compiles into this one and a second shape would be a second
    language to keep in step.
    """

    head: str
    parts: tuple["Code", ...] = ()
    value: Any = None

    def __repr__(self) -> str:
        if self.head == "a number":
            return str(self.value)
        if self.head == "the one it was given":
            return f"#{self.value}"
        if self.head == "nothing":
            return "nothing"
        inside = " ".join(repr(part) for part in self.parts)
        return f"({self.head}{' ' + inside if inside else ''})"


def _plus(a: int, b: int) -> int:
    return a + b


def _minus(a: int, b: int) -> int:
    return a - b


def _times(a: int, b: int) -> int:
    return a * b


def _over(a: int, b: int) -> int:
    if b == 0:
        raise Stuck("nothing goes into nothing")
    return a // b


def _left_over(a: int, b: int) -> int:
    if b == 0:
        raise Stuck("nothing is left over from nothing")
    return a % b


def _below(a: int, b: int) -> int:
    return 1 if a < b else 0


def _same(a: int, b: int) -> int:
    return 1 if a == b else 0


#: The arithmetic. Seven, matching the seven the positional algebra already
#: had, so the floor is not a wider authored vocabulary than the thing it sits
#: under. Four of the seven are shown definable from the other three in
#: :mod:`core.cognition.what_the_floor_can_say`, which is the check that the
#: choice of instruction set is doing no work.
ARITHMETIC: dict[str, Callable[[int, int], int]] = {
    "plus": _plus,
    "minus": _minus,
    "times": _times,
    "over": _over,
    "left over": _left_over,
    "below": _below,
    "same as": _same,
}

#: Every head, in a fixed order. The order is a contract: quotation encodes a
#: head as its position here, so reordering would change what every written
#: term means. Append only.
SIGNATURE: tuple[str, ...] = (
    "a number",
    "nothing",
    "a pair",
    "the first of",
    "the second of",
    "is it a pair",
    "the one it was given",
    "given a thing",
    "of",
    "if",
    "plus",
    "minus",
    "times",
    "over",
    "left over",
    "below",
    "same as",
    "as it is written",
)

_WHERE_IN_SIGNATURE = {head: at for at, head in enumerate(SIGNATURE)}

#: How many parts each head takes. Derived from nothing else, and everything
#: else derives from it: the machine, the encoder, the reader.
HOW_MANY_PARTS: dict[str, int] = {
    "a number": 0,
    "nothing": 0,
    "a pair": 2,
    "the first of": 1,
    "the second of": 1,
    "is it a pair": 1,
    "the one it was given": 0,
    "given a thing": 1,
    "of": 2,
    "if": 3,
    "as it is written": 1,
    **dict.fromkeys(ARITHMETIC, 2),
}


def how_long(code: Code) -> int:
    """Symbols in the term, written out. The ruler nothing can move.

    Walked with a stack rather than by recursion. A term the machine can run
    can be thousands deep, and a length function that overflows on a term the
    evaluator handles would make the ruler the shorter of the two.
    """
    counted = 0
    edge = [code]
    while edge:
        here = edge.pop()
        counted += 1
        edge.extend(here.parts)
    return counted


# ── the machine ───────────────────────────────────────────────────────────


class _Meter:
    """Counts steps, so a term that will not stop is refused rather than run."""

    __slots__ = ("limit", "used")

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.used = 0

    def step(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise OutOfFuel(f"more than {self.limit} steps")


def _an_integer(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Stuck(f"{what} wanted a number and was given {value!r}")
    return value


def _a_pair(value: Any, what: str) -> Pair:
    if not isinstance(value, Pair):
        raise Stuck(f"{what} wanted a pair and was given {value!r}")
    return value


def run(
    code: Code,
    env: Sequence[Any] = (),
    *,
    fuel: int = ENOUGH_STEPS,
    meter: "_Meter | None" = None,
) -> Any:
    """Evaluate a term, spending at most that many steps.

    Written as a loop over an explicit stack rather than as a recursive
    function. Terms here nest as deep as the programs they run, and the
    interpreter written in the floor runs programs — so a Python recursion
    would hit its own limit long before the fuel ran out, and the failure
    would say "recursion" where the truth is "this term is long".
    """
    counter = meter if meter is not None else _Meter(fuel)
    stack: list[tuple[Any, ...]] = []
    here: Code = code
    scope: tuple[Any, ...] = tuple(env)
    value: Any = None
    returning = False

    while True:
        counter.step()
        if not returning:
            head = here.head
            if head == "a number":
                value, returning = int(here.value or 0), True
            elif head == "nothing":
                value, returning = NOTHING, True
            elif head == "the one it was given":
                which = int(here.value or 0)
                if not 0 <= which < len(scope):
                    raise Stuck(f"nothing was given at #{which}")
                value, returning = scope[which], True
            elif head == "given a thing":
                value, returning = Closure(here.parts[0], scope), True
            elif head == "as it is written":
                value, returning = encode(here.parts[0]), True
            elif head == "of":
                stack.append(("call", here.parts[1], scope))
                here = here.parts[0]
            elif head == "if":
                stack.append(("branch", here.parts[1], here.parts[2], scope))
                here = here.parts[0]
            elif head == "a pair":
                stack.append(("left", here.parts[1], scope))
                here = here.parts[0]
            elif head in {"the first of", "the second of", "is it a pair"}:
                stack.append((head,))
                here = here.parts[0]
            elif head in ARITHMETIC:
                stack.append(("arith", head, here.parts[1], scope))
                here = here.parts[0]
            else:
                raise Stuck(f"nothing on the floor called {head!r}")
            continue

        if not stack:
            return value
        frame = stack.pop()
        kind = frame[0]
        if kind == "call":
            stack.append(("apply", value))
            here, scope, returning = frame[1], frame[2], False
        elif kind == "apply":
            work = frame[1]
            if not isinstance(work, Closure):
                raise Stuck(f"asked {work!r} to be a function")
            here, scope, returning = work.body, (value, *work.env), False
        elif kind == "branch":
            taken = frame[1] if _an_integer(value, "a branch") else frame[2]
            here, scope, returning = taken, frame[3], False
        elif kind == "left":
            stack.append(("right", value))
            here, scope, returning = frame[1], frame[2], False
        elif kind == "right":
            value = Pair(frame[1], value)
        elif kind == "the first of":
            value = _a_pair(value, "the first of").first
        elif kind == "the second of":
            value = _a_pair(value, "the second of").second
        elif kind == "is it a pair":
            value = 1 if isinstance(value, Pair) else 0
        elif kind == "arith":
            stack.append(("arith2", frame[1], value))
            here, scope, returning = frame[2], frame[3], False
        elif kind == "arith2":
            work = ARITHMETIC[frame[1]]
            value = work(
                _an_integer(frame[2], frame[1]), _an_integer(value, frame[1])
            )
        else:  # pragma: no cover - the frame kinds are exhaustive
            raise Stuck(f"a frame nothing knows: {kind!r}")


def steps_taken(code: Code, env: Sequence[Any] = (), *, fuel: int = ENOUGH_STEPS) -> int:
    """How much this cost to run. What a budget is spent in."""
    counter = _Meter(fuel)
    run(code, env, meter=counter)
    return counter.used


# ── quotation ─────────────────────────────────────────────────────────────


def from_list(values: Sequence[Any]) -> Any:
    """A list as nested pairs, ending in nothing."""
    made: Any = NOTHING
    for one in reversed(list(values)):
        made = Pair(one, made)
    return made


def as_list(value: Any) -> list[Any]:
    """Back out again. Raises where the value is not a list."""
    out: list[Any] = []
    while isinstance(value, Pair):
        out.append(value.first)
        value = value.second
    if value is not NOTHING:
        raise Stuck("that is not a list")
    return out


def encode(code: Code) -> Any:
    """A term as a value: its head's place, what it holds, and its parts.

    This is the whole of homoiconicity. A term is numbers and pairs, and
    numbers and pairs are what the floor computes over, so a term can be read,
    taken apart and built by a term. Nothing else is needed for the mechanism
    to become an object of the language it works on.
    """
    where = _WHERE_IN_SIGNATURE.get(code.head)
    if where is None:
        raise Stuck(f"nothing on the floor called {code.head!r}")
    return Pair(
        where,
        Pair(int(code.value or 0), from_list([encode(part) for part in code.parts])),
    )


def decode(value: Any) -> Code:
    """A term from its encoding. The inverse, and checked to be one."""
    outer = _a_pair(value, "a written term")
    where = _an_integer(outer.first, "a written head")
    if not 0 <= where < len(SIGNATURE):
        raise Stuck(f"no head numbered {where}")
    inner = _a_pair(outer.second, "a written term")
    held = _an_integer(inner.first, "what a term holds")
    parts = tuple(decode(one) for one in as_list(inner.second))
    head = SIGNATURE[where]
    if len(parts) != HOW_MANY_PARTS[head]:
        raise Stuck(f"{head!r} takes {HOW_MANY_PARTS[head]} parts, not {len(parts)}")
    return Code(head=head, parts=parts, value=held if head in {"a number", "the one it was given"} else None)


# ── writing terms by hand ─────────────────────────────────────────────────
#
# Nothing below is part of the floor. It is a way of writing floor terms with
# names instead of counting binders, which is the difference between an
# interpreter somebody can read and one nobody can check.


@dataclass(frozen=True, slots=True)
class _Named:
    """An expression with names, before the names become distances."""

    kind: str
    parts: tuple[Any, ...] = ()
    value: Any = None


def V(name: str) -> _Named:
    return _Named("var", value=str(name))


def L(name: str, body: Any) -> _Named:
    return _Named("lam", parts=(body,), value=str(name))


def A(work: Any, *given: Any) -> _Named:
    made: Any = work
    for one in given:
        made = _Named("app", parts=(made, one))
    return made


def N(number: int) -> _Named:
    return _Named("num", value=int(number))


NIL = _Named("nil")


def PAIR(one: Any, other: Any) -> _Named:
    return _Named("pair", parts=(one, other))


def FST(one: Any) -> _Named:
    return _Named("first", parts=(one,))


def SND(one: Any) -> _Named:
    return _Named("second", parts=(one,))


def ISPAIR(one: Any) -> _Named:
    return _Named("ispair", parts=(one,))


def IF(test: Any, then: Any, otherwise: Any) -> _Named:
    return _Named("if", parts=(test, then, otherwise))


def QUOTE(code: Code) -> _Named:
    return _Named("quote", value=code)


def LET(name: str, is_: Any, body: Any) -> _Named:
    return A(L(name, body), is_)


def _arith(head: str) -> Callable[[Any, Any], _Named]:
    def made(one: Any, other: Any) -> _Named:
        return _Named(head, parts=(one, other))

    return made


PLUS = _arith("plus")
MINUS = _arith("minus")
TIMES = _arith("times")
OVER = _arith("over")
LEFTOVER = _arith("left over")
BELOW = _arith("below")
SAME = _arith("same as")


def Y(name: str, body: Any) -> _Named:
    """A term that can call itself, out of nothing but functions.

    ``Y("me", L("n", ... A(V("me"), ...) ...))``. The strict fixed point, since
    the machine evaluates arguments before it applies: wrapping the recursive
    use in a function is what stops it unfolding forever before it is asked.

    This is where universality actually comes from. Everything else on the
    floor is finite work; self-application is what lets a finite term describe
    an unbounded one, and it is why the machine has a meter.
    """
    inner = L(
        "self",
        A(V("_fix"), L("given", A(V("self"), V("self"), V("given")))),
    )
    return LET("_fix", L("_fix", A(inner, inner)), A(V("_fix"), L(name, body)))


def build(expression: Any, scope: Sequence[str] = ()) -> Code:
    """Turn a named expression into a term, counting binders so nobody else has to."""
    if isinstance(expression, Code):
        return expression
    if not isinstance(expression, _Named):
        raise Stuck(f"not something to build: {expression!r}")
    kind = expression.kind
    names = list(scope)
    if kind == "var":
        name = str(expression.value)
        for distance, bound in enumerate(reversed(names)):
            if bound == name:
                return Code("the one it was given", value=distance)
        raise Stuck(f"no binder for {name!r}")
    if kind == "lam":
        inside = build(expression.parts[0], (*names, str(expression.value)))
        return Code("given a thing", parts=(inside,))
    if kind == "num":
        return Code("a number", value=int(expression.value))
    if kind == "nil":
        return Code("nothing")
    if kind == "quote":
        return Code("as it is written", parts=(expression.value,))
    heads = {
        "app": "of",
        "pair": "a pair",
        "first": "the first of",
        "second": "the second of",
        "ispair": "is it a pair",
        "if": "if",
    }
    head = heads.get(kind, kind)
    if head not in HOW_MANY_PARTS:
        raise Stuck(f"nothing on the floor called {head!r}")
    return Code(
        head, parts=tuple(build(part, names) for part in expression.parts)
    )


# ── persistence ───────────────────────────────────────────────────────────


def written_down(code: Code) -> dict[str, Any]:
    """The term as plain data, so what she wrote survives a restart."""
    return {
        "head": code.head,
        "value": code.value,
        "parts": [written_down(part) for part in code.parts],
    }


def read_back(row: Any) -> Code | None:
    """A term from what was written down, or nothing where it does not read.

    The arity is checked against :data:`HOW_MANY_PARTS` rather than against a
    second list of head names, because a second list is what let a head that
    ran perfectly well fail to come back.
    """
    if not isinstance(row, dict):
        return None
    head = str(row.get("head") or "")
    wanted = HOW_MANY_PARTS.get(head)
    if wanted is None:
        return None
    parts = []
    for one in row.get("parts") or ():
        part = read_back(one)
        if part is None:
            return None
        parts.append(part)
    if len(parts) != wanted:
        return None
    return Code(head=head, parts=tuple(parts), value=row.get("value"))


def every_code(
    deepest: int = 3,
    *,
    variables: int = 2,
    constants: Sequence[int] = (0, 1, 2),
    also: Sequence[Code] = (),
) -> Iterator[Code]:
    """Every term the floor admits, shortest first, up to that size.

    Here for the same reason the positional algebra has one: a claim that
    nothing shorter says a thing needs the shorter things to have been walked.

    ``also`` is what she has already admitted, offered as leaves. That is the
    only channel by which a long term becomes reachable — shortest-first over
    a universal language reaches a few dozen symbols and no further, which is
    Levin's bound rather than a defect here, and a library is what moves the
    horizon rather than a bigger budget.
    """
    leaves = [
        Code("nothing"),
        *(Code("a number", value=int(k)) for k in constants),
        *(Code("the one it was given", value=n) for n in range(max(0, variables))),
        *also,
    ]
    by_size: dict[int, list[Code]] = {1: list(leaves)}
    yield from leaves
    ones = ("the first of", "the second of", "is it a pair", "given a thing",
            "as it is written")
    twos = ("a pair", "of", *ARITHMETIC)
    for size in range(2, max(2, deepest) + 1):
        grown: list[Code] = []
        for head in ones:
            for part in by_size.get(size - 1, ()):
                made = Code(head, parts=(part,))
                grown.append(made)
                yield made
        for head in twos:
            for left_size in range(1, size - 1):
                for left in by_size.get(left_size, ()):
                    for right in by_size.get(size - 1 - left_size, ()):
                        made = Code(head, parts=(left, right))
                        grown.append(made)
                        yield made
        for test_size in range(1, size - 2):
            for then_size in range(1, size - 1 - test_size):
                rest = size - 1 - test_size - then_size
                for test in by_size.get(test_size, ()):
                    for then in by_size.get(then_size, ()):
                        for otherwise in by_size.get(rest, ()):
                            made = Code("if", parts=(test, then, otherwise))
                            grown.append(made)
                            yield made
        by_size[size] = grown
