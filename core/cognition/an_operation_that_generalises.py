"""Deriving what was done to a pair, as a rule rather than as a table.

Reading a correspondence off examples and keeping it is honest, and it is
weak. A table mapping the pairs she saw to the results she saw refuses every
pair she did not see, so a word derived that way says nothing about a value
outside the examples — which makes it a memory wearing the shape of a word.

The stronger thing is to derive the rule itself. Shown

    (7, 3) -> 4      (9, 2) -> 7      (5, 5) -> 0

a table learns three facts. What is actually there is one, and knowing it
answers a pair nobody has ever shown her.

Which is arrived at the same way everything else here is: by composing, never
by choosing from a list. An operation is an expression over the two values, the
expressions are enumerated shortest first, and one is kept when it survives the
examples it was not fitted on. Nothing in the space was written down as a
candidate; the space is the closure of a handful of ways to combine two
numbers, and any of its points may come out.

Constants are solved for rather than searched. A rule that is "something, and
then add four" has its four read off the first example and checked against the
rest, because a constant that has to be searched for is a constant somebody
chose the range of.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Expression",
    "HOW_TO_COMBINE",
    "an_operation_that_generalises",
    "every_expression",
    "read_back",
    "written_down",
]

logger = logging.getLogger("Aura.AnOperationThatGeneralises")

#: How many of the examples are kept back. A rule fitted to everything it has
#: seen has been tested against nothing, and a table always fits everything.
ENOUGH_HELD_BACK = 2


def _difference(one: Any, other: Any) -> Any:
    return one - other


def _how_far_apart(one: Any, other: Any) -> Any:
    return abs(one - other)


def _sum(one: Any, other: Any) -> Any:
    return one + other


def _product(one: Any, other: Any) -> Any:
    return one * other


def _the_larger(one: Any, other: Any) -> Any:
    return one if one >= other else other


def _the_smaller(one: Any, other: Any) -> Any:
    return one if one <= other else other


def _what_is_left_over(one: Any, other: Any) -> Any:
    if other == 0:
        raise ZeroDivisionError("nothing is left over from nothing")
    return one % other


def _how_many_times(one: Any, other: Any) -> Any:
    if other == 0:
        raise ZeroDivisionError("nothing goes into nothing")
    return one // other


#: Ways of combining two numbers. Small on purpose: what makes the space large
#: is composing them, and a large set of primitives buys the same reach at a
#: much worse rate of coincidence.
HOW_TO_COMBINE: dict[str, Callable[[Any, Any], Any]] = {
    "minus": _difference,
    "how far apart they are": _how_far_apart,
    "added": _sum,
    "multiplied": _product,
    "the larger": _the_larger,
    "the smaller": _the_smaller,
    "what is left over": _what_is_left_over,
    "how many times it goes in": _how_many_times,
}


@dataclass(frozen=True)
class Expression:
    """What was done to a pair, written so it can be read and run.

    ``kind`` is either one of the two values, a constant, or a way of
    combining. It is the same shape at every depth, so an expression built out
    of expressions needs nothing added.
    """

    kind: str
    parts: tuple[Any, ...] = ()
    value: Any = None

    @property
    def name(self) -> str:
        if self.kind == "the first":
            return "the first"
        if self.kind == "the second":
            return "the second"
        if self.kind == "a fixed number":
            return str(self.value)
        inside = " and ".join(part.name for part in self.parts)
        return f"{inside}, {self.kind}"

    def __call__(self, one: Any, other: Any) -> Any:
        if self.kind == "the first":
            return one
        if self.kind == "the second":
            return other
        if self.kind == "a fixed number":
            return self.value
        combine = HOW_TO_COMBINE[self.kind]
        return combine(*(part(one, other) for part in self.parts))

    def how_long(self) -> int:
        if not self.parts:
            return 1
        return 1 + sum(part.how_long() for part in self.parts)


def every_expression(
    constants: Sequence[Any] = (), *, deepest: int = 2
) -> Iterator[Expression]:
    """Every expression over the pair, shortest first, by exact size.

    Built size by size, where an expression of size n is a way of combining one
    of size a with one of size b and a + b + 1 = n. Growing instead by
    squaring everything built so far revisits the small ones at every round: on
    fifteen leaves that turns a search of two thousand into one of twenty-six
    million, and buries the two-symbol answer underneath it.
    """
    leaves = [Expression("the first"), Expression("the second")]
    leaves += [Expression("a fixed number", value=fixed) for fixed in constants]
    by_size: dict[int, list[Expression]] = {1: leaves}
    yield from leaves
    for size in range(3, 2 * max(1, int(deepest)) + 2, 2):
        grown: list[Expression] = []
        for left_size in range(1, size - 1):
            right_size = size - left_size - 1
            for left in by_size.get(left_size, ()):
                for right in by_size.get(right_size, ()):
                    for kind in HOW_TO_COMBINE:
                        made = Expression(kind, parts=(left, right))
                        grown.append(made)
                        yield made
        by_size[size] = grown


def _constants_from(pairs: Sequence[tuple[Any, Any, Any]]) -> tuple[Any, ...]:
    """Numbers the examples themselves put on the table, and keep putting there.

    Solved for, never searched. Every candidate is a difference the examples
    show, and one that shows up in a single example is a coincidence of that
    example — taking the union across all of them floods the search with
    accidents until the real rule sits past the budget. A number has to recur
    to be a number the rule is made of.

    A difference and its negation are one fact read in either direction, so
    both are offered on the same evidence.
    """
    how_often: dict[Any, int] = {}
    for one, other, got in pairs:
        for candidate in {got, got - one, got - other, one - other}:
            if isinstance(candidate, int) and abs(candidate) <= 64:
                how_often[candidate] = how_often.get(candidate, 0) + 1
    recurring = {value for value, seen in how_often.items() if seen >= 2}
    if not recurring:
        recurring = set(how_often)
    both_ways = recurring | {-value for value in recurring}
    return tuple(sorted(both_ways, key=lambda value: (abs(value), value)))


#: Where inverting an operation leaves the operand genuinely free rather than
#: determined. ``max(a, 7) == 7`` says only that a is at most seven, so there
#: is nothing to solve for and the branch falls back to what she can name.
UNCONSTRAINED = None


def _what_the_first_must_be(kind: str, other: Any, got: Any) -> set[Any] | None:
    """Given ``kind(unknown, other) == got``, the values unknown could have had.

    This is what makes the search directed. Enumerating expressions and testing
    them asks "does this happen to work?" a great many times; inverting the
    operation asks "what would have had to be true?" once, and the answer is
    usually a single number. A rule three operations deep is then three
    questions rather than a walk through millions of expressions.

    Nothing is returned where the operation does not determine its operand, and
    that is different from an empty set: empty means no value works, so the
    branch is dead.
    """
    if not isinstance(other, int) or not isinstance(got, int):
        return UNCONSTRAINED
    if kind == "minus":
        return {got + other}
    if kind == "added":
        return {got - other}
    if kind == "multiplied":
        if other == 0:
            return UNCONSTRAINED if got == 0 else set()
        return {got // other} if got % other == 0 else set()
    if kind == "how far apart they are":
        return {other + got, other - got} if got >= 0 else set()
    if kind == "the larger":
        if other > got:
            return set()
        return {got} if other < got else UNCONSTRAINED
    if kind == "the smaller":
        if other < got:
            return set()
        return {got} if other > got else UNCONSTRAINED
    if kind == "how many times it goes in":
        if other == 0:
            return set()
        low = got * other
        return set(range(low, low + abs(other))) if abs(other) <= 32 else UNCONSTRAINED
    if kind == "what is left over":
        return UNCONSTRAINED
    return UNCONSTRAINED


def _what_the_second_must_be(kind: str, one: Any, got: Any) -> set[Any] | None:
    """Given ``kind(one, unknown) == got``, the values unknown could have had."""
    if not isinstance(one, int) or not isinstance(got, int):
        return UNCONSTRAINED
    if kind == "minus":
        return {one - got}
    if kind == "added":
        return {got - one}
    if kind == "multiplied":
        if one == 0:
            return UNCONSTRAINED if got == 0 else set()
        return {got // one} if got % one == 0 else set()
    if kind == "how far apart they are":
        return {one + got, one - got} if got >= 0 else set()
    if kind == "the larger":
        if one > got:
            return set()
        return {got} if one < got else UNCONSTRAINED
    if kind == "the smaller":
        if one < got:
            return set()
        return {got} if one > got else UNCONSTRAINED
    if kind == "what is left over":
        # a % b == got means b divides a - got and b > got.
        if one < got:
            return set()
        room = one - got
        if room == 0:
            return UNCONSTRAINED
        return {
            divisor
            for divisor in range(max(1, got + 1), room + 1)
            if room % divisor == 0
        }
    if kind == "how many times it goes in":
        if got == 0:
            return UNCONSTRAINED
        return {
            divisor
            for divisor in range(1, abs(one) + 1)
            if one // divisor == got
        }
    return UNCONSTRAINED


def _fits(candidate: Expression, examples: Sequence[tuple[Any, Any, Any]]) -> bool:
    try:
        return all(candidate(one, other) == got for one, other, got in examples)
    except (ArithmeticError, TypeError, ValueError):
        return False


def _work_out(
    examples: Sequence[tuple[Any, Any, Any]],
    leaves: Sequence[Expression],
    deepest: int,
) -> Iterator[Expression]:
    """Every expression accounting for these, shortest first, worked out by inverting.

    Yields rather than returns, because the first rule that accounts for the
    half she solved on is not always the one that survives the half she never
    saw — and stopping at the first would throw away the rule that does.

    Shortest first falls out of the shape: everything nameable is tried before
    anything is taken apart, and taking something apart asks the same question
    of a smaller problem.
    """
    for leaf in leaves:
        if _fits(leaf, examples):
            yield leaf
    if deepest <= 0:
        return
    # Inverting says nothing about an operand it does not determine: knowing
    # max(a, 4) is 4 says only that a is at most four. Dropping those branches
    # would lose every rule with one in it, so what she can already name is
    # combined directly as well. Small, and it is what makes the directed
    # search complete rather than merely fast.
    for kind in HOW_TO_COMBINE:
        for left in leaves:
            for right in leaves:
                made = Expression(kind, parts=(left, right))
                if _fits(made, examples):
                    yield made
    for kind in HOW_TO_COMBINE:
        for known in leaves:
            for solve, place in (
                (_what_the_first_must_be, "first"),
                (_what_the_second_must_be, "second"),
            ):
                for choice in _needed(examples, known, kind, solve):
                    smaller = [
                        (one, other, value)
                        for (one, other, _got), value in zip(examples, choice)
                    ]
                    for inner in _work_out(smaller, leaves, deepest - 1):
                        parts = (inner, known) if place == "first" else (known, inner)
                        made = Expression(kind, parts=parts)
                        if _fits(made, examples):
                            yield made


def _needed(
    examples: Sequence[tuple[Any, Any, Any]],
    known: Expression,
    kind: str,
    solve: Callable[[str, Any, Any], set[Any] | None],
) -> list[tuple[Any, ...]]:
    """What the unsolved side had to be at every example, as whole assignments.

    A branch is dropped the moment one example admits no value. Where an
    operation leaves its operand free at some example, nothing here can say
    what it was, so the branch is dropped as well rather than guessed at.
    """
    wanted: list[set[Any]] = []
    for one, other, got in examples:
        try:
            side = known(one, other)
        except (ArithmeticError, TypeError, ValueError):
            return []
        values = solve(kind, side, got)
        if values is UNCONSTRAINED or not values:
            return []
        wanted.append(values)
    settled = [next(iter(values)) for values in wanted]
    if all(len(values) == 1 for values in wanted):
        return [tuple(settled)]
    # Two readings at most, and only where the operation genuinely admits two:
    # how far apart something is says the value was above or below, never which.
    both: list[tuple[Any, ...]] = []
    for pick in (min, max):
        both.append(tuple(pick(values) for values in wanted))
    return both


def an_operation_that_generalises(
    pairs: Sequence[tuple[Any, Any, Any]], *, deepest: int = 3
) -> Expression | None:
    """A rule for what was done to these pairs, which answers pairs it never saw.

    Fitted on half and judged on the half it never saw, so a rule that merely
    memorised the examples cannot come back — which is exactly the failure a
    lookup table makes unavoidable.

    Worked out by inverting, never by enumerating. There is no budget here and
    nothing is cut off part way, so nothing coming back means no rule of this
    depth accounts for the examples rather than that she ran out of room.
    """
    seen = [(one, other, got) for one, other, got in pairs]
    if len(seen) < ENOUGH_HELD_BACK + 1:
        return None
    solving, judging = seen[0::2], seen[1::2]
    if len(judging) < ENOUGH_HELD_BACK:
        return None
    leaves = [Expression("the first"), Expression("the second")]
    leaves += [
        Expression("a fixed number", value=fixed)
        for fixed in _constants_from(solving)
    ]
    # Deepened one step at a time. The recursion goes down before it goes
    # across, so asking it for depth three hands back a three-deep rule while a
    # one-deep rule accounting for the same examples sits unreached — and the
    # shorter rule is the one to believe.
    found = None
    for allowed in range(1, max(1, int(deepest)) + 1):
        found = next(
            (
                candidate
                for candidate in _work_out(solving, leaves, allowed)
                if _fits(candidate, judging)
            ),
            None,
        )
        if found is not None:
            break
    if found is None:
        return None
    logger.info(
        "an operation that generalises: %s, from %d example(s) and held to %d",
        found.name,
        len(solving),
        len(judging),
    )
    return found


def written_down(rule: Expression) -> dict[str, Any]:
    """The rule as plain data, so what she worked out survives a restart.

    The shape, never a pickled object. It reconstructs exactly, a person can
    read it, and it cannot name anything outside ``HOW_TO_COMBINE``.
    """
    return {
        "kind": rule.kind,
        "value": rule.value,
        "parts": [written_down(part) for part in rule.parts],
    }


def read_back(row: Any) -> Expression | None:
    """A rule from what was written down, or nothing when it does not read."""
    if not isinstance(row, dict):
        return None
    kind = str(row.get("kind") or "")
    if kind not in {"the first", "the second", "a fixed number"} and kind not in HOW_TO_COMBINE:
        return None
    parts = tuple(
        part for part in (read_back(one) for one in row.get("parts") or ()) if part
    )
    if kind in HOW_TO_COMBINE and len(parts) != 2:
        return None
    return Expression(kind=kind, parts=parts, value=row.get("value"))
