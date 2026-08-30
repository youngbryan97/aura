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
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

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
    """Every expression over the pair, shortest first.

    Shortest first because the shortest rule that survives what it never saw is
    the one to believe, and because the space grows fast enough that the order
    decides what is reachable at all.
    """
    leaves = [Expression("the first"), Expression("the second")]
    leaves += [Expression("a fixed number", value=fixed) for fixed in constants]
    yield from leaves
    standing = list(leaves)
    for _ in range(max(0, int(deepest))):
        grown: list[Expression] = []
        for kind in HOW_TO_COMBINE:
            for left in standing:
                for right in standing:
                    made = Expression(kind, parts=(left, right))
                    grown.append(made)
                    yield made
        standing = leaves + grown


def _constants_from(pairs: Sequence[tuple[Any, Any, Any]]) -> tuple[Any, ...]:
    """Numbers the examples themselves put on the table.

    Solved for, never searched. Every candidate here is a difference the
    examples show, so a rule needing "and then add four" gets its four from an
    example rather than from a range somebody picked.
    """
    found: set[Any] = set()
    for one, other, got in pairs:
        for candidate in (got, got - one, got - other, one - other):
            if isinstance(candidate, int) and abs(candidate) <= 64:
                found.add(candidate)
    return tuple(sorted(found, key=lambda value: (abs(value), value)))


def an_operation_that_generalises(
    pairs: Sequence[tuple[Any, Any, Any]], *, deepest: int = 2
) -> Expression | None:
    """A rule for what was done to these pairs, which answers pairs it never saw.

    Fitted on half and judged on the half it never saw, so a rule that merely
    memorised the examples cannot come back — which is exactly the failure a
    lookup table makes unavoidable.
    """
    seen = [(one, other, got) for one, other, got in pairs]
    if len(seen) < ENOUGH_HELD_BACK + 1:
        return None
    solving, judging = seen[0::2], seen[1::2]
    if len(judging) < ENOUGH_HELD_BACK:
        return None
    constants = _constants_from(solving)
    for candidate in every_expression(constants, deepest=deepest):
        try:
            if any(candidate(one, other) != got for one, other, got in solving):
                continue
            if any(candidate(one, other) != got for one, other, got in judging):
                continue
        except (ArithmeticError, TypeError, ValueError):
            continue
        logger.info(
            "an operation that generalises: %s, from %d example(s) and held to %d",
            candidate.name,
            len(solving),
            len(judging),
        )
        return candidate
    return None


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
