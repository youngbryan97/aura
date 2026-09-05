"""A rule to find, from sparse examples, that nobody wrote down.

Fluid intelligence, in the only form that can be sealed: the rule is composed
at generation time out of primitives the generator holds, and the composition
is drawn from the freeze seed. So the answer is not in the training data of
anything, because the answer did not exist until the freeze existed.

The primitives are deliberately ordinary — an offset, a mirror, a grouping, a
value map. What makes an instance hard is the composition depth and the fact
that a shallow reading of the examples fits several of them.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ARuleToFind", "invent_the_rules"]


def _offset(k: int) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    return lambda row: tuple(row[k:] + row[:k])


def _mirror(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(reversed(row))


def _swap_ends(depth: int) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    def apply(row: tuple[Any, ...]) -> tuple[Any, ...]:
        made = list(row)
        if len(made) > 2 * depth:
            made[depth], made[-1 - depth] = made[-1 - depth], made[depth]
        return tuple(made)

    return apply


def _group(span: int) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    def apply(row: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(
            row[place]
            for residue in range(span)
            for place in range(residue, len(row), span)
        )

    return apply


def _add(delta: int) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    return lambda row: tuple(
        (one + delta) if isinstance(one, int) else one for one in row
    )


def _times(factor: int) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    return lambda row: tuple(
        (one * factor) if isinstance(one, int) else one for one in row
    )


#: Named so a report can say what the answer was, which is what makes a
#: failure readable rather than a zero.
_PRIMITIVES: tuple[tuple[str, Callable[..., Any], tuple[int, ...]], ...] = (
    ("offset by {0}", _offset, (1, 2, 3)),
    ("mirror", lambda: _mirror, ()),
    ("swap the cells {0} in from each end", _swap_ends, (0, 1, 2)),
    ("group every {0}", _group, (2, 3)),
    ("add {0} to each", _add, (1, 2, 5, -3)),
    ("multiply each by {0}", _times, (2, 3)),
)


@dataclass(frozen=True)
class ARuleToFind:
    """One sealed instance: examples, a question, and the answer."""

    name: str
    said: str
    shown: tuple[tuple[tuple[Any, ...], tuple[Any, ...]], ...]
    asked: tuple[Any, ...]
    answer: tuple[Any, ...]
    depth: int
    held_out: tuple[tuple[tuple[Any, ...], tuple[Any, ...]], ...] = ()

    def is_right(self, said: Any) -> bool:
        try:
            return tuple(said) == self.answer
        except TypeError:
            return False


def _compose(rng: random.Random, depth: int) -> tuple[str, Callable[..., Any]]:
    parts: list[str] = []
    steps: list[Callable[..., Any]] = []
    for _ in range(depth):
        said, make, options = rng.choice(_PRIMITIVES)
        if options:
            chosen = rng.choice(options)
            steps.append(make(chosen))
            parts.append(said.format(chosen))
        else:
            steps.append(make())
            parts.append(said)

    def apply(row: tuple[Any, ...]) -> tuple[Any, ...]:
        for step in steps:
            row = step(row)
        return row

    return ", then ".join(parts), apply


def _apply_the_said(said: str, row: tuple[Any, ...]) -> tuple[Any, ...] | None:
    """Apply a rule from its own description. What lets the world be asked.

    An environment that cannot answer a question about itself is one where
    asking is not a move, and refusing for want of evidence is then the only
    thing a system can do with an ambiguous instance. Rebuilt from the
    description rather than kept as a closure, so the answer comes from the
    same words the instance was named by.
    """

    made: Any = tuple(row)
    for part in str(said).split(", then "):
        step = _from_its_name(part)
        if step is None:
            return None
        try:
            made = step(tuple(made))
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            return None
    return tuple(made)


def _from_its_name(said: str) -> Callable[[tuple[Any, ...]], tuple[Any, ...]] | None:
    """One primitive, from the words that name it."""

    import re

    for pattern, make, _options in _PRIMITIVES:
        if "{0}" not in pattern:
            if pattern == said:
                return make()
            continue
        shape = re.escape(pattern).replace(r"\{0\}", r"(-?\d+)")
        found = re.fullmatch(shape, said)
        if found:
            return make(int(found.group(1)))
    return None


def invent_the_rules(
    seed: int, *, how_many: int = 30, depth: int = 3, shown: int = 3
) -> tuple[ARuleToFind, ...]:
    """Sealed rule-induction instances, drawn from the freeze.

    Every instance shows a few worked examples at different lengths and asks
    for one more. Different lengths on purpose: a rule fitted at one length
    can be a coincidence about that length, and a system that has only ever
    been shown one cannot tell "the ends swap" from "positions 0 and 3 swap".
    """

    rng = random.Random(seed)
    made: list[ARuleToFind] = []
    seen: set[str] = set()
    while len(made) < how_many:
        said, apply = _compose(rng, depth)
        if said in seen:
            continue
        lengths = rng.sample(range(5, 12), shown + 2)
        try:
            pairs = tuple(
                (tuple(range(length)), tuple(apply(tuple(range(length)))))
                for length in lengths
            )
        except (TypeError, ValueError, IndexError):
            continue
        if any(before == after for before, after in pairs):
            continue  # an identity instance asks nothing
        seen.add(said)
        made.append(
            ARuleToFind(
                name=f"rule {len(made)}",
                said=said,
                shown=pairs[:shown],
                asked=pairs[shown][0],
                answer=pairs[shown][1],
                depth=depth,
                held_out=pairs[shown + 1 :],
            )
        )
    return tuple(made)
