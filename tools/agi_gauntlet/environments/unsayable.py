"""Families her language cannot say yet, so acquiring one is visible.

Gate 3 measures getting better at something. Gate 9 is supposed to measure
acquiring a skill she does not have, and it ran the same code — which an
external review noticed and was right about. The difference is not a matter of
degree: improving is a curve on a thing you can already do, and acquiring is a
step from nought.

So the family here is chosen to be outside the base language rather than hard
within it. Everything the positional side can say reads ONE cell per position:
``after[i] = before[g(i, n)]``. These read two and combine them, which is the
shape ``core/cognition/a_rule_with_no_shape.py`` exists for and which the
ladder has to write a maker to reach. Before she grows, the answer is not
merely wrong, it is unavailable; after, it is there and it persists.

Drawn from the freeze seed, so the particular pairing and the particular
operation did not exist until the freeze did.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["AFamilyOutsideTheLanguage", "families_she_cannot_say"]


def _sources(seed: int) -> tuple[Callable[[int, int], int], Callable[[int, int], int], str]:
    rng = random.Random(seed)
    made: list[tuple[Callable[[int, int], int], str]] = [
        (lambda index, size: (index + 1) % size, "the next one round"),
        (lambda index, size: (index + 2) % size, "two on round"),
        (lambda index, size: size - 1 - index, "the mirror of it"),
        (lambda index, size: (index * 2) % size, "twice its place"),
        (lambda index, size: (size - index) % size, "its place from the end"),
    ]
    first, second = rng.sample(made, 2)
    return first[0], second[0], f"{first[1]} and {second[1]}"


@dataclass(frozen=True)
class AFamilyOutsideTheLanguage:
    """One rule that reads two cells per position, and the states shown."""

    name: str
    rule: Callable[[tuple[Any, ...]], tuple[Any, ...]]
    shown: tuple[tuple[int, ...], ...]
    asked: tuple[int, ...]

    @property
    def answer(self) -> tuple[Any, ...]:
        return tuple(self.rule(self.asked))

    def as_a_question(self) -> str:
        body = " ".join(
            f"{' '.join(map(str, state))} becomes "
            f"{' '.join(map(str, self.rule(state)))}."
            for state in self.shown
        )
        return f"{body} What does {' '.join(map(str, self.asked))} become?"


def families_she_cannot_say(
    seed: int, *, how_many: int = 8, size: int = 5, shown: int = 6
) -> list[AFamilyOutsideTheLanguage]:
    """Sealed families, each needing two sources and an operation."""

    rng = random.Random(seed ^ 0x5A17)
    operations: tuple[tuple[str, Callable[[int, int], int]], ...] = (
        ("the larger of", max),
        ("the smaller of", min),
        ("the sum of", lambda a, b: (a + b) % 10),
    )
    found: list[AFamilyOutsideTheLanguage] = []
    for at in range(how_many):
        first, second, said = _sources(seed ^ (at * 7717))
        what, combine = operations[rng.randrange(len(operations))]

        def rule(
            state: tuple[Any, ...],
            _f: Any = first,
            _s: Any = second,
            _c: Any = combine,
        ) -> tuple[Any, ...]:
            n = len(state)
            return tuple(_c(state[_f(i, n)], state[_s(i, n)]) for i in range(n))

        states = [
            tuple(rng.randrange(1, 10) for _ in range(size)) for _ in range(shown + 1)
        ]
        found.append(
            AFamilyOutsideTheLanguage(
                name=f"{what} {said}",
                rule=rule,
                shown=tuple(states[:shown]),
                asked=states[-1],
            )
        )
    return found
