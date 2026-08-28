"""Proving a world is outside the language, rather than failing to find it.

"No form fits" is the same sentence whether the shape is unreachable or the
observations are thin, and the mechanism said it for both. That made every
failure look alike and gave nothing to act on: a world needing another example
and a world needing a different kind of language returned the same nothing.

There is a proof available, and it is cheap.

Every positional program computes ``after[i] = before[f(i, n)]``, where ``f``
sees the position and the length and never the cells. Compose two of them::

    T_f(T_g(x))[i] = T_g(x)[f(i,n)] = x[g(f(i,n), n)]

which is ``T_h`` for ``h(i,n) = g(f(i,n), n)`` — value-blind again. By
induction every finite composition of value-blind forms is value-blind, so no
search depth introduces dependence on the cells. Depth cannot be the answer.

That turns a search failure into a decidable question. For each output position,
intersect the source positions the observations allow. An empty intersection
means no ``f(i, n)`` exists at all — not that this basis lacks it, that none
does — and searching harder is provably wasted.

Deliberately quiet where it should be. Grouping cells by residue is value-blind
and stays inside the envelope, so this says nothing about it; the shape a
person had to add by hand was never a case for a different KIND of language,
and a detector that escalated it would be measuring its own appetite.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["LanguageVerdict", "certify"]


@dataclass(frozen=True)
class LanguageVerdict:
    """What the observations prove about the language, not about the search."""

    #: "outside" — no value-blind rule can explain these, at any depth.
    #: "inside" — a rule exists; whether this basis finds it is a search
    #: question. "undecided" — nothing is refuted yet, so more observations
    #: are the answer and a wider language is not.
    standing: str
    reason: str
    #: The output position whose sources contradict, when there is one. It is
    #: the witness: a person can read the transitions at that position and see
    #: the contradiction without trusting this code.
    position: int | None = None
    length: int | None = None

    @property
    def proven_outside(self) -> bool:
        return self.standing == "outside"


def _sources(before: Sequence[Any], after: Sequence[Any], place: int) -> set[int]:
    """Every position in ``before`` that could have supplied ``after[place]``.

    All of them, never a tie-break. A repeated value that forced a choice here
    would manufacture the contradiction this is trying to detect.
    """

    wanted = after[place]
    return {index for index, value in enumerate(before) if value == wanted}


def certify(transitions: Sequence[Any]) -> LanguageVerdict:
    """What these observations settle about a value-blind language.

    Two proofs, in the order they are cheap.
    """

    observed = [
        (tuple(item.before), tuple(item.after))
        for item in transitions
        if item is not None
    ]
    if not observed:
        return LanguageVerdict("undecided", "nothing was observed")

    for before, after in observed:
        if len(before) != len(after):
            return LanguageVerdict(
                "outside",
                (
                    f"a state of {len(before)} became one of {len(after)}, and "
                    "every rule of this shape puts one cell at every position "
                    "of a state the same length as the one it read"
                ),
                length=len(before),
            )

    # The cells have to be the SAME cells for the rest of this to mean
    # anything.
    #
    # The first version of this proved "mirror, then add one to every cell"
    # outside the language. It is squarely inside it — a value-blind rule about
    # where cells come from, and a map applied to what they hold. Every source
    # set was empty, not because the observations contradicted each other but
    # because nothing in the state held the value being looked for, and empty
    # was being read as contradiction.
    #
    # So: where the cells were transformed, this proof does not apply and does
    # not get claimed. Saying nothing is the correct output of a test whose
    # premise does not hold.
    for before, after in observed:
        if sorted(map(repr, before)) != sorted(map(repr, after)):
            return LanguageVerdict(
                "undecided",
                (
                    "the cells themselves changed, so where each one came from "
                    "cannot be read off the values, and nothing here is proved "
                    "either way"
                ),
            )

    by_length: dict[int, list[tuple[tuple, tuple]]] = defaultdict(list)
    for before, after in observed:
        by_length[len(before)].append((before, after))

    loose = 0
    for length, group in sorted(by_length.items()):
        for place in range(length):
            allowed: set[int] | None = None
            for before, after in group:
                here = _sources(before, after, place)
                allowed = here if allowed is None else (allowed & here)
                if not allowed:
                    break
            if not allowed:
                return LanguageVerdict(
                    "outside",
                    (
                        f"at length {length}, position {place} had to take from "
                        "a different place in different observations, so no "
                        "rule reading only the position and the length can "
                        "say it — and composing such rules only ever makes "
                        "another one"
                    ),
                    position=place,
                    length=length,
                )
            loose += len(allowed) - 1

    if loose:
        # Every position still has more than one story that fits. That is not a
        # language problem and it is not a search problem: it is a shortage of
        # evidence, and it was being reported as the same nothing as the other
        # two.
        return LanguageVerdict(
            "undecided",
            (
                "a repeated cell leaves more than one place it could have come "
                "from, so more than one rule still fits everything shown — "
                "another example would settle it and a wider language would not"
            ),
        )

    return LanguageVerdict(
        "inside",
        (
            "the observations pin one source for every position, so a world "
            "that will not solve is a search question rather than a language "
            "one"
        ),
    )
