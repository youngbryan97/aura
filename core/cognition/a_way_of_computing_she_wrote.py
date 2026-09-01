"""Writing a head, when no word and no maker will do.

The ladder in `sequence_induction` already goes: a word she derives, an
operation she derives, a recipe she composes, a maker she writes. Each rung is
a term in the positional algebra, and every one of them is bounded by the same
thing —
:mod:`core.cognition.what_the_old_language_cannot_say` shows the heads those
terms are built from cap what any of them can say by a polynomial in the length
of the state, and gives a rule inside the range a position lives in that none
of them says. Past that point the answer is not another word. It is a way of
computing the language did not have.

Which is the rung this module adds, and it is the last one, because a head is a
term on a universal floor and there is nothing above universal.

How a candidate is arrived at
-----------------------------
Not from a list of operation kinds, and not from a model asked to name one.
From the correspondence the examples already show. `_where_each_came_from`
reads off, for each length, where each place took its value from. A head is
then a function of four things — the position, the length, and what each of its
two parts says at every position — and the question is which function agrees
with the correspondence. That is an induction over the floor, run the way every
other induction here is run: shortest first, fitted on half the evidence,
judged on the half it never saw, and refused when the half it never saw refuses
it.

What limits it, honestly
------------------------
Shortest-first over a universal language reaches a few dozen symbols. That is
Levin's bound and no budget removes it. A head needing a fixed point is thirty
symbols of pure application before it says anything, and no search here will
ever stumble on one.

What moves the horizon is the library. Everything she has already admitted is
offered as a leaf, so a head that is unreachable today because its pieces are
missing becomes short tomorrow because they are not. That is not a workaround —
it is the developmental claim itself, and it is what
`tests/test_a_head_reaches_further_once_she_has_one.py` measures by taking the
first head away and watching the second become unreachable at the same budget.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

from core.cognition.the_floor_she_stands_on import (
    Code,
    OutOfFuel,
    Stuck,
    every_code,
    from_list,
    how_long,
    run,
)

__all__ = [
    "AWayOfComputing",
    "WHAT_A_HEAD_IS_GIVEN",
    "a_way_of_computing_she_wrote",
    "as_a_head",
    "what_each_part_says",
]

logger = logging.getLogger("Aura.AWayOfComputingSheWrote")

#: What one candidate may spend on one position. A search checks millions.
_A_CANDIDATE_MAY_SPEND = 4_000

#: How long one head synthesis may run. The same allowance a maker gets, for
#: the same reason: the caller says how long there is and the widening spends
#: it on the likeliest first.
_AS_LONG_AS_A_MAKER_GETS = 20.0

#: What a head is given, innermost first. A head is written with names and read
#: with distances, and this is the one place the two have to agree.
#:
#: Both forms of each part, because they cost different amounts to use. What a
#: part says HERE is one symbol and is what nearly every head wants; what it
#: says everywhere is a list, and reading a list at a computed place needs a
#: fixed point, which is thirty symbols before it says anything. Offering only
#: the second would have made every head unreachable; offering only the first
#: would have made a head unable to do what ``through`` does.
WHAT_A_HEAD_IS_GIVEN: tuple[str, ...] = (
    "everything the second part says",
    "everything the first part says",
    "what the second part says here",
    "what the first part says here",
    "how long the state is",
    "where it is",
)


@dataclass(frozen=True)
class AWayOfComputing:
    """A head she wrote, with what it was fitted on and what it survived."""

    body: Code
    #: The names of the words that went into its parts.
    over: tuple[str, ...]
    #: Lengths it was fitted at, and lengths it was only judged at.
    fitted_at: tuple[int, ...]
    judged_at: tuple[int, ...]

    @property
    def how_long(self) -> int:
        return how_long(self.body)

    def describes(self) -> str:
        return (
            f"{self.how_long} symbols over {', '.join(self.over)}, fitted at "
            f"{self.fitted_at} and held to {self.judged_at}"
        )


def what_each_part_says(
    word: Callable[[int, int], int], size: int
) -> Any:
    """Everything a word says at a state this long, as a list on the floor.

    Finite because a word is a pure function of where it is asked and how long
    the thing is. Handing this rather than one number is what lets a head read
    a part somewhere other than here.
    """
    return from_list([int(word(at, size)) % max(1, size) for at in range(size)])


def as_a_head(body: Code) -> Code:
    """Close a body over everything a head is given, outermost binder last."""
    made = body
    for _ in WHAT_A_HEAD_IS_GIVEN:
        made = Code("given a thing", parts=(made,))
    return made


def _computes(
    body: Code,
    wanted: dict[int, tuple[int, ...]],
    tables: dict[int, tuple[Any, Any]],
) -> bool:
    """Whether this body is the correspondence she saw.

    Stops at the first disagreement, because nearly every candidate disagrees
    at the first place and building the whole answer first does the work of a
    term that was already wrong.
    """
    for size, found in wanted.items():
        if size <= 0 or size not in tables:
            return False
        first, second = tables[size]
        here_first, here_second = _here(first), _here(second)
        for at in range(size):
            try:
                said = run(
                    body,
                    (second, first, here_second[at], here_first[at], size, at),
                    fuel=_A_CANDIDATE_MAY_SPEND,
                )
            except (OutOfFuel, Stuck, RecursionError, TypeError, ValueError):
                return False
            if not isinstance(said, int) or said % size != found[at]:
                return False
    return True


def _still_says_something_at_a_length_it_never_saw(
    body: Code,
    tables: dict[int, tuple[Any, Any]],
) -> bool:
    """Whether it names a place inside a state it was not fitted to.

    The same test a maker gets, for the same reason. The family says nothing
    about an unseen length, so what can be checked there is that the head does
    not raise and does not point outside the thing.
    """
    for size, (first, second) in tables.items():
        here_first, here_second = _here(first), _here(second)
        for at in range(size):
            try:
                said = run(
                    body,
                    (second, first, here_second[at], here_first[at], size, at),
                    fuel=_A_CANDIDATE_MAY_SPEND,
                )
            except (OutOfFuel, Stuck, RecursionError, TypeError, ValueError):
                return False
            if not isinstance(said, int):
                return False
    return True


def _here(table: Any) -> list[int]:
    """A table back as plain numbers, so the search does not walk it per place."""
    from core.cognition.the_floor_she_stands_on import as_list

    return [int(one) for one in as_list(table)]


def _pairs_of_words(names: Sequence[str]) -> Iterator[tuple[str, str]]:
    for first in names:
        for second in names:
            yield first, second


def a_way_of_computing_she_wrote(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    *,
    now_sayable: Callable[[], bool],
    words: dict[str, Any] | None = None,
    already: Sequence[Code] = (),
    deepest: int = 3,
    within: float = _AS_LONG_AS_A_MAKER_GETS,
    most_words: int = 6,
) -> AWayOfComputing | None:
    """Write a way of computing for the family in front of her, or nothing.

    ``already`` is what she has admitted before, offered as leaves. Nothing
    here is a list of operation kinds; the leaves are numbers, the four things
    a head is given, and her own past work.
    """
    from core.cognition.an_invented_kind import addressings
    from core.cognition.one_algebra import _where_each_came_from  # noqa: PLC2701

    if now_sayable():
        return None
    wanted = _where_each_came_from(transitions)
    if len(wanted) < 2:
        # One length is not enough to hold anything back, and a head fitted to
        # everything it has seen has been judged on nothing.
        return None
    every = dict(words) if words is not None else addressings()
    if not every:
        return None

    lengths = sorted(wanted)
    fitted_at = tuple(lengths[0::2])
    judged_at = tuple(lengths[1::2])
    if not fitted_at or not judged_at:
        return None

    from core.cognition.how_she_learns_to_look import in_the_order_worth_trying
    from core.cognition.one_algebra import _tells_her_the_answer  # noqa: PLC2701
    from core.cognition.what_it_costs_to_say import _symbols  # noqa: PLC2701

    ordered = in_the_order_worth_trying(
        every, _tells_her_the_answer, wanted, shortest=_symbols
    )[: max(1, most_words)]

    unseen = max(lengths) + 1
    began = time.monotonic()
    bodies: list[Code] = []
    stream = every_code(
        deepest=deepest,
        variables=len(WHAT_A_HEAD_IS_GIVEN),
        constants=(0, 1, 2),
        also=tuple(already),
    )

    def so_far() -> Iterator[Code]:
        yield from bodies
        for body in stream:
            bodies.append(body)
            yield body

    for first_name, second_name in _pairs_of_words(ordered):
        first_word, second_word = every[first_name], every[second_name]
        try:
            tables = {
                size: (
                    what_each_part_says(first_word, size),
                    what_each_part_says(second_word, size),
                )
                for size in (*lengths, unseen)
            }
        except (ArithmeticError, IndexError, TypeError, ValueError):
            continue
        fitting = {size: wanted[size] for size in fitted_at}
        judging = {size: wanted[size] for size in judged_at}
        unseen_only = {unseen: tables[unseen]}
        for body in so_far():
            if time.monotonic() - began >= within:
                logger.info("gave up writing a way of computing after %.1fs", within)
                return None
            if not _computes(body, fitting, tables):
                continue
            if not _computes(body, judging, tables):
                # It fitted the half it saw and not the half it did not. That
                # is the failure a table always makes, and the reason the
                # evidence is split before anything is searched.
                continue
            if not _still_says_something_at_a_length_it_never_saw(body, unseen_only):
                continue
            found = AWayOfComputing(
                body=as_a_head(body),
                over=(first_name, second_name),
                fitted_at=fitted_at,
                judged_at=judged_at,
            )
            logger.info("she wrote a way of computing: %s", found.describes())
            return found
    return None
