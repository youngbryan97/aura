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
    A,
    Code,
    L,
    OutOfFuel,
    Stuck,
    V,
    build,
    every_code,
    from_list,
    how_long,
    run,
)

__all__ = [
    "AWayOfComputing",
    "WHAT_A_HEAD_IS_GIVEN",
    "WHERE_A_STEP_READS_THE_ONE_BEFORE",
    "a_way_by_recurrence",
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
    "itself, given two numbers",
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
    #: Candidates walked before this one. What the answer cost, in the unit
    #: the search that failed is also counted in.
    found_at: int = 0
    #: Whether it is defined by what it says at the place before. The only
    #: shape here that reaches past what the positional algebra already says.
    by_recurrence: bool = False
    #: What was actually written, before it was closed over the seven things a
    #: head is given. Kept rather than recovered from the closed head, because
    #: the closed head carries the fixed point it is given and taking that
    #: apart to find the body again is a guess about the wrapper's shape.
    written: Code | None = None

    @property
    def how_long(self) -> int:
        """Symbols of what she wrote, not of the wrapper it arrives in."""
        return how_long(self.written if self.written is not None else self.body)

    def describes(self) -> str:
        return (
            f"{self.how_long} symbols over {', '.join(self.over)}, fitted at "
            f"{self.fitted_at} and held to {self.judged_at}, found after "
            f"{self.found_at:,} candidate(s)"
            + (", by what it says at the place before" if self.by_recurrence else "")
        )


def what_each_part_says(
    word: Callable[[int, int], int], size: int
) -> Any:
    """Everything a word says at a state this long, as a list on the floor.

    Finite because a word is a pure function of two numbers: where it is asked
    and how long the state is. Handing this rather than one number is what lets
    a head read a part somewhere other than here.
    """
    return from_list([int(word(at, size)) % max(1, size) for at in range(size)])


def _twice_over_itself(work: Any) -> Any:
    """The strict fixed point for a function of two numbers.

    Not a head and not a primitive. Written out of ``given a thing`` and
    ``of``, which is where the floor's universality comes from in the first
    place, and exhibited here only so a head does not have to find it.
    """
    inner = L(
        "x",
        A(work, L("a", L("b", A(V("x"), V("x"), V("a"), V("b"))))),
    )
    return A(inner, inner)


def as_a_head(body: Code) -> Code:
    """Close a body over everything a head is given, including itself.

    A head that cannot refer to itself can only compose what the positional
    algebra already composes, so every head it could write is a shorter name —
    measured, on 120 families out of 120, before this existed. What makes a
    head able to say something new is a fixed point, and a fixed point written
    out of application alone is thirty-eight symbols before it computes
    anything, which no shortest-first search reaches.

    So the fixed point is supplied rather than found. It adds no meanings, by
    the substitution argument in
    :mod:`core.cognition.what_growth_cannot_do` — a term over the floor is
    eliminable — and it moves what is reachable, which is the only quantity
    that was ever going to move. What it costs is one degree of freedom made
    explicit: ``itself`` re-enters on the two numbers a head is given and
    keeps the position, the length and the tables, because those are the
    context of the place being asked about rather than what a recursion counts
    down.

    The seven bindings arrive in the order :data:`WHAT_A_HEAD_IS_GIVEN` names
    them, and the body is re-bound to exactly those seven so nothing from the
    fixed point's own workings is in scope.
    """
    rebound: Any = body
    for _ in range(7):
        rebound = Code("given a thing", parts=(rebound,))
    step = L(
        "itself",
        L(
            "first",
            L(
                "second",
                A(
                    rebound,
                    V("itself"),
                    V("where"),
                    V("many"),
                    V("first"),
                    V("second"),
                    V("all first"),
                    V("all second"),
                ),
            ),
        ),
    )
    return build(
        L(
            "where",
            L(
                "many",
                L(
                    "here first",
                    L(
                        "here second",
                        L(
                            "all first",
                            L(
                                "all second",
                                A(
                                    _twice_over_itself(step),
                                    V("here first"),
                                    V("here second"),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )


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
    closed = as_a_head(body)
    for size, found in wanted.items():
        if size <= 0 or size not in tables:
            return False
        first, second = tables[size]
        here_first, here_second = _here(first), _here(second)
        for at in range(size):
            said = _ask(closed, at, size, here_first[at], here_second[at], first, second)
            if said is None or said % size != found[at]:
                return False
    return True


def _ask(
    closed: Code,
    at: int,
    size: int,
    here_first: int,
    here_second: int,
    first: Any,
    second: Any,
) -> int | None:
    """Run a closed head at one place, or nothing where it refuses."""
    given: Any = closed
    try:
        work = run(given, fuel=_A_CANDIDATE_MAY_SPEND)
        for one in (at, size, here_first, here_second, first, second):
            work = run(work.body, (one, *work.env), fuel=_A_CANDIDATE_MAY_SPEND)
        return int(work)
    except (OutOfFuel, Stuck, RecursionError, TypeError, ValueError, AttributeError):
        return None


def _still_says_something_at_a_length_it_never_saw(
    body: Code,
    tables: dict[int, tuple[Any, Any]],
) -> bool:
    """Whether it names a place inside a state it was not fitted to.

    The same test a maker gets, for the same reason. The family says nothing
    about an unseen length, so what can be checked there is that the head does
    not raise and does not point outside the thing.
    """
    closed = as_a_head(body)
    for size, (first, second) in tables.items():
        here_first, here_second = _here(first), _here(second)
        for at in range(size):
            if _ask(closed, at, size, here_first[at], here_second[at], first, second) is None:
                return False
    return True


def _here(table: Any) -> list[int]:
    """A table back as plain numbers, so the search does not walk it per place."""
    from core.cognition.the_floor_she_stands_on import as_list

    return [int(one) for one in as_list(table)]


#: The slot a step uses for the answer at the place before it. Outside the
#: seven a head is given, so a step can be searched over eight variables and
#: then have this one replaced by the call that produces it.
WHERE_A_STEP_READS_THE_ONE_BEFORE = 7

#: Which of the seven a recurrence counts down. What a part says HERE is the
#: only one of them that moves with the place being asked about while the
#: length and the tables stay put, so it is the only one a step could count
#: down without the question changing underneath it.
_WHAT_IT_COUNTS_DOWN = 3

#: Where a head refers to itself, and where the second number is read.
_ITSELF, _THE_OTHER_NUMBER = 6, 2


def _the_one_before(
    wanted: dict[int, tuple[int, ...]],
    here: dict[int, tuple[list[int], list[int]]],
) -> tuple[list[tuple[tuple[int, ...], int]], list[tuple[tuple[int, ...], int]]] | None:
    """Split the family into a step and a base, or nothing where it does not split.

    A recurrence needs the answer at one place to be reachable from the answer
    at the place before it. What counts as "before" is one less of the number
    the first part says here, and that is only well defined where the first
    part says a different number at every place — so this refuses rather than
    guesses where it does not.

    Returns the step examples, each an environment and what it must give, and
    the base examples for where the count has run out.
    """
    step: list[tuple[tuple[int, ...], int]] = []
    base: list[tuple[tuple[int, ...], int]] = []
    for size, found in wanted.items():
        if size <= 1 or size not in here:
            return None
        first, second = here[size]
        if len(set(first)) != size:
            # The count repeats, so "the place before" names more than one
            # place and nothing here can say which.
            return None
        answer = {first[at]: found[at] for at in range(size)}
        for at in range(size):
            count = first[at]
            where = (second[at], second[at], second[at], count, size, at, 0)
            if count == 0:
                base.append((where, found[at]))
                continue
            if count - 1 not in answer:
                return None
            step.append(((*where, answer[count - 1]), found[at]))
    if not step or not base:
        return None
    return step, base


def _something_that_fits(
    examples: Sequence[tuple[tuple[int, ...], int]],
    *,
    variables: int,
    already: Sequence[Code],
    deepest: int,
    within: float,
    started: float,
    avoid: int | None = None,
    reads_once: int | None = None,
) -> Code | None:
    """The shortest term over these variables giving these answers.

    Enumeration, and it is enough here because the thing being searched for is
    a STEP rather than a whole rule — three or four symbols where the rule it
    builds is fourteen. That difference is the entire reason this works, and
    it is the same difference inverting an operation buys in
    `an_operation_that_generalises`.
    """
    for candidate in every_code(
        deepest=deepest, variables=variables, constants=(0, 1, 2), also=tuple(already)
    ):
        if time.monotonic() - started >= within:
            return None
        if avoid is not None and _reads(candidate, avoid):
            continue
        if reads_once is not None and _how_often_it_reads(candidate, reads_once) > 1:
            # A step that looks at the one before twice makes the head cost
            # twice as much at every count, so what it costs doubles with the
            # thing it counts down. Measured rather than reasoned about: the
            # first doubling step found was `the one before plus the one
            # before`, and the head it built ran out of fuel at length nine.
            continue
        fits = True
        for where, said in examples:
            try:
                got = run(candidate, where, fuel=_A_CANDIDATE_MAY_SPEND)
            except (OutOfFuel, Stuck, RecursionError, TypeError, ValueError):
                fits = False
                break
            if got != said:
                fits = False
                break
        if fits:
            return candidate
    return None


def _how_often_it_reads(code: Code, which: int) -> int:
    counted = 0
    edge = [code]
    while edge:
        here = edge.pop()
        if here.head == "the one it was given" and int(here.value or 0) == which:
            counted += 1
        edge.extend(here.parts)
    return counted


def _reads(code: Code, which: int) -> bool:
    return _how_often_it_reads(code, which) > 0


def _put_the_call_in(body: Code) -> Code:
    """Replace the slot standing for the one before with the call that gives it."""
    if body.head == "the one it was given":
        if int(body.value or 0) != WHERE_A_STEP_READS_THE_ONE_BEFORE:
            return body
        return Code(
            "of",
            parts=(
                Code(
                    "of",
                    parts=(
                        Code("the one it was given", value=_ITSELF),
                        Code(
                            "minus",
                            parts=(
                                Code("the one it was given", value=_WHAT_IT_COUNTS_DOWN),
                                Code("a number", value=1),
                            ),
                        ),
                    ),
                ),
                Code("the one it was given", value=_THE_OTHER_NUMBER),
            ),
        )
    return Code(
        body.head,
        parts=tuple(_put_the_call_in(part) for part in body.parts),
        value=body.value,
    )


def a_way_by_recurrence(
    wanted: dict[int, tuple[int, ...]],
    here: dict[int, tuple[list[int], list[int]]],
    *,
    already: Sequence[Code] = (),
    deepest: int = 5,
    within: float = 8.0,
) -> Code | None:
    """A head defined by what it says at the place before, or nothing.

    The thing that makes a head able to say what the positional algebra cannot
    is a fixed point, and a fixed point is fourteen symbols before it computes
    anything — past any shortest-first search. So it is not searched for. The
    family is asked whether its answers stand in a recurrence, and if they do
    only the STEP is searched, which is three or four symbols.

    That is a schema, and calling it anything else would be dishonest. It is
    the inversion of the fixed point the floor already has, used to direct the
    search — the same relationship `_what_the_first_must_be` has to arithmetic
    in `an_operation_that_generalises`. By the substitution argument it adds no
    meanings; what it changes is what can be found.
    """
    split = _the_one_before(wanted, here)
    if split is None:
        return None
    step_examples, base_examples = split
    began = time.monotonic()
    step = _something_that_fits(
        step_examples,
        variables=WHERE_A_STEP_READS_THE_ONE_BEFORE + 1,
        already=already,
        deepest=deepest,
        within=within / 2,
        started=began,
        avoid=_ITSELF,
        reads_once=WHERE_A_STEP_READS_THE_ONE_BEFORE,
    )
    if step is None or not _reads(step, WHERE_A_STEP_READS_THE_ONE_BEFORE):
        # A step that never looks at the one before is not a recurrence, and
        # admitting it as one would dress an ordinary term as a fixed point.
        return None
    base = _something_that_fits(
        base_examples,
        variables=len(WHAT_A_HEAD_IS_GIVEN) - 1,
        # A base is what it says when the count has run out, and that is
        # short or it is not a base at all.
        already=already,
        deepest=deepest,
        within=within,
        started=began,
        avoid=_ITSELF,
    )
    if base is None:
        return None
    return Code(
        "if",
        parts=(
            Code(
                "same as",
                parts=(
                    Code("the one it was given", value=_WHAT_IT_COUNTS_DOWN),
                    Code("a number", value=0),
                ),
            ),
            base,
            _put_the_call_in(step),
        ),
    )


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
    by_recurrence: bool = True,
) -> AWayOfComputing | None:
    """Write a way of computing for the family in front of her, or nothing.

    ``already`` is what she has admitted before, offered as leaves. Nothing
    here is a list of operation kinds; the leaves are numbers, the seven things
    a head is given, and her own past work.

    ``by_recurrence`` is the lesion. Turned off, the only route left is
    enumeration, and enumeration reaches short bodies — which are the ones the
    positional algebra already says. It is here so the claim that the
    recurrence schema is what does the work can be tested by removing it
    rather than argued for.
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
    walked = 0
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

        # A recurrence first, because it is the only thing here that finds a
        # head the positional algebra cannot already say. Enumeration finds
        # short bodies, and a short body composes what the positional terms
        # compose — measured, 120 families out of 120 classified as a shorter
        # name before this existed. What a recurrence costs is one pass over a
        # few hundred thousand steps, against a whole family's search.
        here_at = {
            size: (_here(tables[size][0]), _here(tables[size][1]))
            for size in lengths
        }
        found_by_recurrence = a_way_by_recurrence(
            {size: wanted[size] for size in lengths},
            here_at,
            already=tuple(already),
            within=max(1.0, within / 3),
        )
        if (
            by_recurrence
            and found_by_recurrence is not None
            and _computes(found_by_recurrence, fitting, tables)
            and _computes(found_by_recurrence, judging, tables)
            and _still_says_something_at_a_length_it_never_saw(
                found_by_recurrence, unseen_only
            )
        ):
            walked += 1
            found = AWayOfComputing(
                body=as_a_head(found_by_recurrence),
                over=(first_name, second_name),
                fitted_at=fitted_at,
                judged_at=judged_at,
                found_at=walked,
                by_recurrence=True,
                written=found_by_recurrence,
            )
            logger.info("she wrote a way of computing: %s", found.describes())
            return found

        for body in so_far():
            walked += 1
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
                found_at=walked,
                written=body,
            )
            logger.info("she wrote a way of computing: %s", found.describes())
            return found
    return None
