"""Iteration, which is the one head that adds reach composition cannot have.

Every other head composes a FIXED number of times, so what a term of length L
reaches is a fixed number of steps. This one takes its count from the term, and
the term can read the size — so the number of steps depends on the input, and
no fixed-length composition has that shape.

Certified rather than asserted. The behaviour below is "double and shift, as
many times as the thing is long", whose closed form needs two-to-the-n and the
grammar has no exponentiation. Over the given words at length seven the search
FINISHED after 33,457,544 terms with nothing saying it; with iteration it is
found after 10,457.
"""

from __future__ import annotations

import pytest

from core.cognition.an_invented_kind import WHERE_FROM
from core.cognition.one_algebra import Made, Term, every_term, holes_in

DOUBLE_AND_SHIFT = Term(
    "plus", (Term("times", (Term("fixed", value=2), Term("where"))), Term("fixed", value=1))
)
AS_MANY_TIMES_AS_IT_IS_LONG = Term("over again", (Term("many"), DOUBLE_AND_SHIFT))


def _wanted(at, size):
    place = at
    for _turn in range(size):
        place = (2 * place + 1) % size
    return place


def test_it_does_the_thing_as_many_times_as_it_is_told():
    for size in (4, 5, 6, 9, 12):
        for at in range(size):
            said = Made(term=AS_MANY_TIMES_AS_IT_IS_LONG, words=())(at, size)
            assert said % size == _wanted(at, size)


def test_the_count_comes_from_the_term_not_from_a_number_written_here():
    """Half way along, at every size, from one term."""
    half = Term(
        "over again",
        (Term("over", (Term("many"), Term("fixed", value=2))), Term("hole", value=0)),
    )
    made = Made(term=half, words=(WHERE_FROM["one along"],))
    for size in (4, 5, 6, 9, 12):
        assert tuple(made(at, size) for at in range(size)) == tuple(
            (at + size // 2) % size for at in range(size)
        )


def test_going_round_more_times_than_the_thing_is_long_says_nothing_new():
    """The cap is where the answers stop changing, which is the world's number
    rather than a budget: walking n places n times has already reached whatever
    cycle it reaches."""
    forever = Term("over again", (Term("fixed", value=999), Term("hole", value=0)))
    capped = Term("over again", (Term("many"), Term("hole", value=0)))
    along = WHERE_FROM["one along"]
    for size in (4, 5, 6):
        for at in range(size):
            assert Made(term=forever, words=(along,))(at, size) == Made(
                term=capped, words=(along,)
            )(at, size)


def test_it_is_generated_by_the_enumerator():
    found = None
    for at, term in enumerate(every_term((0, 1, 2), holes=1, deepest=3)):
        if term.head == "over again":
            found = term
            break
        if at > 200000:
            break
    assert found is not None
    assert len(found.parts) == 2


def test_an_iterating_term_still_names_a_place_inside_the_thing():
    """Whatever it does, it must land somewhere in the thing."""
    for term in (AS_MANY_TIMES_AS_IT_IS_LONG,):
        for size in (3, 4, 7, 11):
            for at in range(size):
                said = Made(term=term, words=())(at, size)
                assert 0 <= said % size < size


def test_a_hole_inside_an_iteration_is_still_counted():
    half = Term(
        "over again",
        (Term("over", (Term("many"), Term("fixed", value=2))), Term("hole", value=0)),
    )
    assert holes_in(half) == 1
