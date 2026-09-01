"""She originates the semantics, with no candidate and no list of kinds supplied.

The prompt's Experiment A, at the scale this repository can actually run it.
What goes in is a task interface and data: before and after states, and a
success criterion that is equality. What does not go in is a target operator, a
candidate implementation, a name, or a list of the kinds of operation she is
allowed to invent.

What comes out is an executable term on the floor, fitted on half the lengths
and judged on the other half, which then computes the family at lengths it was
never shown.

The honest limits are tested too. A shortest-first search over a universal
language reaches a few dozen symbols and no further; that is Levin's bound and
no budget removes it. What moves the horizon is the library, and the test that
says so takes the first head away and watches the second become unreachable at
the same budget.
"""

from __future__ import annotations

import pytest

from core.cognition.a_way_of_computing_she_wrote import (
    WHAT_A_HEAD_IS_GIVEN,
    a_way_of_computing_she_wrote,
    as_a_head,
)
from core.cognition.an_invented_kind import WHERE_FROM
from core.cognition.one_algebra import (
    DERIVED_HEADS,
    Term,
    forget_the_head,
    read_back,
    run,
    the_head_she_wrote,
    written_down,
)
from core.cognition.the_floor_she_stands_on import Code, how_long


@pytest.fixture
def _clean_registry():
    before = dict(DERIVED_HEADS)
    DERIVED_HEADS.clear()
    yield
    DERIVED_HEADS.clear()
    DERIVED_HEADS.update(before)


def _family(rule, sizes):
    """Before and after states only. No rule name, no operator, no hint."""
    made = []
    for size in sizes:
        before = tuple(range(100, 100 + size))
        made.append((before, tuple(before[rule(at, size) % size] for at in range(size))))
    return made


def _both_places(at, size):
    return at + (at + 1) % size


def test_she_writes_one_from_the_examples_alone() -> None:
    found = a_way_of_computing_she_wrote(
        _family(_both_places, (4, 5, 6, 7)),
        now_sayable=lambda: False,
        words=dict(WHERE_FROM),
        within=30.0,
    )
    assert found is not None
    assert found.fitted_at and found.judged_at
    assert set(found.fitted_at) & set(found.judged_at) == set()
    assert found.how_long <= 20


def test_what_she_wrote_computes_the_family_at_lengths_it_never_saw(
    _clean_registry,
) -> None:
    found = a_way_of_computing_she_wrote(
        _family(_both_places, (4, 5, 6, 7)),
        now_sayable=lambda: False,
        words=dict(WHERE_FROM),
        within=30.0,
    )
    assert found is not None
    the_head_she_wrote("what she wrote", 2, found.body)
    term = Term("what she wrote", parts=(Term("hole", value=0), Term("hole", value=1)))
    words = tuple(WHERE_FROM[one] for one in found.over)
    for size in (9, 11, 16):
        got = tuple(run(term, at, size, words) for at in range(size))
        assert got == tuple(_both_places(at, size) % size for at in range(size)), size


def test_what_she_wrote_survives_a_restart(_clean_registry) -> None:
    found = a_way_of_computing_she_wrote(
        _family(_both_places, (4, 5, 6, 7)),
        now_sayable=lambda: False,
        words=dict(WHERE_FROM),
        within=30.0,
    )
    assert found is not None
    from core.cognition.the_floor_she_stands_on import read_back as read_code
    from core.cognition.the_floor_she_stands_on import written_down as write_code

    assert read_code(write_code(found.body)) == found.body
    the_head_she_wrote("what she wrote", 2, found.body)
    term = Term("what she wrote", parts=(Term("where"), Term("many")))
    assert read_back(written_down(term)) == term


def test_nothing_is_written_where_something_already_says_it() -> None:
    """A head is the most expensive thing she can add, so it is the last resort."""
    found = a_way_of_computing_she_wrote(
        _family(_both_places, (4, 5, 6, 7)),
        now_sayable=lambda: True,
        words=dict(WHERE_FROM),
        within=30.0,
    )
    assert found is None


def test_a_family_with_one_length_holds_nothing_back() -> None:
    found = a_way_of_computing_she_wrote(
        _family(_both_places, (5,)),
        now_sayable=lambda: False,
        words=dict(WHERE_FROM),
        within=30.0,
    )
    assert found is None


def test_it_refuses_where_the_correspondence_contradicts_itself() -> None:
    """A record that is not a function of what she reads admits no head."""
    muddled = [((1, 2, 3), (1, 2, 3)), ((4, 5, 6), (5, 4, 6)), ((7, 8, 9), (7, 8, 9))]
    found = a_way_of_computing_she_wrote(
        muddled, now_sayable=lambda: False, words=dict(WHERE_FROM), within=10.0
    )
    assert found is None


def test_what_a_head_is_given_matches_what_the_interpreter_hands_it() -> None:
    """Six things, in one order, agreed between the writer and the runner."""
    from core.cognition import one_algebra

    body = as_a_head(Code("the one it was given", value=0))
    assert how_long(body) == 1 + len(WHAT_A_HEAD_IS_GIVEN)
    source = one_algebra._a_head_she_wrote.__doc__ or ""
    assert "floor" in source
    the_head_she_wrote("hands back the last thing", 2, body)
    try:
        # The innermost binder is everything the second part says, so this
        # head hands back a list; run() asks for a number and refuses.
        with pytest.raises(ValueError):
            run(
                Term("hands back the last thing", parts=(Term("where"), Term("many"))),
                0,
                4,
                (lambda at, size: at, lambda at, size: 0),
            )
    finally:
        forget_the_head("hands back the last thing")


def test_the_library_is_what_moves_the_horizon(_clean_registry) -> None:
    """A head unreachable at a budget becomes reachable once its piece exists.

    The developmental claim, as a measurement rather than an argument. The
    same search, the same budget, the same family: it fails with an empty
    library and succeeds with one entry, and the entry is the thing the answer
    is built out of.
    """
    from core.cognition.the_floor_she_stands_on import (
        A,
        IF,
        L,
        MINUS,
        N,
        SAME,
        TIMES,
        V,
        Y,
        build,
    )

    doubling = build(
        Y(
            "twice",
            L(
                "k",
                IF(
                    SAME(V("k"), N(0)),
                    N(1),
                    TIMES(N(2), A(V("twice"), MINUS(V("k"), N(1)))),
                ),
            ),
        )
    )
    assert how_long(doubling) > 25, "the piece has to be past the search horizon"

    family = _family(lambda at, size: 2**at, (4, 5, 6, 7))

    cold = a_way_of_computing_she_wrote(
        family, now_sayable=lambda: False, words=dict(WHERE_FROM), within=20.0
    )
    warm = a_way_of_computing_she_wrote(
        family,
        now_sayable=lambda: False,
        words=dict(WHERE_FROM),
        already=(doubling,),
        within=20.0,
    )
    assert cold is None, "reachable without the piece, so the piece proved nothing"
    assert warm is not None, "unreachable even with the piece"
    # What it wrote is the piece plus a little, which is what "the library
    # moved the horizon" has to mean: not that a longer search succeeded, but
    # that the answer was short once the piece was a leaf.
    assert warm.how_long <= how_long(doubling) + len(WHAT_A_HEAD_IS_GIVEN) + 4


def test_a_head_is_weighed_in_terms_to_walk_before_it_is_kept() -> None:
    """The most expensive thing she can admit, priced on both sides.

    A word is one more thing to put in a hole. A head is one more shape at
    every node of every term, so it multiplies. The trade is the same unit as
    everywhere else here — terms she would otherwise walk — and there is no
    threshold to argue about: it pays when reaching the answer, plus the
    branches it adds, is cheaper than the search that walked the whole space
    and returned nothing.
    """
    from core.cognition.keeping_the_language_small import (
        what_a_head_costs_the_search,
    )

    pays = what_a_head_costs_the_search(
        "one that pays",
        without=60_000,
        with_it=90_000,
        found_at=800,
        walked_without_finding=5_000_000,
    )
    assert pays.pays
    assert pays.adds == 30_000

    costs = what_a_head_costs_the_search(
        "one that does not",
        without=60_000,
        with_it=9_000_000,
        found_at=4_000_000,
        walked_without_finding=5_000_000,
    )
    assert not costs.pays


def test_what_it_wrote_records_what_the_answer_cost() -> None:
    found = a_way_of_computing_she_wrote(
        _family(_both_places, (4, 5, 6, 7)),
        now_sayable=lambda: False,
        words=dict(WHERE_FROM),
        within=30.0,
    )
    assert found is not None
    assert found.found_at > 0
    assert "candidate" in found.describes()
