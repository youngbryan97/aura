"""A head now earns its place, and these are the two things that let it.

For a while none did. The head search found a candidate on essentially every
family and the growth classifier said the same thing every time: the positional
language already says it, in five symbols or fewer. 120 families out of 120.
The mechanism was complete and inert.

Two causes, and both were in the search rather than in the theory.

**A head could not refer to itself.** So every head it could write composed what
the positional algebra already composes, and of course the positional algebra
said it. What makes a head able to say something else is a fixed point, and a
fixed point written out of application alone is thirty-eight symbols before it
computes anything — past anything a shortest-first search reaches. It is
supplied now as one of the seven things a head is given. By the substitution
argument that adds no meanings; it moves what is reachable, which was always
the only quantity that could move.

**The step was searched for rather than solved.** Even with a fixed point in
hand, the shortest doubling head is fourteen symbols and enumeration does not
reach fourteen. So it is not enumerated. The family is asked whether its
answers stand in a recurrence — whether the answer at a place is reachable from
the answer at the place before — and if they do, only the STEP is searched.
That is three to five symbols. It is the same move `an_operation_that_generalises`
makes when it inverts an operation instead of walking the space, and it is a
schema, which is worth saying plainly rather than dressing up.

What that buys, measured: doubling, factorial and triangular numbers, each
written from before-and-after states alone, each holding at lengths nine,
eleven and thirteen that were neither fitted nor judged, and two of the three
admitted through the full gate — priced, classified, installed.
"""

from __future__ import annotations

import logging
import math

import pytest

from core.cognition.a_way_of_computing_she_wrote import (
    a_way_by_recurrence,
    a_way_of_computing_she_wrote,
    as_a_head,
)
from core.cognition.an_invented_kind import WHERE_FROM, induce_from
from core.cognition.one_algebra import DERIVED_HEADS, Term, run
from core.cognition.the_floor_she_stands_on import from_list, how_long
from core.cognition.sequence_induction import _a_way_of_computing  # noqa: PLC2701


@pytest.fixture(autouse=True)
def _clean():
    """Both registries, because admitting a head also widens the vocabulary.

    A head is only reachable through a word written over it, so the full gate
    installs both. Restoring one and not the other leaves a word behind that
    every later test in the process then searches over — which is an
    order-dependence defect introduced by the test rather than by the code.
    """
    heads, words = dict(DERIVED_HEADS), dict(WHERE_FROM)
    DERIVED_HEADS.clear()
    yield
    DERIVED_HEADS.clear()
    DERIVED_HEADS.update(heads)
    WHERE_FROM.clear()
    WHERE_FROM.update(words)


def _family(rule, sizes=(4, 5, 6, 7)):
    """Before and after states only. No rule name, no operator, no hint."""
    made = []
    for size in sizes:
        before = tuple(range(100, 100 + size))
        made.append((before, tuple(before[rule(at, size) % size] for at in range(size))))
    return made


_RULES = {
    "doubling": lambda at, size: pow(2, at, size),
    "factorial": lambda at, size: math.factorial(at) % size,
    "triangular": lambda at, size: (at * (at + 1) // 2) % size,
}


@pytest.mark.parametrize("name", sorted(_RULES))
def test_she_writes_a_recursive_head_from_the_states_alone(name: str) -> None:
    found = a_way_of_computing_she_wrote(
        _family(_RULES[name]),
        now_sayable=lambda: False,
        words=dict(WHERE_FROM),
        within=20.0,
    )
    assert found is not None, name
    assert found.by_recurrence, "found without the recurrence, so it was already short"
    assert found.written is not None
    assert how_long(found.written) <= 24


@pytest.mark.parametrize("name", sorted(_RULES))
def test_it_holds_at_lengths_it_was_neither_fitted_nor_judged_at(name: str) -> None:
    rule = _RULES[name]
    found = a_way_of_computing_she_wrote(
        _family(rule),
        now_sayable=lambda: False,
        words=dict(WHERE_FROM),
        within=20.0,
    )
    assert found is not None
    from core.cognition.one_algebra import the_head_she_wrote

    the_head_she_wrote("what she wrote", 2, found.body)
    term = Term("what she wrote", parts=(Term("hole", value=0), Term("hole", value=1)))
    words = tuple(WHERE_FROM[one] for one in found.over)
    for size in (9, 11, 13):
        got = tuple(run(term, at, size, words) for at in range(size))
        assert got == tuple(rule(at, size) % size for at in range(size)), (name, size)


def test_taking_the_recurrence_away_takes_the_head_with_it() -> None:
    """The lesion. Without the schema, only enumeration is left.

    Enumeration reaches short bodies, and a short body composes what the
    positional algebra already composes — which is where this started.
    """
    family = _family(_RULES["doubling"])
    with_it = a_way_of_computing_she_wrote(
        family, now_sayable=lambda: False, words=dict(WHERE_FROM), within=20.0
    )
    without = a_way_of_computing_she_wrote(
        family,
        now_sayable=lambda: False,
        words=dict(WHERE_FROM),
        within=20.0,
        by_recurrence=False,
    )
    assert with_it is not None and with_it.by_recurrence
    assert without is None, "reachable without the schema, so the schema proved nothing"


def test_a_step_that_reads_the_one_before_twice_is_refused() -> None:
    """A fact about the recursion rather than a matter of taste.

    A step reading the one before twice makes the head cost twice as much at
    every count, so what it costs doubles with the thing it counts down.
    Measured: the first doubling step found was `the one before plus the one
    before`, and the head it built ran out of fuel at length nine.
    """
    from core.cognition.a_way_of_computing_she_wrote import (
        WHERE_A_STEP_READS_THE_ONE_BEFORE,
        _how_often_it_reads,
    )

    sizes = (4, 5, 6, 7)
    wanted = {n: tuple(pow(2, at, n) for at in range(n)) for n in sizes}
    here = {n: ([at for at in range(n)], [0] * n) for n in sizes}
    body = a_way_by_recurrence(wanted, here)
    assert body is not None
    assert _how_often_it_reads(body, WHERE_A_STEP_READS_THE_ONE_BEFORE) == 0
    # The call replaces the slot, so what survives is the call, once.
    from core.cognition.a_way_of_computing_she_wrote import _ITSELF

    assert _how_often_it_reads(body, _ITSELF) == 1


def test_a_family_whose_count_repeats_is_refused_rather_than_guessed() -> None:
    """"The place before" needs the count to name one place, and it must say so."""
    sizes = (4, 5)
    wanted = {n: tuple(range(n)) for n in sizes}
    # The first word says the same thing everywhere, so there is no order to
    # count down and no unique place before.
    here = {n: ([0] * n, [0] * n) for n in sizes}
    assert a_way_by_recurrence(wanted, here) is None


def test_a_head_written_this_way_passes_the_whole_gate() -> None:
    """Priced, classified and installed, not merely found.

    The gate above the search asks what it costs the search and which of the
    three things "the language grew" means. A head that is only a shorter name
    comes straight back out, which is what happened to every head written
    before a fixed point was available.
    """
    family = _family(_RULES["doubling"])
    said = _a_way_of_computing(
        family, lambda: induce_from(list(family)) is not None
    )
    assert said is not None, "found and then refused by the gate"
    assert "COMPUTING" in said
    assert DERIVED_HEADS, "nothing was kept"
    kept = next(iter(DERIVED_HEADS.values()))
    assert kept.kind, "kept without saying which kind of growth it was"
