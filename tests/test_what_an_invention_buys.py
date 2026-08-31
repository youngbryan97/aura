"""Criterion 6, stated so that it can be true.

"She needed the word she made" has two readings and only one can hold. The
strong one — the behaviour is not expressible without it — is false by one line:
a word she made is a term over words she had, so any expression mentioning it
can have it substituted away and denotes the same thing. Naming is a ``let``.

The weak one is about reach. She searches to a bounded length; a behaviour whose
shortest saying is twenty-three symbols is not findable at a horizon of nine,
however expressible it is. Give her a word saying the awkward part in one symbol
and the behaviour is three symbols long.

This file was written after claiming the strong reading and being wrong. The
experiment walked 18,658,023 terms, found nothing, and concluded necessity — it
had skipped every term with no hole in it, and the answer is a twenty-three
symbol term with no holes at all.
"""

from __future__ import annotations

import pytest

from core.cognition.an_invented_kind import WHERE_FROM
from core.cognition.one_algebra import Made, Term, every_term, holes_in, what_it_rests_on
from core.cognition.what_an_invention_buys import (
    the_horizon_of,
    the_shortest_way_to_say,
)
from core.cognition.what_growth_cannot_do import naming_cannot_add_a_meaning

WHERE, MANY = Term("where"), Term("many")
ZERO, ONE, TWO = Term("fixed", value=0), Term("fixed", value=1), Term("fixed", value=2)
ODD_N = Term("same as", (Term("left over", (MANY, TWO)), ONE))
EVEN_A = Term("same as", (Term("left over", (WHERE, TWO)), ZERO))
WRITTEN_OUT = Term(
    "plus",
    (
        WHERE,
        Term(
            "plus",
            (ODD_N, Term("times", (TWO, Term("times", (Term("minus", (ONE, ODD_N)), EVEN_A))))),
        ),
    ),
)


def _target(at, size):
    here, partner = WHERE_FROM["here"], WHERE_FROM["its partner"]
    inner = here(at, size) if size % 2 == 1 else partner(at, size)
    return (inner + 1) % size


def _says_it(word):
    try:
        return all(
            int(word(at, size)) % size == _target(at, size)
            for size in (4, 5, 6, 7)
            for at in range(size)
        )
    except (ArithmeticError, IndexError, TypeError, ValueError):
        return False


def _the_word_she_made():
    parity = Made(
        term=Term("if", (ODD_N, Term("hole", value=0), Term("hole", value=1))),
        words=(WHERE_FROM["here"], WHERE_FROM["its partner"]),
        built_from=("here", "its partner"),
    )
    return Made(
        term=Term("plus", (ONE, Term("hole", value=0))),
        words=(parity,),
        built_from=("the parity word",),
    )


def test_the_behaviour_is_sayable_with_no_word_at_all():
    """Twenty-three symbols, no holes, and it says the target exactly."""
    written = Made(term=WRITTEN_OUT, words=())
    assert holes_in(WRITTEN_OUT) == 0
    assert _says_it(written)
    for size in range(2, 40):
        for at in range(size):
            assert int(written(at, size)) % size == _target(at, size)


def test_naming_added_no_meaning():
    """The whole content of the substitution theorem, on this word, in code."""
    assert naming_cannot_add_a_meaning(
        _says_it,
        a_word_she_made=_the_word_she_made(),
        unfolds_to=lambda: Made(term=WRITTEN_OUT, words=()),
    )


def test_what_it_bought_was_twenty_symbols():
    """Nothing became sayable. Something became findable."""
    named = _the_word_she_made()
    assert named.term.how_long() == 3
    assert WRITTEN_OUT.how_long() == 23
    assert the_horizon_of(4) == 9
    # Out of reach at the horizon she searches to, and in reach with the word.
    assert WRITTEN_OUT.how_long() > the_horizon_of(4) >= named.term.how_long()


def test_a_search_says_whether_it_finished():
    """Not finding something inside a budget is not the same as its not being
    there, and only one of those is a fact about the language."""
    _size, _how, finished = the_shortest_way_to_say(
        _says_it, dict(WHERE_FROM), up_to=5, holes=1, within=0.0
    )
    assert finished is False
    size, how, done = the_shortest_way_to_say(
        lambda word: True, dict(WHERE_FROM), up_to=1, holes=1, within=30.0
    )
    assert done is True and size == 1 and how


def test_closed_terms_are_walked():
    """Skipping every term with no hole is what turned a twenty-three symbol
    closed answer into a proof of impossibility."""
    walked = sum(
        1
        for at, term in enumerate(every_term((0, 1, 2), holes=2, deepest=3))
        if holes_in(term) == 0 and at < 20000
    )
    assert walked > 0


def test_branching_is_generated_at_all():
    """`if` takes three parts, so one plus three odd sizes is even, and the
    enumerator walked only odd sizes. Not one had ever been produced."""
    found = None
    for at, term in enumerate(every_term((0, 1), holes=1, deepest=3)):
        if term.head == "if":
            found = term
            break
        if at > 100000:
            break
    assert found is not None
    assert len(found.parts) == 3


def test_a_word_carries_what_it_was_built_from():
    """Provenance read off the construction, not off the spelling."""
    named = _the_word_she_made()
    assert named.built_from == ("the parity word",)
    words = {"the parity word": named.words[0], "it": named}
    assert what_it_rests_on("it", words) >= {"here", "its partner"}
    assert what_it_rests_on("here", words) == frozenset()
