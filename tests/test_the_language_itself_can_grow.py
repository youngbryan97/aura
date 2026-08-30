"""When the language that GENERATES hypotheses is the thing that is missing.

She can compose a meaning out of a small algebra and admit it to the
interpreter without anybody adding a branch. That is the set of expressions
growing — K(t+1) = K(t) + one induced kind.

The algebra those expressions are built from did not grow. Five ways to say
where a value comes from and four ways to combine a pair were written down by
a person, and every meaning she could ever induce was a point in their closure.
A family outside it is not merely unsolved: it is unsayable, and no amount of
searching finds it, because the search is over the wrong set.

    A(t+1) = A(t) + one operation she derived

Derived rather than chosen. Where the values are distinct, where each one came
from can be READ OFF; where the values were made rather than moved, what was
done with a pair can be read off instead. If nothing the language already says
produces it, what was read off IS a new word — and once it is a word, every
hypothesis she can form afterwards may use it.
"""

from __future__ import annotations

import pytest

from core.cognition.an_invented_kind import (
    WHAT_OF_IT,
    WHERE_FROM,
    every_meaning,
    induce_from,
)
from core.cognition.widening_the_language import (
    DerivedAddressing,
    DerivedOperation,
    an_addressing_nobody_wrote,
    an_operation_nobody_wrote,
    widen_with_addressing,
    widen_with_operation,
)


@pytest.fixture(autouse=True)
def _the_language_she_was_given():
    """Every test starts from the authored algebra and leaves it that way."""
    where, what = dict(WHERE_FROM), dict(WHAT_OF_IT)
    yield
    WHERE_FROM.clear()
    WHERE_FROM.update(where)
    WHAT_OF_IT.clear()
    WHAT_OF_IT.update(what)


def moved(perm):
    states = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 1, 2, 6), (4, 7, 2, 8)]
    return [(s, tuple(s[i] for i in perm)) for s in states]


def made(how):
    states = [(9, 2, 7, 4), (5, 1, 8, 3), (6, 6, 2, 9), (3, 7, 1, 5)]
    return [(s, how(s)) for s in states]


def a_difference(s):
    return tuple(abs(s[i] - s[(i + 1) % len(s)]) for i in range(len(s)))


# ── a way of saying where a value comes from ─────────────────────────────

def test_an_addressing_is_read_off_rather_than_guessed():
    found = an_addressing_nobody_wrote(moved((2, 0, 3, 1)))
    assert found is not None
    assert found.at[4] == (2, 0, 3, 1)


def test_and_refused_when_the_language_already_says_it():
    """A language does not need a second name for something it can express."""
    mirrored = moved((3, 2, 1, 0))
    assert an_addressing_nobody_wrote(
        mirrored, already=list(WHERE_FROM.values())
    ) is None


def test_repeated_values_say_nothing_about_where_anything_came_from():
    assert an_addressing_nobody_wrote([((1, 1, 2), (1, 2, 1))]) is None


def test_and_examples_that_disagree_describe_no_correspondence():
    assert an_addressing_nobody_wrote(
        [((1, 2, 3), (2, 1, 3)), ((4, 5, 6), (6, 5, 4))]
    ) is None


# ── a way of combining two values ────────────────────────────────────────

def test_an_operation_is_read_off_where_the_values_were_made():
    here = WHERE_FROM["here"]
    along = WHERE_FROM["one along"]
    done = an_operation_nobody_wrote(made(a_difference), here, along)
    assert done is not None
    assert done(9, 2) == 7


def test_and_refused_when_the_language_already_combines_that_way():
    here = WHERE_FROM["here"]
    done = an_operation_nobody_wrote(
        made(lambda s: s), here, here, already=list(WHAT_OF_IT.values())
    )
    assert done is None


# ── and the language actually grows ──────────────────────────────────────

def test_a_derived_addressing_widens_the_space_of_hypotheses():
    before = len(list(every_meaning()))
    found = an_addressing_nobody_wrote(moved((2, 0, 3, 1)))
    assert widen_with_addressing("the way these move", found)
    assert len(list(every_meaning())) > before


def test_and_the_family_becomes_sayable():
    pairs = moved((2, 0, 3, 1))
    assert induce_from(pairs) is None
    widen_with_addressing(
        "the way these move", an_addressing_nobody_wrote(pairs)
    )
    assert induce_from(pairs) is not None


def test_a_derived_operation_widens_it_too():
    before = len(list(every_meaning()))
    done = an_operation_nobody_wrote(
        made(a_difference), WHERE_FROM["here"], WHERE_FROM["one along"]
    )
    assert widen_with_operation("what was done with these", done)
    assert len(list(every_meaning())) > before


def test_a_word_is_not_added_twice():
    found = an_addressing_nobody_wrote(moved((2, 0, 3, 1)))
    assert widen_with_addressing("a word", found)
    assert widen_with_addressing("a word", found) == ""


# ── and a derived word refuses what it has never seen ────────────────────

def test_an_addressing_read_off_one_length_says_nothing_about_another():
    found = DerivedAddressing("read off four", {4: (1, 0, 3, 2)})
    assert found(0, 4) == 1
    with pytest.raises(IndexError):
        found(0, 6)


def test_an_operation_says_nothing_about_a_pair_it_never_saw():
    done = DerivedOperation("read off two", {(1, 2): 3})
    assert done(1, 2) == 3
    with pytest.raises(KeyError):
        done(4, 5)


def test_and_a_meaning_built_on_one_refuses_rather_than_throws():
    """Which is what makes a derived word safe to put in the language."""
    from core.cognition.an_invented_kind import Induced

    widen_with_addressing(
        "only at four", DerivedAddressing("only at four", {4: (1, 0, 3, 2)})
    )
    meaning = Induced("only at four", "only at four", "as it is")
    assert meaning.read((1, 2, 3, 4)) == (2, 1, 4, 3)
    assert meaning.read((1, 2, 3)) is None
