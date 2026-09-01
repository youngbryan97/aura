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


# ── and a way of INVENTING words, which is a different thing again ───────

from core.cognition.an_invented_kind import WAYS_TO_BUILD, addressings  # noqa: E402
from core.cognition.widening_the_language import (  # noqa: E402
    a_way_of_building_nobody_wrote,
    one_after_another,
)


@pytest.fixture(autouse=True)
def _the_ways_she_was_given():
    ways = dict(WAYS_TO_BUILD)
    WAYS_TO_BUILD.clear()
    yield
    WAYS_TO_BUILD.clear()
    WAYS_TO_BUILD.update(ways)


def mirrored_then_stepped(s):
    size = len(s)
    turned = [s[size - 1 - i] for i in range(size)]
    return tuple(turned[(i + 1) % size] for i in range(size))


def needs_two_addressings():
    return [(t, mirrored_then_stepped(t)) for t in
            [(1, 2, 3, 4), (5, 6, 7, 8), (9, 1, 2, 6), (4, 7, 2, 8)]]


def test_a_family_no_single_word_can_say():
    assert induce_from(needs_two_addressings()) is None


def test_she_derives_a_way_of_MAKING_words():
    assert a_way_of_building_nobody_wrote(needs_two_addressings()) == "one after another"


def test_and_then_the_family_is_sayable():
    a_way_of_building_nobody_wrote(needs_two_addressings())
    found = induce_from(needs_two_addressings())
    assert found is not None
    assert ", then " in found.name


def test_it_enlarges_what_she_could_say_about_anything():
    """The property a derived WORD cannot have.

    A word is read off what she was shown and says nothing beyond it. A way of
    building takes every word she has — including the derived ones — and makes
    more out of them, so it reaches families she has never met.
    """
    before_words = len(addressings())
    before_meanings = len(list(every_meaning()))

    # One derived WORD: read off what she was shown, so it adds a fixed
    # number of meanings and no more.
    # A name the authored algebra does not already hold: widening with
    # "one along" replaces the word of that name and adds nothing.
    assert "two along" not in WHERE_FROM
    widen_with_addressing(
        "two along", DerivedAddressing("two along", {4: (2, 3, 0, 1)})
    )
    after_a_word = len(list(every_meaning()))
    word_added = after_a_word - before_meanings

    # One WAY OF BUILDING: it takes every word she has, including the one
    # just derived, so what it adds scales with the vocabulary.
    a_way_of_building_nobody_wrote(needs_two_addressings())
    after_a_way = len(list(every_meaning()))
    way_added = after_a_way - after_a_word

    assert len(addressings()) > before_words
    # The property, and not a chosen multiple of it: a word adds an
    # increment, a way of building multiplies. The assertion here used to be
    # "more than ten times the meanings", which is a number nobody measured -
    # the real figure is 6.5x on this algebra, and the test had been red for
    # however long that was true.
    assert word_added > 0, "a derived word added no meanings at all"
    assert way_added > word_added * 2, (
        f"a way of building added {way_added} where a word added {word_added}; "
        "it is not reaching families the word could not"
    )
    assert after_a_way > before_meanings * 2


def test_and_it_applies_to_words_she_derives_later():
    """Which is what makes it a way of growing rather than a growth."""
    a_way_of_building_nobody_wrote(needs_two_addressings())
    with_it = len(addressings())
    widen_with_addressing(
        "something new", DerivedAddressing("something new", {4: (1, 0, 3, 2)})
    )
    assert len(addressings()) > with_it + 1, (
        "a word derived afterwards was not put through the way of building"
    )


def test_a_way_of_building_that_changes_nothing_is_not_admitted():
    """Enlarging the search for nothing is worse than not enlarging it."""
    already = [((1, 2, 3), (3, 2, 1)), ((4, 5, 6), (6, 5, 4)),
               ((7, 8, 9), (9, 8, 7)), ((2, 9, 4), (4, 9, 2))]
    assert induce_from(already) is not None
    assert a_way_of_building_nobody_wrote(already) is None
    assert WAYS_TO_BUILD == {}


def test_it_is_not_admitted_twice():
    a_way_of_building_nobody_wrote(needs_two_addressings())
    assert a_way_of_building_nobody_wrote(needs_two_addressings()) is None


def test_the_way_of_building_composes_two_words_in_turn():
    made = one_after_another({"a": lambda i, n: (i + 1) % n, "b": lambda i, n: n - 1 - i})
    assert "a, then b" in made
    assert made["a, then b"](0, 4) == 2


def test_a_word_is_never_composed_with_itself():
    made = one_after_another({"a": lambda i, n: i})
    assert made == {}


def test_a_word_that_refuses_a_length_is_not_the_word_that_already_says_this():
    """A refusal is an answer, and letting it escape crashed a claim.

    DerivedAddressing raises IndexError when asked at a length it was never
    seen at — deliberately, because a correspondence read off four cells says
    nothing about six. _already_said_by caught TypeError, ValueError and
    ZeroDivisionError and not that one, so a language holding any word of
    another length made every claim prediction that walked the closure die
    with 'was never seen at length 4'. It passed alone and failed in a run of
    twenty-five files, which is the order-dependence shape, not a flake.
    """
    from core.cognition.widening_the_language import (
        DerivedAddressing,
        an_addressing_nobody_wrote,
    )

    # A word the language knows only at length three.
    only_at_three = DerivedAddressing(name="short", at={3: (2, 0, 1)})
    with pytest.raises(IndexError):
        only_at_three(0, 4)

    # Deriving at length four must not die on it.
    derived = an_addressing_nobody_wrote(
        [
            ((1, 2, 3, 4), (4, 1, 2, 3)),
            ((5, 6, 7, 8), (8, 5, 6, 7)),
        ],
        already={"short": only_at_three},
    )
    assert derived is not None
    assert derived(0, 4) == 3
