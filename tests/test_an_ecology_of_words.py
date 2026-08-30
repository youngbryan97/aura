"""A language that only grows eventually cannot be searched.

Two gates already ask whether something is worth adding. Neither is ever asked
again, so until now nothing ever left. These check that something does, that it
is the right something, and that nothing she worked out goes with it.
"""

from __future__ import annotations

import pytest

from core.cognition import an_invented_kind as kinds
from core.cognition.a_kind_of_thing_she_named import (
    KINDS_OF_THING,
    a_kind_of_thing_she_named,
    a_way_of_building_over,
)
from core.cognition.an_ecology_of_words import (
    a_maker_that_stopped_paying,
    a_season,
    forget,
    what_can_be_forgotten,
    what_each_word_earned,
)


@pytest.fixture(autouse=True)
def _left_as_found():
    was = (
        dict(kinds.WAYS_TO_BUILD),
        dict(KINDS_OF_THING),
        dict(kinds.KINDS),
        dict(kinds.WHERE_FROM),
    )
    for store in (kinds.WAYS_TO_BUILD, KINDS_OF_THING, kinds.KINDS):
        store.clear()
    # Words another test file derived are still in the language at setup, so a
    # fixture that only restores what it found starts from someone else's
    # language.
    for name in [
        one
        for one, word in list(kinds.WHERE_FROM.items())
        if type(word).__name__ == "DerivedAddressing"
    ]:
        kinds.WHERE_FROM.pop(name, None)
    try:
        yield
    finally:
        for store, before in zip(
            (kinds.WAYS_TO_BUILD, KINDS_OF_THING, kinds.KINDS, kinds.WHERE_FROM), was
        ):
            store.clear()
            store.update(before)


def _family(sizes, when_even, when_odd):
    return [
        (
            one,
            tuple(
                one[(when_even if len(one) % 2 == 0 else when_odd)(at, len(one)) % len(one)]
                for at in range(len(one))
            ),
        )
        for one in sizes
    ]


SHOWN = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 1, 2, 6), (4, 7, 2, 8)]
SHOWN += [(1, 2, 3, 4, 5), (6, 7, 8, 9, 1), (2, 4, 6, 8, 3), (5, 1, 9, 3, 7)]
UNSEEN = [(3, 1, 4, 2), (5, 9, 2, 6), (3, 1, 4, 1, 5), (9, 2, 6, 5, 3)]


def _a_language_that_has_grown():
    far, along = kinds.WHERE_FROM["the far end"], kinds.WHERE_FROM["one along"]
    shown = _family(SHOWN, far, along)
    maker, _over = a_way_of_building_over(a_kind_of_thing_she_named(shown))
    kinds.WAYS_TO_BUILD["over parity"] = maker
    kinds.admit("mixed by parity", kinds.induce_from(shown))
    return maker, _family(UNSEEN, along, far)


def _nothing_reaching(_words):
    return {
        f"always the {n}th": (lambda at, size, n=n: n % max(1, size))
        for n in range(7, 13)
    }


def test_a_maker_that_reaches_nothing_is_retired():
    _maker, held = _a_language_that_has_grown()
    kinds.WAYS_TO_BUILD["always the nth"] = _nothing_reaching
    worth = a_maker_that_stopped_paying(
        "always the nth", _nothing_reaching, held_out=[held]
    )
    assert not worth.keep_it
    assert worth.reaches == 0


def test_a_maker_that_still_reaches_is_kept():
    maker, held = _a_language_that_has_grown()
    worth = a_maker_that_stopped_paying("over parity", maker, held_out=[held])
    assert worth.keep_it
    assert worth.reaches > 0


def test_a_season_takes_the_useless_maker_and_its_words_with_it():
    _maker, held = _a_language_that_has_grown()
    kinds.WAYS_TO_BUILD["always the nth"] = _nothing_reaching
    before = len(kinds.addressings())
    after = a_season(held_out=[held])
    assert "always the nth" not in kinds.WAYS_TO_BUILD
    assert len(kinds.addressings()) < before
    assert after.smaller_by > 0


def test_the_season_counts_what_a_retired_maker_took_with_it():
    """It said "nothing forgotten" while dropping six words, because retiring a
    maker and dropping a word are two paths and one was not writing the record."""
    _maker, held = _a_language_that_has_grown()
    kinds.WAYS_TO_BUILD["always the nth"] = _nothing_reaching
    after = a_season(held_out=[held])
    assert after.forgotten
    assert any("always the nth" in one for one in after.forgotten)


def test_two_makers_that_make_the_same_words_leave_one():
    maker, held = _a_language_that_has_grown()
    kinds.WAYS_TO_BUILD["also over parity"] = maker
    a_season(held_out=[held])
    assert len(kinds.WAYS_TO_BUILD) == 1


def test_what_she_settled_stays_sayable_through_a_season():
    """Losing knowledge to save search is the wrong trade at any price."""
    _maker, held = _a_language_that_has_grown()
    kinds.WAYS_TO_BUILD["always the nth"] = _nothing_reaching
    a_season(held_out=[held])
    assert kinds.interpretation_of("mixed by parity") is not None


def test_a_word_a_settled_meaning_names_is_never_dropped():
    _maker, _held = _a_language_that_has_grown()
    used = [one for one in what_each_word_earned() if one.used_by]
    assert used
    kept = forget([one.name for one in used])
    assert kept.forgotten == ()


def test_a_made_word_is_not_dropped_directly_because_it_would_come_back():
    """Its maker rebuilds it on the next call, so the saving never happens."""
    _maker, _held = _a_language_that_has_grown()
    made = [one for one in what_each_word_earned() if one.sort == "made"]
    assert made
    assert forget([one.name for one in made]).forgotten == ()


def test_a_word_the_source_gave_her_is_never_weighed():
    _maker, _held = _a_language_that_has_grown()
    assert all(one.sort != "given" for one in what_each_word_earned())
    assert forget(["here", "the far end"]).forgotten == ()


def test_nothing_is_forgotten_from_a_language_that_has_not_grown():
    assert what_can_be_forgotten() == ()
    assert a_season().forgotten == ()
