"""Once she can invent freely, she reinvents the same thing in different clothes.

Measured after a single invention: thirty words, twenty-five distinct
behaviours, five duplicates — and the plainest was "if how many there are left
over 2 then [here] else [here]", which is "here" at six times the length.

Every duplicate is another branch at every step of every search, bought for
nothing.
"""

from __future__ import annotations

import pytest

from core.cognition import an_invented_kind as kinds
from core.cognition.a_kind_of_thing_she_named import (
    KINDS_OF_THING,
    a_kind_of_thing_she_named,
    a_way_of_building_over,
)
from core.cognition.one_thing_many_spellings import (
    how_it_behaves,
    one_of_each,
    the_other_spellings,
)

FOURS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 1, 2, 6), (4, 7, 2, 8)]
FIVES = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 1), (2, 4, 6, 8, 3), (5, 1, 9, 3, 7)]


@pytest.fixture(autouse=True)
def _left_as_found():
    ways, named = dict(kinds.WAYS_TO_BUILD), dict(KINDS_OF_THING)
    kinds.WAYS_TO_BUILD.clear()
    KINDS_OF_THING.clear()
    try:
        yield
    finally:
        for holds, was in ((kinds.WAYS_TO_BUILD, ways), (KINDS_OF_THING, named)):
            holds.clear()
            holds.update(was)


def _a_language_that_has_grown():
    far = kinds.WHERE_FROM["the far end"]
    along = kinds.WHERE_FROM["one along"]

    def either(at, size):
        return far(at, size) if size % 2 == 0 else along(at, size)

    family = [
        (one, tuple(one[either(at, len(one)) % len(one)] for at in range(len(one))))
        for one in FOURS + FIVES
    ]
    named = a_kind_of_thing_she_named(family)
    assert named is not None
    maker, _over = a_way_of_building_over(named)
    kinds.WAYS_TO_BUILD["over parity"] = maker
    return family


def test_a_grown_language_holds_no_two_words_that_do_the_same_thing():
    _a_language_that_has_grown()
    said = [how_it_behaves(word) for word in kinds.addressings().values()]
    said = [one for one in said if one is not None]
    assert len(said) == len(set(said))


def test_the_cheapest_spelling_is_the_one_kept():
    """Cheapest on the ruler she cannot move, because names are what a maker
    can make cheap."""
    from core.cognition.the_ruler_she_cannot_move import what_it_costs_to_be

    _a_language_that_has_grown()
    words = kinds.addressings()
    assert what_it_costs_to_be(words["here"]) == 1


def test_the_other_spellings_are_kept_rather_than_thrown_away():
    """A concept keeps its alternatives; a cheaper one changes the spelling."""
    _a_language_that_has_grown()
    also = the_other_spellings(kinds.addressings()["here"])
    assert any("else" in one for one in also)


def test_nothing_expressible_is_lost_by_removing_the_duplicates():
    family = _a_language_that_has_grown()
    assert kinds.induce_from(family) is not None


def test_a_word_that_will_not_say_what_it_does_is_kept():
    """Dropping something because it declined to be measured is worse."""

    def refuses(at, size):
        raise ValueError("not at that size")

    kept = one_of_each({"refuses": refuses, "here": kinds.WHERE_FROM["here"]})
    assert "refuses" in kept


def test_two_names_for_one_behaviour_collapse_to_one():
    here = kinds.WHERE_FROM["here"]
    kept = one_of_each({"here": here, "also here": here})
    assert len(kept) == 1
