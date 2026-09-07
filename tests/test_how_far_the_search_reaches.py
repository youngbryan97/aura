"""A universal language, and a search that walks 380 terms of it."""
from __future__ import annotations

import collections

import pytest

from core.cognition.how_far_the_search_reaches import (
    how_far_it_reaches,
    how_many_at,
    how_many_up_to,
    what_the_library_buys,
    what_the_search_reached,
)

#: What the operator search stands on: nothing, three constants, one variable.
FLOOR_LEAVES = 5


@pytest.mark.parametrize("size", [1, 2, 3])
def test_the_count_agrees_with_the_generator_it_counts(size: int) -> None:
    """Counted rather than generated, and the two must not disagree.

    A denominator computed by a different rule from the thing it divides is a
    ratio of two different questions.
    """
    from core.cognition.the_floor_she_stands_on import every_code, how_long

    seen = collections.Counter(
        how_long(one)
        for one in every_code(deepest=3, variables=1, constants=(0, 1, 2), also=())
    )
    assert how_many_at(size, leaves=FLOOR_LEAVES) == seen[size]


def test_the_whole_space_at_depth_three_is_the_number_the_code_already_knew() -> None:
    """`an_operator_she_invents` says it "walked the same 380 terms"."""
    assert how_many_up_to(3, leaves=FLOOR_LEAVES) == 380


def test_a_cap_above_the_space_is_a_search_that_exhausted_it() -> None:
    """Reporting the cap as the count reads as sampling when it was exhaustive."""
    reached = what_the_search_reached(
        deepest=3, leaves=FLOOR_LEAVES, would_examine=4000
    )
    assert reached.there_were == 380
    assert reached.examined == 380, "it cannot walk more terms than exist"
    assert reached.exhausted
    assert reached.share_examined == 1.0


def test_a_cap_below_the_space_is_a_sample_and_says_so() -> None:
    reached = what_the_search_reached(
        deepest=4, leaves=FLOOR_LEAVES, would_examine=4000
    )
    assert reached.there_were > reached.would_examine
    assert not reached.exhausted
    assert 0.0 < reached.share_examined < 1.0
    assert reached.to_dict()["one_in"] >= 1


def test_a_library_entry_is_eventually_worth_more_than_a_depth() -> None:
    """The only thing that moves a shortest-first horizon is better leaves."""
    few = what_the_library_buys(deepest=3, floor_leaves=FLOOR_LEAVES, from_her_library=2)
    assert few["a_library_entry_is_worth"] == "less than a depth"

    many = what_the_library_buys(
        deepest=3, floor_leaves=FLOOR_LEAVES, from_her_library=20
    )
    assert many["a_library_entry_is_worth"] == "more than a depth"
    assert many["times_larger"] > few["times_larger"]


def test_more_leaves_and_more_depth_both_grow_the_space() -> None:
    assert how_many_up_to(3, leaves=6) > how_many_up_to(3, leaves=5)
    assert how_many_up_to(4, leaves=5) > how_many_up_to(3, leaves=5)


def test_nothing_is_admitted_below_size_one() -> None:
    assert how_many_at(0, leaves=5) == 0
    assert how_many_at(-3, leaves=5) == 0
    assert how_many_up_to(1, leaves=5) == 5


def test_the_report_carries_both_the_reach_and_what_widens_it() -> None:
    said = how_far_it_reaches(deepest=3, leaves=FLOOR_LEAVES, would_examine=4000)
    assert said["reach"]["there_were"] == 380
    assert said["library"]["terms_at_one_more_depth"] > 380


def test_what_her_library_actually_offers_is_reported_rather_than_assumed() -> None:
    """The mechanism that moves the horizon, and how much is in it."""
    from core.cognition.what_she_already_knows_how_to_say import (
        what_she_already_knows_how_to_say,
    )

    hers = what_she_already_knows_how_to_say()
    said = how_far_it_reaches(
        deepest=3,
        leaves=FLOOR_LEAVES + len(hers),
        would_examine=4000,
        from_her_library=len(hers),
    )
    assert said["library"]["from_her_library"] == len(hers)
    # No assertion about the size: an empty library is a fact about this
    # process rather than a failure, and the point is that it is reported.
    assert said["reach"]["there_were"] >= 380
