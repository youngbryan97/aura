"""Something later against something now, without a rate anybody chose.

A Stellaris agenda that pays after ten years, in a game they might not be in.
Go's territory against influence: points already theirs, against a position
that will become points if the game goes the way it looks like going.
"""

from __future__ import annotations

from core.cognition.what_it_is_worth_by_the_time_it_comes import (
    HowFarSheUsuallyGets,
    already_hers,
    what_it_comes_to,
)

#: A small certain thing now, and a large thing that takes a while.
OPTIONS = {"take the points now": (10.0, 0.0), "build the thing": (100.0, 50.0)}


def test_in_a_world_that_lasts_she_builds() -> None:
    steady = HowFarSheUsuallyGets()
    for _ in range(20):
        steady.a_run_lasted(200.0)
    assert what_it_comes_to(OPTIONS, steady)[0].name == "build the thing"


def test_in_a_world_that_does_not_she_takes_the_points() -> None:
    """The same two options, and nothing about them changed."""
    precarious = HowFarSheUsuallyGets()
    for _ in range(20):
        precarious.a_run_lasted(12.0)
    assert what_it_comes_to(OPTIONS, precarious)[0].name == "take the points now"


def test_having_been_nowhere_she_is_neither_a_builder_nor_a_hoarder() -> None:
    fresh = HowFarSheUsuallyGets()
    assert fresh.still_here_in(50.0) == 0.5
    assert "not finished a run" in fresh.describe()


def test_one_long_run_is_not_a_promise() -> None:
    barely = HowFarSheUsuallyGets()
    barely.a_run_lasted(200.0)
    assert barely.still_here_in(50.0) < 1.0


def test_a_payoff_now_has_nothing_taken_off_it() -> None:
    precarious = HowFarSheUsuallyGets()
    for _ in range(10):
        precarious.a_run_lasted(1.0)
    now = next(
        one for one in what_it_comes_to(OPTIONS, precarious) if one.arrives_in == 0
    )
    assert now.still_here == 1.0
    assert now.comes_to == now.pays


def test_settled_and_likely_are_not_added_up() -> None:
    """A lead made of things that might become points is not a lead when
    there is no time left for them to become anything."""
    things = [{"safe": True}, {"safe": True}, {"safe": False}]
    settled, likely = already_hers(things, lambda one: one["safe"])
    assert (settled, likely) == (2.0, 1.0)
