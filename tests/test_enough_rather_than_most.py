"""Enough, rather than the most — and both halves from the same sentence.

Strong Go players count in the late game and then simplify if ahead or
complicate if behind. It is not a taste for risk. It is what winning is when
winning is a threshold.
"""

from __future__ import annotations

from core.cognition.enough_rather_than_most import (
    how_likely_each_is,
    the_one_most_likely_to_do,
)

#: The quiet move always gains a little. The wild one usually gains nothing
#: and occasionally gains a lot. On average the quiet one is better.
QUIET = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
WILD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0]
OPTIONS = {"the quiet one": QUIET, "the wild one": WILD}


def test_a_long_way_behind_she_takes_the_one_that_might_not_come_off() -> None:
    """The quiet move loses slowly, and losing slowly is still losing."""
    took = the_one_most_likely_to_do(OPTIONS, needs=15.0, has=0.0)
    assert took is not None
    assert took.name == "the wild one", took.describe()
    # And it is the WORSE option on the ordinary measure, which is the point.
    by_name = {one.name: one for one in how_likely_each_is(OPTIONS, needs=15.0)}
    assert by_name["the wild one"].on_average < by_name["the quiet one"].on_average


def test_needing_only_a_little_she_takes_the_sure_thing() -> None:
    took = the_one_most_likely_to_do(OPTIONS, needs=2.0, has=0.0)
    assert took is not None
    assert took.name == "the quiet one", took.describe()
    assert took.clears_it == 1.0


def test_far_enough_ahead_she_simplifies_without_being_told_to() -> None:
    """Both clear the bar every time, so the narrower spread wins — which is
    the whole of what simplifying when ahead means."""
    took = the_one_most_likely_to_do(OPTIONS, needs=10.0, has=10.0)
    assert took is not None
    assert took.name == "the quiet one", took.describe()
    assert took.clears_it == 1.0
    assert took.how_spread == 0.0


def test_the_same_options_go_two_ways_on_where_she_stands() -> None:
    """Nothing about the options changed. Only how much she still needs."""
    behind = the_one_most_likely_to_do(OPTIONS, needs=15.0)
    ahead = the_one_most_likely_to_do(OPTIONS, needs=15.0, has=14.0)
    assert behind is not None and ahead is not None
    assert behind.name != ahead.name
    assert behind.name == "the wild one" and ahead.name == "the quiet one"


def test_a_bar_nothing_reaches_is_not_a_bar_she_can_steer_by() -> None:
    took = the_one_most_likely_to_do(OPTIONS, needs=1000.0)
    assert took is not None
    assert took.clears_it == 0.0
    # Falls back to the ordinary measure, because there is nothing else left.
    assert took.name == "the quiet one"


def test_nothing_to_choose_from_is_nothing() -> None:
    assert the_one_most_likely_to_do({}, needs=1.0) is None
    assert the_one_most_likely_to_do({"a": []}, needs=1.0) is None
