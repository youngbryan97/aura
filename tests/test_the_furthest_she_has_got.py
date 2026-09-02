"""Practising rather than repeating: keep what worked, vary where it stopped.

The counter in the corner of the Ninja Gaiden recording says P-6. He is not
reacting his way through on the sixth go; he is replaying what he already
knows and thinking only where he died last time.
"""

from __future__ import annotations

import random

from core.cognition.the_furthest_she_has_got import TheFurthestSheHasGot

DOORS = ["a", "b", "c", "d"]
#: A route that only one order gets through. Nothing says which.
THE_WAY = ["c", "a", "d", "b"]


def _how_far(took):
    """How many of these acts led somewhere, which the world knows and she
    does not."""
    got = 0
    for want, one in zip(THE_WAY, took, strict=False):
        if one != want:
            break
        got += 1
    return got


def test_she_finds_a_route_by_keeping_what_worked() -> None:
    roll = random.Random(4)
    got = TheFurthestSheHasGot()
    for _attempt in range(60):
        took = list(got.replay())
        while len(took) < len(THE_WAY):
            worth = got.worth_trying_at_the_frontier(DOORS) if len(took) == got.frontier else DOORS
            if not worth:
                break
            took.append(roll.choice(worth))
            if _how_far(took) < len(took):
                break
        got.an_attempt_ended(took, got_to=_how_far(took))
        if got.frontier == len(THE_WAY):
            break
    assert list(got.replay()) == THE_WAY, got.describe(DOORS)
    # Four doors and four places. Blundering about would take far longer than
    # this; the point is that the part that worked is never re-derived.
    assert got.attempts < 20, got.attempts


def test_it_does_not_try_the_same_thing_in_the_same_place_twice() -> None:
    got = TheFurthestSheHasGot()
    got.an_attempt_ended(["a"], got_to=0)
    assert "a" in got.already_failed_here()
    assert "a" not in got.worth_trying_at_the_frontier(DOORS)
    got.an_attempt_ended(["b"], got_to=0)
    assert got.worth_trying_at_the_frontier(DOORS) == ["c", "d"]


def test_everything_failing_in_one_place_says_to_change_something_earlier() -> None:
    """Which is a finding, not a dead end. Somewhere before it is wrong."""
    got = TheFurthestSheHasGot()
    for door in DOORS:
        got.an_attempt_ended([door], got_to=0)
    assert got.stuck_at_the_frontier(DOORS)
    assert got.worth_trying_at_the_frontier(DOORS) == []
    assert "something earlier has to change" in got.describe(DOORS)


def test_getting_further_moves_the_thinking_forward() -> None:
    got = TheFurthestSheHasGot()
    assert got.an_attempt_ended(["c", "b"], got_to=1) is True
    assert got.replay() == ("c",)
    assert got.frontier == 1
    assert "b" in got.already_failed_here()
    assert got.an_attempt_ended(["c", "a", "b"], got_to=2) is True
    assert got.replay() == ("c", "a")
    assert got.frontier == 2
    # That attempt tried "b" at the new place and it did not work there
    # either, so it is ruled out THERE — which is a different fact from it
    # being ruled out one step earlier.
    assert got.already_failed_here() == frozenset({"b"})
    assert got.already_failed_here(at=1) == frozenset({"b"})
    assert got.worth_trying_at_the_frontier(DOORS) == ["a", "c", "d"]


def test_a_route_keeps_between_sittings() -> None:
    got = TheFurthestSheHasGot()
    got.an_attempt_ended(["c", "b"], got_to=1)
    again = TheFurthestSheHasGot.from_memory(got.as_memory())
    assert again.replay() == ("c",)
    assert again.already_failed_here() == frozenset({"b"})
    assert again.attempts == 1
