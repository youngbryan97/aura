"""Whether a thing can survive, and who runs out first when two cannot.

Go decides whether a group lives by whether it has TWO separate spaces the
other player can never fill. One is not enough. The difference between one and
two is the difference between a thing that is alive and a thing that is merely
still there, and nothing about size changes it.

The tests are services and their inputs, because it is the same question.
"""

from __future__ import annotations

from core.cognition.two_ways_out import (
    how_long_it_holds,
    not_worth_another_move,
    the_ones_worth_defending,
    who_wins_the_race,
)

BIG_ONE_INPUT = {"name": "the big one", "inputs": ["the primary feed"], "room": 9}
SMALL_TWO_INPUTS = {"name": "the small one", "inputs": ["a feed", "a mirror"], "room": 2}


def _stands(thing):
    return how_long_it_holds(
        thing, ways_out=lambda one: one["inputs"], room=lambda one: one["room"]
    )


def test_two_ways_out_is_alive_and_one_is_only_still_standing() -> None:
    big = _stands(BIG_ONE_INPUT)
    small = _stands(SMALL_TWO_INPUTS)
    assert not big.alive, big.describe()
    assert small.alive, small.describe()
    # And the one in danger is the LARGER one, which is the point.
    assert big.steps_left > small.steps_left


def test_the_ones_worth_defending_are_the_ones_with_one_way_out() -> None:
    """Not the biggest and not the weakest."""
    already_gone = {"name": "gone", "inputs": [], "room": 0}
    worth = the_ones_worth_defending(
        [BIG_ONE_INPUT, SMALL_TWO_INPUTS, already_gone],
        ways_out=lambda one: one["inputs"],
        room=lambda one: one["room"],
    )
    assert [one[0]["name"] for one in worth] == ["the big one"]


def test_a_race_is_counted_and_not_fought() -> None:
    quick = how_long_it_holds(
        {"inputs": ["one"], "room": 3}, ways_out=lambda o: o["inputs"], room=lambda o: o["room"]
    )
    slow = how_long_it_holds(
        {"inputs": ["one"], "room": 7}, ways_out=lambda o: o["inputs"], room=lambda o: o["room"]
    )
    who, by = who_wins_the_race(slow, quick, i_move_first=True)
    assert who == "mine" and by == 5, "more room and moving first wins by five"
    who, _by = who_wins_the_race(quick, slow, i_move_first=True)
    assert who == "theirs"


def test_moving_first_is_worth_exactly_one_step() -> None:
    even = how_long_it_holds(
        {"inputs": ["one"], "room": 4}, ways_out=lambda o: o["inputs"], room=lambda o: o["room"]
    )
    assert who_wins_the_race(even, even, i_move_first=True)[0] == "mine"
    assert who_wins_the_race(even, even, i_move_first=False)[0] == "neither"


def test_a_thing_that_cannot_be_killed_gets_the_shared_room() -> None:
    """The asymmetry that decides most of these, and why one way out is worth
    so much less than two."""
    alive = how_long_it_holds(
        {"inputs": ["a", "b"], "room": 1}, ways_out=lambda o: o["inputs"], room=lambda o: o["room"]
    )
    not_alive = how_long_it_holds(
        {"inputs": ["a"], "room": 6}, ways_out=lambda o: o["inputs"], room=lambda o: o["room"]
    )
    who, _by = who_wins_the_race(alive, not_alive, shared=4)
    assert who == "mine", "the one that cannot be killed wins however little room it has"


def test_a_race_already_lost_is_not_worth_another_move() -> None:
    """Which is the point of counting: the moves are worth something else."""
    losing = how_long_it_holds(
        {"inputs": ["one"], "room": 2}, ways_out=lambda o: o["inputs"], room=lambda o: o["room"]
    )
    winning = how_long_it_holds(
        {"inputs": ["one"], "room": 9}, ways_out=lambda o: o["inputs"], room=lambda o: o["room"]
    )
    assert not_worth_another_move(losing, winning)
    assert not not_worth_another_move(winning, losing)
