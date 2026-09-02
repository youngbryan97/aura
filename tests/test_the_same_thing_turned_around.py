"""Finding out that four corners are one corner, from her own record.

Two stones on a Go board and nothing can be searched: the branching is three
hundred and fifty nine. Strong players are not looking further, they are
looking at fewer things — the four corners are the same corner, so everything
ever learned about one is about all of them.

Nothing below is told which turnings exist.
"""

from __future__ import annotations

import random

from core.cognition.the_same_thing_turned_around import (
    turnings_that_hold,
    which_are_the_same,
)

PLACES = [(row, col) for row in range(2) for col in range(3)]
ACTS = ["left", "right"]


def _slid(row: list) -> list:
    kept = [one for one in row if one is not None]
    return kept + [None] * (len(row) - len(kept))


def _move(places: dict, act: str) -> dict:
    out: dict = {}
    for row in range(2):
        line = [places.get((row, col)) for col in range(3)]
        if act == "right":
            line = list(reversed(_slid(list(reversed(line)))))
        else:
            line = _slid(line)
        for col, one in enumerate(line):
            if one is not None:
                out[(row, col)] = one
    return out


def _a_record(seed: int = 3, how_many: int = 24):
    """What she watched: a situation, what she did, what it became.

    Every value distinct, because a place holding the same thing as another
    gives more than one way to line two situations up, and guessing which is
    how a coincidence becomes a belief.
    """
    roll = random.Random(seed)
    watched = []
    for _ in range(how_many):
        values = roll.sample(range(10, 99), roll.randrange(2, 5))
        where = roll.sample(PLACES, len(values))
        before = dict(zip(where, values, strict=True))
        act = roll.choice(ACTS)
        watched.append((before, act, _move(before, act)))
    return watched


def test_she_finds_the_turning_and_what_it_does_to_her_acts() -> None:
    held = turnings_that_hold(
        _a_record(), every_place=PLACES, acts=ACTS, expect=_move
    )
    assert held, "a world with a mirror in it and she found none"
    mirrored = [
        one
        for one in held
        if one.turns_acts_into.get("left") == "right"
        and one.turns_acts_into.get("right") == "left"
    ]
    assert mirrored, [one.describe() for one in held]
    turning = mirrored[0]
    # And what it does to places is the mirror, which nothing told it.
    assert turning.sends[(0, 0)] == (0, 2)
    assert turning.sends[(0, 2)] == (0, 0)
    assert turning.sends[(1, 1)] == (1, 1)


def test_a_situation_and_its_turning_get_one_name() -> None:
    """Which is the whole saving: one experience becomes as many as there are
    turnings of it."""
    held = turnings_that_hold(
        _a_record(), every_place=PLACES, acts=ACTS, expect=_move
    )
    assert held
    here = {(0, 0): 12, (1, 2): 34}
    there = {(0, 2): 12, (1, 0): 34}
    assert which_are_the_same(here, held) == which_are_the_same(there, held)
    # And two situations that are NOT turnings of each other do not.
    other = {(0, 0): 12, (1, 1): 34}
    assert which_are_the_same(here, held) != which_are_the_same(other, held)


def test_a_world_with_no_turning_in_it_yields_none() -> None:
    """Said rather than invented. A map that lines up one pair of situations
    is a coincidence, and only surviving the whole record makes it a fact."""
    roll = random.Random(5)
    lopsided = []
    for _ in range(24):
        values = roll.sample(range(10, 99), 3)
        before = dict(zip(roll.sample(PLACES, 3), values, strict=True))
        # A world where the act does something different in EVERY place, so
        # no relabelling of places can leave it looking the same. Doing
        # something different per column is not enough — the rows would then
        # be interchangeable, and interchangeable is exactly what a turning
        # is.
        after = {
            (row, col): what + row * 3 + col
            for (row, col), what in before.items()
        }
        lopsided.append((before, "poke", after))
    held = turnings_that_hold(
        lopsided, every_place=PLACES, acts=["poke"],
        expect=lambda places, act: {
            (row, col): what + row * 3 + col for (row, col), what in places.items()
        },
    )
    assert all(one.sends != {} for one in held)
    for one in held:
        # Anything it did keep has to be the identity, which is no claim.
        assert all(where == goes for where, goes in one.sends.items()), one.describe()
