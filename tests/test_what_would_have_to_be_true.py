"""Wanting the near thing instead of the far one.

A bag of numbers. Two of the same combine into one of twice the size, and a
one can always be picked up. Nothing is laid out; there are no rows and no
corners and no board.

The point of the tests is the horizon. Looking one step ahead cannot see an
eight from here however carefully it looks, and no amount of looking harder
inside one step will change that. Walking back from the eight to what would
make an eight near lands on something one step away, and then the far thing is
reached by wanting the near one.
"""

from __future__ import annotations

import random

from core.cognition.what_would_have_to_be_true import (
    a_way_to_get_there,
    what_would_have_to_be_true,
)

Bag = tuple[int, ...]
PICK_UP = "pick up a one"


def _acts(bag: Bag) -> list:
    return [PICK_UP, *sorted({one for one in bag if bag.count(one) >= 2})]


def _do(bag: Bag, act) -> Bag | None:
    if act == PICK_UP:
        return tuple(sorted(bag + (1,)))
    if bag.count(act) < 2:
        return None
    left = list(bag)
    left.remove(act)
    left.remove(act)
    return tuple(sorted(left + [act * 2]))


def _in_reach(bag: Bag, wanted, *, steps: int = 1) -> bool:
    here = {bag}
    for _ in range(steps):
        after = set()
        for was in here:
            for act in _acts(was):
                got = _do(was, act)
                if got is None:
                    continue
                if wanted(got):
                    return True
                after.add(got)
        here = after
    return bool(wanted(bag))


def _has_an_eight(bag: Bag) -> bool:
    return 8 in bag


HERE: Bag = (1, 1, 2, 2, 4)


def _somewhere_like(seed: int = 7, how_many: int = 40) -> list[Bag]:
    roll = random.Random(seed)
    return [
        tuple(sorted(roll.choice([1, 2, 4]) for _ in range(roll.randrange(3, 7))))
        for _ in range(how_many)
    ]


def test_looking_ahead_cannot_see_it_from_here() -> None:
    assert not _in_reach(HERE, _has_an_eight)


def test_what_would_make_it_near_is_two_of_the_thing_below_it() -> None:
    nearer = what_would_have_to_be_true(
        _has_an_eight, _somewhere_like(), _in_reach
    )
    assert nearer is not None
    assert nearer.name == "it holds two 4", nearer.name
    assert nearer.tells_them_apart > 0.9


def test_the_walk_back_lands_on_something_she_can_do_now() -> None:
    way = a_way_to_get_there(
        _has_an_eight,
        HERE,
        somewhere_like=_somewhere_like(),
        in_reach=_in_reach,
        called="an eight",
    )
    assert way.reached, way.describe()
    assert [one.name for one in way.steps] == ["it holds two 4", "an eight"]
    first = way.want_first
    assert first is not None
    assert _in_reach(HERE, first.holds)


def test_it_says_so_when_nothing_it_has_seen_makes_the_thing_nearer() -> None:
    way = a_way_to_get_there(
        lambda bag: 4096 in bag,
        HERE,
        somewhere_like=_somewhere_like(),
        in_reach=_in_reach,
        called="a very large one",
    )
    assert not way.reached
    assert "tells the near places from the far" in way.why


def test_a_want_already_met_needs_no_walk() -> None:
    way = a_way_to_get_there(
        lambda bag: 4 in bag,
        HERE,
        somewhere_like=_somewhere_like(),
        in_reach=_in_reach,
        called="a four",
    )
    assert way.reached
    assert [one.name for one in way.steps] == ["a four"]
