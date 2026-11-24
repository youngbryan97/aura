"""Positioning in time, in worlds that are not boards (core/cognition/where_forces_can_meet.py)."""
from __future__ import annotations

import pytest

from core.cognition.where_forces_can_meet import (
    Mover,
    arrivals,
    balance,
    intended,
    nearer,
    overloaded,
    time_to_reach,
)

pytestmark = pytest.mark.unit

#: A road network: towns and the roads between them.
ROADS = {
    "port": ["mill", "ford"], "mill": ["port", "hill", "gate"], "ford": ["port", "gate"],
    "hill": ["mill", "keep"], "gate": ["mill", "ford", "keep", "well"], "keep": ["hill", "gate"],
    "well": ["gate", "farm"], "farm": ["well"],
}


def by_road(_mover, place, _standing):
    return ROADS[place]


def test_how_soon_a_courier_reaches_each_town() -> None:
    courier = Mover("c", 0, "port")
    assert time_to_reach(courier, by_road, [], horizon=3) == {
        "port": 0, "mill": 1, "ford": 1, "hill": 2, "gate": 2, "keep": 3, "well": 3}


def test_what_stands_in_a_town_is_met_there_and_not_passed() -> None:
    courier = Mover("c", 0, "port")
    blocked = time_to_reach(courier, by_road, ["mill", "ford"], horizon=4)
    assert blocked == {"port": 0, "mill": 1, "ford": 1}


def test_who_can_be_at_a_place_and_how_soon() -> None:
    movers = [Mover("a", 0, "port"), Mover("b", 0, "farm"), Mover("x", 1, "keep")]
    came = arrivals(movers, by_road, "gate", horizon=3)
    assert [(m.name, t) for m, t in came[0]] == [("a", 2), ("b", 2)]
    assert [(m.name, t) for m, t in came[1]] == [("x", 1)]


def test_the_balance_at_the_places_that_matter_by_time() -> None:
    movers = [Mover("a", 0, "port"), Mover("b", 0, "farm"), Mover("x", 1, "keep")]
    counts = balance(movers, by_road, ["gate"], horizon=3)
    assert counts[0] == [0, 2, 2] and counts[1] == [1, 1, 1]


def test_a_defender_that_alone_holds_two_places_is_overloaded() -> None:
    """Their one guard at the gate is the only one of theirs in time for both the mill and the well."""
    movers = [Mover("guard", 1, "gate"), Mover("raider", 0, "hill"), Mover("other", 0, "farm")]
    found = overloaded(movers, by_road, ["mill", "well"], side=1, horizon=2)
    assert [(m.name, sorted(places)) for m, places in found] == [("guard", ["mill", "well"])]


def test_maneuvering_is_the_next_step_that_closes_in() -> None:
    courier = Mover("c", 0, "port")
    steps = nearer(courier, by_road, [], ["keep"], horizon=4)
    assert steps[0][1] == 2 and {there for there, left in steps if left == 2} == {"mill", "ford"}


def grid_reach(_mover, place, standing):
    r, c = place
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        there = (r + dr, c + dc)
        if 0 <= there[0] < 6 and 0 <= there[1] < 6:
            yield there


def test_moves_that_converge_where_she_can_be_stronger_are_intended() -> None:
    """Units on an open field: three of hers can be at their camp before more than its own one can answer there."""
    movers = [Mover("u1", 0, (2, 5)), Mover("u2", 0, (3, 3)), Mover("u3", 0, (1, 4)),
              Mover("camp", 1, (5, 5)), Mover("far", 1, (0, 0))]
    matters = {1: [(5, 5), (4, 5), (5, 4)], 0: [(0, 5)]}
    assert balance(movers, grid_reach, matters[1], horizon=6)[0][2] > balance(movers, grid_reach, matters[1], 6)[1][2]
    shares = intended(movers, grid_reach, side=0, matters=matters, horizon=6)
    assert shares and abs(sum(shares.values()) - 1.0) < 1e-9
    # Each of hers closing in is intended, and nothing that moves away.
    assert ((2, 5), (3, 5)) in shares and ((3, 3), (4, 3)) in shares
    assert ((2, 5), (1, 5)) not in shares


def test_where_she_cannot_be_stronger_nothing_is_intended_toward_it() -> None:
    movers = [Mover("u", 0, (0, 0)), Mover("camp", 1, (5, 5)), Mover("d1", 1, (4, 4)), Mover("d2", 1, (5, 3))]
    assert intended(movers, grid_reach, side=0, matters={1: [(5, 5), (4, 5), (5, 4)]}, horizon=6) == {}


def test_what_must_be_kept_safe_goes_where_fewer_of_theirs_can_come() -> None:
    """Three of theirs gather on her keep's left sooner than hers can answer there: she intends to move it right."""
    keep_zone = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]
    movers = [Mover("keep", 0, (0, 2)), Mover("h", 0, (0, 5)),
              Mover("a", 1, (2, 0)), Mover("b", 1, (2, 1)), Mover("c", 1, (1, 0))]
    shares = intended(movers, grid_reach, side=0, matters={0: keep_zone, 1: [(5, 5)]}, horizon=4)
    assert ((0, 2), (0, 3)) in shares
    assert ((0, 2), (0, 1)) not in shares
