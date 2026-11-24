"""A small win that costs no ground is taken, and one that lasts is worth more (core/cognition/money_in_the_bank.py),
in a world that is not a board: units on an open field."""
from __future__ import annotations

import pytest

from core.cognition.money_in_the_bank import Act, ground_lost, lasting_gain, what_to_bank
from core.cognition.where_forces_can_meet import Mover

pytestmark = pytest.mark.unit

SIZE = 6


def field_reach(_mover, place, standing):
    r, c = place
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        there = (r + dr, c + dc)
        if 0 <= there[0] < SIZE and 0 <= there[1] < SIZE:
            yield there


def moved(movers, name, to):
    return tuple(Mover(m.name, m.side, to if m.name == name else m.at, m.worth) for m in movers)


#: Her guard by her keep in one corner, a raider of hers mid-field; their runner near its own camp in the other.
BEFORE = (Mover("keep", 0, (0, 0)), Mover("guard", 0, (1, 0)), Mover("raider", 0, (3, 4)),
          Mover("camp", 1, (5, 5)), Mover("runner", 1, (4, 5)))
MATTERS = {0: [(0, 0), (0, 1), (1, 0)], 1: [(5, 5), (5, 4), (4, 5)]}


def test_closing_their_way_out_is_a_gain_that_lasts() -> None:
    hemmed = moved(BEFORE, "raider", (4, 4))  # beside their runner and camp: fewer ways out for either
    assert lasting_gain(BEFORE, hemmed, field_reach, 0, MATTERS, horizon=2) > 0


def test_taking_her_guard_away_from_her_keep_loses_ground() -> None:
    away = moved(BEFORE, "guard", (2, 0))
    assert ground_lost(BEFORE, away, field_reach, 0, MATTERS, horizon=3)
    assert not ground_lost(BEFORE, moved(BEFORE, "raider", (4, 4)), field_reach, 0, MATTERS, horizon=3)


def test_among_acts_her_judgement_cannot_tell_apart_she_banks_and_never_gives_ground() -> None:
    acts = [
        Act("drift", 0.0, moved(BEFORE, "raider", (2, 4))),          # judged best, banks nothing
        Act("hem", -0.1, moved(BEFORE, "raider", (4, 4))),           # as good within the judgement's unsteadiness
        Act("guard off", -0.05, moved(BEFORE, "guard", (2, 0))),     # as good, gives ground at her keep
        Act("all in", -5.0, moved(BEFORE, "raider", (5, 4))),        # far worse by her judgement
    ]
    chosen = what_to_bank(acts, BEFORE, field_reach, 0, MATTERS, horizon=2, within=0.2)
    assert chosen is not None and chosen.name == "hem"


def test_where_nothing_banks_nothing_is_chosen() -> None:
    acts = [Act("drift", 0.0, moved(BEFORE, "raider", (2, 4)))]
    assert what_to_bank(acts, BEFORE, field_reach, 0, MATTERS, horizon=2, within=0.2) is None
