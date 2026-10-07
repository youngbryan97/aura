"""Where a game's words ask for a place to be gone over, she goes over it, nearest first and sweeping, rather than wait.

LIVE 2026-10-07 a painter told to give "the entire campground a fresh coat of paint" stood waiting for something to
go to, and its paint ran out where it stood. Covering is read from the words in any phrasing (paint, mow, fill in,
explore), and the aim is measured in cells of her own thing's size; nothing here knows a game.
"""
from __future__ import annotations

import pytest

from core.agency.what_the_rules_said import WhatTheRulesSaid

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("words", "covers"), [
    ("Help Lazlo spruce things up by giving the entire campground a fresh coat of paint. Use the arrow keys to move.", True),
    ("Mow the whole lawn before the storm.", True),
    ("Fill in every square of the grid with colour.", True),
    ("Explore the cave and find the key.", True),
    ("Catch the fruit. Not the bombs.", False),
    ("Click the orange targets. Do not hit the green ones.", False),
    ("Keep the ball from getting past you.", False),
])
def test_the_words_say_whether_the_ground_is_the_point(words, covers):
    assert WhatTheRulesSaid.read(words).covers is covers


class _Thing:
    def __init__(self, x, y, w=10.0, h=10.0):
        self.x, self.y, self.w, self.h = x, y, w, h


def _aim(covered, at, heading=(0.0, 0.0)):
    from core.agency.playing_as_it_happens import _Choosing

    choosing = object.__new__(_Choosing)
    choosing.mine = _Thing(*at)
    choosing.covered = (covered, 10.0)

    class _Moves:
        shape = (40, 60)  # rows, columns: six cells across, four down

    class _Hers:
        last_velocity = heading

    choosing.moves, choosing.hers = _Moves(), _Hers()
    return choosing._ground_not_yet_covered()


def test_she_goes_to_the_nearest_ground_not_yet_gone_over():
    covered = {(0, 0), (1, 0), (2, 0)}
    assert _aim(covered, (25, 5)) in ((35.0, 5.0), (25.0, 15.0))  # the next cell along, or the one under her


def test_ties_go_the_way_she_is_already_going_so_she_sweeps():
    assert _aim({(2, 1)}, (25, 15), heading=(1.0, 0.0)) == (35.0, 15.0)
    assert _aim({(2, 1)}, (25, 15), heading=(-1.0, 0.0)) == (15.0, 15.0)


def test_when_everything_is_gone_over_there_is_nowhere_left():
    every = {(c, r) for c in range(6) for r in range(4)}
    assert _aim(every, (25, 15)) is None


def test_where_the_words_do_not_ask_for_it_there_is_no_such_aim():
    from core.agency.playing_as_it_happens import _Choosing

    choosing = object.__new__(_Choosing)
    choosing.mine, choosing.covered = _Thing(5, 5), None
    assert choosing._ground_not_yet_covered() is None
