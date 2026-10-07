"""A world that stands still until she moves in it is played as it happens, and gone round where nothing calls her.

LIVE 2026-10-07, a top-down shooter: its room held still, it was taken for a
screen to be stepped through, each tap of an arrow moved its hero a few
pixels, which she took for nothing, and for three hours she clicked the words
on its scoreboard. A person presses a key for a moment and watches: what she
steers keeps going for as long as the key is down, where a menu's highlight
jumps once and is still.
"""
from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.playing_as_it_happens import _out_of_her_reach, _Run, goes_with, it_goes_while_held
from core.agency.which_one_answers_to_her import REALLY_MOVES, _Speeds

pytestmark = pytest.mark.unit

FPS = 20


class _World:
    """A room drawn at FPS pictures a second; ``held`` is the key down, and ``move`` what that does each picture."""

    def __init__(self, move):
        self.n, self.held, self.move = 0, "", move
        self.x, self.y, self.lit = 200.0, 180.0, 0

    async def look(self):
        at = self.n / FPS
        self.n += 1
        self.move(self, at)
        picture = np.full((360, 480, 3), (40, 30, 60), np.uint8)
        picture[300:360, :] = (90, 70, 50)
        x, y = int(self.x), int(self.y)
        picture[y - 12:y + 12, x - 12:x + 12] = (230, 140, 40)
        return picture, at

    async def down(self, key):
        self.held = key

    async def up(self, key):
        self.held = ""


def _walks(world, _at):
    if world.held == "right":
        world.x += 6.0


def _a_menu(world, _at):
    # A highlight that jumps to the next item when the key goes down, and stays there.
    if world.held == "right" and not world.lit:
        world.lit, world.x = 1, world.x + 80.0


def _sways(world, at):
    world.x = 200.0 + 3.0 * math.sin(at * 12.0)


def test_what_she_steers_goes_on_going_while_the_key_is_held():
    world = _World(_walks)
    assert asyncio.run(it_goes_while_held(world.look, world, "right"))
    assert world.held == "", "the key is let go"


def test_a_highlight_that_jumps_once_is_not_a_world_that_moves_under_her_hand():
    world = _World(_a_menu)
    assert not asyncio.run(it_goes_while_held(world.look, world, "right"))


def test_a_key_that_moves_nothing_and_a_figure_that_sways_are_not_either():
    assert not asyncio.run(it_goes_while_held(_World(lambda *_: None).look, _World(lambda *_: None), "right"))
    world = _World(_sways)
    assert not asyncio.run(it_goes_while_held(world.look, world, "right"))


def _thing(number, x, y, path, w=20.0, h=20.0, moved=True):
    return SimpleNamespace(number=number, x=x, y=y, w=w, h=h, path=path, moved=moved, kind=1)


def test_her_shadow_goes_with_her_and_another_player_does_not():
    hers = [(t / 10, 100.0 + 10 * t, 100.0) for t in range(8)]
    mine = _thing(1, *hers[-1][1:], hers)
    shadow = _thing(2, mine.x, mine.y + 15, [(at, x, y + 15) for at, x, y in hers])
    other = _thing(3, mine.x + 20, mine.y, [(at, 220.0 - 3 * i, 100.0) for i, (at, _x, _y) in enumerate(hers)])
    assert goes_with(shadow, mine)
    assert not goes_with(other, mine)


def test_where_she_holds_a_way_and_does_not_go_what_lies_that_way_is_out_of_her_reach():
    run = _Run(keys=["left", "right"], began=0.0, held="right", cell=20.0)
    mine = _thing(1, 100.0, 100.0, [])
    choosing = SimpleNamespace(mine=mine, ways={"right": (60.0, 0.0)}, moves=SimpleNamespace(shape=(200, 300)),
                               target=lambda: ((None, None), "wait", None))
    _out_of_her_reach(run, choosing, 0.0)
    _out_of_her_reach(run, choosing, 0.7)
    assert (6, 5) in run.barred and (14, 5) in run.barred, "the cells beyond the wall, to the edge"
    assert (4, 5) not in run.barred, "not the cells behind her"


def test_where_she_makes_for_a_place_and_gets_no_nearer_that_place_is_out_of_her_reach():
    run = _Run(keys=["left", "right"], began=0.0, cell=20.0)
    mine = _thing(1, 100.0, 100.0, [])
    choosing = SimpleNamespace(mine=mine, ways={}, moves=SimpleNamespace(shape=(200, 300)),
                               target=lambda: ((150.0, 30.0), "cover", None))
    for at in (0.0, 0.5, 1.0, 1.6):
        _out_of_her_reach(run, choosing, at)
    assert (7, 1) in run.barred


def test_standing_still_under_a_key_that_moved_her_is_a_wall_while_her_other_keys_still_move_her():
    speeds = _Speeds()
    for i in range(6):
        speeds.add("up", 0.0, -80.0, at=i * 0.1)
        speeds.add("left", -80.0, 0.0, at=i * 0.1)
    for i in range(20):
        speeds.add("up", 0.0, 0.0, at=0.6 + i * 0.05)
    assert math.hypot(*speeds.typical("up")) > REALLY_MOVES, "up still moves her: she is against a wall"
    for i in range(40):
        speeds.add("up", 0.0, 0.0, at=10.0 + i * 0.05)
    assert math.hypot(*speeds.typical("up")) <= REALLY_MOVES, "nothing has moved her for a while: up is not taken to be blocked"
