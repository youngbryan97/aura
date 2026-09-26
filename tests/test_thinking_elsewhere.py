"""Her look-ahead in a process of its own chooses what it would choose here, and the rest of her cannot slow it.

On her own clock the search reached 3.40 moves ahead alone and 2.56 beside six
busy threads in the same process; in its own process, 3.32 beside the same six.
"""

from __future__ import annotations

import random

import pytest

from core.agency import thinking_elsewhere
from core.agency.looking_ahead import look_ahead
from core.perception.how_it_moves import HowItMoves
from core.perception.what_the_world_does import WhatTheWorldDoes
from tools.measure_getting_there import WORLDS


def _learned():
    world = WORLDS["four by four"]
    roll = random.Random(3)
    knows, arrivals = HowItMoves(), WhatTheWorldDoes()
    state, names, boards = world.start(roll), list(world.acts), []
    for step in range(300):
        move = roll.choice(names)
        after = world.act(state, move)
        knows.watched(state, move, after)
        if after.as_text() != state.as_text():
            after = world.something_turns_up(after, roll)
            arrivals.watched(knows.expect(state, move), after)
        state = after if not world.over(after) else world.start(roll)
        if step % 30 == 0:
            boards.append(state)
    return knows, arrivals, names, boards


@pytest.mark.asyncio
async def test_the_child_chooses_what_she_would_choose_here():
    knows, arrivals, names, boards = _learned()
    for board in boards[:6]:
        thought = await thinking_elsewhere.think_it_through(
            knows, board, names, toward="2048", approach="", budget_s=1.0,
            world=arrivals, weights=None, depth=2,
        )
        assert thought is not None
        there, saw = thought
        here = look_ahead(knows, board, names, toward="2048", budget_s=1.0, world=arrivals, depth=2)
        assert saw == 2
        assert max(there, key=lambda act: there[act][0]) == max(here, key=lambda act: here[act][0])


@pytest.mark.asyncio
async def test_without_a_child_she_thinks_here(monkeypatch):
    knows, arrivals, names, boards = _learned()
    monkeypatch.setattr(thinking_elsewhere, "_the_child", lambda: None)
    assert await thinking_elsewhere.think_it_through(
        knows, boards[0], names, toward="2048", approach="", budget_s=0.1, world=arrivals, weights=None,
    ) is None
