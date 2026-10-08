"""A wait on what plays by itself that is cut off counts as watched, so the next look does not wait again.

LIVE 2026-10-08 a game's title never stopped moving; each eight-second look
began by waiting on it, was cut off before the wait ended, and three empty
looks ended the game before she had pressed anything.
"""
from __future__ import annotations

import asyncio

import pytest

from core.perception.watching_it_happen import Watching
from core.skills.screen_pursuit_on_a_page import OnAPage


@pytest.mark.unit
def test_a_cut_off_wait_is_not_waited_again():
    async def go():
        watching = Watching(take=None)
        watching.playing_on_its_own = lambda *_: True  # a title that never stops moving
        try:
            await asyncio.wait_for(watching.watch(at_most_s=30.0), timeout=0.6)
        except TimeoutError:
            pass
        return await asyncio.wait_for(watching.watch(at_most_s=30.0), timeout=0.6)

    assert asyncio.run(go()) == 0.0


@pytest.mark.unit
def test_a_wait_inside_a_look_is_shorter_than_a_look():
    assert OnAPage.WATCH_IN_A_LOOK_S < 8.0
