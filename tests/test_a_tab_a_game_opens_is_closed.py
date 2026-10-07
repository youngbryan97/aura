"""A tab a game opens on its own while she plays it is closed, and her game brought back in front.

LIVE 2026-10-07 a blank tab stood in front of the game she was playing, and
whoever was watching saw nothing.
"""
from __future__ import annotations

import asyncio

import pytest

from core.skills import sovereign_browser_drawing as drawing

pytestmark = pytest.mark.unit


class _Tab:
    def __init__(self):
        self.closed = self.in_front = False

    async def close(self):
        self.closed = True

    async def bring_to_front(self):
        self.in_front = True


def test_the_tab_is_closed_and_the_game_is_in_front_again(monkeypatch):
    said = []
    monkeypatch.setattr(drawing, "_tell", said.append)
    game, advert, another = _Tab(), _Tab(), _Tab()
    closing = drawing._Closing(game)

    async def play():
        closing.opened(advert)
        closing.opened(another)
        await asyncio.sleep(0.01)

    asyncio.run(play())
    assert advert.closed and another.closed and game.in_front
    assert said == ["The game opened another tab of its own; I closed it and went back to the game."]
