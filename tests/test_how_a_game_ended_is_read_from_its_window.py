"""Where the game stops letting her move and her own model cannot say why, the window's words say how it ended.

LIVE 21 November 2026 she said "The game won't let me move any more, so it's over" four times against Stockfish, with
En Croissant reading "0-1 Black wins" beside the board, and her account of one such game called it "a quiet, even
little battle that ended over before anyone broke through". She had lost.
"""
from __future__ import annotations

import asyncio

import pytest

import core.skills.screen_pursuit_a_way_on as way_on
import core.skills.screen_pursuit_board_of_pieces as board_of_pieces

#: The window as it read after one of those games.
AFTER_THE_GAME = ["Aura vs. Stockfish", "WHITE", "Aura", "BLACK", "Stockfish", "0-1", "Black wins", "New Game",
                  "Analyze"]


@pytest.fixture
def window(monkeypatch):
    async def read(_app):
        return {"ok": True, "layout": [{"text": text, "center_x": 0.5, "center_y": 0.1 * i}
                                       for i, text in enumerate(AFTER_THE_GAME)]}

    monkeypatch.setattr(way_on, "_its_own_pixels_read", read)
    asked: list[str] = []

    def answering(answer):
        async def her_model(ask, prompt, schema, most, *, matters):
            asked.append(prompt)
            return schema(ended=answer)

        monkeypatch.setattr("core.cognition.what_things_are.asked_patiently", her_model)

    return asked, answering


def test_the_window_says_she_lost(window) -> None:
    asked, answering = window
    answering("I lost")
    ended, words = asyncio.run(board_of_pieces.what_the_window_says_of_the_end("en-croissant"))
    assert ended == "lost" and "Black wins" in words
    assert "Black wins" in asked[0]


def test_a_window_that_does_not_say_leaves_the_ending_unknown(window) -> None:
    _asked, answering = window
    answering("the window does not say")
    ended, _words = asyncio.run(board_of_pieces.what_the_window_says_of_the_end("en-croissant"))
    assert ended == ""
