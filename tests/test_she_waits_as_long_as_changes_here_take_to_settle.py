"""She waits for a board to stand still as long as its changes have been seen to take, not a fixed half second.

LIVE 21 November 2026, game 16 against Stockfish: seeing each of their moves stand still took 0.6 s of her clock on
every turn, two looks half a second apart, where En Croissant's moves had finished drawing before her first look.
"""
from __future__ import annotations

import asyncio
import time

import core.agency.playing_pieces_by_kind as playing
from core.cognition.how_long_a_change_takes_to_settle import REMEMBERED, HowLongItSettles
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME


def test_the_wait_is_the_longest_change_watched_of_late() -> None:
    settles = HowLongItSettles(0.5)
    assert settles.wait() == 0.5
    for _ in range(REMEMBERED):
        settles.watched(0.0)
    assert settles.wait() == 0.0
    settles.watched(0.12)
    assert settles.wait() == 0.12


def test_a_reading_caught_mid_change_brings_back_the_first_wait() -> None:
    settles = HowLongItSettles(0.5)
    for _ in range(REMEMBERED):
        settles.watched(0.0)
    settles.too_soon()
    assert settles.wait() == 0.5


def test_a_board_whose_changes_finish_at_once_is_believed_after_two_quick_looks(monkeypatch) -> None:
    app = BoardApplication(NOBODYS_GAME, seed=5)
    seat = playing.a_board_of_two_sides_at_its_start(app.draw())
    assert seat is not None
    frames = iter([1, 2, 2])

    async def look():
        # A board drawn in two steps: what a picture caught during the first step shows less than the second.
        picture = app.draw()
        if next(frames, 2) == 1:
            picture = picture.copy()
            picture[: picture.shape[0] // 2] = picture[0, 0]
        return picture

    async def watch() -> float:
        began = time.monotonic()
        still = await playing._seen_still(seat, look)
        assert still is not None
        return time.monotonic() - began

    for _ in range(REMEMBERED):
        seat.settles.watched(0.0)
    assert asyncio.run(watch()) < 0.25
