"""A board that does not stand still for a while after her move is watched on, and the game goes on when it settles
(core/agency/playing_pieces_by_kind.py).

LIVE 25 November 2026, against Stockfish in En Croissant: ten seconds after her fourteenth move the board had not
stood still, the game was given up as "the board went out of sight", and the keys pressed to find it again took the
board back to its start.
"""
from __future__ import annotations

import asyncio
import time

import pytest

import core.agency.playing_pieces_by_kind as playing
from core.agency.pieces_by_their_kind import FIRST
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME

#: How long the board flickers after her first move, against how long one wait for it to stand still runs.
FLICKERS_S, SETTLES_WITHIN_S = 4.0, 1.0


class _ABoardThatFlickers(BoardApplication):
    """After her first move, one empty square of the board changes on every look for a while: a mark that comes and
    goes. Then the board stands still again."""

    def __init__(self, *args, drops_frames: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.drops_frames = drops_frames
        self.flicker_until = 0.0
        self.flickers = 0

    def make(self, move) -> None:
        super().make(move)
        if len(self.made) == 1:
            self.flicker_until = time.monotonic() + FLICKERS_S

    async def look(self):
        picture = await super().look()
        if time.monotonic() < self.flicker_until:
            self.flickers += 1
            if self.drops_frames:
                return None
            empty = next((r, c) for r in range(self.family.rows) for c in range(self.family.columns)
                         if (r, c) not in self.position.cells)
            x, y = self._centre_of(empty)
            half = self.square // 3
            picture = picture.copy()
            picture[int(y) - half:int(y) + half, int(x) - half:int(x) + half] = 255 * (self.flickers % 2)
        return picture

    def _centre_of(self, square):
        row, column = square
        return (self.margin + column * self.square + self.square / 2,
                self.margin + (self.family.rows - 1 - row) * self.square + self.square / 2)


@pytest.mark.parametrize("drops_frames", [False, True])
def test_a_board_temporarily_unreadable_is_watched_on_and_the_game_goes_on(monkeypatch, drops_frames) -> None:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 1.0)
    monkeypatch.setattr(playing, "SETTLES_WITHIN_S", SETTLES_WITHIN_S)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    app = _ABoardThatFlickers(NOBODYS_GAME, seed=5, drops_frames=drops_frames)
    said: list[str] = []

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        assert seat is not None
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=12.0, say=said.append,
                                                     thinks_s=0.1)
        finally:
            stop.set()
            await answering

    came = asyncio.run(run())
    assert app.flickers > 0 and app.position.turn in (FIRST, 1 - FIRST)
    assert came["ended"] != "the board went out of sight", said[-3:]
    assert came["moves"] >= 3, (came.get("ended"), came.get("moves"), said[-4:])
