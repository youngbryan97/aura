"""Where the game shows her clock, her whole turn is held to its share of it: picking pieces up as well as thinking.

LIVE 20 November 2026 in a three-minute game against Stockfish only her search was held to the share. Picking up six
pieces a turn and seeing each move shown were not, and she had spent 2:36 of her three minutes by her tenth move.
"""
from __future__ import annotations

import asyncio

import core.agency.playing_pieces_by_kind as playing
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME


class _SlowHands(BoardApplication):
    """Each click takes a tenth of a second, and the clicks made before her first move are counted."""

    clicks_before_her_first_move: int | None = None

    async def click(self, x_share: float, y_share: float) -> bool:
        await asyncio.sleep(0.1)
        return await super().click(x_share, y_share)

    def make(self, move) -> None:
        if self.clicks_before_her_first_move is None:
            self.clicks_before_her_first_move = self.clicks
        super().make(move)


def _first_move(monkeypatch, clock) -> _SlowHands:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.2)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    monkeypatch.setattr(playing._Rehearsing, "moves_left", lambda self, made: 10)
    app = _SlowHands(NOBODYS_GAME, seed=5)

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        assert seat is not None
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=4.0, thinks_s=0.1,
                                                     clock=clock)
        finally:
            stop.set()
            await answering

    asyncio.run(run())
    assert app.clicks_before_her_first_move is not None, "she never moved"
    return app


def test_with_little_on_her_clock_she_picks_up_few_pieces_before_she_moves(monkeypatch) -> None:
    async def four_seconds_left() -> float:
        return 4.0

    app = _first_move(monkeypatch, four_seconds_left)
    # A tenth of her four seconds for the turn: one piece picked up (two clicks), perhaps the one she moves picked up
    # first (two more), and the move (two more).
    assert app.clicks_before_her_first_move <= 6, app.clicks_before_her_first_move


def test_with_no_clock_she_picks_up_what_she_needs_to(monkeypatch) -> None:
    app = _first_move(monkeypatch, None)
    assert app.clicks_before_her_first_move > 6, app.clicks_before_her_first_move
