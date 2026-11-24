"""A board that stops taking any move has refused none of them: the moves it did not take are not learned as no moves.

LIVE 24 November 2026, move 14 against Stockfish in En Croissant: the board stopped taking her moves. Her d2-d3, a
legal move, was learned at once as a move the pawn does not have ("The game didn't take that move; I've learned it
isn't one"), and every piece she picked up after it offered nothing. A move the board does not take is now held until
the board is seen taking another; then it was refused, and if the board takes nothing more, her hands did not reach it.
"""
from __future__ import annotations

import asyncio

import core.agency.playing_pieces_by_kind as playing
from core.agency.how_the_pieces_move import HowThePiecesMove
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME


class _StopsTaking(BoardApplication):
    """A board that takes her first moves and then takes nothing: its clicks reach nothing."""

    def __init__(self, *args, takes: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.takes = takes

    async def click(self, x_share: float, y_share: float) -> bool:
        if len([m for m in self.made if self.position.cells.get(m.end, (None,))[0] == 0]) >= self.takes:
            self.clicks += 1
            return True
        return await super().click(x_share, y_share)


def test_moves_a_board_did_not_take_before_it_took_nothing_are_not_learned_as_refused(monkeypatch) -> None:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.5)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    app = _StopsTaking(NOBODYS_GAME, seed=5, takes=2)
    refused: list = []
    monkeypatch.setattr(HowThePiecesMove, "refused",
                        lambda self, position, move, family=None: refused.append((position, move)))
    said: list[str] = []

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=6.0, say=said.append,
                                                     thinks_s=0.05)
        finally:
            stop.set()
            await answering

    came = asyncio.run(run())
    assert came["moves"] >= 1
    # Nothing the board did not take, once it had stopped taking anything, was learned as a refusal.
    assert refused == [], [move for _at, move in refused]
