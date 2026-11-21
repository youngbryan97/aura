"""A board that changes in a way no move she can read explains is read again as it stands; the game goes on.

LIVE 20 November 2026 against Stockfish in En Croissant she said "I've lost track of the board; I'll stop here" 22
seconds after her tenth move, on a board she could read square by square, and her clock ran out seven seconds later.
"""
from __future__ import annotations

import asyncio

import core.agency.playing_pieces_by_kind as playing
from core.agency.pieces_by_their_kind import FIRST, SECOND, Position
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME


class _ABoardThatChangesUnseen(BoardApplication):
    """After her second move, one of her pieces is shown as the other side's: no move makes that."""

    def make(self, move) -> None:
        super().make(move)
        if len(self.made) == 4 and self.position.turn == FIRST:
            square = next(sq for sq, (side, _k) in sorted(self.position.cells.items()) if side == FIRST)
            cells = dict(self.position.cells)
            cells[square] = (SECOND, cells[square][1])
            self.position = Position(cells=cells, turn=FIRST, unmoved=self.position.unmoved - {square})


def test_she_goes_by_the_board_as_it_stands_and_plays_on(monkeypatch) -> None:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 1.0)
    monkeypatch.setattr(playing, "MISREAD_AT_MOST", 3)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    app = _ABoardThatChangesUnseen(NOBODYS_GAME, seed=5)
    said: list[str] = []

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        assert seat is not None
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=20.0, say=said.append,
                                                     thinks_s=0.1)
        finally:
            stop.set()
            await answering

    came = asyncio.run(run())
    assert came["ended"] != "lost track of the board", said[-4:]
    assert "I've lost the thread of the moves, so I'll go by the board as it stands." in said
    assert came["moves"] >= 4, f"she did not play on after reading the board again: {said[-4:]}"
    # Her move was made though the board showed more than she could read: no rule is learned from a refusal that
    # never happened.
    assert not any("didn't take that move" in line for line in said), [line for line in said if "didn't" in line]
    # A kind she has not seen move is still said in English when it is someone's.
    assert not any(" my a " in line or " their a " in line for line in said)
