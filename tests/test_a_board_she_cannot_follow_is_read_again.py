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


def test_a_piece_whose_look_is_unread_keeps_the_side_she_holds_it_to_be() -> None:
    """LIVE 21 November 2026 a glow on her king under attack read it as a piece of the other side, of no kind."""
    believed = {(0, 4): (FIRST, "4"), (7, 0): (SECOND, "0")}
    shown = {(0, 4): (SECOND, None), (7, 0): (SECOND, "0"), (2, 2): (SECOND, None)}
    read = playing._read_against(shown, believed)
    assert read[(0, 4)] == (FIRST, None)
    assert read[(2, 2)] == (SECOND, None), "a piece on a square she held empty keeps the side it was read as"


def test_an_ending_her_model_infers_is_asked_of_the_game(monkeypatch) -> None:
    """Her model finds no move once; the game offers her moves, so the game goes on."""
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.5)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    real_moves = playing.moves
    calls = {"n": 0}

    def no_move_the_first_time(family, position):
        calls["n"] += 1
        return [] if calls["n"] == 1 else real_moves(family, position)

    monkeypatch.setattr(playing, "moves", no_move_the_first_time)
    app = BoardApplication(NOBODYS_GAME, seed=5)
    said: list[str] = []

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=8.0, say=said.append,
                                                     thinks_s=0.1)
        finally:
            stop.set()
            await answering

    came = asyncio.run(run())
    # The game may still end for real inside the run; what matters is that the ending her model inferred first did not.
    moved = [i for i, line in enumerate(said) if line.startswith("I'll move")]
    ended = [i for i, line in enumerate(said) if "no move left" in line or "won't let me move" in line]
    assert came["moves"] >= 2 and len(moved) >= 2, said[-3:]
    assert not ended or ended[0] > moved[1], said[: (ended[0] + 1) if ended else None][-3:]


def test_an_unread_look_where_nothing_stood_is_set_aside_where_the_board_is_explained_without_it() -> None:
    """LIVE 22 November 2026, game 19: a look of nothing read on h1, beside her king in check, left the board
    unexplained; she went by the board with the phantom on it."""
    believed = {(0, 4): (FIRST, "4"), (7, 0): (SECOND, "0")}
    shown = {(0, 4): (FIRST, "4"), (7, 0): (SECOND, "0"), (0, 7): (FIRST, None)}
    assert playing._without_marks_where_nothing_stood(shown, believed) == {(0, 4): (FIRST, "4"), (7, 0): (SECOND, "0")}
    # A look she cannot read on a square she held a piece on stays: the piece is there, its look changed.
    marked = {(0, 4): (FIRST, None), (7, 0): (SECOND, "0")}
    assert playing._without_marks_where_nothing_stood(marked, believed) == marked


class _ABoardThatShowsAMarkBesideAMove(BoardApplication):
    """After each move of hers, a square she holds empty is drawn with a look of nothing: a mark, not a piece."""

    def draw(self):
        import numpy as np

        picture = np.array(super().draw(), copy=True)
        if len(self.made) >= 2:
            empty = next((sq for sq in sorted((r, c) for r in range(self.family.rows) for c in range(self.family.columns))
                          if sq not in self.position.cells and sq[0] == 3), None)
            if empty is not None:
                x, y = self._centre(empty)
                s = self.square // 3
                picture[int(y - s):int(y + s), int(x - s):int(x + s)] = (40, 200, 220)
        return picture


def _played(app, monkeypatch, caplog, seconds: float = 20.0) -> tuple[dict, list[tuple[float, str]]]:
    import logging
    import time

    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 1.0)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    said: list[tuple[float, str]] = []

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=seconds,
                                                     say=lambda line: said.append((time.time(), line)), thinks_s=0.1)
        finally:
            stop.set()
            await answering

    with caplog.at_level(logging.INFO, logger="Aura.PlayingPiecesByKind"):
        came = asyncio.run(run())
    return came, said


def test_a_board_read_unexplained_the_same_twice_is_gone_by_at_once(monkeypatch, caplog) -> None:
    """LIVE 22 November 2026, game 19: twenty looks before she went by the board took 24 s of a turn whose share was
    0.6 s. A board read the same twice, each reading itself of a board standing still, has settled."""
    _came, said = _played(_ABoardThatChangesUnseen(NOBODYS_GAME, seed=5), monkeypatch, caplog)
    first = next(r.created for r in caplog.records if r.getMessage().startswith("the board shows what no move"))
    lost = next(t for t, line in said if line.startswith("I've lost the thread"))
    # Twenty readings no move explains, each two pictures and a rest, would take seconds; two are enough.
    assert lost - first < 2.0, lost - first


def test_a_mark_on_a_square_she_held_empty_does_not_lose_her_the_thread(monkeypatch, caplog) -> None:
    app = _ABoardThatShowsAMarkBesideAMove(NOBODYS_GAME, seed=5)
    came, said = _played(app, monkeypatch, caplog, seconds=6.0)
    assert came["moves"] >= 2
    assert not any(line.startswith("I've lost the thread") for _t, line in said), [line for _t, line in said][-6:]
