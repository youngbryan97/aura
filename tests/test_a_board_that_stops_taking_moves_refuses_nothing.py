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
from core.agency.pieces_by_their_kind import FIRST
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME


class _StopsTaking(BoardApplication):
    """A board that takes her first moves and then takes nothing: its clicks reach nothing."""

    def __init__(self, *args, takes: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.takes = takes
        self.accepted = 0

    def make(self, move) -> None:
        self.accepted += self.position.turn == FIRST
        super().make(move)

    async def click(self, x_share: float, y_share: float) -> bool:
        if self.accepted >= self.takes:
            self.clicks += 1
            return True
        return await super().click(x_share, y_share)


def test_moves_a_board_did_not_take_before_it_took_nothing_are_not_learned_as_refused(monkeypatch) -> None:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.5)
    monkeypatch.setattr(playing, "_the_most_a_look_may_take", lambda *_args: 0.25)
    monkeypatch.setattr(playing, "MOVE_SHOWS_S", 0.2)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    app = _StopsTaking(NOBODYS_GAME, seed=5, takes=2)
    refused: list = []
    monkeypatch.setattr(HowThePiecesMove, "refused",
                        lambda self, position, move, family=None: refused.append((position, move)))
    said: list[str] = []
    ending_reads: list[bool] = []

    async def read_end() -> tuple[str, str]:
        ending_reads.append(True)
        return "", ""

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=6.0, say=said.append,
                                                     thinks_s=0.05, read_end=read_end)
        finally:
            stop.set()
            await answering

    came = asyncio.run(run())
    assert came["moves"] >= 1
    assert ending_reads == [True]
    assert came["ended"] == "result unconfirmed"
    assert sum(line.startswith("I played") for line in said) == came["moves"]
    assert all(not line.startswith("I'll move") for line in said)
    # Nothing the board did not take, once it had stopped taking anything, was learned as a refusal.
    assert refused == [], [move for _at, move in refused]


class _NotYetBegun(BoardApplication):
    """A board shown before its game is begun: it draws the pieces and takes nothing."""

    async def click(self, x_share: float, y_share: float) -> bool:
        self.clicks += 1
        return True


def test_pieces_picked_up_on_a_board_not_yet_begun_are_not_learned_as_going_nowhere(monkeypatch) -> None:
    """LIVE 24 November 2026 her first looks were at the setup page of a game, whose board takes nothing until the game
    is begun; every piece she picked up there was learned as going nowhere."""
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.5)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    learned_nothing: list = []
    original = HowThePiecesMove.offered

    def offered(self, position, square, squares, family=None):
        if not squares:
            learned_nothing.append(square)
        return original(self, position, square, squares, family=family)

    monkeypatch.setattr(HowThePiecesMove, "offered", offered)
    app = _NotYetBegun(NOBODYS_GAME, seed=5)

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=3.0, say=lambda _line: None,
                                                 thinks_s=0.05)

    came = asyncio.run(run())
    assert came["ended"] == "not taking moves"
    assert learned_nothing == []


def test_a_mate_her_rules_read_and_the_game_offers_nothing_against_is_how_the_game_ended() -> None:
    """LIVE 25 November 2026 she was mated by a knight against Stockfish in En Croissant. Her rules read the end, the
    game offered her nothing, and the window said nothing in words; she said "I couldn't get another move to register
    or verify a result" and picked the same board up three times more. Where her rules read the end and the game
    offers nothing, the two agree, and that is how it ended. A board that only stops taking moves, with no end her
    rules read, is still not a result."""
    from core.agency.pieces_by_their_kind import Position
    from core.agency.playing_pieces_by_kind import _Game, _the_end

    learned = HowThePiecesMove.from_the_start(8, 8, dict(NOBODYS_GAME.start))
    said: list[str] = []
    game = _Game(learned=learned, believed=Position(cells=dict(NOBODYS_GAME.start), turn=FIRST))
    came = _the_end("won:1", game, said.append, observed=("", ""))
    assert came["ended"] == "lost"
    assert said[-1].startswith("The game offers me no move, and by its rules as I know them I lost")
    said.clear()
    came = _the_end("no move", game, said.append, observed=("", ""))
    assert came["ended"] == "result unconfirmed"
