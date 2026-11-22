"""A move after which a lost game went wrong is not played again from the same place (WhatFailedWhere).

LIVE 21 November 2026 she played the same opening against Stockfish in three games of six, nine moves alike in games 13
and 14, and lost each: the engine answers the same way every time, and so did she. The lesson from the Ghosts 'n Goblins
and Ninja Gaiden recordings (core/cognition/the_furthest_she_has_got.py): what was tried at a place and failed is not
worth trying there again.
"""
from __future__ import annotations

import asyncio

import core.agency.playing_pieces_by_kind as playing
from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.pieces_by_their_kind import FIRST, Position
from core.cognition.the_furthest_she_has_got import WhatFailedWhere
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME


def test_what_failed_where_is_kept_between_sittings() -> None:
    failed = WhatFailedWhere()
    failed.failed_at("a place", "a move")
    again = WhatFailedWhere.from_memory(failed.as_memory())
    assert again.already_failed_at("a place") == {"a move"}
    assert again.already_failed_at("another place") == frozenset()


def test_the_move_after_which_her_judgement_fell_furthest_is_ruled_out_where_she_made_it() -> None:
    start = dict(NOBODYS_GAME.start)
    game = playing._Game(learned=HowThePiecesMove.from_the_start(7, 7, start),
                         believed=Position(cells=start, turn=FIRST, unmoved=frozenset(start)))
    game.judged = [("p1", "m1", 80), ("p2", "m2", 60), ("p3", "m3", -340), ("p4", "m4", -400)]
    playing._where_it_went_wrong(game)
    assert game.went_wrong.ruled_out == {"p2": {"m2"}}


def _first_move(went_wrong: WhatFailedWhere, said: list[str], monkeypatch) -> str:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.5)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    app = BoardApplication(NOBODYS_GAME, seed=5)

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=2.0, say=said.append,
                                                     thinks_s=0.05, went_wrong=went_wrong)
        finally:
            stop.set()
            await answering

    came = asyncio.run(run())
    assert came["went_wrong"] is went_wrong
    return came["the_moves"][0]


def test_a_game_that_comes_back_to_the_place_goes_another_way(monkeypatch) -> None:
    said: list[str] = []
    first = _first_move(WhatFailedWhere(), said, monkeypatch)
    start = dict(playing.a_board_of_two_sides_at_its_start(BoardApplication(NOBODYS_GAME, seed=5).draw()).start)
    place = playing._a_place(Position(cells=start, turn=FIRST, unmoved=frozenset(start)))
    went_wrong = WhatFailedWhere()
    # Her first move as the board wrote it ("d2-d3"), as a move in her own squares.
    files = "abcdefghijklmnopqrstuvwxyz"
    a, b = first.replace("x", "-").split("-")
    went_wrong.failed_at(place, f"{int(a[1:]) - 1},{files.index(a[0])}-{int(b[1:]) - 1},{files.index(b[0])}")
    said_again: list[str] = []
    assert _first_move(went_wrong, said_again, monkeypatch) != first
    assert any("from here before and it went wrong" in line for line in said_again), said_again
