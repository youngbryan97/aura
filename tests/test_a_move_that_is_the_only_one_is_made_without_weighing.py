"""A move that is the only one is made without weighing it (core/cognition/when_the_situation_decides.py).

A lesson from draughts, where a capture is compulsory: weighing a decision already made is thought spent looking like
diligence. On a board played against a clock it is also time: LIVE 21 November 2026 her clock ran out in four of seven
games against Stockfish.
"""
from __future__ import annotations

import asyncio

import core.agency.playing_pieces_by_kind as playing
import core.agency.searching_by_kind as searching
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME


def test_the_only_move_is_made_without_a_search(monkeypatch) -> None:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.5)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    real_moves, real_search = playing.moves, searching.search
    turns = {"seen": 0}
    searched_on: list[int] = []

    def one_move_on_her_first_turn(family, position):
        legal = real_moves(family, position)
        turns["seen"] += 1
        return legal[:1] if turns["seen"] == 1 else legal

    def search_spy(*args, **kwargs):
        searched_on.append(turns["seen"])
        return real_search(*args, **kwargs)

    monkeypatch.setattr(playing, "moves", one_move_on_her_first_turn)
    monkeypatch.setattr(searching, "search", search_spy)
    app = BoardApplication(NOBODYS_GAME, seed=5)
    said: list[str] = []

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=4.0, say=said.append,
                                                     thinks_s=0.1)
        finally:
            stop.set()
            await answering

    came = asyncio.run(run())
    assert came["moves"] >= 1
    first = next(line for line in said if line.startswith("I played"))
    assert first.endswith("It's the only move I have"), first
    assert 1 not in searched_on, "her first turn, with one move, was searched"
