"""She expects the reply her search found best for the other side, and counts whether it came.

Her search's line says what it expects back: her move, then the reply it found best for them. Whether the reply came
as expected is counted, and said ("They played ..., as I expected."). Searching the expected position while they
thought was tried and taken out: over 80 games against Stockfish held to depth 4 it scored 0.36 where she scored 0.44
without it, and a test with a table of its own for each arm found it changed none of her moves (22 November 2026).
"""
from __future__ import annotations

from types import SimpleNamespace

import core.agency.playing_pieces_by_kind as playing
from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.pieces_by_their_kind import FIRST, Position, after, moves
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME


def _a_game() -> playing._Game:
    start = dict(NOBODYS_GAME.start)
    return playing._Game(learned=HowThePiecesMove.from_the_start(7, 7, start),
                         believed=Position(cells=start, turn=FIRST, unmoved=frozenset(start)))


def _her_move_and_their_reply(game: playing._Game):
    family = game.learned.family(unseen_go_every_way=True)
    mine = moves(family, game.believed)[0]
    there = after(family, game.believed, mine)
    reply, other = moves(family, there)[:2]
    return family, mine, reply, other


def test_she_expects_the_reply_her_search_found_and_counts_whether_it_came() -> None:
    game = _a_game()
    family, mine, reply, other = _her_move_and_their_reply(game)
    found = SimpleNamespace(line=[mine, reply], by="compiled")
    playing._expect_back(game, family, mine, found)
    assert game.expected is not None and game.expected[0] == reply
    assert game.expected[1] == after(family, after(family, game.believed, mine), reply)
    assert playing._was_expected(game, reply) is True
    playing._expect_back(game, family, mine, found)
    assert playing._was_expected(game, other) is False
    assert (game.foreseen, game.foreseeable) == (1, 2)
    assert game.expected is None


def test_nothing_is_expected_where_her_search_did_not_look_past_her_move() -> None:
    game = _a_game()
    family, mine, reply, _ = _her_move_and_their_reply(game)
    playing._expect_back(game, family, mine, SimpleNamespace(line=[mine], by="compiled"))
    assert game.expected is None
    playing._expect_back(game, family, mine, SimpleNamespace(line=[mine, reply], by="the only move"))
    assert game.expected is None
    assert playing._was_expected(game, reply) is False and game.foreseeable == 0
