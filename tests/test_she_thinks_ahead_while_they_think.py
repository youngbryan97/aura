"""While the other side thinks, she thinks on the position she expects their answer to leave her.

Her search's line says what it expects back: her move, then the reply it found best for them. Waiting is time she
would otherwise spend watching, so she spends it searching the position that reply would leave her, for about as long
as their answers take. What she finds stays in her search's table: measured 22 November 2026, a 0.3 s search begun
after 2 s of it reached 12 moves deep where one begun cold reached 9. Whether the reply came as expected is counted, and
said.
"""
from __future__ import annotations

import time
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


def test_she_thinks_on_the_expected_position_for_about_as_long_as_their_answers_take() -> None:
    game = _a_game()
    family, mine, reply, _ = _her_move_and_their_reply(game)
    playing._expect_back(game, family, mine, SimpleNamespace(line=[mine, reply], by="compiled"))
    game.answers_s = [1.0, 2.0, 30.0]
    asked: list = []

    def search(family_, position, *, seconds, weights, earlier, threads):
        asked.append((position, seconds, threads))
        return SimpleNamespace(move=None)

    rehearsing = SimpleNamespace(weights_for=lambda family_: None)
    playing._think_ahead(game, rehearsing, search)
    playing._think_ahead(game, rehearsing, search)  # once a turn of theirs
    game.thinking_ahead.result(timeout=5)
    assert len(asked) == 1
    position, seconds, threads = asked[0]
    assert position == game.expected[1]
    assert seconds == 2.0  # the middling answer, not the one long one
    assert threads >= 1
    # Their answer lets the thinking go; the next turn of theirs thinks again.
    playing._was_expected(game, reply)
    assert game.thinking_ahead is None


def test_the_longest_she_thinks_ahead_is_bounded() -> None:
    game = _a_game()
    family, mine, reply, _ = _her_move_and_their_reply(game)
    playing._expect_back(game, family, mine, SimpleNamespace(line=[mine, reply], by="compiled"))
    game.answers_s = [500.0]
    asked: list = []
    playing._think_ahead(game, SimpleNamespace(weights_for=lambda f: None),
                         lambda *a, seconds, **k: asked.append(seconds))
    deadline = time.monotonic() + 5
    while not asked and time.monotonic() < deadline:
        time.sleep(0.01)
    assert asked == [playing.AHEAD_AT_MOST_S]
