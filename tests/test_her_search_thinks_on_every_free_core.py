"""Her search thinks on the cores nothing else is using, its searchers sharing one table (rust_extensions/aura_board_search).

LIVE 21 November 2026 her search ran on one core of eighteen, and four times the time to think took her from 0.396 to
0.781 against a searcher held to depth 4. Searchers in threads of their own, over the same position, each find what the
others can use, because what any of them finds goes into the one table they share.
"""
from __future__ import annotations

import os

import core.agency.playing_pieces_by_kind as playing
from core.agency.pieces_by_their_kind import moves
from core.agency.searching_by_kind import first_weights, search
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

#: A position with something to look at.
MIDDLE = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -"


def test_searchers_in_threads_find_a_legal_move_and_look_at_more() -> None:
    position = from_fen(MIDDLE)
    legal = moves(CHESS, position)
    alone = search(CHESS, position, seconds=0.5, weights=first_weights(CHESS))
    together = search(CHESS, position, seconds=0.5, weights=first_weights(CHESS), threads=4)
    assert together.move in legal and together.line and together.line[0] == together.move
    assert together.nodes > alone.nodes


def test_a_move_set_aside_is_set_aside_by_every_searcher() -> None:
    position = from_fen(MIDDLE)
    first = search(CHESS, position, seconds=0.3, weights=first_weights(CHESS), threads=4).move
    other = search(CHESS, position, seconds=0.3, weights=first_weights(CHESS), threads=4, set_aside=[first]).move
    assert other is not None and (other.start, other.end) != (first.start, first.end)


def test_she_thinks_on_the_cores_nothing_else_is_using(monkeypatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 18)
    monkeypatch.setattr(os, "getloadavg", lambda: (5.4, 5.0, 4.0))
    assert playing._cores_free() == 12
    monkeypatch.setattr(os, "getloadavg", lambda: (40.0, 30.0, 20.0))
    assert playing._cores_free() == 1
