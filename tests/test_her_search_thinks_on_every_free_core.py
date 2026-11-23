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
    monkeypatch.setattr(playing, "_her_own_cores", lambda: 0.0)
    monkeypatch.setattr(os, "cpu_count", lambda: 18)
    monkeypatch.setattr(os, "getloadavg", lambda: (5.4, 5.0, 4.0))
    assert playing._cores_free() == 12
    monkeypatch.setattr(os, "getloadavg", lambda: (40.0, 30.0, 20.0))
    assert playing._cores_free() == 1


def test_her_own_searchers_are_not_someone_else_using_the_machine(monkeypatch) -> None:
    """LIVE 23 November 2026, game 26: a turn on fifteen threads looked at 9.4 million positions, and the load those
    threads left was taken for someone else's, so three turns later she searched on one thread and saw 0.6 million."""
    import time

    clock = {"wall": 100.0, "used": 50.0}
    monkeypatch.setattr(playing, "monotonic_time", lambda: clock["wall"])
    monkeypatch.setattr(time, "process_time", lambda: clock["used"])
    monkeypatch.setattr(playing, "_HER_OWN", {"used": None, "at": None, "cores": 0.0})
    monkeypatch.setattr(os, "cpu_count", lambda: 18)
    monkeypatch.setattr(os, "getloadavg", lambda: (1.0, 1.0, 1.0))
    assert playing._cores_free() == 17
    # A minute of her own searching on fifteen cores, beside someone else's four: the machine's load comes up to it as
    # its one-minute average does, five seconds at a time.
    import math

    load = 1.0
    for _ in range(12):
        clock["wall"] += 5.0
        clock["used"] += 5.0 * 15
        load = load * math.exp(-5.0 / 60.0) + 19.0 * (1 - math.exp(-5.0 / 60.0))
        playing._her_own_cores()
    monkeypatch.setattr(os, "getloadavg", lambda: (load, 12.0, 6.0))
    # Counted as someone else's, her own load would leave her five cores; it leaves her what the other four do not use.
    assert int(18 - load) <= 6
    assert playing._cores_free() >= 14
