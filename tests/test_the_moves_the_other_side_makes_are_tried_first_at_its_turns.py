"""The moves the other side has been seen to make are tried early at its turns in her search.

Whoever she plays has habits: the same piece brought out the same way, the same answer to the same threat. Her model of
what they answered where (core/agency/thinking_on_their_time.py) is keyed by whole positions, so it helps only where a
position comes back. The moves they made, by their squares, are now kept through a game and handed to her search with
the share of their moves each was, and the search counts them as history at their turns, so they are looked at early
(habit expected_weight). It models the other side; her own moves are not taken from theirs.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from core.agency.searching_by_kind import first_weights, placed, search
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

E2, E4 = (1, 4), (3, 4)


def test_the_moves_they_were_seen_to_make_reach_the_compiled_search_with_their_shares() -> None:
    sent = json.loads(placed(CHESS.the_start(), expected={(E2, E4): 0.75, ((0, 6), (2, 5)): 0.25}))
    assert sent["expected"] == [[0, 6, 2, 5, 0.25], [1, 4, 3, 4, 0.75]]
    assert "expected" not in json.loads(placed(CHESS.the_start()))


def test_their_moves_change_her_search_only_where_her_habits_weigh_them() -> None:
    pytest.importorskip("aura_board_search")
    after_e4 = CHESS.the_start()
    seen = {((6, 3), (4, 3)): 0.5, ((7, 6), (5, 5)): 0.5}
    plain = first_weights(CHESS)
    weighing = replace(plain, habits={"expected_weight": 50_000.0})

    def lines(weights, expected):
        found = search(CHESS, after_e4, seconds=30.0, nodes=60_000, weights=weights, expected=expected)
        return found.nodes, found.score, [(m.start, m.end) for m in found.line]

    # Unweighed, what they were seen to make changes nothing.
    assert lines(plain, seen) == lines(plain, None)
    # Weighed, it changes the order her search looks in.
    assert lines(weighing, seen) != lines(weighing, None)
