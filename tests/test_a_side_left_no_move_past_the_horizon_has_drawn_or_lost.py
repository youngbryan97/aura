"""Past the horizon of her compiled search, a side left no move has drawn or lost by the rules, as in the search.

From the suggestions of 23 November 2026: a move that left the other side no move, and not attacked, was scored where
the look past the horizon stood, at +569, a queen up, where her rules of chess have it drawn. The look past the horizon
stood on how the position was judged, and never asked whether the side to move had a move at all. It now asks, at the
horizon, where none of the moves it made was one the rules allow (habit ends_past_the_horizon; measured 23 November 2026
in her model of chess, 160 games from each of two sets of openings at equal positions searched, 0.512 +- 0.035 and
0.500 +- 0.034 against the look as it was, and no slower).
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from core.agency.pieces_by_their_kind import moves
from core.agency.searching_by_kind import described, first_weights, placed
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

engine = pytest.importorskip("aura_board_search")

#: Their royal on h8, hers on f7, her strongest piece on g5: to g6, it leaves them no move and not attacked.
BOXED_IN = from_fen("7k/5K2/8/6Q1/8/8/8/8 w - -")
TO_G6 = ((4, 6), (5, 6))


def _scored(family, *, ends: float) -> int:
    weights = json.loads(first_weights(family).as_json())
    weights["habits"] = {**(weights.get("habits") or {}), "ends_past_the_horizon": ends}
    others = [(m.start[0], m.start[1], m.end[0], m.end[1], m.becomes) for m in moves(family, BOXED_IN)
              if (m.start, m.end) != TO_G6]
    found = engine.search(described(family), placed(BOXED_IN), json.dumps(weights), 5.0, 1, 16, 0, others)
    assert tuple(found["move"][:4]) == (*TO_G6[0], *TO_G6[1])
    return int(found["score"])


def test_a_side_left_no_move_and_not_attacked_has_drawn_where_the_rules_draw_it() -> None:
    assert _scored(CHESS, ends=1.0) == 0
    assert _scored(CHESS, ends=0.0) > 300, "judged as it stood, as before"


def test_a_side_left_no_move_has_lost_where_the_rules_have_it_lose() -> None:
    losing = replace(CHESS, no_move_unattacked="loses")
    assert _scored(losing, ends=1.0) > 900_000


def test_the_position_the_review_found_scores_its_stalemate_as_a_draw() -> None:
    # From k7/8/1QK5/8/8/8/8/8 w - -, her king to d7 leaves theirs no move and not attacked: +569 at depth one, as
    # the review of 23 November 2026 found it.
    review = from_fen("k7/8/1QK5/8/8/8/8/8 w - -")
    others = [(m.start[0], m.start[1], m.end[0], m.end[1], m.becomes) for m in moves(CHESS, review)
              if (m.start, m.end) != ((5, 2), (6, 3))]
    found = engine.search(described(CHESS), placed(review), first_weights(CHESS).as_json(), 5.0, 1, 16, 0, others)
    assert tuple(found["move"][:4]) == (5, 2, 6, 3) and int(found["score"]) == 0
