"""Her search leaves alone a line whose verdict no margin can change, and still finds what a position offers.

Near the end of a look, a position already past what she needs by more than one of the least pieces for each ply left
is not looked into further; a quiet move late in the order, or one that even with such a margin cannot lift her to what
she needs, is not tried; a take that leaves her short even with what it takes and two of the least pieces is not
followed. The later a move comes in the order, the shorter the look it is given, and quiet moves tried and found wanting
are tried later next time. Measured 22 November 2026 against Stockfish held to depth 5 at 0.2 s a move, 40 games on
each of two openings' seeds: 0.487 and 0.463 with these, 0.388 and 0.263 without.

None of it knows a game. A margin is how far one quiet move moved her judgement at the position the search began
from, at most: a property of the weights she judges by. On her offline weights of 22 November 2026 it was 94, about
one pawn; on her live ones, which count each square a piece reaches nearly half a pawn, 357.
"""
from __future__ import annotations

from dataclasses import replace

from core.agency.searching_by_kind import Weights, described, first_weights, placed, search
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

#: A position with quiet moves to make.
MIDDLE = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -"


def test_a_mate_two_moves_away_is_still_found() -> None:
    # The front rook checks on the back row; taking it lets the second through.
    position = from_fen("r5k1/5ppp/8/8/8/8/3R1PPP/3R2K1 w - -")
    found = search(CHESS, position, seconds=0.5, weights=first_weights(CHESS))
    assert (found.move.start, found.move.end) == ((1, 3), (7, 3)), found
    assert found.mate


def test_a_piece_left_where_it_can_be_taken_is_taken() -> None:
    position = from_fen("4k3/8/8/3q4/8/4N3/8/4K3 w - -")
    found = search(CHESS, position, seconds=0.3, weights=first_weights(CHESS))
    assert (found.move.start, found.move.end) == ((2, 4), (4, 3)), found


def _margin(weights) -> int:
    import aura_board_search as engine

    position = from_fen(MIDDLE)
    return engine.search(described(CHESS), placed(position), weights.as_json(), 0.05)["margin"]


def test_the_margin_is_as_wide_as_a_quiet_move_moves_her_judgement() -> None:
    # Each square a piece reaches counted a tenth of the scale, then four tenths: the second moves with every move.
    plain = replace(first_weights(CHESS), mobility={kind: 10.0 for kind in CHESS.kinds})
    restless = replace(plain, mobility={kind: 40.0 for kind in CHESS.kinds})
    assert _margin(restless) > _margin(plain) > 0


def test_where_no_quiet_move_moves_her_judgement_there_are_no_margins_and_she_still_moves() -> None:
    weights = Weights(worth={kind: 0.0 for kind in CHESS.kinds})
    assert _margin(weights) == 0
    found = search(CHESS, from_fen(MIDDLE), seconds=0.2, weights=weights)
    assert found.move is not None
