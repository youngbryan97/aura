"""Her search leaves alone a line whose verdict no margin can change, and still finds what a position offers.

Near the end of a look, a position already past what she needs by more than one of the least pieces for each ply left
is not looked into further; a quiet move late in the order, or one that even with such a margin cannot lift her to what
she needs, is not tried; a take that leaves her short even with what it takes and two of the least pieces is not
followed. The later a move comes in the order, the shorter the look it is given, and quiet moves tried and found wanting
are tried later next time. Measured 22 November 2026 against Stockfish held to depth 5 at 0.2 s a move, 40 games on
each of two openings' seeds: 0.487 and 0.463 with these, 0.388 and 0.263 without.

None of it knows a game: the margins are counted in the least worth her own rehearsal has given any kind.
"""
from __future__ import annotations

from core.agency.searching_by_kind import first_weights, search
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen


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


def test_with_no_kind_worth_anything_yet_there_are_no_margins_and_she_still_moves() -> None:
    position = from_fen("4k3/8/8/3q4/8/4N3/8/4K3 w - -")
    weights = first_weights(CHESS)
    weights.worth = {kind: 0.0 for kind in weights.worth}
    found = search(CHESS, position, seconds=0.2, weights=weights)
    assert found.move is not None
