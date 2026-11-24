"""A move her rules allow on few sightings of the kind going that way is looked at before it is made.

LIVE 23 November 2026: one misread look gave her royal kind two squares far up the board, and her rules had it ride
the whole file. Preparing after, she worked out a move of it from e1 to e4. Her king is picked up seldom, so what the
game would have shown against the ride was not seen.
"""
from __future__ import annotations

from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.pieces_by_their_kind import Move
from core.agency.playing_pieces_by_kind import SEEN_TO_BELIEVE
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

ALONE = from_fen("4k3/8/8/8/8/8/8/4K3 w - -")
E1 = (0, 4)
AROUND = {(1, 4), (1, 3), (1, 5), (0, 3), (0, 5)}


def test_far_squares_read_in_one_look_rest_on_too_little_and_a_step_seen_often_does_not() -> None:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    for _ in range(3):
        rules.offered(ALONE, E1, AROUND, family=CHESS)
    rules.offered(ALONE, E1, AROUND | {(5, 4), (6, 4)}, family=CHESS)  # a misreading: e6 and e7 taken for offered
    # Never seen passing over a piece: as far as e6 and e7 only along a clear way, and not relied on.
    assert [s.sure for s in rules.family().kinds["K"].steps if s.offsets == ((1, 0),) and s.rides > 1] == [False, False]
    assert rules.seen_going_there(ALONE, Move(E1, (5, 4))) < SEEN_TO_BELIEVE
    assert rules.seen_going_there(ALONE, Move(E1, (1, 4))) >= SEEN_TO_BELIEVE
    # Its image across the board counts for it: a step seen going left is the same evidence for going right.
    assert rules.seen_going_there(ALONE, Move(E1, (1, 5))) >= SEEN_TO_BELIEVE


def test_a_crossing_rests_on_how_often_one_was_seen() -> None:
    """That her rules can express a crossing is no evidence one was seen: it rests on the crossings she has seen made
    or been offered (suggestions of 23 November 2026; it counted as seen a billion times before)."""
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    board = from_fen("4k3/8/8/8/8/8/8/4K2R w K -")
    castling = Move(E1, (0, 6), partner=((0, 7), (0, 5)))
    assert rules.seen_going_there(board, castling) == 0
    for _ in range(SEEN_TO_BELIEVE):
        rules.watched(board, castling)
    assert rules.seen_going_there(board, castling) >= SEEN_TO_BELIEVE
    kept = HowThePiecesMove.from_memory(rules.as_memory())
    assert kept is not None and kept.seen_going_there(board, castling) >= SEEN_TO_BELIEVE
