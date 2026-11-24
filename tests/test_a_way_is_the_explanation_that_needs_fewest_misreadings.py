"""How far a kind goes along a way is the explanation of its offers that needs the fewest squares misread.

LIVE 23 November 2026: her royal kind, offered one square forward many times and six and seven squares once, was held
to ride the whole file. A piece that rides past a square can stop on it, and the game offers every square it passes,
so that look needed four squares misread as not offered for a line, and only its two far offers misread (or a leap)
for a step of one. Her bishop's way back to the right, offered at one to four squares and at six, needs one square
misread either way, and stays a line.
"""
from __future__ import annotations

from core.agency.how_the_pieces_move import SEEN_TO_BELIEVE, HowThePiecesMove
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

E1 = (0, 4)


def _fresh() -> HowThePiecesMove:
    return HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)


def test_far_offers_with_the_squares_between_never_offered_are_a_step_and_leaps_not_yet_sure() -> None:
    rules = _fresh()
    open_file = from_fen("k7/8/8/8/8/8/8/4K3 w - -")
    around = {(1, 4), (1, 3), (1, 5), (0, 3), (0, 5)}
    for _ in range(SEEN_TO_BELIEVE):
        rules.offered(open_file, E1, around, family=CHESS)
    rules.offered(open_file, E1, around | {(6, 4), (7, 4)})  # a misreading: e7 and e8 taken for offered
    forward = [s for s in rules.family().kinds["K"].steps if s.offsets[0][1] == 0 and s.offsets[0][0] > 0]
    assert ((1, 0), 1, True) in [(s.offsets[0], s.rides, s.sure) for s in forward]
    assert {(s.offsets[0], s.sure) for s in forward if s.offsets[0] != (1, 0)} == {((6, 0), False), ((7, 0), False)}
    assert not any(s.rides == 0 for s in forward)


def test_a_line_with_one_square_never_offered_stays_a_line() -> None:
    rules = _fresh()
    c1 = (0, 2)
    # Along the diagonal to h6: d2, e3, f4 and h6 offered, g5 missed by the reading.
    rules.offered(from_fen("4k3/8/8/8/8/8/8/2B1K3 w - -"), c1, {(1, 3), (2, 4), (3, 5), (5, 7), (1, 1), (2, 0)})
    assert [s.rides for s in rules.family().kinds["B"].steps if s.offsets == ((1, 1),)] == [0]


def test_two_pieces_changing_places_are_relied_on_as_far_as_moves_of_that_sort_were_seen() -> None:
    rules = _fresh()
    unmoved = from_fen("4k3/8/8/8/8/8/8/4K2R w K -")
    around = {(1, 4), (1, 3), (1, 5), (0, 3), (0, 5)}
    rules.offered(unmoved, E1, around | {(0, 7)})
    swaps = [s for s in rules.family().kinds["K"].steps if s.may == "swaps"]
    assert swaps and not any(s.sure for s in swaps)
    for _ in range(SEEN_TO_BELIEVE - 1):
        rules.offered(unmoved, E1, around | {(0, 7)})
    # The way seen is relied on; its mirror image, an inference, waits on the image being seen.
    assert [s.sure for s in rules.family().kinds["K"].steps if s.may == "swaps" and s.offsets == ((0, 3),)] == [True]


def test_three_far_squares_one_look_showed_beyond_squares_other_looks_withheld_are_not_a_line() -> None:
    """One corrupted look showing three far squares (suggestions of 23 November 2026): looks before it showed the
    squares between empty and not offered, which a line would have offered."""
    from core.agency.pieces_by_their_kind import Move

    rules = _fresh()
    open_file = from_fen("k7/8/8/8/8/8/8/4K3 w - -")
    around = {(1, 4), (1, 3), (1, 5), (0, 3), (0, 5)}
    for _ in range(SEEN_TO_BELIEVE):
        rules.offered(open_file, E1, around)
    rules.offered(open_file, E1, around | {(4, 4), (5, 4), (6, 4)})  # one misreading: e5, e6 and e7
    assert rules.seen_going_there(open_file, Move(E1, (5, 4))) < SEEN_TO_BELIEVE, "one look, however many squares"
    assert rules.seen_going_there(open_file, Move(E1, (1, 4))) >= SEEN_TO_BELIEVE
    assert not any(s.sure for s in rules.family().kinds["K"].steps if s.offsets[0][0] >= 4)


def test_a_move_by_far_squares_one_look_showed_beyond_a_gap_is_checked_before_it_is_made() -> None:
    """One corrupted look showing three far squares: a leap she is not sure of, so a move by it is checked first."""
    from core.agency.pieces_by_their_kind import Move

    rules = _fresh()
    open_file = from_fen("k7/8/8/8/8/8/8/4K3 w - -")
    around = {(1, 4), (1, 3), (1, 5), (0, 3), (0, 5)}
    for _ in range(SEEN_TO_BELIEVE):
        rules.offered(open_file, E1, around)
    rules.offered(open_file, E1, around | {(4, 4), (5, 4), (6, 4)})  # e5, e6 and e7 in one misread look
    assert not rules.sure_of(open_file, Move(E1, (5, 4)))
    assert rules.sure_of(open_file, Move(E1, (1, 4)))
