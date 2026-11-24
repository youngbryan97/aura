"""A square the game does not offer a piece, where her rules say it goes, counts against the way it goes there.

LIVE 23 November 2026, after game 29: her rules had her royal kind leap two forward and three across, and ride
forward the whole board, each on one offer seen once. Whenever she picked the piece up after, the game showed those
squares empty and not offered, and nothing of that counted. Her search gave her king, and Stockfish's, ways out of
attacks that were none: at the ninth move, in check, it chose a leap of her king to b3.
"""
from __future__ import annotations

from core.agency.how_the_pieces_move import ROYAL_AFTER, HowThePiecesMove
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

#: Her bishop on e2 with their rook on a8, and the squares the game offers it there.
HERE = from_fen("r1bq1rk1/pp4pp/2pb4/3p1p2/4p3/4Q3/PPPBBPPP/RN2R1K1 w - -")
E2, A8 = (1, 4), (7, 0)
OFFERS = {(2, 3), (3, 2), (4, 1), (5, 0), (2, 5), (3, 6), (4, 7), (0, 3), (0, 5)}
E1 = (0, 4)


def _knowing_what_is_kept_safe(rules: HowThePiecesMove) -> HowThePiecesMove:
    rules.kept_off_attacked["K"] = ROYAL_AFTER + 1
    return rules


def _leaps(rules: HowThePiecesMove) -> list:
    return [s for s in rules.family().kinds["B"].steps
            if abs(s.offsets[0][1]) and s.offsets[0][0] * 2 == abs(s.offsets[0][1]) * 3]


def test_a_leap_read_once_goes_when_the_game_withholds_it() -> None:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    rules.offered(HERE, E2, OFFERS, family=CHESS)
    rules.offered(HERE, E2, OFFERS | {A8}, family=CHESS)  # a misreading: a8 taken for offered
    _knowing_what_is_kept_safe(rules)
    assert _leaps(rules)
    # Picked up again: the game shows what it showed before, and not a8.
    rules.offered(HERE, E2, OFFERS, family=rules.family(unseen_go_every_way=True))
    assert not _leaps(rules)
    assert any(s.offsets == ((1, -1),) and s.rides == 0 for s in rules.family().kinds["B"].steps)


def test_a_move_withheld_for_what_it_leaves_open_takes_nothing_from_the_way() -> None:
    """Their queen on a5 holds her bishop on d2 to the line between it and her royal piece."""
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    d2 = (1, 3)
    rules.offered(from_fen("6k1/8/8/8/8/8/3B4/4K3 w - -"), d2,
                  {(2, 2), (3, 1), (4, 0), (2, 4), (3, 5), (4, 6), (5, 7), (0, 2)}, family=CHESS)
    _knowing_what_is_kept_safe(rules)
    pinned = from_fen("4k3/8/8/q7/8/8/3B4/4K3 w - -")
    rules.offered(pinned, d2, {(2, 2), (3, 1), (4, 0)}, family=rules.family(unseen_go_every_way=True))
    for direction in ((1, 1), (-1, -1)):
        way = rules.seen["B"][direction]
        assert not way.refused_going_at and not way.refused_taking_at
    assert any(s.offsets == ((1, 1),) for s in rules.family().kinds["B"].steps)


def test_a_leap_read_far_once_goes_when_the_game_withholds_it_and_a_line_stops_short_of_a_withheld_square() -> None:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    alone = from_fen("k7/8/8/8/8/8/8/4K3 w - -")
    around = {(1, 4), (1, 3), (1, 5), (0, 3), (0, 5)}
    rules.offered(alone, E1, around, family=CHESS)
    rules.offered(alone, E1, around | {(6, 4)}, family=CHESS)  # a misreading: e7 taken for offered
    # Six squares up with the five between never offered: a step of one, and six squares a misreading or a way that
    # far she is not sure of, along a clear way only, since it was never seen passing over a piece.
    assert [(s.rides, s.sure) for s in rules.family().kinds["K"].steps if s.offsets == ((1, 0),)] == [(1, False),
                                                                                                    (6, False)]
    _knowing_what_is_kept_safe(rules)
    rules.offered(alone, E1, around, family=rules.family(unseen_go_every_way=True))
    assert [s.rides for s in rules.family().kinds["K"].steps if s.offsets == ((1, 0),)] == [1]
    # A run read unbroken, e2 to e4 with e5 empty and not offered, is a way three squares long; withheld past e2
    # with e3 and e4 empty, it stops at e2.
    lined = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    lined.offered(alone, E1, around | {(2, 4), (3, 4)}, family=CHESS)
    assert [s.rides for s in lined.family().kinds["K"].steps if s.offsets == ((1, 0),)] == [3]
    _knowing_what_is_kept_safe(lined)
    lined.offered(alone, E1, around, family=lined.family(unseen_go_every_way=True))
    assert [s.rides for s in lined.family().kinds["K"].steps if s.offsets == ((1, 0),)] == [1]


def test_a_distance_offered_unmoved_and_withheld_once_moved_is_an_unmoved_one() -> None:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    side_steps = {(1, 4), (1, 3), (1, 5), (0, 3), (0, 5)}
    unmoved = from_fen("4k3/8/8/8/8/8/8/4K2R w K -")
    moved = from_fen("4k3/8/8/8/8/8/8/4K2R w - -")
    rules.offered(unmoved, E1, side_steps | {(0, 6)}, family=CHESS)
    rules.offered(moved, E1, side_steps | {(0, 6)}, family=CHESS)  # a misreading of a piece that has moved
    _knowing_what_is_kept_safe(rules)
    rules.offered(moved, E1, side_steps, family=rules.family(unseen_go_every_way=True))
    way = rules.seen["K"][(0, 1)]
    assert 2 not in way.offered and 2 in way.offered_unmoved
    assert not way.refused_going_at.get(2), "withheld from a piece that had moved, it is not refused to one unmoved"
    # Withheld from it unmoved, where what else such a move needs is not in her rules, it counts against nothing.
    rules.offered(unmoved, E1, side_steps, family=rules.family(unseen_go_every_way=True))
    assert 2 in rules.seen["K"][(0, 1)].offered_unmoved


def test_an_offer_of_nothing_says_nothing() -> None:
    rules = _knowing_what_is_kept_safe(HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start))
    rules.offered(HERE, E2, OFFERS, family=CHESS)
    rules.offered(HERE, E2, set(), family=rules.family(unseen_go_every_way=True))
    assert not any(way.refused_going_at or way.refused_taking_at for way in rules.seen["B"].values())


def test_a_partner_square_offered_for_a_crossing_is_the_crossing_and_not_two_pieces_changing_places() -> None:
    """En Croissant offers her king the rook's square as well as the square castling lands it on."""
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    unmoved = from_fen("4k3/8/8/8/8/8/8/4K2R w K -")
    rules.offered(unmoved, E1, {(1, 4), (1, 3), (1, 5), (0, 3), (0, 5)}, family=CHESS)
    rules.crosses["K"] = "R"
    rules.offered(unmoved, E1, {(1, 4), (1, 3), (1, 5), (0, 3), (0, 5), (0, 6), (0, 7)},
                  family=rules.family(unseen_go_every_way=True))
    assert not rules.seen["K"][(0, 1)].swapped


def test_two_pieces_changing_places_withheld_is_not_a_way() -> None:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    around = {(1, 4), (1, 3), (1, 5), (0, 3), (0, 5)}
    unmoved = from_fen("4k3/8/8/8/8/8/8/4K2R w K -")
    # Before she knows of any crossing, the rook's square offered is read as the two changing places.
    rules.offered(unmoved, E1, around | {(0, 7)})
    assert any(s.may == "swaps" for s in rules.family().kinds["K"].steps)
    _knowing_what_is_kept_safe(rules)
    # In check from their rook on e8: the game offers her king its side steps, and not the rook's square.
    checked = from_fen("4r1k1/8/8/8/8/8/8/4K2R w K -")
    rules.offered(checked, E1, {(1, 3), (1, 5), (0, 3), (0, 5)}, family=rules.family(unseen_go_every_way=True))
    assert not any(s.may == "swaps" for s in rules.family().kinds["K"].steps)
