"""A distance a kind goes stands while the game offered it more often than it refused it.

LIVE 23 November 2026, game 26: one misread offer gave her bishop a leap six forward and four across (e2 to a8). The
game refused that move four times, each refusal cost her about eight seconds, and the leap stayed among the bishop's
steps, because an offer once read was kept whatever the game said after. The game's word against the game's word:
what it offered, counted, against what it refused her, counted.
"""
from __future__ import annotations

from core.agency.how_the_pieces_move import ROYAL_AFTER, HowThePiecesMove
from core.agency.pieces_by_their_kind import FIRST, Move
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

#: Her bishop on e2, their rook on a8, nothing between that matters to a leap.
HERE = from_fen("r1bq1rk1/pp4pp/2pb4/3p1p2/4p3/4Q3/PPPBBPPP/RN2R1K1 w - -")
E2, A8, D3 = (1, 4), (7, 0), (2, 3)
#: What the game offers that bishop there: along its four lines to the first piece. A misreading adds a square to
#: these; the squares shown are still shown.
OFFERS = {D3, (3, 2), (4, 1), (5, 0), (2, 5), (3, 6), (4, 7), (0, 3), (0, 5)}


def _rules() -> HowThePiecesMove:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    # How a bishop goes along its lines, as offered on an open board; and the kind the game keeps safe, as she has
    # seen it kept off attacked squares.
    rules.offered(HERE, E2, OFFERS, family=CHESS)
    rules.kept_off_attacked["K"] = ROYAL_AFTER + 1
    return rules


def _leaps_of(rules: HowThePiecesMove) -> list:
    """The bishop's steps along the misread way, six forward and four across, or its mirror image."""
    return [s for s in rules.family().kinds["B"].steps
            if abs(s.offsets[0][1]) and s.offsets[0][0] * 2 == abs(s.offsets[0][1]) * 3]


def test_an_offer_misread_once_and_refused_once_is_no_longer_a_way() -> None:
    """Her pawn on h2 is left attacked by their bishop whatever she does; that is not why the game refused her."""
    rules = _rules()
    rules.offered(HERE, E2, OFFERS | {A8}, family=CHESS)  # a misreading: a8 taken for offered
    assert _leaps_of(rules)
    rules.refused(HERE, Move(E2, A8, taken_at=A8), family=CHESS)
    assert not _leaps_of(rules)
    # The bishop's own lines are untouched.
    assert any(s.offsets == ((1, -1),) for s in rules.family().kinds["B"].steps)


def test_a_distance_offered_more_often_than_refused_stands() -> None:
    rules = _rules()
    for _ in range(3):
        rules.offered(HERE, E2, OFFERS | {A8}, family=CHESS)
    rules.refused(HERE, Move(E2, A8, taken_at=A8), family=CHESS)
    assert _leaps_of(rules)
    rules.refused(HERE, Move(E2, A8, taken_at=A8), family=CHESS)
    rules.refused(HERE, Move(E2, A8, taken_at=A8), family=CHESS)
    assert not _leaps_of(rules)
    # Offered again, and more often than refused, it is a way again.
    rules.offered(HERE, E2, OFFERS | {A8}, family=CHESS)
    assert _leaps_of(rules)


def test_counts_are_kept_and_a_record_from_before_them_counts_its_offer_once() -> None:
    rules = _rules()
    rules.offered(HERE, E2, OFFERS | {A8}, family=CHESS)
    kept = rules.as_memory()
    again = HowThePiecesMove.from_memory(kept)
    assert again is not None and _leaps_of(again)
    way = again.seen["B"][(3, -2)]
    assert way.took_at == {2: 1} and not way.went_at.get(2)
    # A record written before offers were counted.
    for ways in kept["seen"].values():
        for seen in ways.values():
            for name in ("went_at", "took_at", "refused_going_at", "refused_taking_at"):
                seen.pop(name, None)
    old = HowThePiecesMove.from_memory(kept)
    assert old is not None and _leaps_of(old)
    old.refused(HERE, Move(E2, A8, taken_at=A8), family=CHESS)
    assert not _leaps_of(old)


def test_a_refused_take_leaves_the_kind_going_that_far_to_an_empty_square() -> None:
    rules = _rules()
    for _ in range(2):
        rules.offered(from_fen("6k1/8/8/8/8/8/4B3/4K3 w - -"), E2, {D3, (3, 2)}, family=CHESS)
    rules.refused(HERE, Move(E2, D3, taken_at=D3), family=CHESS)
    rules.refused(HERE, Move(E2, D3, taken_at=D3), family=CHESS)
    way = rules.seen["B"][(1, -1)]
    assert 1 in way.offered and way.would_not_take == 2


def test_a_refusal_that_what_the_move_leaves_open_explains_takes_nothing_from_the_way() -> None:
    """Her royal piece open to their rook once the bishop stands aside: the game refuses the bishop's move for that,
    and the bishop still goes along its line."""
    pinned = from_fen("4r1k1/8/8/8/8/8/4B3/4K3 w - -")
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    rules.offered(from_fen("6k1/8/8/8/8/8/4B3/4K3 w - -"), E2, {D3}, family=CHESS)
    assert pinned.cells[E2] == (FIRST, "B")
    rules.refused(pinned, Move(E2, D3), family=CHESS)
    assert any(s.offsets == ((1, -1),) for s in rules.family().kinds["B"].steps)


def test_the_image_of_a_way_refused_where_it_lands_is_not_one() -> None:
    """LIVE 23 November 2026, game 27: the misread leap, refused as e2 to a8, came back as its image, d2 to h8."""
    rules = _rules()
    rules.offered(HERE, E2, OFFERS | {A8}, family=CHESS)
    images = [s for s in _leaps_of(rules) if s.offsets[0][1] > 0]
    assert images, "the image is inferred from the step seen"
    d2, h8 = (1, 3), (7, 7)
    there = from_fen("r1bq1rkr/pp4pp/2pb4/3p1p2/4p3/4Q3/PPPBBPPP/RN2R1K1 w - -")
    rules.refused(there, Move(d2, h8, taken_at=h8), family=CHESS)
    assert not [s for s in _leaps_of(rules) if s.offsets[0][1] > 0]
