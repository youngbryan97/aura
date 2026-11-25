"""A kind she has seen taken, with the game going on after, is not the kind the game keeps safe.

LIVE 21 November 2026 her record of En Croissant's board had been kept through misreadings since mended: ten moves
taken for refusals, nine of them leaving a pawn open. It held the pawns' kind as the one to keep safe, and she called
nearly every move an attack on it and said she could force a win on her first move.
"""
from __future__ import annotations

from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.pieces_by_their_kind import FIRST, SECOND, Move, Position

START = {(0, 0): (FIRST, "K"), (1, 0): (FIRST, "P"), (1, 1): (FIRST, "P"),
         (7, 7): (SECOND, "K"), (6, 1): (SECOND, "P"), (6, 2): (SECOND, "P")}


def _learned_wrongly() -> HowThePiecesMove:
    learned = HowThePiecesMove.from_the_start(8, 8, START)
    learned.refusals, learned.refused_leaving_open = 10, {"P": 10}
    return learned


def test_a_kind_taken_and_played_on_is_not_royal() -> None:
    learned = _learned_wrongly()
    assert learned.royal("P"), "the misread record does make pawns royal"
    position = Position(cells={**START, (2, 1): (SECOND, "P")}, turn=FIRST, unmoved=frozenset(START))
    learned.watched(position, Move((1, 0), (2, 1), taken_at=(2, 1)))
    assert learned.royal("P"), "one capture alone does not say the game went on"
    after = Position(cells={(0, 0): (FIRST, "K"), (2, 1): (FIRST, "P"), (1, 1): (FIRST, "P"), (7, 7): (SECOND, "K"),
                            (6, 1): (SECOND, "P"), (6, 2): (SECOND, "P")}, turn=SECOND, unmoved=frozenset())
    learned.watched(after, Move((6, 2), (5, 2)))
    assert not learned.royal("P")


def test_what_falsified_it_is_kept_for_the_next_game() -> None:
    learned = _learned_wrongly()
    learned.taken_and_played_on["P"] = 1
    back = HowThePiecesMove.from_memory(learned.as_memory())
    assert back is not None and not back.royal("P")


def test_a_kind_she_no_longer_reads_is_nothing_a_piece_becomes() -> None:
    learned = HowThePiecesMove.from_the_start(8, 8, START)
    learned.becomes = {"P": ["14", "K"]}
    learned.became_at = {"P": {7: {"14": 1, "K": 1}}}  # each seen on the far row
    learned.seen.setdefault("14", {})
    assert learned.family().kinds["P"].becomes == ("K",)
