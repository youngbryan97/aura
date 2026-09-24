"""A place she could not read is not evidence against how the thing moves.

The readings carried unread places from 2026-09-18 and nothing that learns
the rule looked at them. LIVE 2026-09-24: "her rule missed on right: (0,2)
said '8' saw None", again and again, and the true rule fell to 12 of the last
50 moves on a board her eyes read correctly whenever it was still.
"""
from __future__ import annotations

from core.perception.how_it_moves import HowItMoves
from core.perception.what_is_there import Arrangement, Cell


def _board(*tiles: tuple[int, int, str], unknown: tuple[tuple[int, int], ...] = ()) -> Arrangement:
    return Arrangement(
        4, 4, tuple(Cell(r, c, s, (0.0, 0.0)) for r, c, s in tiles),
        places_seen=True, unknown=unknown,
    )


def test_a_pair_with_an_unread_tile_is_not_graded():
    knows = HowItMoves()
    before = _board((0, 0, "8"), (0, 1, "4"))
    # Right slides the row to ". . 8 4"; the 8 did not read in the picture after.
    after = _board((0, 3, "4"), unknown=((0, 2),))
    knows.watched(before, "right", after)
    assert knows.unreadable == 1
    assert not knows.tried, "a rule was graded against a tile nobody could read"


def test_nor_is_one_whose_start_could_not_be_read():
    knows = HowItMoves()
    before = _board((0, 1, "4"), unknown=((0, 0),))
    after = _board((0, 2, "8"), (0, 3, "4"))
    knows.watched(before, "right", after)
    assert knows.unreadable == 1
    assert not knows.tried


def test_the_same_pair_read_in_full_is_graded():
    """The guard is about the unread place, not about the move."""
    knows = HowItMoves()
    before = _board((0, 0, "8"), (0, 1, "4"))
    after = _board((0, 2, "8"), (0, 3, "4"))
    knows.watched(before, "right", after)
    assert knows.unreadable == 0
    assert knows.tried and any(knows.right.values())


def test_an_unread_place_that_is_furniture_does_not_stop_the_grading():
    knows = HowItMoves()
    knows.counters.add((3, 3))
    before = _board((0, 0, "8"), (0, 1, "4"), unknown=((3, 3),))
    after = _board((0, 2, "8"), (0, 3, "4"), unknown=((3, 3),))
    knows.watched(before, "right", after)
    assert knows.unreadable == 0
