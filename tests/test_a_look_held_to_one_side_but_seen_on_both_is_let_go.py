"""A look held to be one side's that stands at both ends of a board at its start is a kind both sides have
(core/agency/playing_pieces_by_kind.py ``a_board_of_two_sides_at_its_start``).

LIVE 25 November 2026 her kept looks for En Croissant held the shapes of two kinds to the lighter side: the darker
side's pieces of those kinds read as the lighter side's, the board of a new game was not seen as a game at its start,
and six runs running she pressed controls around a board she could have played.
"""
from __future__ import annotations

import pytest

from core.agency.playing_pieces_by_kind import a_board_of_two_sides_at_its_start
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

pytestmark = pytest.mark.unit


def test_looks_held_to_one_side_but_standing_at_both_ends_are_believed_by_their_colour_again() -> None:
    picture = BoardApplication(CHESS, seed=1).draw()
    first = a_board_of_two_sides_at_its_start(picture)
    assert first is not None
    looks = first.looks
    seen = looks.read(picture, first.lattice)
    top = {cell[1] for line in seen.cells[:4] for cell in line if cell is not None}
    bottom = {cell[1] for line in seen.cells[4:] for cell in line if cell is not None}
    shared = sorted(top & bottom)
    assert len(shared) >= 2
    lighter = seen.cells[-1][0][0] if seen.cells[-1][0] else seen.cells[0][0][0]
    for kind in shared[:2]:
        looks.side_of_kind[kind] = lighter  # misread once, and kept
    again = a_board_of_two_sides_at_its_start(picture, looks)
    assert again is not None
    assert not set(shared[:2]) & set(looks.side_of_kind)
