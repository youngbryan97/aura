"""The thing's grid is worked out from the thing, not from what surrounds it.

LIVE 2026-08-31, on a clean reading of one native window — no browser, no
adverts, no other application: the tiles landed on columns 0, 3, 4 and 6 of a
nine-column lattice that a score, a title and a New Game button had defined
between them. No rule about sliding along a row could match, because those were
not rows of the board. Forty-eight moves, and "how this moves is not worked out
yet".

The band is the outline of the places that answer to her, and an outline is
coarser than the places are: furniture inside a board's outline is inside the
box. The places themselves are not coarse, and they are already counted.
"""

from __future__ import annotations

from core.perception.where_it_responds import (
    Responsive,
    the_places_that_answer,
    what_is_there,
)

BOARD = [(0.40, 0.20, "2"), (0.40, 0.30, "4"), (0.50, 0.20, "8"), (0.50, 0.30, "16")]
FURNITURE = [
    (0.22, 0.38, "SCORE"),
    (0.22, 0.45, "BEST"),
    (0.24, 0.38, "476"),
    (0.29, 0.44, "New Game"),
]


def _seen(cells):
    return {
        "layout": [
            {"center_y": y, "center_x": x, "text": said} for y, x, said in cells
        ]
    }


def _places(cells):
    return frozenset(
        (int(round(x * 100)), int(round(y * 100))) for _y, x, _said in cells
    ) & frozenset(
        (int(round(x * 100)), int(round(y * 100))) for y, x, _said in cells
    )


def test_the_furniture_defines_columns_the_board_does_not_have():
    whole = what_is_there(_seen(BOARD + FURNITURE), None)
    assert (whole.rows, whole.columns) != (2, 2)


def test_and_the_places_that_answer_give_the_board_its_own_grid():
    only = what_is_there(
        _seen(BOARD + FURNITURE),
        None,
        answering=frozenset(
            (int(round(x * 100)), int(round(y * 100))) for y, x, _s in BOARD
        ),
    )
    assert (only.rows, only.columns) == (2, 2)
    assert sorted(cell.says for cell in only.cells) == ["16", "2", "4", "8"]


def test_before_she_knows_which_places_answer_she_uses_what_she_has():
    """Not a reason to read nothing."""
    both = what_is_there(_seen(BOARD + FURNITURE), None, answering=None)
    assert both.cells
    assert both.rows >= 2


def test_cropping_to_nothing_is_not_a_reading():
    """Places that match none of what is on screen must not empty it."""
    nowhere = frozenset({(9, 9), (8, 8), (7, 7)})
    still = what_is_there(_seen(BOARD + FURNITURE), None, answering=nowhere)
    assert len(still.cells) == len(BOARD + FURNITURE)


def test_nothing_is_claimed_before_anything_has_answered():
    assert the_places_that_answer(Responsive()) == frozenset()
