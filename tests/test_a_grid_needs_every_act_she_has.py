"""A grid worked out from what moves needs her to have moved every way she can.

What moves under one act is wherever that act puts things. Press up twenty
times on a board and every tile is against the top of it, so the places that
have ever moved are two rows deep — and a grid drawn through them is two rows
deep.

LIVE 2026-09-04: ten acts into a game, all of them up, seven places in two
rows, and a four by four board became a two by three lattice. Every reading
after that was cropped to six cells, no rule could match one, nothing looked
ahead, and she pressed the same key three hundred times.
"""

from __future__ import annotations

from core.perception.the_lattice_she_holds import TheLatticeSheHolds

#: Four rows and four columns of a board, in hundredths of the window.
BOARD = frozenset(
    (25 + 17 * column, 34 + 14 * row) for row in range(4) for column in range(4)
)
#: What is left after twenty presses of up: everything against the top.
A_STRIP = frozenset((x, y) for x, y in BOARD if y <= 48)

ACTS = ("up", "down", "left", "right")


def _twice(lattice: TheLatticeSheHolds, places, **rest) -> bool:
    lattice.built_from(places, 10, **rest)
    return lattice.built_from(places, 20, **rest)


def test_one_act_tried_is_not_enough_to_draw_a_grid():
    lattice = TheLatticeSheHolds()
    assert not _twice(lattice, A_STRIP, tried={"up"}, available=ACTS)
    assert not lattice.held


def test_every_act_tried_lets_it_through():
    lattice = TheLatticeSheHolds()
    assert _twice(lattice, BOARD, tried=set(ACTS), available=ACTS)
    assert (lattice.rows, lattice.columns) == (4, 4)


def test_the_strip_a_single_act_leaves_is_the_grid_this_prevents():
    """Drawn through it, it is a real lattice — of a board that is not there."""
    lattice = TheLatticeSheHolds()
    assert _twice(lattice, A_STRIP, tried=set(ACTS), available=ACTS)
    assert lattice.rows < 4


def test_a_caller_that_names_no_acts_is_unaffected():
    lattice = TheLatticeSheHolds()
    assert _twice(lattice, BOARD)
    assert (lattice.rows, lattice.columns) == (4, 4)


def test_more_acts_tried_than_she_has_is_still_enough():
    lattice = TheLatticeSheHolds()
    assert _twice(lattice, BOARD, tried={*ACTS, "escape"}, available=ACTS)


def test_the_pursuit_names_the_acts_it_has_taken():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    at = source.index('responds["lattice"].built_from(')
    nearby = source[at : at + 500]
    assert 'tried=responds["state"].tried' in nearby
    assert "available=move_keys" in nearby
