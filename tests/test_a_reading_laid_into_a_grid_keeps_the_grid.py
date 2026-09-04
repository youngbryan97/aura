"""Cropping furniture must not re-address the thing it leaves behind.

A reading laid into a lattice has empty places, and they are places rather
than gaps. Working the rows out again from whatever happens to be occupied
moves every address the moment a row empties — so two readings a move apart
are in different frames of reference however alike they look, which is the
exact thing holding a lattice exists to prevent.

LIVE 2026-09-04 on a correctly read four by four board: the top row emptied,
the reading became three rows, and what had been rows one to three were
compared against rows nought to two of the frame before. Row nought disagreed
with every rule every time while the rest matched, and the true rule sat at
59% of 29 all game — never trusted, so nothing ever looked ahead.
"""

from __future__ import annotations

from core.perception.what_is_there import Arrangement, Cell

DOWN = (0.34, 0.48, 0.62, 0.76)
ACROSS = (0.25, 0.42, 0.58, 0.74)


def _board(rows: list[list[str]], *, gridded: bool = True) -> Arrangement:
    cells = tuple(
        Cell(row=r, column=c, says=said, at=(ACROSS[c], DOWN[r]))
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    )
    return Arrangement(
        rows=len(rows),
        columns=len(rows[0]),
        cells=cells,
        down_at=DOWN if gridded else (),
        across_at=ACROSS if gridded else (),
    )


EMPTY_TOP = _board([["", "", "", ""], ["2", "", "", "8"], ["4", "8", "32", "2"], ["4", "8", "2", ""]])


def test_an_empty_row_of_a_gridded_reading_is_still_a_row():
    left = EMPTY_TOP.without({(0, 0)})
    assert left.rows == 4
    assert {(cell.row, cell.column) for cell in left.cells if cell.says == "2"} >= {(1, 0)}


def test_a_line_that_was_wholly_cropped_does_go():
    whole_row = {(0, column) for column in range(4)}
    assert EMPTY_TOP.without(whole_row).rows == 3


def test_a_wholly_cropped_column_goes_too():
    whole_column = {(row, 3) for row in range(4)}
    assert EMPTY_TOP.without(whole_column).columns == 3


def test_two_readings_a_move_apart_keep_the_same_addresses():
    """The failure this prevents, stated as the thing it broke."""
    before = _board([["2", "8", "", "8"], ["", "2", "4", "4"], ["4", "8", "32", "2"], ["2", "2", "4", "4"]])
    after = _board([["", "", "", ""], ["2", "8", "", ""], ["4", "8", "32", "2"], ["4", "8", "2", ""]])
    counters = {(0, 0)}
    assert before.without(counters).rows == after.without(counters).rows == 4


def test_a_reading_with_no_grid_still_finds_the_thing_by_cropping():
    """Before a lattice there is nothing to keep, and cropping is how she
    finds the board inside a window."""
    loose = _board([["", "", "", ""], ["", "2", "4", ""], ["", "8", "16", ""]], gridded=False)
    assert loose.without({(0, 0)}).rows == 2


def test_nothing_dropped_changes_nothing():
    assert EMPTY_TOP.without(set()) is EMPTY_TOP
