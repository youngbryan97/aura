"""Position is not decoration.

A board where the largest tile sits in a corner and a board where it sits in
the middle are different positions with the same contents. Flattened to
"2 4 8 64" they are the same string — so nothing downstream can hold a
corner, an edge, a column or a row, and no approach about any of them can
survive contact with what she reads.

Bryan, 2026-08-26, watching her play: "she never employs the best and super
common strategy of keep your highest block in the corner." She could not. She
was never shown one.
"""
from __future__ import annotations

from core.perception.where_it_responds import within


def _cell(text: str, x: float, y: float) -> dict:
    return {"text": text, "center_x": x, "center_y": y, "x": x, "y": y}


def _grid(values: list[list[str]], *, x0=0.40, y0=0.30, step=0.07) -> dict:
    layout = [
        _cell(value, x0 + column * step, y0 + row * step)
        for row, line in enumerate(values)
        for column, value in enumerate(line)
    ]
    return {"ok": True, "text": " ".join(v for line in values for v in line), "layout": layout}


BOARD = [
    ["2", "4", "8", "4"],
    ["16", "32", "2", "8"],
    ["4", "8", "128", "2"],
    ["64", "2", "4", "16"],
]


def test_a_board_reads_as_a_board():
    read = within(_grid(BOARD), (0.30, 0.20, 0.75, 0.70))
    assert read.splitlines() == ["2 4 8 4", "16 32 2 8", "4 8 128 2", "64 2 4 16"]


def test_the_same_values_in_a_different_arrangement_read_differently():
    """This is the whole point: contents alone cannot tell the two apart."""
    moved = [row[:] for row in BOARD]
    moved[0][0], moved[3][0] = moved[3][0], moved[0][0]
    here = within(_grid(BOARD), (0.30, 0.20, 0.75, 0.70))
    there = within(_grid(moved), (0.30, 0.20, 0.75, 0.70))
    assert here != there
    assert sorted(here.split()) == sorted(there.split()), "same contents, different position"


def test_rows_are_found_from_the_spacing_that_is_there():
    """Not a fixed tolerance. A dense layout and a sparse one both have rows,
    and each one's rows are closer together than they are to the next."""
    tight = within(_grid(BOARD, step=0.02), (0.0, 0.0, 1.0, 1.0))
    wide = within(_grid(BOARD, step=0.15), (0.0, 0.0, 1.0, 1.0))
    assert len(tight.splitlines()) == 4
    assert len(wide.splitlines()) == 4


def test_a_single_row_stays_one_row():
    one = within(_grid([["Name", "Email", "Phone"]]), (0.0, 0.0, 1.0, 1.0))
    assert one.splitlines() == ["Name Email Phone"]


def test_reading_cells_out_of_order_still_lays_them_out_in_order():
    board = _grid(BOARD)
    board["layout"].reverse()
    read = within(board, (0.30, 0.20, 0.75, 0.70))
    assert read.splitlines()[0] == "2 4 8 4"
    assert read.splitlines()[-1] == "64 2 4 16"


def test_nothing_inside_the_band_falls_back_to_the_whole_reading():
    board = _grid(BOARD)
    assert within(board, (0.90, 0.90, 0.99, 0.99)) == board["text"]
