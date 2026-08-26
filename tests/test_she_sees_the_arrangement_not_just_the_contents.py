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


def test_a_gap_holds_its_place_so_columns_mean_the_same_thing_in_every_row():
    """An empty cell produces no text, so a row with gaps came out short and
    the second entry in it could be the second column or the fourth. A corner
    is a column as much as a row.

    LIVE 2026-08-26, read off the real board: "2 32 16 64 / 4 2 / 2 4 16 / 2"
    — four rows, and no way to say which column anything was in.
    """
    from core.perception.where_it_responds import EMPTY_CELL

    sparse = {
        "ok": True,
        "text": "64 2 2 4 2 16 32 4",
        "layout": [
            _cell("64", 0.54, 0.30), _cell("2", 0.61, 0.30),
            _cell("2", 0.47, 0.40), _cell("4", 0.54, 0.40), _cell("2", 0.61, 0.40),
            _cell("2", 0.40, 0.50), _cell("16", 0.47, 0.50), _cell("32", 0.54, 0.50),
        ],
    }
    rows = within(sparse, (0.30, 0.20, 0.75, 0.70)).splitlines()
    assert rows[0].split() == [EMPTY_CELL, EMPTY_CELL, "64", "2"]
    assert rows[1].split() == [EMPTY_CELL, "2", "4", "2"]
    assert rows[2].split() == ["2", "16", "32", EMPTY_CELL]
    # Every row is the same width, which is what makes a column a column.
    assert len({len(row.split()) for row in rows}) == 1
    # Every row now has its columns in the same places.
    assert rows[0].split()[2] == "64"


def test_a_full_grid_is_unchanged_by_the_gap_handling():
    read = within(_grid(BOARD), (0.30, 0.20, 0.75, 0.70))
    assert read.splitlines() == ["2 4 8 4", "16 32 2 8", "4 8 128 2", "64 2 4 16"]


def test_the_bottom_left_of_a_sparse_board_is_the_bottom_left():
    from core.agency.standing_strategy import where_it_sits

    sparse = {
        "ok": True,
        "text": "x",
        "layout": [
            _cell("4", 0.54, 0.30),
            _cell("128", 0.40, 0.50), _cell("2", 0.54, 0.50),
        ],
    }
    read = within(sparse, (0.30, 0.20, 0.75, 0.70))
    assert "bottom-left" in where_it_sits("128", read)
    assert "bottom-left" not in where_it_sits("4", read)
