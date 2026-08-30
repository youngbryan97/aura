"""Six tiles on a four-by-four board occupy three columns. The fourth is real.

A grid is defined by its pitch, not by what happens to sit on it. Built from
occupied positions alone, a reading is three columns wide on one glance and
four on the next, and nothing downstream can model a thing that changes shape.

LIVE 2026-08-30, read off the running game on play2048.co: columns at 0.184,
0.596 and 0.811, gaps of 0.412 and 0.215 — and 0.412 is twice 0.215. There is
a column at 0.399 with nothing in it.
"""

from __future__ import annotations

import pytest

from core.perception.the_thing_itself import the_thing_itself
from core.perception.what_is_there import arranged, the_places_nothing_sits_in

#: What the OCR actually returned from the canvas, positions and all: the
#: score, the best score, and six tiles.
OFF_THE_SCREEN = [
    (0.01235, 0.40578, "24"),
    (0.01308, 0.58675, "6068"),
    (0.42006, 0.81063, "2"),
    (0.58430, 0.18377, "2"),
    (0.58285, 0.81063, "4"),
    (0.74564, 0.18563, "4"),
    (0.74491, 0.59608, "2"),
    (0.74564, 0.81157, "8"),
]


def test_a_missing_column_is_put_back():
    filled = the_places_nothing_sits_in([0.184, 0.596, 0.811])
    assert len(filled) == 4
    assert filled[1] == pytest.approx(0.399, abs=0.01)


def test_evenly_spaced_positions_gain_nothing():
    assert the_places_nothing_sits_in([0.42, 0.583, 0.746]) == (0.42, 0.583, 0.746)


@pytest.mark.parametrize(
    "ragged",
    [
        [0.10, 0.23, 0.61, 0.66],
        [0.05, 0.40, 0.44, 0.90],
        [0.2, 0.8],
    ],
)
def test_positions_that_are_not_one_lattice_gain_nothing(ragged):
    """Inventing a lattice is worse than reading a short one."""
    assert the_places_nothing_sits_in(ragged) == tuple(sorted(ragged))


def test_the_real_reading_comes_out_as_a_board():
    got = arranged(OFF_THE_SCREEN)
    assert (got.rows, got.columns) == (4, 4)
    assert got.occupied() == 8


def test_and_the_score_row_is_not_part_of_the_board():
    board = the_thing_itself(arranged(OFF_THE_SCREEN))
    assert board.columns == 4
    assert board.occupied() == 6
    said = board.as_text()
    assert "6068" not in said
    assert "24" not in said


def test_the_tiles_land_in_the_places_they_are_actually_in():
    board = the_thing_itself(arranged(OFF_THE_SCREEN))
    rows = [line.split() for line in board.as_text().splitlines()]
    assert rows[0][3] == "2"
    assert rows[1][0] == "2"
    assert rows[1][3] == "4"
    assert rows[2] == ["4", ".", "2", "8"]
