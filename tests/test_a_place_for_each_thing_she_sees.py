"""What she sees has a place for each thing in it, not just a shape on a line.

Flattened to a string, a board where the largest tile sits in a corner and a
board where it sits in the middle are the same string — so nothing downstream
can hold a corner, and no approach about one can be checked. This is the
structure the string is a rendering of.

Nothing here knows what it is looking at: the same object describes a game
board, a timetable, a price list and a form, because the only thing it assumes
is that the thing is laid out.
"""

from __future__ import annotations

import pytest

from core.perception.what_is_there import EMPTY_CELL, Arrangement, arranged

#: A 4x4 board with three empty places, as a reading hands it over: (y, x, text).
BOARD = [
    (0.30, 0.20, "8"), (0.30, 0.35, "2"), (0.30, 0.50, "2"), (0.30, 0.65, "2"),
    (0.45, 0.20, "2"), (0.45, 0.50, "32"), (0.45, 0.65, "4"),
    (0.60, 0.20, "64"), (0.60, 0.35, "2"), (0.60, 0.50, "32"),
    (0.75, 0.35, "4"), (0.75, 0.50, "8"), (0.75, 0.65, "2"),
]

TIMETABLE = [
    (0.10, 0.10, "Mon"), (0.10, 0.30, "Tue"), (0.10, 0.50, "Wed"),
    (0.22, 0.10, "09:00"), (0.22, 0.30, "Standup"), (0.22, 0.50, "Review"),
    (0.34, 0.10, "11:00"), (0.34, 0.50, "1:1"),
]


@pytest.fixture
def board() -> Arrangement:
    return arranged(BOARD)


# ── the rows and columns that are really there ───────────────────────────

def test_the_shape_is_found_from_the_spacing_that_is_present(board):
    assert (board.rows, board.columns) == (4, 4)


def test_a_gap_is_a_place_rather_than_a_missing_word(board):
    assert board.at(1, 1) is None
    assert board.occupied() == 13
    assert board.empty() == 3
    assert board.places() == 16


def test_a_different_thing_finds_its_own_shape():
    timetable = arranged(TIMETABLE)
    assert (timetable.rows, timetable.columns) == (3, 3)
    assert timetable.at(0, 0).says == "Mon"
    assert timetable.at(2, 1) is None


# ── the places a plan is phrased about ───────────────────────────────────

def test_a_corner_is_something_that_can_be_asked_about(board):
    corners = board.corners()
    assert corners["top-left"].says == "8"
    assert corners["top-right"].says == "2"
    assert corners["bottom-right"].says == "2"
    assert corners["bottom-left"] is None


def test_an_edge_is_a_row_or_a_column_of_places(board):
    bottom = [cell.says if cell else EMPTY_CELL for cell in board.edges()["bottom"]]
    assert bottom == [EMPTY_CELL, "4", "8", "2"]
    left = [cell.says if cell else EMPTY_CELL for cell in board.edges()["left"]]
    assert left == ["8", "2", "64", EMPTY_CELL]


@pytest.mark.parametrize(
    ("row", "column", "said"),
    [(0, 0, "top-left"), (3, 3, "bottom-right"), (0, 2, "top"), (2, 0, "left"), (1, 1, "middle")],
)
def test_where_a_thing_sits_is_said_the_way_a_person_says_it(board, row, column, said):
    at = board.at(row, column)
    if at is None:
        at = type(board.cells[0])(row=row, column=column, says="x", at=(0.0, 0.0))
    assert board.place_of(at) == said


def test_nowhere_is_not_a_place(board):
    assert board.place_of(None) == ""


# ── what is here ─────────────────────────────────────────────────────────

def test_the_largest_thing_is_found_where_things_are_numbers(board):
    assert board.largest().says == "64"
    assert board.place_of(board.largest()) == "left"


def test_a_thing_can_be_found_by_what_it_says(board):
    assert board.where_is("32") == ((1, 2), (2, 2))
    assert board.where_is("4096") == ()
    assert board.where_is("") == ()


@pytest.mark.parametrize(
    ("said", "value"),
    [("1,024", 1024.0), ("Score: 88", 88.0), ("$14.00", 14.0), ("-3", -3.0)],
)
def test_a_number_written_the_way_people_write_it_is_a_number(said, value):
    only = arranged([(0.1, 0.1, said)])
    assert only.cells[0].number() == value


@pytest.mark.parametrize("said", ["Standup", "—", "09:00", ""])
def test_what_is_not_a_number_is_not_read_as_one(said):
    only = arranged([(0.1, 0.1, said)])
    assert not only.cells or only.cells[0].number() is None


def test_where_nothing_is_a_number_there_is_no_largest():
    assert arranged([(0.1, 0.1, "Mon"), (0.1, 0.3, "Tue")]).largest() is None


# ── how it is said ───────────────────────────────────────────────────────

def test_the_rendering_keeps_the_gaps_so_a_column_stays_a_column(board):
    assert board.as_text().splitlines() == [
        "8 2 2 2",
        f"2 {EMPTY_CELL} 32 4",
        f"64 2 32 {EMPTY_CELL}",
        f"{EMPTY_CELL} 4 8 2",
    ]


def test_the_rendering_is_what_the_reader_already_produced(board):
    from core.perception.where_it_responds import _laid_out

    assert _laid_out(BOARD) == board.as_text()


# ── the shape of a position, for recognising one like it ─────────────────

def test_two_positions_approached_the_same_way_look_the_same(board):
    """The record has to key on the kind of position, or nothing ever matches."""
    same_shape = arranged([(y, x, said) for y, x, said in BOARD])
    assert board.as_shape() == same_shape.as_shape()


def test_moving_the_largest_thing_changes_the_kind_of_position(board):
    moved = arranged(
        [(y, x, "2" if said == "64" else said) for y, x, said in BOARD]
        + [(0.30, 0.35, "64")]
    )
    assert moved.as_shape() != board.as_shape()


def test_the_shape_says_where_the_largest_thing_is(board):
    assert "largest:64@left" in board.as_shape()


# ── nothing there ────────────────────────────────────────────────────────

def test_an_empty_reading_is_empty_rather_than_broken():
    nothing = arranged([])
    assert (nothing.rows, nothing.columns) == (0, 0)
    assert nothing.as_text() == ""
    assert nothing.as_shape() == "empty"
    assert nothing.corners() == {}
    assert nothing.edges() == {}
    assert nothing.largest() is None


def test_blank_readings_are_not_things():
    assert arranged([(0.1, 0.1, "   "), (0.2, 0.2, "")]).occupied() == 0


def test_one_thing_is_a_place_of_its_own():
    only = arranged([(0.5, 0.5, "OK")])
    assert (only.rows, only.columns) == (1, 1)
    assert only.as_text() == "OK"
