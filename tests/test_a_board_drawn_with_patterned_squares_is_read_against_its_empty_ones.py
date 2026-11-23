"""A board whose squares are drawn with more than flat ground is read against its own empty squares.

The picture is a web page's draughts board of 11 October 2026 (tests/fixtures/a_board_of_two_sides.png), drawn in two
patterned pinks: read square by square for a silhouette on flat ground, every one of its 64 squares held a piece. At a
game's start the rows between the two sides are empty, so what they look like is what nothing looks like there.

A figure stands on one piece of each side. It is drawn taller than its square, and its top is joined to the piece
below; and the top side's figure is drawn in colours nearer the bottom side's, so its side is the one it began on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import core.agency.playing_pieces_by_kind as playing
from core.perception.pieces_by_their_look import Looks

PICTURE = Path(__file__).parent / "fixtures" / "a_board_of_two_sides.png"


def _read() -> list[list]:
    picture = np.asarray(Image.open(PICTURE).convert("RGB"))
    lattice = playing._ruled_by_its_period(picture)
    assert lattice is not None and (lattice.rows, lattice.columns) == (8, 8)
    looks = Looks()
    looks.read(picture, lattice)
    looks.the_empty_squares_are(picture, lattice, (3, 4))
    return looks.read(picture, lattice).cells, looks


def test_the_empty_rows_read_empty_and_every_piece_where_it_stands() -> None:
    cells, _looks = _read()
    held = ["".join("x" if cell is not None else "." for cell in line) for line in cells]
    # Rows from the top of the picture: the figure's top, spilling up from the bottom row, is read as part of it.
    assert held == ["x.x.x.x.", ".x.x.x.x", "x.x.x.x.", "........", "........", ".x.x.x.x", "x.x.x.x.", ".x.x.x.x"], held


def test_the_two_sides_chips_are_told_apart_by_the_colour_of_their_rings() -> None:
    cells, looks = _read()
    top = {cells[0][0][0], cells[1][1][0], cells[2][2][0]}
    bottom = {cells[5][1][0], cells[6][0][0], cells[7][1][0]}
    assert len(top) == 1 and len(bottom) == 1 and top != bottom, (top, bottom, looks.side_colours)


def test_she_sits_at_it_as_the_board_of_two_sides_it_is() -> None:
    from core.perception.a_board_seen import a_board_in

    picture = np.asarray(Image.open(PICTURE).convert("RGB"))
    seat = playing.a_board_of_two_sides_at_its_start(picture)
    assert seat is not None and (seat.rows, seat.columns) == (8, 8)
    # As the page's own reader saw it (core/perception/a_board_seen.py): her side at the bottom, a figure on each side.
    told = a_board_in(picture).says().splitlines()
    mine = {(seat.rows - 1 - r, c) for r, line in enumerate(told) for c, ch in enumerate(line) if ch in "bB"}
    theirs = {(seat.rows - 1 - r, c) for r, line in enumerate(told) for c, ch in enumerate(line) if ch in "aA"}
    assert {sq for sq, (side, _k) in seat.start.items() if side == 0} == mine
    assert {sq for sq, (side, _k) in seat.start.items() if side == 1} == theirs
    figures = {sq for sq, (_side, kind) in seat.start.items()
               if sum(1 for _s, k in seat.start.values() if k == kind) == 1}
    assert figures == {(seat.rows - 1 - r, c) for r, line in enumerate(told) for c, ch in enumerate(line) if ch in "AB"}
