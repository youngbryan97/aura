"""Tiles that will not read are not what an empty place looks like.

Her eyes learn the empty look from the commonest look among places that said
nothing. Tiles that would not read say nothing too, and when they outnumber
the empty places their look was taken for empty: every one of them read as an
empty place from then on. LIVE 2026-09-24, "her rule missed on up: (0,1) said
'8' saw None", an 8 at rest read as an empty place.
"""
from __future__ import annotations

from core.perception import what_the_pixels_show as pixels
from core.perception.what_the_pixels_show import Looker

from tests.test_the_places_are_seen_not_inferred import _a_board

EIGHT = (121, 177, 242)
EMPTY = {(0, 0), (1, 2), (3, 3)}


def _a_board_of_eights():
    places = [(row, column) for row in range(4) for column in range(4) if (row, column) not in EMPTY]
    board = _a_board({spot: EIGHT for spot in places})
    for row, column in places:
        x, y = 47 + column * 110 + 48, 117 + row * 110 + 48
        board[y - 14 : y + 14, x - 9 : x + 9] = (250, 250, 250)
    return board


def test_the_empty_look_is_the_one_with_nothing_drawn_in_it(monkeypatch):
    monkeypatch.setattr(pixels, "recognize_text", lambda image: [])
    monkeypatch.setattr(Looker, "_read_as_a_strip", lambda self, image, grid, spots: {})
    looker = Looker()
    grid = looker.read(_a_board_of_eights())["grids"][0]
    unsure = {tuple(spot) for spot in grid["unsure"]}
    assert len(unsure) == 13, "thirteen tiles that did not read were read as empty"
    assert not unsure & EMPTY


def test_a_learned_empty_look_is_not_replaced_by_tiles_that_would_not_read(monkeypatch):
    monkeypatch.setattr(pixels, "recognize_text", lambda image: [])
    monkeypatch.setattr(Looker, "_read_as_a_strip", lambda self, image, grid, spots: {})
    looker = Looker()
    looker.read(_a_board({}))
    looker.read(_a_board_of_eights())
    grid = looker.read(_a_board_of_eights())["grids"][0]
    assert len(grid["unsure"]) == 13
