"""The places of a laid-out thing are in the picture, empty ones included.

LIVE 2026-09-17, the 2048 desktop app. A reading made only of recognised text
had no way to see a square with nothing in it, so the grid was inferred from
where tiles had been seen over many moves, and after thirty moves she had
learned nothing about how the board moved: every pair was "not the thing
itself" or "a different frame". And macOS Vision returned a "2" and an "8"
from a clean capture while leaving out a "4" and a "2" in plain sight.

These are drawn pictures, not screenshots, so what is asserted is the reading
of structure: panels of one size at one pitch are a grid, a place with nothing
in it is still a place, text lying across the places belongs to none of them,
and a frame held between glances survives the jitter of finding an edge a
pixel over.

Aura's own process refuses OpenCV on macOS (its media stack collides with the
one speech recognition loads), and the first live look found 0 panels because
the reader had quietly needed it. The pictures here are read with OpenCV
refused the same way.
"""
from __future__ import annotations

import builtins

import numpy as np
import pytest

from core.perception.what_the_pixels_show import Looker, grids_in, panels_in
from core.perception.where_it_responds import what_is_there


@pytest.fixture(autouse=True)
def _no_opencv(monkeypatch):
    real = builtins.__import__

    def refusing(name, *args, **kwargs):
        if name.split(".", 1)[0] == "cv2":
            raise ImportError("OpenCV is refused in this process")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refusing)


def _fill(picture: np.ndarray, corner: tuple[int, int], far: tuple[int, int], colour) -> None:
    (x0, y0), (x1, y1) = corner, far
    picture[y0 : y1 + 1, x0 : x1 + 1] = colour


def _a_board(filled: dict[tuple[int, int], tuple[int, int, int]], side: int = 4) -> np.ndarray:
    """A window with a title bar of text, a button, and a grid of squares."""
    picture = np.full((620, 520, 3), (238, 244, 249), np.uint8)
    _fill(picture, (330, 30), (480, 70), (100, 120, 140))
    _fill(picture, (40, 110), (480, 550), (160, 173, 187))
    pitch, size, left, top = 110, 96, 47, 117
    for row in range(side):
        for column in range(side):
            colour = filled.get((row, column), (180, 193, 205))
            x, y = left + column * pitch, top + row * pitch
            _fill(picture, (x, y), (x + size, y + size), colour)
    return picture


def test_a_picture_larger_than_the_working_size_is_read_the_same():
    small = _a_board({(0, 0): (218, 228, 238), (3, 2): (121, 177, 242)})
    large = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    first, second = grids_in(panels_in(small))[0], grids_in(panels_in(large))[0]
    assert (second.rows, second.columns) == (4, 4)
    assert max(abs(a - b) for a, b in zip(first.down_at, second.down_at, strict=True)) < 0.01


def test_a_grid_is_found_with_its_empty_places():
    picture = _a_board({(0, 0): (218, 228, 238), (3, 2): (121, 177, 242)})
    grids = grids_in(panels_in(picture))
    assert len(grids) == 1
    grid = grids[0]
    assert (grid.rows, grid.columns) == (4, 4)
    # Every place is there, not only the two with something in them.
    assert grid.where(grid.across_at[1], grid.down_at[2]) == (2, 1)


def test_one_button_is_not_a_grid():
    picture = np.full((300, 300, 3), 240, np.uint8)
    _fill(picture, (40, 40), (140, 90), (90, 90, 90))
    assert grids_in(panels_in(picture)) == []


def _words_at(grid, spots: dict[tuple[int, int], str]) -> list[dict]:
    return [
        {
            "text": said,
            "center_x": grid.across_at[column],
            "center_y": grid.down_at[row],
            "width": grid.cell_width * 0.3,
            "height": grid.cell_height * 0.3,
        }
        for (row, column), said in spots.items()
    ]


def test_what_each_place_says_is_read_into_it_and_empty_places_stay_empty():
    picture = _a_board({(0, 0): (218, 228, 238), (1, 3): (121, 177, 242)})
    grid = grids_in(panels_in(picture))[0]
    reading = Looker().read(picture, words=_words_at(grid, {(0, 0): "2", (1, 3): "8"}))
    says = reading["grids"][0]["says"]
    assert says[0] == "2"
    assert says[1 * 4 + 3] == "8"
    assert says.count("") == 14


def test_text_lying_across_the_places_belongs_to_none_of_them():
    picture = _a_board({(0, 0): (218, 228, 238)})
    grid = grids_in(panels_in(picture))[0]
    across = {
        "text": "Game over!",
        "center_x": grid.across_at[1],
        "center_y": grid.down_at[1],
        "width": grid.cell_width * 3.0,
        "height": grid.cell_height * 0.4,
    }
    words = _words_at(grid, {(0, 0): "2"}) + [across]
    reading = Looker().read(picture, words=words)
    assert "Game over!" not in reading["grids"][0]["says"]
    assert reading["grids"][0]["covered"] is True


def test_a_look_already_read_is_recognised_without_reading_it_again():
    colour = (121, 177, 242)
    picture = _a_board({(0, 0): colour, (2, 2): colour})
    grid = grids_in(panels_in(picture))[0]
    looker = Looker()
    looker.read(picture, words=_words_at(grid, {(0, 0): "8", (2, 2): "8"}))
    # The same look, and the recognition pass that would have read it is not
    # given the word this time.
    later = looker.read(picture, words=_words_at(grid, {(0, 0): "8"}))
    assert later["grids"][0]["says"][2 * 4 + 2] == "8"


def _observation(down, across, says):
    return {
        "ok": True,
        "text": " ".join(s for s in says if s),
        "layout": [],
        "grids": [
            {
                "rows": 4,
                "columns": 4,
                "down_at": list(down),
                "across_at": list(across),
                "cell_width": 0.15,
                "cell_height": 0.12,
                "says": list(says),
            }
        ],
    }


def test_an_arrangement_comes_from_the_grid_in_the_pixels():
    says = ["2"] + [""] * 14 + ["4"]
    seen = what_is_there(_observation((0.3, 0.42, 0.54, 0.66), (0.2, 0.35, 0.5, 0.65), says), None)
    assert (seen.rows, seen.columns) == (4, 4)
    assert seen.occupied() == 2
    assert seen.at(3, 3).says == "4"


def test_the_frame_held_between_glances_survives_an_edge_found_a_pixel_over():
    from core.perception.the_lattice_she_holds import TheLatticeSheHolds

    lattice = TheLatticeSheHolds()
    says = ["2"] + [""] * 15
    first = what_is_there(
        _observation((0.3, 0.42, 0.54, 0.66), (0.2, 0.35, 0.5, 0.65), says), None, lattice=lattice
    )
    second = what_is_there(
        _observation((0.301, 0.419, 0.541, 0.66), (0.2, 0.351, 0.5, 0.649), says),
        None,
        like=first,
        lattice=lattice,
    )
    assert second.down_at == first.down_at
    assert second.across_at == first.across_at


def test_places_seen_in_the_picture_are_a_thing_laid_out_however_few_are_filled():
    from core.skills.screen_pursuit_bearings import _is_a_thing_laid_out

    says = ["2"] + [""] * 14 + ["2"]
    seen = what_is_there(_observation((0.3, 0.42, 0.54, 0.66), (0.2, 0.35, 0.5, 0.65), says), None)
    assert seen.places_seen is True
    assert _is_a_thing_laid_out(seen)
    # Being seen is how it was read, not what it is.
    from dataclasses import replace

    assert replace(seen, places_seen=False) == seen
    assert not _is_a_thing_laid_out(replace(seen, places_seen=False))


def test_words_outside_the_grid_are_read_again_only_when_that_part_changes(monkeypatch):
    """Reading every word of a window again each glance is most of what a look costs."""
    from core.perception import what_the_pixels_show as pixels

    picture = _a_board({(0, 0): (218, 228, 238)})
    grid = grids_in(panels_in(picture))[0]
    times: list[int] = []

    def counted(image):
        times.append(1)
        return [
            {"text": "Score 12", "center_x": 0.78, "center_y": 0.08, "width": 0.2, "height": 0.04},
            {"text": "2", "center_x": grid.across_at[0], "center_y": grid.down_at[0],
             "width": grid.cell_width * 0.3, "height": grid.cell_height * 0.3},
        ]

    monkeypatch.setattr(pixels, "recognize_text", counted)
    looker = Looker()
    first = looker.read(picture)
    assert [run["text"] for run in first["layout"]].count("Score 12") == 1

    # A tile moves: inside the grid, so the words around it are the same words.
    moved = _a_board({(1, 1): (218, 228, 238)})
    again = looker.read(moved)
    assert len(times) == 1
    assert [run["text"] for run in again["layout"]].count("Score 12") == 1

    # The score changes: that is outside the grid, and it is read again.
    changed = _a_board({(1, 1): (218, 228, 238)})
    changed[70:100, 340:470] = (60, 70, 80)
    looker.read(changed)
    assert len(times) == 2


def test_a_place_she_has_read_something_from_is_not_what_empty_looks_like(monkeypatch):
    """On a board of mostly one value, the commonest unread look IS that value."""
    from core.perception import what_the_pixels_show as pixels

    pale = (232, 240, 246)
    mostly = {(row, column): pale for row in range(4) for column in range(4)} 
    del mostly[(0, 0)]
    del mostly[(3, 3)]
    picture = _a_board(mostly)
    grid = grids_in(panels_in(picture))[0]
    looker = Looker()
    # She reads one of them, so that look is known to carry something.
    looker.read(picture, words=_words_at(grid, {(1, 1): "2"}))
    monkeypatch.setattr(pixels, "recognize_text", lambda image: [])
    says = looker.read(picture)["grids"][0]["says"]
    # Fourteen places hold the same pale thing; two are empty.
    assert says.count("") == 2


def test_one_place_left_to_read_is_read_beside_the_ones_she_knows(monkeypatch):
    """Recognition reads a line, not a square: one digit alone comes back empty."""
    from core.perception import what_the_pixels_show as pixels

    picture = _a_board({(0, 0): (218, 228, 238), (1, 1): (242, 177, 121), (2, 2): (99, 177, 242)})
    grid = grids_in(panels_in(picture))[0]
    looker = Looker()
    looker.read(picture, words=_words_at(grid, {(0, 0): "2", (1, 1): "8"}))
    asked: list[list[tuple[int, int]]] = []
    real = Looker._read_as_a_strip

    def watched(self, image, grid_, spots):
        asked.append(list(spots))
        return real(self, image, grid_, spots)

    monkeypatch.setattr(Looker, "_read_as_a_strip", watched)
    monkeypatch.setattr(pixels, "recognize_text", lambda image: [])
    looker.read(picture)
    assert asked, "nothing was read as a strip"
    # The one place she has not read went in with the places she knows.
    assert len(asked[0]) >= 3
