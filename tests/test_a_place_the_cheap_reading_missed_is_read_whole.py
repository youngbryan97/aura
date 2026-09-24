"""A place the cheap reading could not read sends her back to a whole reading.

Where the words around a grid have not changed, her eyes reuse them and read
each place from how it looks, or from a strip of places laid side by side.
The strip reader depends on its company: a lone 8 came back as "00" or as
nothing. With the words around the grid unchanged, nothing else ever read that
place again. LIVE 2026-09-24 an 8 at rest went unread for 65 pictures running,
and 472 of her 794 moves taught her nothing.
"""
from __future__ import annotations

from core.perception import what_the_pixels_show as pixels
from core.perception.what_the_pixels_show import Looker, grids_in, panels_in

from tests.test_the_places_are_seen_not_inferred import _a_board

TWO, EIGHT = (218, 228, 238), (121, 177, 242)


def _with_an_eight(board):
    """A tile with something drawn in it; a flat square is how an empty place looks."""
    x, y = 47 + 110 + 48, 117 + 110 + 48
    board[y - 14 : y + 14, x - 9 : x + 9] = (250, 250, 250)
    return board


def _words(grid, spots):
    return [{"text": "Score 12", "center_x": 0.78, "center_y": 0.08, "width": 0.2, "height": 0.04}] + [
        {
            "text": said,
            "center_x": grid.across_at[column],
            "center_y": grid.down_at[row],
            "width": grid.cell_width * 0.3,
            "height": grid.cell_height * 0.3,
        }
        for (row, column), said in spots.items()
    ]


def _eyes(monkeypatch, says_on_each_reading):
    readings: list[int] = []

    def whole(image):
        readings.append(1)
        grid = grids_in(panels_in(image))[0]
        return _words(grid, says_on_each_reading(len(readings)))

    monkeypatch.setattr(pixels, "recognize_text", whole)
    monkeypatch.setattr(Looker, "_read_as_a_strip", lambda self, image, grid, spots: {})
    return readings


def test_a_new_look_the_strip_cannot_read_is_read_whole(monkeypatch):
    readings = _eyes(monkeypatch, lambda n: {(0, 0): "2"} if n == 1 else {(0, 0): "2", (1, 1): "8"})
    looker = Looker()
    looker.read(_a_board({(0, 0): TWO}))
    # An 8 arrives inside the grid; the words around it are the same words.
    again = looker.read(_with_an_eight(_a_board({(0, 0): TWO, (1, 1): EIGHT})))
    grid = again["grids"][0]
    assert grid["says"][1 * 4 + 1] == "8"
    assert not grid["unsure"]
    assert len(readings) == 2, "the whole picture is read again for the place the strip missed"


def test_a_board_the_cheap_reading_reads_in_full_costs_no_whole_reading(monkeypatch):
    readings = _eyes(monkeypatch, lambda n: {(0, 0): "2", (1, 1): "8"})
    looker = Looker()
    board = _with_an_eight(_a_board({(0, 0): TWO, (1, 1): EIGHT}))
    looker.read(board)
    looker.read(board)
    assert len(readings) == 1


def test_what_the_cheap_reading_held_is_not_learned_when_it_is_read_again(monkeypatch):
    """The first pass is a draft: its lessons go with it."""
    _eyes(monkeypatch, lambda n: {(0, 0): "2"} if n == 1 else {(0, 0): "2", (1, 1): "8"})
    looker = Looker()
    looker.read(_a_board({(0, 0): TWO}))
    looker.read(_with_an_eight(_a_board({(0, 0): TWO, (1, 1): EIGHT})), learn=False)
    held = [what for kind, what in looker.held_lessons if kind == "seen"]
    assert sorted(text for _look, text in held) == ["2", "8"]
