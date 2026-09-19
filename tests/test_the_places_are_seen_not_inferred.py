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


def test_two_things_that_look_alike_are_read_rather_than_guessed(monkeypatch):
    """A 256 was read as a 128 in half the glances of a game.

    Their surrounds are a few units apart and the digits are a tenth of the
    square, so nearest-look alone answers with whichever she happened to read
    first. When two things she has read are both this close, she cannot tell
    them apart and the place is read again.
    """
    from core.perception import what_the_pixels_show as pixels

    looker = Looker()
    one = _a_board({(0, 0): (114, 207, 237)})
    other = _a_board({(0, 0): (97, 204, 237)})
    grid = grids_in(panels_in(one))[0]
    looker.read(one, words=_words_at(grid, {(0, 0): "128"}))
    looker.read(other, words=_words_at(grid, {(0, 0): "256"}))
    # Both remembered, and both within the distance of this look.
    look = looker._look_of(other, grid.place(0, 0))
    assert looker.recognised(look) is None

    # And the reading of that board does not claim the wrong one.
    monkeypatch.setattr(pixels, "recognize_text", lambda image: [])
    says = looker.read(other)["grids"][0]["says"]
    assert says[0] in ("", "256")


def _growing(scale: float) -> np.ndarray:
    """A board with one thing at (1, 1) drawn at this share of its square."""
    picture = _a_board({})
    pitch, size, left, top = 110, 96, 47, 117
    x, y = left + 1 * pitch, top + 1 * pitch
    half = int(size * scale / 2)
    if half > 0:
        middle_x, middle_y = x + size // 2, y + size // 2
        _fill(picture, (middle_x - half, middle_y - half), (middle_x + half, middle_y + half), (97, 204, 237))
    return picture


def test_a_thing_still_growing_into_its_place_is_not_a_settled_reading(monkeypatch):
    """LIVE 2026-09-18: a freshly made 256 was missing from readings called settled.

    A tile that has just been made grows into its square. At its first
    frames it reads as an empty place, and two of those in a row say the
    same thing, so a reading was settled while the tile was still arriving.
    What has to stop is how each place looks, as well as what it says.
    """
    import time

    from core.perception import what_the_pixels_show as pixels

    frames = [_growing(scale) for scale in (0.1, 0.2, 0.45, 0.8, 1.0, 1.0, 1.0)]
    showing = {"at": -1}

    def take():
        showing["at"] = min(showing["at"] + 1, len(frames) - 1)
        return frames[showing["at"]]

    def strip(self, image, grid_, spots):
        # Recognition finds the number once the thing is its full size.
        if showing["at"] >= 4 and (1, 1) in spots:
            return {(1, 1): "256"}
        return {}

    monkeypatch.setattr(pixels, "recognize_text", lambda image: [])
    monkeypatch.setattr(Looker, "_read_as_a_strip", strip)
    looker = Looker()
    # What it says alone agrees across the first two frames: both empty.
    early = [pixels.what_a_reading_says(Looker().read(frame)) for frame in frames[:2]]
    assert early[0] == early[1]

    _picture, reading, still = pixels.settled_reading(
        take, looker, wait=True, within_s=5.0, began=time.monotonic()
    )
    assert still
    assert reading["grids"][0]["says"][1 * 4 + 1] == "256"


def test_a_place_that_cannot_be_read_is_not_the_same_as_an_empty_one():
    from core.perception.what_the_pixels_show import what_a_reading_says

    empty = {"grids": [{"rows": 1, "columns": 2, "says": ["", "2"], "unsure": []}], "layout": []}
    arriving = {"grids": [{"rows": 1, "columns": 2, "says": ["", "2"], "unsure": [[0, 0]]}], "layout": []}
    assert what_a_reading_says(empty) != what_a_reading_says(arriving)


def test_a_look_that_resembles_something_read_is_never_what_empty_looks_like(monkeypatch):
    """LIVE 2026-09-18: the empty look became a 2, and every 2 read as nothing.

    A dimmed 4 learned under a finished game's message sits within reach of a
    plain 2, so every 2 was a look she could not tell apart and none of them
    was recognised. Silent and many, early in a game, they outnumbered the
    empty places and were taken for what empty looks like.
    """
    from core.perception import what_the_pixels_show as pixels

    plain_two, dimmed_four = (218, 228, 238), (212, 224, 238)
    looker = Looker()
    # Both learned, and both within the distance of the plain 2's look.
    first = _a_board({(0, 0): plain_two})
    grid = grids_in(panels_in(first))[0]
    looker.read(first, words=_words_at(grid, {(0, 0): "2"}))
    second = _a_board({(0, 0): dimmed_four})
    looker.read(second, words=_words_at(grid, {(0, 0): "4"}))
    look = looker._look_of(first, grid.place(0, 0))
    assert looker.recognised(look) is None

    # A board with more of those 2s than empty places, and nothing read.
    twos = {(row, column): plain_two for row in range(4) for column in range(4) if (row + column) % 3}
    board = _a_board(twos)
    monkeypatch.setattr(pixels, "recognize_text", lambda image: [])
    monkeypatch.setattr(Looker, "_read_as_a_strip", lambda self, image, grid_, spots: {})
    reading = looker.read(board)
    says, unsure = reading["grids"][0]["says"], reading["grids"][0]["unsure"]
    # None of the 2s is taken for an empty place: each is either read or,
    # here where nothing can be read, said to be unsure.
    for (row, column) in twos:
        assert says[row * 4 + column] or [row, column] in unsure
    empty_look = looker.blank.get((4, 4))
    assert empty_look is not None
    assert looker._apart(empty_look, looker._look_of(board, grid.place(0, 0))) < 1.0  # (0, 0) is empty


def test_nothing_is_learned_from_a_grid_with_something_lying_across_it():
    looker = Looker()
    dimmed = _a_board({(1, 1): (212, 224, 238)})
    grid = grids_in(panels_in(dimmed))[0]
    across = {
        "text": "Game over!",
        "center_x": grid.across_at[1],
        "center_y": grid.down_at[1],
        "width": grid.cell_width * 3.0,
        "height": grid.cell_height * 0.4,
    }
    words = _words_at(grid, {(2, 2): "Tr."}) + [across]
    reading = looker.read(dimmed, words=words)
    assert reading["grids"][0]["covered"] is True
    assert looker.seen == []


def test_a_place_that_could_not_be_read_is_not_an_empty_place():
    """LIVE 2026-09-18: an unread 32 was an empty place, and the move read as a 32 vanishing."""
    says = ["4", "8", "16", ""] + [""] * 12
    observation = _observation((0.3, 0.42, 0.54, 0.66), (0.2, 0.35, 0.5, 0.65), says)
    observation["grids"][0]["unsure"] = [[0, 3]]
    seen = what_is_there(observation, None)
    assert seen.unknown == ((0, 3),)
    assert seen.at(0, 3) is None
    # What it is not is not part of what it is: two readings of one state
    # still compare equal.
    observation["grids"][0]["unsure"] = []
    assert what_is_there(observation, None) == seen


def test_a_reading_with_a_place_it_could_not_read_is_not_settled(monkeypatch):
    """What it says and how it looks can both stop changing with a place unread."""
    import time

    from core.perception import what_the_pixels_show as pixels

    frames = [_growing(1.0)] * 6
    showing = {"at": -1}

    def take():
        showing["at"] = min(showing["at"] + 1, len(frames) - 1)
        return frames[showing["at"]]

    def strip(self, image, grid_, spots):
        # Recognition gets it on the fourth picture of the same still thing.
        if showing["at"] >= 3 and (1, 1) in spots:
            return {(1, 1): "32"}
        return {}

    monkeypatch.setattr(pixels, "recognize_text", lambda image: [])
    monkeypatch.setattr(Looker, "_read_as_a_strip", strip)
    _picture, reading, still = pixels.settled_reading(
        take, Looker(), wait=True, within_s=5.0, began=time.monotonic()
    )
    assert still
    assert reading["grids"][0]["says"][1 * 4 + 1] == "32"
    assert not pixels.anything_unread(reading)


def test_a_pair_with_a_place_she_could_not_read_teaches_nothing():
    from screen_pursuit_support import pursuit_source
    from source_contract import in_order

    in_order(
        pursuit_source(),
        'getattr(pending["arranged"], "unknown", ())',
        'dropped["a place she could not read"] += 1',
        'knows.watched(pending["arranged"], previous.chosen.name, laid_out)',
    )


def test_a_board_with_something_lying_over_it_is_not_a_state():
    """LIVE 2026-09-18: "16 Tr. 4" read off a board under "Try again" went to her rule."""
    says = ["2", "4", "8", "16"] + [""] * 12
    observation = _observation((0.3, 0.42, 0.54, 0.66), (0.2, 0.35, 0.5, 0.65), says)
    observation["grids"][0]["covered"] = True
    seen = what_is_there(observation, None)
    assert len(seen.unknown) == 16


def test_one_empty_place_left_is_known_to_be_empty(monkeypatch):
    """LIVE 2026-09-19: with one gap on the board it was "could not be read" on every look."""
    from core.perception import what_the_pixels_show as pixels

    colours = [(218, 228, 238), (200, 224, 237), (121, 177, 242), (99, 150, 245)]
    full = {(row, column): colours[(row + column) % 4] for row in range(4) for column in range(4)}
    del full[(2, 0)]
    board = _a_board(full)
    # A digit in each numbered place, so they are not flat either.
    for (row, column) in full:
        x, y = 47 + column * 110 + 40, 117 + row * 110 + 40
        board[y : y + 16, x : x + 4] = (80, 90, 100)
    monkeypatch.setattr(pixels, "recognize_text", lambda image: [])
    # Recognition reads every numbered place and finds nothing in the gap.
    monkeypatch.setattr(
        Looker, "_read_as_a_strip",
        lambda self, image, grid_, spots: {spot: "8" for spot in spots if spot != (2, 0)},
    )
    looker = Looker()
    reading = looker.read(board)
    unsure = reading["grids"][0]["unsure"]
    assert [2, 0] not in unsure
    assert reading["grids"][0]["says"][2 * 4 + 0] == ""
    # And what empty looks like is learned from it, for the next reading.
    assert looker.blank.get((4, 4)) is not None


def test_a_flat_colour_in_a_grid_with_no_words_is_not_assumed_empty(monkeypatch):
    """Where no place carries words, a flat colour may be all the content there is."""
    from core.perception import what_the_pixels_show as pixels

    board = _a_board({(0, 0): (121, 177, 242), (1, 1): (99, 150, 245)})
    monkeypatch.setattr(pixels, "recognize_text", lambda image: [])
    monkeypatch.setattr(Looker, "_read_as_a_strip", lambda self, image, grid_, spots: {})
    reading = Looker().read(board)
    assert [0, 0] in reading["grids"][0]["unsure"]
