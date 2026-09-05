"""Finding the laid-out thing inside a page that is mostly not it.

A reading of a screen is a reading of everything on it — the heading, the
score, the navigation, the footer, the cookie notice — and somewhere in the
middle, the thing she is actually acting on. Handed all of it as one
arrangement, every part of the machinery downstream is working on the wrong
object: the shape is called open because two hundred places is not small, no
rule about movement can match because most of the page never moves, and two
readings a second apart disagree about how many rows there are because the
furniture drifts.

LIVE 2026-08-29 on play2048.co: readings of 12x17 and then 7x7 of a board that
is four by four. After eighty-four moves, "how this moves is not worked out
yet". With nothing worked out there is nothing to look ahead over, so every
single move cost a full language generation — five seconds of prefill and up
to twenty of decode, about twenty-eight seconds a move, where the same loop
plays several a second once it has a model.

What tells the thing from the page is that it is a LATTICE. Its rows are
evenly spaced, its columns are evenly spaced, and the spacings are the same
ones repeated — because it was laid out by something that draws grids, and the
prose around it was not. Nothing here knows what a game is. It knows that a
regular block of cells is one object and a paragraph is not, which is as true
of a spreadsheet, a calendar, a seating plan or a set of search filters.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from core.perception.what_is_there import Arrangement, Cell

__all__ = ["ENOUGH_TO_BE_A_LATTICE", "REGULAR_ENOUGH", "the_thing_itself"]

logger = logging.getLogger("Aura.TheThingItself")

#: The least a block can be and still be a thing laid out rather than a few
#: words that happen to line up. Two by two is a coincidence; three by three
#: is a grid, and it is the smallest board anybody plays on.
ENOUGH_TO_BE_A_LATTICE = 3

#: How close two spacings have to be to count as the same one, as a share of
#: the spacing itself. Text rendering wobbles by a pixel or two and a reading
#: measures from the middle of a word, so exact equality never happens.
REGULAR_ENOUGH = 0.25

#: How full a candidate block has to be. A board mid-game has empty places in
#: it; a region of prose that happens to span a rectangle has almost nothing
#: on the lattice points.
DENSE_ENOUGH = 0.5


def _runs_of_regular_spacing(values: Sequence[float]) -> list[tuple[int, int]]:
    """Every stretch of these positions that is evenly spaced.

    Returned as index ranges into ``values``, which must be sorted. A stretch
    of three or more positions with the same gap between each is a lattice
    along one axis; anything shorter is not evidence of one.

    Every such stretch, not only the longest. A page's furniture can be spaced
    loosely enough that the whole width reads as one run — measured on
    play2048.co, six columns whose gaps ranged 0.115 to 0.143 all counted as
    the same spacing — and the board is a tighter lattice sitting inside it.
    Offering only the maximal run hides the thing inside the page it is on.
    """
    if len(values) < ENOUGH_TO_BE_A_LATTICE:
        return []
    gaps = [values[n + 1] - values[n] for n in range(len(values) - 1)]
    runs: list[tuple[int, int]] = []
    for start in range(len(gaps)):
        for end in range(start + ENOUGH_TO_BE_A_LATTICE - 2, len(gaps)):
            stretch = gaps[start : end + 1]
            widest, narrowest = max(stretch), min(stretch)
            if widest - narrowest > REGULAR_ENOUGH * max(narrowest, 1e-9):
                break
            runs.append((start, end + 1))
    return runs


def _trimmed(
    reading: Arrangement, first: int, last: int, across: tuple[int, int], *, down: bool
) -> tuple[int, int]:
    """Drop rows or columns at the edges of a block that hold nothing.

    A regular spacing can run on past the thing into the page beside it, and
    an empty margin is not part of what she is acting on however evenly it is
    spaced.
    """
    other_first, other_last = across

    def holds(index: int) -> bool:
        for other in range(other_first, other_last + 1):
            row, column = (index, other) if down else (other, index)
            if reading.at(row, column) is not None:
                return True
        return False

    while first < last and not holds(first):
        first += 1
    while last > first and not holds(last):
        last -= 1
    return first, last


def _same_place(one: float, other: float, pitch: float) -> bool:
    """Whether two measured positions are the same place on the lattice."""
    return abs(one - other) <= REGULAR_ENOUGH * max(pitch, 1e-9)


def _where_it_was(
    reading: Arrangement, before: Arrangement
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """The block the last reading was cropped to, found again in this one.

    A thing does not change size between two glances at it. What changes is
    how much of it happens to be occupied, and a board whose top row has just
    emptied has no top row for anything to infer — so found afresh each time,
    the thing shrinks and grows under her and no rule survives the comparison.
    """
    if before is None or not before.down_at or not before.across_at:
        return None
    if before.rows * before.columns < ENOUGH_TO_BE_A_LATTICE ** 2:
        return None
    down, across = list(reading.down_at), list(reading.across_at)
    if not down or not across:
        return None
    row_pitch = (before.down_at[-1] - before.down_at[0]) / max(1, before.rows - 1)
    column_pitch = (before.across_at[-1] - before.across_at[0]) / max(1, before.columns - 1)

    def find(want: float, among: list[float], pitch: float) -> int | None:
        for index, value in enumerate(among):
            if _same_place(value, want, pitch):
                return index
        return None

    top = find(before.down_at[0], down, row_pitch)
    bottom = find(before.down_at[-1], down, row_pitch)
    left = find(before.across_at[0], across, column_pitch)
    right = find(before.across_at[-1], across, column_pitch)
    if None in (top, bottom, left, right) or top > bottom or left > right:
        return None
    return (top, bottom), (left, right)


def the_thing_itself(
    reading: Arrangement, like: Arrangement | None = None
) -> Arrangement:
    """The largest regular block inside a reading, or the reading unchanged.

    Unchanged when there is no block worth calling one, because cropping on no
    evidence is how a reading of a page becomes a reading of nothing. A thing
    that IS the whole reading comes back as itself.

    ``like`` is the block the last reading was cropped to. Found again rather
    than worked out afresh, because a thing does not change size between two
    glances at it, and re-deciding every time makes it grow and shrink under
    her.
    """
    if reading is None or reading.rows < ENOUGH_TO_BE_A_LATTICE:
        return reading
    if reading.columns < ENOUGH_TO_BE_A_LATTICE:
        return reading

    down = list(reading.down_at) or [float(n) for n in range(reading.rows)]
    across = list(reading.across_at) or [float(n) for n in range(reading.columns)]
    if len(down) != reading.rows or len(across) != reading.columns:
        return reading

    held_before = _where_it_was(reading, like) if like is not None else None
    if held_before is not None:
        (top, bottom), (left, right) = held_before
        return _cropped_to(reading, top, bottom, left, right)

    best: tuple[int, tuple[int, int], tuple[int, int]] | None = None
    for row_run in _runs_of_regular_spacing(down) or [(0, reading.rows - 1)]:
        for column_run in _runs_of_regular_spacing(across) or [(0, reading.columns - 1)]:
            rows = row_run[1] - row_run[0] + 1
            columns = column_run[1] - column_run[0] + 1
            if rows < ENOUGH_TO_BE_A_LATTICE or columns < ENOUGH_TO_BE_A_LATTICE:
                continue
            held = sum(
                1
                for row in range(row_run[0], row_run[1] + 1)
                for column in range(column_run[0], column_run[1] + 1)
                if reading.at(row, column) is not None
            )
            if held < DENSE_ENOUGH * rows * columns:
                continue
            # The biggest block that is dense enough, not the densest.
            #
            # Density already rules out prose, and trimming already removes an
            # empty margin, so what is left to choose between are real
            # candidates — and preferring the densest of those crops a row off
            # a board whose top row is nearly empty, which is most boards for
            # most of a game. LIVE 2026-08-29: "the thing itself is 3x4 inside
            # a reading of 4x4", losing a row she was playing on.
            if best is None or rows * columns > best[0]:
                best = (rows * columns, row_run, column_run)

    if best is None:
        return reading
    _size, (top, bottom), (left, right) = best
    top, bottom = _trimmed(reading, top, bottom, (left, right), down=True)
    left, right = _trimmed(reading, left, right, (top, bottom), down=False)
    if bottom - top + 1 < ENOUGH_TO_BE_A_LATTICE or right - left + 1 < ENOUGH_TO_BE_A_LATTICE:
        return reading
    if (bottom - top + 1, right - left + 1) == (reading.rows, reading.columns):
        return reading

    logger.info(
        "the thing itself is %dx%d inside a reading of %dx%d",
        bottom - top + 1, right - left + 1, reading.rows, reading.columns,
    )
    return _cropped_to(reading, top, bottom, left, right)


def _cropped_to(
    reading: Arrangement, top: int, bottom: int, left: int, right: int
) -> Arrangement:
    """The block of a reading between those rows and columns, as its own thing."""
    return Arrangement(
        rows=bottom - top + 1,
        columns=right - left + 1,
        cells=tuple(
            Cell(row=cell.row - top, column=cell.column - left, says=cell.says, at=cell.at)
            for cell in reading.cells
            if top <= cell.row <= bottom and left <= cell.column <= right
        ),
        down_at=tuple(list(reading.down_at)[top : bottom + 1]),
        across_at=tuple(list(reading.across_at)[left : right + 1]),
    )
