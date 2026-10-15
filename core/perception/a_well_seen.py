"""A well seen: the cells a falling-piece game is played in, what stands in each, and the piece coming down.

A person shown a game of falling pieces sees the well before anything in it: one tall stretch of plain ground with
walls either side, cut into cells of one size, with things resting on its floor and one thing higher up that is on its
way down. That is read here from the picture alone. The ground is a colour the picture has a great deal of, in one
piece taller than it is wide. The cells are the spacing its ruled lines keep, or where it is not ruled, the size of the
smallest thing in it, and either must divide the well into a whole number of columns. What stands in a cell is told
from the ground by how much of the cell's middle is some other colour, and one kind from another by how it looks.
What rests is what reaches the floor through its neighbours; the piece is whatever else is in the well.

Nothing here knows a game: not the number of columns, nor what the pieces are made of, nor how they clear. Blocks that
fill their cells, round tokens with ground showing between them, a piece that comes down a cell at a time or a little
at each picture are all seen the same way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.verify import invariant

__all__ = ["PieceSeen", "WellSeen", "a_ground_like_a_wells", "a_well_in", "the_piece_in", "what_rests", "what_stands_in"]

Cell = tuple[int, int]  # column, row; row 0 at the top

#: The fewest columns a well has and the most; the fewest rows.
FEWEST_COLUMNS, MOST_COLUMNS, FEWEST_ROWS = 4, 16, 6
#: How far from the ground's colour a pixel may be and still be ground, over its three channels together.
SAME_GROUND = 40
#: How much of the picture the ground's one piece must cover to be a well's, and how much of its own box.
OF_THE_PICTURE, OF_ITS_BOX = 0.04, 0.35
#: How much taller than wide a well is, at the least.
TALLER_THAN_WIDE = 0.8
#: How far a well's width may be from a whole number of cells, in cells.
A_WHOLE_NUMBER = 0.3
#: How much of a cell's middle is some other colour than the ground where the cell holds something.
HOLDS = 0.25
#: How unlike two looks may be and still be one kind: the mean difference between their small pictures, of 255.
SAME_LOOK = 20.0
#: The small picture a cell's look is kept as, a side of it in pixels.
LOOK_PX = 6
#: How much of a cell, each way, a thing must span to be part of a piece and not a speck, a line or a wall's edge.
A_FAIR_PART = 0.4
#: How far in from the well's sides its walls' own edges may show, in cells.
THE_WALLS_EDGE = 0.15
#: How thick a ruled line may be, in pixels each side of its middle.
A_LINE_PX = 2
_SQUARE = np.ones((3, 3), dtype=bool)
#: How many of the commonest colours are tried as the ground.
GROUNDS_TRIED = 4


@dataclass
class WellSeen:
    """Where a well is in the picture, how it is cut into cells, and the looks its kinds have been told apart by."""

    left: float
    top: float
    wide: float
    tall: float
    columns: int
    rows: int
    ground: tuple[int, int, int]
    #: One small picture for each kind seen, in the order they were first seen: a cell's kind is its place here.
    looks: list[np.ndarray] = field(default_factory=list)

    def box(self, column: float, row: float) -> tuple[int, int, int, int]:
        """A cell's left, top, right and bottom in the picture's pixels."""
        return (round(self.left + column * self.wide), round(self.top + row * self.tall),
                round(self.left + (column + 1) * self.wide), round(self.top + (row + 1) * self.tall))

    def centre(self, column: float, row: float) -> tuple[float, float]:
        return self.left + (column + 0.5) * self.wide, self.top + (row + 0.5) * self.tall

    def kept(self) -> list[float]:
        """The well as numbers, for keeping between one stretch of play and the next."""
        return [self.left, self.top, self.wide, self.tall, self.columns, self.rows, *self.ground]

    def valid(self) -> bool:
        return (all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in (self.left, self.top, self.wide, self.tall))
                and min(self.wide, self.tall) > 0
                and type(self.columns) is int and FEWEST_COLUMNS <= self.columns <= MOST_COLUMNS
                and type(self.rows) is int and self.rows >= FEWEST_ROWS
                and math.isfinite(self.left+self.columns*self.wide)
                and math.isfinite(self.top+self.rows*self.tall)
                and len(self.ground) == 3 and all(type(v) is int and 0 <= v <= 255 for v in self.ground))

    @classmethod
    def from_kept(cls, kept: Any) -> WellSeen | None:
        try:
            if any(isinstance(n, bool) for n in kept):
                return None
            left, top, wide, tall, columns, rows, red, green, blue = (float(n) for n in kept)
        except (TypeError, ValueError):
            return None
        if (not all(math.isfinite(n) for n in (left, top, wide, tall, columns, rows, red, green, blue))
                or any(n != int(n) for n in (columns, rows, red, green, blue))):
            return None
        well = cls(left, top, wide, tall, int(columns), int(rows), (int(red), int(green), int(blue)))
        return well if well.valid() else None


@dataclass
class PieceSeen:
    """The piece coming down: its parts by column and row from its own corner with each part's kind, the column its
    corner is over, how far down its top is in cells, and whether all of it is in the well yet."""

    parts: tuple[tuple[int, int, int], ...]
    column: int
    row: float
    whole: bool

    def shape(self) -> tuple[int, int]:
        return (1 + max(dc for dc, _dr, _k in self.parts), 1 + max(dr for _dc, dr, _k in self.parts))


def _picture(picture: Any) -> np.ndarray:
    return np.asarray(picture)[..., :3].astype(np.int16)


def _is_ground(image: np.ndarray, ground: tuple[int, int, int]) -> np.ndarray:
    return np.abs(image - np.asarray(ground, dtype=np.int16)).sum(axis=2) < SAME_GROUND


def _no_lines(there: np.ndarray) -> np.ndarray:
    """What is there without what is only a few pixels thick: ruled lines and specks go, things stay."""
    from scipy import ndimage

    return ndimage.binary_opening(there, _SQUARE, iterations=A_LINE_PX)


def _commonest_colours(image: np.ndarray) -> list[tuple[int, int, int]]:
    """The colours the picture has most of, each as the middle colour of its pixels."""
    coarse = (image >> 3).astype(np.int32)
    keys = (coarse[..., 0] << 10) | (coarse[..., 1] << 5) | coarse[..., 2]
    counts = np.bincount(keys.ravel(), minlength=1 << 15)
    found = []
    for key in np.argsort(counts)[::-1][:GROUNDS_TRIED]:
        if counts[key] < OF_THE_PICTURE * keys.size:
            break
        of_it = image[keys == key]
        found.append(tuple(int(v) for v in np.median(of_it, axis=0)))
    return found


def _the_ground_box(image: np.ndarray, ground: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    """The box of the largest single piece of ground, where it is big, tall and mostly ground: left, top, right, bottom."""
    from scipy import ndimage

    # Lines ruled across the ground, of whatever colour, do not cut it into pieces: gaps a few pixels wide are closed.
    labelled, count = ndimage.label(ndimage.binary_closing(_is_ground(image, ground), _SQUARE, iterations=A_LINE_PX))
    if not count:
        return None
    sizes = np.bincount(labelled.ravel())[1:]
    biggest = int(np.argmax(sizes)) + 1
    down, across = ndimage.find_objects(labelled)[biggest - 1]
    wide, tall = across.stop - across.start, down.stop - down.start
    if sizes[biggest - 1] < OF_THE_PICTURE * labelled.size or tall < TALLER_THAN_WIDE * wide:
        return None
    if sizes[biggest - 1] < OF_ITS_BOX * wide * tall:
        return None
    return across.start, down.start, across.stop, down.stop


def _ruled(profile: np.ndarray) -> tuple[float, float] | None:
    """Where a profile of edge strengths is ruled at one spacing: the first line and the spacing."""
    if profile.size < 12 or float(profile.max()) <= 0.0:
        return None
    strong = max(2.0, 0.5 * float(np.percentile(profile, 99)))
    at = [i for i in range(1, profile.size - 1) if profile[i] >= strong and profile[i] == profile[i - 1:i + 2].max()]
    lines: list[float] = []
    for one in at:
        if lines and one - lines[-1] <= 3:
            lines[-1] = (lines[-1] + one) / 2.0
        else:
            lines.append(float(one))
    gaps = np.diff(lines)
    if gaps.size < 2:
        return None
    spacing = float(np.median(gaps))
    if spacing < 6 or float(np.mean(np.abs(gaps - spacing) <= 2.0)) < 0.7:
        return None
    return lines[0], spacing


def _the_pitch(image: np.ndarray, ground: tuple[int, int, int], box: tuple[int, int, int, int]) -> float | None:
    """How far apart the cells of an unruled well are: the least distance between the middles of two things that
    stand side by side or one on the other. A thing may be narrower than its cell; two of them are a cell apart."""
    from scipy import ndimage

    left, top, right, bottom = box
    labelled, count = ndimage.label(_no_lines(~_is_ground(image[top:bottom, left:right], ground)))
    least = (right - left) / (2.0 * MOST_COLUMNS)
    things = [((across.start + across.stop) / 2.0, (down.start + down.stop) / 2.0, across.stop - across.start)
              for down, across in ndimage.find_objects(labelled)
              if across.stop - across.start >= least and down.stop - down.start >= least]
    apart = [max(abs(x - u), abs(y - v)) for i, (x, y, size) in enumerate(things) for u, v, _other in things[i + 1:]
             if min(abs(x - u), abs(y - v)) <= 0.2 * size < 0.6 * size <= max(abs(x - u), abs(y - v))]
    return float(min(apart)) if count and apart else None


def a_well_in(picture: Any, known: WellSeen | None = None) -> WellSeen | None:
    """The well in a picture, where there is one. A well found before is looked for in the same place first: its
    ground must still be there."""
    image = _picture(picture)
    if known is not None and known.valid():
        left, top, right, bottom = known.box(0, 0)[0], known.box(0, 0)[1], known.box(known.columns, 0)[0], \
            known.box(0, known.rows)[1]
        if 0 <= top < bottom <= image.shape[0] and 0 <= left < right <= image.shape[1] and \
                float(_is_ground(image[top:bottom, left:right], known.ground).mean()) >= 0.1:
            return known
    for ground in _commonest_colours(image):
        box = _the_ground_box(image, ground)
        if box is None:
            continue
        well = _cut_into_cells(image, ground, box)
        if well is not None:
            return well
    return None


def a_ground_like_a_wells(picture: Any) -> bool:
    """Whether a picture has a stretch of plain ground big and tall enough to be a well's, whatever is or is not in
    it yet: an empty well that is not ruled shows its cells only once something stands in it."""
    image = _picture(picture)
    return any(_the_ground_box(image, ground) is not None for ground in _commonest_colours(image))


def _cut_into_cells(image: np.ndarray, ground: tuple[int, int, int], box: tuple[int, int, int, int]) -> WellSeen | None:
    left, top, right, bottom = box
    grey = image[top:bottom, left:right].mean(axis=2)
    # Lines ruled the whole way across show in the middle picture of every row; a thing standing in the way does not.
    across = _ruled(np.median(np.abs(np.diff(grey, axis=1)), axis=0))
    down = _ruled(np.median(np.abs(np.diff(grey, axis=0)), axis=1))
    if across is not None:
        wide = across[1]
        tall = down[1] if down is not None and abs(down[1] - wide) <= 0.25 * wide else wide
        first = left + across[0] + 0.5
        start = first - wide * round((first - left) / wide)
        first_down = top + (down[0] + 0.5 if down is not None else 0.0)
        start_down = first_down - tall * round((first_down - top) / tall)
    else:
        pitch = _the_pitch(image, ground, box)
        if pitch is None:
            return None
        columns = round((right - left) / pitch)
        if columns < 1:
            return None
        wide = tall = (right - left) / columns
        start, start_down = float(left), bottom - tall * round((bottom - top) / tall)
    columns = round((right - start) / wide)
    rows = round((bottom - start_down) / tall)
    if not FEWEST_COLUMNS <= columns <= MOST_COLUMNS or rows < FEWEST_ROWS:
        return None
    if abs((right - start) / wide - columns) > A_WHOLE_NUMBER:
        return None
    return WellSeen(float(start), float(start_down), float(wide), float(tall), int(columns), int(rows), ground)


def _the_look(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """A small picture of the middle of a box: what it is known again by. Each of its pixels is the mean of the part
    of the middle it stands for, worked here in plain arithmetic: her main process does not load a picture library
    (core/media/safe_imports.py), and LIVE 2026-10-11 asking for one here ended every game she opened."""
    left, top, right, bottom = box
    inset_x, inset_y = (right - left) // 6, (bottom - top) // 6
    middle = image[top + inset_y:bottom - inset_y, left + inset_x:right - inset_x].astype(np.float32)
    tall, wide = middle.shape[:2]
    if tall < 1 or wide < 1:
        return np.zeros((LOOK_PX, LOOK_PX, 3), dtype=np.float32)
    rows = np.minimum(np.linspace(0, tall, LOOK_PX + 1).astype(int), tall)
    columns = np.minimum(np.linspace(0, wide, LOOK_PX + 1).astype(int), wide)
    small = np.zeros((LOOK_PX, LOOK_PX, 3), dtype=np.float32)
    for row in range(LOOK_PX):
        for column in range(LOOK_PX):
            part = middle[rows[row]:max(rows[row] + 1, rows[row + 1]), columns[column]:max(columns[column] + 1, columns[column + 1])]
            if part.size:
                small[row, column] = part.reshape(-1, 3).mean(axis=0)
    return small


def _kind_of(look: np.ndarray, well: WellSeen) -> int:
    """Which kind a look is: the kind it is most like, where it is like one; else a kind not seen before."""
    best, least = -1, SAME_LOOK
    for kind, known in enumerate(well.looks):
        apart = float(np.abs(look - known).mean())
        if apart < least:
            best, least = kind, apart
    if best < 0:
        well.looks.append(look)
        return len(well.looks) - 1
    return best


def _holds(image: np.ndarray, well: WellSeen, box: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = box
    left, top = max(0, left), max(0, top)
    if right - left < 3 or bottom - top < 3:
        return False
    inset_x, inset_y = (right - left) // 5, (bottom - top) // 5
    middle = image[top + inset_y:bottom - inset_y, left + inset_x:right - inset_x]
    return middle.size > 0 and float((~_is_ground(middle, well.ground)).mean()) >= HOLDS


def what_stands_in(picture: Any, well: WellSeen) -> dict[Cell, int]:
    """What stands in each cell of the well that is not empty: its kind, by its look."""
    image = _picture(picture)
    cells: dict[Cell, int] = {}
    for column in range(well.columns):
        for row in range(well.rows):
            box = well.box(column, row)
            if box[3] > image.shape[0] or box[2] > image.shape[1] or box[0] < 0 or box[1] < 0:
                continue
            if _holds(image, well, box):
                cells[(column, row)] = _kind_of(_the_look(image, box), well)
    return cells


def what_rests(cells: dict[Cell, int], well: WellSeen) -> dict[Cell, int]:
    """The cells that rest: those that reach the floor through their neighbours. What else is in the well is on its
    way down."""
    resting: set[Cell] = set()
    frontier = [cell for cell in cells if cell[1] == well.rows - 1]
    while frontier:
        cell = frontier.pop()
        if cell in resting:
            continue
        resting.add(cell)
        column, row = cell
        frontier += [near for near in ((column + 1, row), (column - 1, row), (column, row - 1), (column, row + 1))
                     if near in cells and near not in resting]
    return {cell: cells[cell] for cell in resting}


def the_piece_in(picture: Any, well: WellSeen, resting: dict[Cell, int]) -> PieceSeen | None:
    """The piece coming down: what is in the well that is neither ground nor one of the cells that rest. It is read
    where it is, not where the cells are, so a piece between two rows is read whole."""
    from scipy import ndimage

    image = _picture(picture)
    if not well.valid():
        return None
    left, top = round(well.left), round(well.top)
    right, bottom = round(well.left + well.columns * well.wide), round(well.top + well.rows * well.tall)
    right, bottom = min(right, image.shape[1]), min(bottom, image.shape[0])
    if not (0 <= left < right and 0 <= top < bottom):
        return None
    there = ~_is_ground(image[top:bottom, left:right], well.ground)
    for column, row in resting:
        x0, y0, x1, y1 = well.box(column, row)
        there[max(0, y0 - top):max(0, y1 - top), max(0, x0 - left):max(0, x1 - left)] = False
    # The walls' own edges show just inside the well all the way round, and are no part of what is in it.
    edge_x, edge_y = max(1, round(THE_WALLS_EDGE * well.wide)), max(1, round(THE_WALLS_EDGE * well.tall))
    there[:edge_y, :] = there[-edge_y:, :] = False
    there[:, :edge_x] = there[:, -edge_x:] = False
    # Specks and ruled lines are not a piece: only what is a fair part of a cell both ways is.
    labelled, count = ndimage.label(_no_lines(there))
    if not count:
        return None
    least_wide, least_tall = A_FAIR_PART * well.wide, A_FAIR_PART * well.tall
    spans = [(down, across) for down, across in ndimage.find_objects(labelled)
             if across.stop - across.start >= least_wide and down.stop - down.start >= least_tall]
    if not count or not spans:
        return None
    x0, x1 = float(min(across.start for _down, across in spans)), float(max(across.stop for _down, across in spans))
    y0, y1 = float(min(down.start for down, _across in spans)), float(max(down.stop for down, _across in spans))
    across = max(1, round((x1 - x0) / well.wide))
    down = max(1, round((y1 - y0) / well.tall))
    corner = round(((x0 + x1) / 2.0) / well.wide - across / 2.0)
    upper = ((y0 + y1) / 2.0) / well.tall - down / 2.0
    parts = []
    for dc in range(across):
        for dr in range(down):
            box = well.box(corner + dc, upper + dr)
            if (corner + dc, round(upper + dr)) in resting and abs(upper - round(upper)) < 0.1:
                continue
            if _holds(image, well, box):
                parts.append((dc, dr, _kind_of(_the_look(image, (max(0, box[0]), max(0, box[1]), box[2], box[3])), well)))
    if not parts:
        return None
    least_dc, least_dr = min(dc for dc, _dr, _k in parts), min(dr for _dc, dr, _k in parts)
    parts = [(dc - least_dc, dr - least_dr, kind) for dc, dr, kind in parts]
    return PieceSeen(tuple(parts), int(corner + least_dc), float(upper + least_dr), whole=y0 > 0.5 * well.tall)


@invariant('perception.unseen_well_geometry_is_not_a_piece', scope='perception',
           owner='core/perception/a_well_seen.py', observational=False)
def _well_geometry_invariant() -> tuple:
    well = WellSeen(80., 80., 8., 8., 8, 16, (30, 30, 30))
    assert the_piece_in(np.zeros((32, 32, 3), np.uint8), well, {}) is None
    kept = well.kept()
    kept[0] = math.nan
    assert WellSeen.from_kept(kept) is None
    return ()
