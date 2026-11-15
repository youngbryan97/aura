"""The ground a place's code says is to be gone over, as it lies in her picture: which cells there are, which of them
she has marked, and which are left.

Without the code she measures ground in cells of her own size over the whole picture. That is wrong in two ways a
reading of the code puts right. The code's cells are its own size and begin where it says, so her cells and its
cells disagree about what is done. And the picture holds more than the ground: offline 2026-10-13 a painter whose
floor began 40 pixels below the top of its picture, under its score bar, went to the "ground" beside the bar, held
the up arrow against the top of the floor for eighteen seconds, and painted nothing more. With the code's grid
(core/perception/a_world_as_its_code_makes_it.py) only its cells are ground, a cell is marked when the point of her
the code marks by is over it, and what is left is exactly what the code still counts as not done.

Nothing here names a game: a grid, a point of her body and a rectangle she is kept in, from whatever code said them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = ["GroundInHerPicture", "ground_in_her_picture", "mark_the_ground_under_her", "stage_in_her_picture"]

#: The furthest, in cells, she can have gone straight along a row or a column between two pictures for the cells
#: between to be taken as gone over: further than that, she was put somewhere, not walked there.
ALONG_AT_MOST = 4
#: How far inside a row (or a column), as a share of a cell, the point of her that marks must be at both ends of a way
#: she went along it for the cells between to be taken as marked: along a row's edge, which side the world marks is
#: anyone's guess, and a cell taken as marked that is not is never gone back for. Offline 2026-10-13 her body went
#: down a column's edge on the paint world's own code, and the four cells beside it she took as marked were never
#: painted.
SURELY_INSIDE = 0.05
#: How far a colour seen at a cell's middle may be from the colour the code marks in, on its most different channel,
#: and be a mark: what a picture's packing leaves of a flat fill.
MARKS_SEEN_WITHIN = 40
#: How long she goes for a cell within a few cells of it, in seconds, before she leaves it for others and comes back
#: later, and how long it is left. A cell her steps keep carrying her past is one she gets on another way round.
MISSED_AFTER_S = 2.5
LEFT_FOR_S = 12.0


def stage_in_her_picture(stage: tuple[float, float], shape: tuple[int, int],
                         lies: tuple[float, float, float, float] | None = None) -> tuple[float, float, float] | None:
    """How a stage of ``stage`` pixels lies in a picture of ``shape`` (rows, columns): its scale and where its corner
    is. ``lies`` is where the thing that draws it was found in the picture (left, top, width, height, as shares of the
    picture), the whole picture where not found; a stage drawn into a box of another shape is shown whole and centred
    in it, as a player shows a movie."""
    wide, tall = float(stage[0]), float(stage[1])
    rows, columns = float(shape[0]), float(shape[1])
    if wide <= 0 or tall <= 0 or rows <= 0 or columns <= 0:
        return None
    left, top, width, height = lies or (0.0, 0.0, 1.0, 1.0)
    box_x, box_y, box_w, box_h = left * columns, top * rows, width * columns, height * rows
    scale = min(box_w / wide, box_h / tall)
    return scale, box_x + (box_w - wide * scale) / 2.0, box_y + (box_h - tall * scale) / 2.0


@dataclass
class GroundInHerPicture:
    """A code's grid in her picture's working pixels: where cell (0, 0) begins, a cell's width and height, how many
    columns and rows, where the point of her that marks a cell is from the middle of her body, the rectangle that
    point is kept in, and the share of cells that wins."""

    origin: tuple[float, float]
    cell: tuple[float, float]
    cells: tuple[int, int]
    mark: tuple[float, float] = (0.0, 0.0)
    reach: tuple[float, float, float, float] | None = None
    wins_at: float | None = None
    #: The cells she has marked, by column and row.
    marked: set[tuple[int, int]] = field(default_factory=set)
    #: Where the point of her that marks was last seen, and when.
    last: tuple[float, float] | None = None
    now: float = 0.0
    #: The cells she has lately gone for while near them, each with since when and when last; and the cells she kept
    #: missing, each until when it is left.
    going_for: dict[tuple[int, int], tuple[float, float]] = field(default_factory=dict)
    left_until: dict[tuple[int, int], float] = field(default_factory=dict)
    #: The colour the code marks ground gone over in, where it says (red, green, blue): then what is marked is seen.
    drawn_in: tuple[int, int, int] | None = None

    def cell_of(self, x: float, y: float) -> tuple[int, int] | None:
        column = math.floor((x - self.origin[0]) / self.cell[0] + 1e-9)
        row = math.floor((y - self.origin[1]) / self.cell[1] + 1e-9)
        if 0 <= column < self.cells[0] and 0 <= row < self.cells[1]:
            return column, row
        return None

    def middle_of(self, column: int, row: int) -> tuple[float, float]:
        return self.origin[0] + (column + 0.5) * self.cell[0], self.origin[1] + (row + 0.5) * self.cell[1]

    def mark_of(self, x: float, y: float) -> tuple[float, float]:
        """The point that marks, for her body's middle at (x, y)."""
        return x + self.mark[0], y + self.mark[1]

    def marked_by(self, mine: Any) -> tuple[int, int] | None:
        """The cell her body marks where it is now, kept as marked; None off the grid. Gone straight along a row or a
        column since she was last seen, and not far, she went over the cells between, and those are kept as marked
        too: the world marks the ground under her as she goes, not only where she is seen. Offline 2026-10-13,
        deciding four times a second on the paint world's own code, she went over a row between two pictures each
        time and never saw herself in it, and went up and down across it for six seconds to paint it."""
        here = self.mark_of(float(mine.x), float(mine.y))
        self.now = max(self.now, float(getattr(mine, "seen", self.now) or self.now))
        cell = self.cell_of(*here)
        if cell is not None:
            self.marked.add(cell)
        last, self.last = self.last, here
        if last is None:
            return cell
        dx, dy = here[0] - last[0], here[1] - last[1]
        if abs(dx) > ALONG_AT_MOST * self.cell[0] or abs(dy) > ALONG_AT_MOST * self.cell[1]:
            return cell
        # Along a row (or down a column) she stayed surely inside of at both ends, every cell between its ends.
        along = 0 if abs(dy) < self.cell[1] / 2 else 1 if abs(dx) < self.cell[0] / 2 else None
        start, end = self.cell_of(*last), cell
        if along is None or start is None or end is None or start[1 - along] != end[1 - along]:
            return cell
        line = 1 - along
        if not all(SURELY_INSIDE <= (point[line] - self.origin[line]) / self.cell[line] - end[line] <= 1 - SURELY_INSIDE
                   for point in (last, here)):
            return cell
        for k in range(min(start[along], end[along]) + 1, max(start[along], end[along])):
            self.marked.add((k, end[1]) if along == 0 else (end[0], k))
        return cell

    def within_reach(self, column: int, row: int) -> bool:
        """Whether the point that marks can be brought over a cell at all: some of the cell is inside where the point
        is kept."""
        if self.reach is None:
            return True
        left, top, right, bottom = self.reach
        x0, y0 = self.origin[0] + column * self.cell[0], self.origin[1] + row * self.cell[1]
        return x0 <= right + 1e-6 and x0 + self.cell[0] >= left - 1e-6 and y0 <= bottom + 1e-6 and \
            y0 + self.cell[1] >= top - 1e-6

    def left_to_do(self) -> list[tuple[int, int]]:
        """The cells not yet marked that she can reach, less those she kept missing, while others are left."""
        left = [(c, r) for r in range(self.cells[1]) for c in range(self.cells[0])
                if (c, r) not in self.marked and self.within_reach(c, r)]
        kept = [cell for cell in left if self.left_until.get(cell, -math.inf) <= self.now]
        return kept or left

    def going_to(self, cell: tuple[int, int], near: bool) -> None:
        """She goes for ``cell``, ``near`` it or not: going for it while near it, on and off, for long without marking
        it, she leaves it for a while. Offline 2026-10-13, deciding four times a second on the paint world's own code,
        each step carried her past one cell going right and past another going left, and she went for each in turn
        for fifteen seconds."""
        self.going_for = {kept: (since, last) for kept, (since, last) in self.going_for.items()
                          if self.now - last <= MISSED_AFTER_S / 2 and kept not in self.marked}
        if not near:
            self.going_for.pop(cell, None)
            return
        since, _last = self.going_for.get(cell, (self.now, self.now))
        self.going_for[cell] = (since, self.now)
        if self.now - since >= MISSED_AFTER_S:
            self.left_until[cell] = self.now + LEFT_FOR_S
            del self.going_for[cell]

    def where_to_be_for(self, column: int, row: int) -> tuple[float, float]:
        """Where her body's middle is to be for the point that marks to be over the middle of a cell, as near it as
        where that point is kept allows."""
        x, y = self.middle_of(column, row)
        if self.reach is not None:
            left, top, right, bottom = self.reach
            x, y = min(max(x, left), right), min(max(y, top), bottom)
        return x - self.mark[0], y - self.mark[1]

    def saw(self, marked: set[tuple[int, int]], looked_at: set[tuple[int, int]]) -> None:
        """What the picture shows of the ground: of the cells ``looked_at``, those in ``marked`` are marked and the rest
        are not; a cell not looked at (something stands over it) is as it was."""
        self.marked = (self.marked - looked_at) | (marked & looked_at)

    def seen_in(self, picture: Any, shape: tuple[int, int], things: Any = ()) -> None:
        """The marks read off ``picture`` (rows, columns, colour) of a view ``shape`` (rows, columns) in her working
        pixels, by the colour the code marks in, at the middle of every cell no thing stands over. Offline 2026-10-13,
        deciding three times a second, she marked only where her body was seen and the cells it passed between
        pictures, and went back over painted floor for cells she could not be sure of; a person looks at the floor."""
        import numpy as np

        if self.drawn_in is None or picture is None:
            return
        picture = np.asarray(picture)
        if picture.ndim != 3 or picture.shape[2] < 3 or not shape[0] or not shape[1]:
            return
        down, across = picture.shape[0] / shape[0], picture.shape[1] / shape[1]
        # What stands over the ground hides it; the marks themselves, seen as things, do not.
        boxes = [(t.x - t.w / 2, t.y - t.h / 2, t.x + t.w / 2, t.y + t.h / 2) for t in things
                 if len(tuple(getattr(t, "colour", ()) or ())) != 3
                 or max(abs(int(a) - int(b)) for a, b in zip(t.colour, self.drawn_in, strict=True)) > MARKS_SEEN_WITHIN]
        cells, places = [], []
        for row in range(self.cells[1]):
            for column in range(self.cells[0]):
                x0, y0 = self.origin[0] + column * self.cell[0], self.origin[1] + row * self.cell[1]
                x1, y1 = x0 + self.cell[0], y0 + self.cell[1]
                if any(x0 < right and left < x1 and y0 < bottom and top < y1 for left, top, right, bottom in boxes):
                    continue
                cx, cy = int((x0 + x1) / 2 * across), int((y0 + y1) / 2 * down)
                if 0 <= cx < picture.shape[1] and 0 <= cy < picture.shape[0]:
                    cells.append((column, row))
                    places.append((cy, cx))
        if not cells:
            return
        rows, columns = np.asarray(places).T
        colours = picture[rows, columns, :3].astype(np.int16)
        near = np.abs(colours - np.asarray(self.drawn_in, dtype=np.int16)).max(axis=1) <= MARKS_SEEN_WITHIN
        self.saw({cell for cell, yes in zip(cells, near, strict=True) if yes}, set(cells))

    def done(self) -> float:
        reachable = sum(1 for r in range(self.cells[1]) for c in range(self.cells[0]) if self.within_reach(c, r))
        return len(self.marked) / reachable if reachable else 0.0


def ground_in_her_picture(world: Any, shape: tuple[int, int],
                          lies: tuple[float, float, float, float] | None = None) -> GroundInHerPicture | None:
    """The ground a code's reading names, laid on a picture of ``shape`` (rows, columns) of its stage, drawn where
    ``lies`` says (see ``stage_in_her_picture``); None where the reading names no ground, or not enough of it to lay
    down (its stage, its cells)."""
    ground = getattr(world, "ground", None)
    stage = getattr(world, "stage", None)
    if ground is None or not stage or not ground.cells:
        return None
    laid = stage_in_her_picture(stage, shape, lies)
    if laid is None:
        return None
    scale, left, top = laid
    hers = getattr(world, "hers", None)
    size = getattr(hers, "size", None)
    # Where the code does not say which point of her marks, it is taken to be her middle; and where it does not say her
    # size either (a Flash body placed by its registration point), the point it keeps inside its rectangle is that
    # same middle.
    marks_at = ground.marks_at
    if marks_at is None and size is not None:
        marks_at = (size[0] / 2, size[1] / 2)
    mark = (0.0, 0.0)
    if marks_at is not None and size is not None:
        mark = ((marks_at[0] - size[0] / 2) * scale, (marks_at[1] - size[1] / 2) * scale)
    reach = None
    kept = getattr(hers, "kept_inside", None)
    if kept is not None:
        at = marks_at if size is not None and marks_at is not None else (0.0, 0.0)
        x0, y0, x1, y1 = kept
        reach = (left + (x0 + at[0]) * scale, top + (y0 + at[1]) * scale,
                 left + (x1 + at[0]) * scale, top + (y1 + at[1]) * scale)
    return GroundInHerPicture(origin=(left + ground.origin[0] * scale, top + ground.origin[1] * scale),
                              cell=(ground.cell[0] * scale, ground.cell[1] * scale), cells=tuple(ground.cells),
                              mark=mark, reach=reach, wins_at=ground.wins_at, drawn_in=_rgb(getattr(ground, "drawn_in", "")))


def _rgb(colour: Any) -> tuple[int, int, int] | None:
    hex_ = str(colour or "").lstrip("#")
    hex_ = "".join(c * 2 for c in hex_) if len(hex_) == 3 else hex_
    try:
        return (int(hex_[0:2], 16), int(hex_[2:4], 16), int(hex_[4:6], 16)) if len(hex_) == 6 else None
    # not a failure: a colour that is not hex is none
    except ValueError:
        return None


def mark_the_ground_under_her(run: Any, moves: Any, mine: Any, happened: list[dict[str, Any]]) -> GroundInHerPicture | None:
    """Each picture of play: the cell of her own size her body is in, as before; and where the guide to the place holds
    a reading of its code that names its ground, that ground laid on her picture, with the cell of it under the point
    of her that marks marked. A new screen is new ground (a level begun is bare). The ground, or None."""
    if any(h.get("what") == "new screen" for h in happened):
        run.ground = None
    if mine is None:
        return run.ground or None
    run.cell = run.cell or max(4.0, mine.w, mine.h)
    run.covered.add((int(mine.x // run.cell), int(mine.y // run.cell)))
    if run.ground is None:
        from core.cognition.a_guide_to_a_place import THE_GUIDE

        guide = THE_GUIDE.get()
        world = getattr(guide, "world_its_code_makes", None)
        lies = getattr(guide, "stage_in_her_picture", None)
        # Looked for once a screen: False where the guide names no ground.
        run.ground = ground_in_her_picture(world, moves.shape, lies) or False
    if run.ground:
        run.ground.marked_by(mine)
        run.ground.seen_in(getattr(run, "picture", None), moves.shape, list(getattr(moves, "things", {}).values()))
    return run.ground or None
