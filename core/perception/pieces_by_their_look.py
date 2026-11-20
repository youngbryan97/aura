"""Pieces told apart by their look: the squares of a board, which side stands on each, and which pieces look alike.

A person who sits down at a board they have never seen sees three things before they know the game: where the squares
are, which pieces are whose (one side light, the other dark, or any two colours), and which pieces are the same kind
(the four on the corners look alike, so do the eight in a row in front). Which kind is which, and what each may do,
comes after, from the rules they know and where each kind stands at the start (core/agency/pieces_by_their_kind.py).

The squares are found as squares, not as edges: a board's empty squares are regions of one flat colour, all of one
size, on one lattice. Where pieces stand the ground still shows at the square's corners, so a square is on the board
where its corners are ground. LIVE-like 19 November 2026: the edge comb of core/perception/a_board_seen.py read En
Croissant's board as nine by nine (the panel beside it added an edge at the board's spacing) and a smaller window of it
as three by six.

A piece is what differs from the ground of its own square, measured against that square's corners, so a square tinted
for the last move is read against its own tint; its silhouette, holes filled, is its look. Two pieces of one kind look
alike whatever their side, so looks are compared as silhouettes and sides by colour. How alike two looks must be to be
one kind is not a constant: it is the widest gap in how far apart the looks on this board are.

Nothing here knows a game, a piece set or a colour scheme.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.verify import invariant

__all__ = ["Lattice", "Looks", "PiecesSeen", "squares_of_one_ground"]

#: The fewest squares a side of a board has, and the most.
FEWEST, MOST = 3, 19
#: Colour levels per channel when telling flat regions apart: flat ground differs by less than one level.
LEVELS_SHIFT = 3
#: How many of the most common flat colours are looked at for squares.
COLOURS_LOOKED_AT = 10
#: How unlike its ground a pixel must be to belong to a piece, in the largest channel difference.
UNLIKE_GROUND = 36
#: The smallest share of a square a piece covers; a move hint drawn on an empty square covers less.
A_PIECE_COVERS = 0.06
#: The side of the grid a look is compared on.
LOOK_SIDE = 24
#: The longest side a picture is read at; a larger one is halved first.
READ_AT_MOST = 1800


@dataclass(frozen=True)
class Lattice:
    """Where a picture is ruled into squares: the left and top of the board, a square's width and height, in the
    picture's pixels, and how many columns and rows."""

    left: float
    top: float
    width: float
    height: float
    columns: int
    rows: int

    def centre(self, row: int, column: int) -> tuple[float, float]:
        return self.left + (column + 0.5) * self.width, self.top + (row + 0.5) * self.height

    def square(self, picture: np.ndarray, row: int, column: int) -> np.ndarray:
        x0 = int(round(self.left + column * self.width))
        y0 = int(round(self.top + row * self.height))
        return picture[y0:y0 + int(self.height), x0:x0 + int(self.width)]


def _halved(picture: np.ndarray) -> tuple[np.ndarray, int]:
    factor = 1
    while max(picture.shape[:2]) > READ_AT_MOST:
        tall, wide = (picture.shape[0] // 2) * 2, (picture.shape[1] // 2) * 2
        picture = picture[:tall:2, :wide:2]
        factor *= 2
    return picture, factor


def _flat_squares(picture: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Regions of one flat colour shaped like squares: left, top, width, height."""
    from scipy import ndimage

    levels = np.asarray(picture[..., :3], dtype=np.int32) >> LEVELS_SHIFT
    code = (levels[..., 0] << 16) | (levels[..., 1] << 8) | levels[..., 2]
    flat = np.zeros(code.shape, bool)
    flat[:-1, :-1] = (code[:-1, :-1] == code[:-1, 1:]) & (code[:-1, :-1] == code[1:, :-1])
    if not flat.any():
        return []
    values, counts = np.unique(code[flat], return_counts=True)
    found: list[tuple[int, int, int, int]] = []
    for value in values[np.argsort(counts)[::-1][:COLOURS_LOOKED_AT]]:
        mask = code == value
        labels, count = ndimage.label(mask)
        if not count:
            continue
        sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
        for index, where in enumerate(ndimage.find_objects(labels)):
            if where is None:
                continue
            tall, wide = where[0].stop - where[0].start, where[1].stop - where[1].start
            if min(tall, wide) < 8 or abs(tall - wide) > 0.12 * max(tall, wide):
                continue
            if sizes[index] < 0.25 * tall * wide:
                continue
            found.append((where[1].start, where[0].start, wide, tall))
    return found


def _ground_of(square: np.ndarray) -> np.ndarray:
    """The colour of a square's ground, from its four corners."""
    side = min(square.shape[:2])
    inset, patch = max(1, side // 30), max(2, side // 12)
    corners = [square[inset:inset + patch, inset:inset + patch],
               square[inset:inset + patch, -inset - patch:-inset],
               square[-inset - patch:-inset, inset:inset + patch],
               square[-inset - patch:-inset, -inset - patch:-inset]]
    return np.median(np.concatenate([c.reshape(-1, c.shape[-1])[:, :3] for c in corners]).astype(np.float32), axis=0)


def _corners_flat(square: np.ndarray) -> bool:
    """Whether a square's corners are one flat ground, as a board square's are."""
    side = min(square.shape[:2])
    inset, patch = max(1, side // 30), max(2, side // 12)
    corners = [square[inset:inset + patch, inset:inset + patch],
               square[inset:inset + patch, -inset - patch:-inset],
               square[-inset - patch:-inset, inset:inset + patch],
               square[-inset - patch:-inset, -inset - patch:-inset]]
    means = [c.reshape(-1, c.shape[-1])[:, :3].astype(np.float32).mean(axis=0) for c in corners]
    spread = [float(c.reshape(-1, c.shape[-1])[:, :3].astype(np.float32).std(axis=0).max()) for c in corners]
    return max(spread) < 6.0 and max(float(np.abs(m - means[0]).max()) for m in means) < 12.0


def squares_of_one_ground(picture: np.ndarray) -> Lattice | None:
    """The board in a picture, found from its squares: the size most flat square regions share, the lattice they lie
    on, and its extent, grown or trimmed to the squares whose corners are flat ground; None where there is none."""
    picture = np.asarray(picture)
    if picture.ndim != 3 or min(picture.shape[:2]) < FEWEST * 8:
        return None
    small, factor = _halved(picture)
    boxes = _flat_squares(small)
    if len(boxes) < FEWEST * 2:
        return None
    widths = np.array([b[2] for b in boxes])
    tally = np.bincount(widths)
    near = np.convolve(tally, np.ones(5, dtype=int), mode="same")
    size = int(near.argmax())
    kept = [b for b in boxes if abs(b[2] - size) <= max(2, 0.08 * size) and abs(b[3] - size) <= max(2, 0.08 * size)]
    if len(kept) < FEWEST * 2:
        return None
    width = float(np.median([b[2] for b in kept]))
    height = float(np.median([b[3] for b in kept]))
    xs = np.array([b[0] for b in kept], float)
    ys = np.array([b[1] for b in kept], float)
    x_phase = float(np.median(np.mod(xs, width)))
    y_phase = float(np.median(np.mod(ys, height)))
    columns = np.round((xs - x_phase) / width).astype(int)
    rows = np.round((ys - y_phase) / height).astype(int)

    def square_at(row: int, column: int) -> np.ndarray | None:
        x0, y0 = x_phase + column * width, y_phase + row * height
        if x0 < 0 or y0 < 0 or x0 + width > small.shape[1] or y0 + height > small.shape[0]:
            return None
        square = small[int(round(y0)):int(round(y0 + height)), int(round(x0)):int(round(x0 + width))]
        return square if square.size else None

    # The ground of each of the two colours squares come in by turns, from the flat squares found: a board's own
    # squares have it at their corners, and the flat page around a board does not.
    grounds: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        square = square_at(row, column)
        if square is not None:
            grounds[(row + column) % 2].append(_ground_of(square))
    ground = {parity: np.median(np.stack(found), axis=0) for parity, found in grounds.items() if found}
    if not ground:
        return None
    if len(ground) == 1:
        ground[1 - next(iter(ground))] = next(iter(ground.values()))

    def on_board(row: int, column: int) -> bool:
        square = square_at(row, column)
        return (square is not None and _corners_flat(square)
                and float(np.abs(_ground_of(square) - ground[(row + column) % 2]).max()) <= UNLIKE_GROUND / 2)

    # The board is the largest run of such squares joined side to side, from the lattice the flat squares lie on.
    reach = MOST
    r_lo, r_hi = int(rows.min()) - reach, int(rows.max()) + reach
    c_lo, c_hi = int(columns.min()) - reach, int(columns.max()) + reach
    on = {(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1) if on_board(r, c)}
    best: set[tuple[int, int]] = set()
    unseen = set(on)
    while unseen:
        first = unseen.pop()
        group, frontier = {first}, [first]
        while frontier:
            r, c = frontier.pop()
            for nxt in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nxt in unseen:
                    unseen.discard(nxt)
                    group.add(nxt)
                    frontier.append(nxt)
        if len(group) > len(best):
            best = group
    if not best:
        return None
    r0, r1 = min(r for r, _ in best), max(r for r, _ in best)
    c0, c1 = min(c for _, c in best), max(c for _, c in best)
    n_columns, n_rows = c1 - c0 + 1, r1 - r0 + 1
    if not (FEWEST <= n_columns <= MOST and FEWEST <= n_rows <= MOST):
        return None
    # A board is filled with its squares; a scatter of icons of one size is not.
    if len(best) < 0.75 * n_columns * n_rows:
        return None
    return Lattice(left=(x_phase + c0 * width) * factor, top=(y_phase + r0 * height) * factor,
                   width=width * factor, height=height * factor, columns=n_columns, rows=n_rows)


def _silhouette(square: np.ndarray) -> np.ndarray | None:
    """What stands on a square, as a filled silhouette, or None where it is empty."""
    from scipy import ndimage

    ground = _ground_of(square)
    unlike = np.abs(square[..., :3].astype(np.float32) - ground).max(axis=2) > UNLIKE_GROUND
    filled = ndimage.binary_fill_holes(unlike)
    labels, count = ndimage.label(filled)
    if not count:
        return None
    sizes = ndimage.sum(filled, labels, index=np.arange(1, count + 1))
    largest = labels == (1 + int(np.argmax(sizes)))
    return largest if largest.sum() >= A_PIECE_COVERS * largest.size else None


def _look(silhouette: np.ndarray, square: np.ndarray | None = None) -> np.ndarray:
    """A piece's look: its outline filled, and beside it the lines drawn inside it. Pieces that share an outline and
    differ only in what is drawn on them (a mark, a letter, a hole) are told apart by the second; the lines are where
    brightness changes sharply, whichever way, so a light piece and a dark piece of one kind draw them alike."""
    from PIL import Image
    from scipy import ndimage

    outline = Image.fromarray(silhouette.astype(np.uint8) * 255).resize((LOOK_SIDE, LOOK_SIDE), Image.BILINEAR)
    outline_look = np.asarray(outline, dtype=np.float32) / 255.0
    inner = np.zeros_like(outline_look)
    if square is not None:
        # At the look's own scale, so a board drawn small and one drawn large give the same lines.
        grey = Image.fromarray(square[..., :3].astype(np.uint8)).convert("L").resize((LOOK_SIDE, LOOK_SIDE),
                                                                                      Image.BILINEAR)
        grey_look = np.asarray(grey, dtype=np.float32)
        sharp = np.hypot(ndimage.sobel(grey_look, axis=0), ndimage.sobel(grey_look, axis=1))
        inside = ndimage.binary_erosion(outline_look > 0.5, iterations=2)
        if inside.any():
            scale = float(np.percentile(sharp[inside], 90)) or 1.0
            inner = np.where(inside, np.clip(sharp / scale, 0.0, 1.0), 0.0).astype(np.float32)
    return np.concatenate([outline_look, inner], axis=0)


def _apart(one: np.ndarray, other: np.ndarray) -> float:
    return float(np.abs(one - other).mean())


#: How much wider than the jumps among themselves the jump between two groups of differences must be for there to be
#: two groups: differences that are noise (one kind drawn on two grounds, or by two sides) and differences between kinds.
TWO_GROUPS = 2.5


def _widest_gap(looks: list[np.ndarray]) -> float:
    """The cut between alike and unlike, from the steps of the tree that joins every look to its nearest (a
    single-linkage tree). Steps of exactly nothing are pieces drawn identically. Among the rest, a jump of more than
    ``TWO_GROUPS`` times splits noise from kinds; where there is no such jump, the rest are all differences between
    kinds if any pieces were drawn identically, and all noise of one kind if none were."""
    count = len(looks)
    if count < 2:
        return 0.0
    apart = np.array([[_apart(a, b) for b in looks] for a in looks])
    joined, steps = {0}, []
    nearest = apart[0].copy()
    while len(joined) < count:
        length, nxt = min((nearest[j], j) for j in range(count) if j not in joined)
        steps.append(float(length))
        joined.add(nxt)
        nearest = np.minimum(nearest, apart[nxt])
    steps.sort()
    some = [step for step in steps if step > 1e-9]
    identical = len(some) < len(steps)
    if len(some) >= 2:
        ratio, at = max((some[i + 1] / some[i], i) for i in range(len(some) - 1))
        if ratio >= TWO_GROUPS:
            return float(np.sqrt(some[at] * some[at + 1]))
    if some and identical:
        return some[0] / 2.0
    return (some[-1] if some else 0.0) * 2.0 + 1e-9


@dataclass
class PiecesSeen:
    """The board read: ``cells[row][column]`` is None for an empty square, else (side, look): side 0 is the lighter
    side, look an index into ``Looks.kinds``."""

    lattice: Lattice
    cells: list[list[tuple[int, int] | None]]

    def says(self) -> str:
        return "\n".join(" ".join("." if cell is None else f"{'ab'[cell[0]]}{cell[1]}" for cell in row)
                         for row in self.cells)

    def occupied(self) -> list[tuple[int, int]]:
        return [(r, c) for r, row in enumerate(self.cells) for c, cell in enumerate(row) if cell is not None]


@dataclass
class Looks:
    """The looks of the pieces of one board, kept from one picture to the next: each kind's silhouette, the cut between
    alike and unlike, and the colours of the two sides."""

    kinds: list[np.ndarray] = field(default_factory=list)
    cut: float = 0.0
    sides: list[float] = field(default_factory=list)

    def as_memory(self) -> dict:
        """The looks she knows on this board, kept with what she learned of the game, so the next sitting calls each
        kind by the same name."""
        return {"kinds": [np.round(kind, 2).tolist() for kind in self.kinds], "cut": self.cut, "sides": list(self.sides)}

    @classmethod
    def from_memory(cls, kept: dict) -> Looks:
        try:
            return cls(kinds=[np.asarray(kind, dtype=np.float32) for kind in kept.get("kinds", [])],
                       cut=float(kept.get("cut", 0.0)), sides=[float(s) for s in kept.get("sides", [])])
        except (TypeError, ValueError):
            return cls()

    def read(self, picture: np.ndarray, lattice: Lattice | None = None) -> PiecesSeen | None:
        """The board in ``picture`` read square by square; None where there is no board."""
        picture = np.asarray(picture)
        lattice = lattice or squares_of_one_ground(picture)
        if lattice is None:
            return None
        found: dict[tuple[int, int], tuple[np.ndarray, float]] = {}

        for row in range(lattice.rows):
            for column in range(lattice.columns):
                square = lattice.square(picture, row, column)
                if square.size == 0:
                    continue
                silhouette = _silhouette(square)
                if silhouette is None:
                    continue
                colour = float(np.median(square[..., :3][silhouette].astype(np.float32)))
                found[(row, column)] = (_look(silhouette, square), colour)
        if not self.kinds and found:
            self._learn([v[0] for v in found.values()], [v[1] for v in found.values()])
        cells: list[list[tuple[int, int] | None]] = [[None] * lattice.columns for _ in range(lattice.rows)]
        for (row, column), (look, colour) in found.items():
            cells[row][column] = (self._side_of(colour), self._kind_of(look))
        return PiecesSeen(lattice, cells)

    def _learn(self, looks: list[np.ndarray], colours: list[float]) -> None:
        self.cut = _widest_gap(looks)
        for look in looks:
            self._kind_of(look)
        values = sorted(colours)
        gaps = [(values[i + 1] - values[i], i) for i in range(len(values) - 1)]
        if gaps:
            _gap, at = max(gaps)
            light = [v for v in values[at + 1:]]
            dark = [v for v in values[:at + 1]]
            self.sides = [float(np.median(light)), float(np.median(dark))]
        else:
            self.sides = [values[0]] if values else []

    def _side_of(self, colour: float) -> int:
        if not self.sides:
            return 0
        return int(np.argmin([abs(colour - s) for s in self.sides]))

    def _kind_of(self, look: np.ndarray) -> int:
        """The kind a look is, a new one where it is like none kept; a kind's look is kept as the first seen of it."""
        if self.kinds:
            apart = [_apart(look, kind) for kind in self.kinds]
            nearest = int(np.argmin(apart))
            if apart[nearest] <= self.cut:
                return nearest
        self.kinds.append(look)
        return len(self.kinds) - 1


@invariant("perception.pieces_by_their_look_widest_gap_splits_alike_from_unlike", scope="perception",
           owner="core/perception/pieces_by_their_look.py", observational=False)
def _widest_gap_invariant() -> tuple:
    """Looks drawn identically per kind, and looks that differ a little per kind, are cut between kinds."""
    blank = np.zeros((2 * LOOK_SIDE, LOOK_SIDE), np.float32)
    one, two = blank.copy(), blank.copy()
    one[4:20, 8:16] = 1.0
    two[2:22, 4:20] = 1.0
    cut = _widest_gap([one, one, two, two])
    assert 0.0 < cut < _apart(one, two)
    nudged = one.copy()
    nudged[4, 8] = 0.5
    cut = _widest_gap([one, nudged, one, two, two, two])
    assert _apart(one, nudged) < cut < _apart(one, two)
    return ()
