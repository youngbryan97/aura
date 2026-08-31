"""What she is looking at, with a place for each thing in it.

A reading of a screen arrives as text and positions. Flattened to a string it
loses the only thing that makes a position a position: "2 4 8 64" describes a
board where the 64 sits in a corner and a board where it sits in the middle,
and they are different situations that want different moves.

:func:`arranged` recovers the rows and columns that are really there and hands
back something with an index, so a corner, an edge, a row and a column are
things code can ask about and things she can hold a plan about. The string she
reads is one rendering of it rather than the whole of it.

Nothing here knows what it is looking at. Rows and columns are found from the
spacing that is present, and a cell says whatever was read in it — a tile, a
price, a seat, a cell of a spreadsheet, a field of a form. The only thing this
module assumes is that the thing is laid out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "Arrangement",
    "Cell",
    "arranged",
    "holds_in",
    "the_part_laid_out_regularly",
    "EMPTY_CELL",
]

#: What an empty place looks like when the rest of the row is not empty. A gap
#: has to be visible for a position to mean anything.
EMPTY_CELL = "."

#: The least a gap between two things has to be, as a share of the screen,
#: before they are on different rows. Below this the reading's own noise
#: decides the layout.
_SAME_ROW = 0.008

#: A number as it is written on a screen, with the separators people use.
_A_NUMBER = re.compile(r"^[^\d\-+]*([-+]?\d[\d,. ]*)\D*$")


@dataclass(frozen=True)
class Cell:
    """One thing, and where it sits."""

    row: int
    column: int
    says: str
    at: tuple[float, float]

    def number(self) -> float | None:
        """What this says as a number, when it says one."""
        found = _A_NUMBER.match(self.says.strip())
        if not found:
            return None
        try:
            return float(found.group(1).replace(",", "").replace(" ", ""))
        except ValueError:
            return None


@dataclass(frozen=True)
class Arrangement:
    """Everything she can see in the part that answers to her, in its places.

    Deliberately incurious about what the thing is. Every question here is
    about position and content, so the same object describes a game board, a
    timetable, a seating plan and a spreadsheet, and an approach phrased about
    a corner or a bottom row can be checked against any of them.
    """

    rows: int
    columns: int
    cells: tuple[Cell, ...]
    #: Where the columns and rows were found to be, so a later reading of the
    #: same thing can be placed in the same grid rather than inferring its own.
    down_at: tuple[float, ...] = ()
    across_at: tuple[float, ...] = ()

    # ── what is where ────────────────────────────────────────────────────

    def at(self, row: int, column: int) -> Cell | None:
        for cell in self.cells:
            if cell.row == row and cell.column == column:
                return cell
        return None

    def row_at(self, row: int) -> tuple[Cell | None, ...]:
        return tuple(self.at(row, column) for column in range(self.columns))

    def column_at(self, column: int) -> tuple[Cell | None, ...]:
        return tuple(self.at(row, column) for row in range(self.rows))

    def corners(self) -> dict[str, Cell | None]:
        """The four corners, named the way a person names them."""
        if not self.rows or not self.columns:
            return {}
        last_row, last_column = self.rows - 1, self.columns - 1
        return {
            "top-left": self.at(0, 0),
            "top-right": self.at(0, last_column),
            "bottom-left": self.at(last_row, 0),
            "bottom-right": self.at(last_row, last_column),
        }

    def edges(self) -> dict[str, tuple[Cell | None, ...]]:
        """The four edges, named the way a person names them."""
        if not self.rows or not self.columns:
            return {}
        return {
            "top": self.row_at(0),
            "bottom": self.row_at(self.rows - 1),
            "left": self.column_at(0),
            "right": self.column_at(self.columns - 1),
        }

    # ── what is here ─────────────────────────────────────────────────────

    def places(self) -> int:
        return self.rows * self.columns

    def occupied(self) -> int:
        return len(self.cells)

    def empty(self) -> int:
        return max(0, self.places() - self.occupied())

    def numbers(self) -> tuple[float, ...]:
        """Everything here that is written as a number."""
        return tuple(
            value for value in (cell.number() for cell in self.cells) if value is not None
        )

    def largest(self) -> Cell | None:
        """The biggest number here, when anything here is a number."""
        numbered = [(cell.number(), cell) for cell in self.cells if cell.number() is not None]
        if not numbered:
            return None
        return max(numbered, key=lambda row: row[0])[1]

    def where_is(self, said: str) -> tuple[tuple[int, int], ...]:
        """Every place something is, by what it says."""
        wanted = " ".join(str(said or "").split()).lower()
        if not wanted:
            return ()
        return tuple(
            (cell.row, cell.column) for cell in self.cells if cell.says.strip().lower() == wanted
        )

    def place_of(self, cell: Cell | None) -> str:
        """Where a cell sits, said the way a person would say it.

        The vocabulary a plan is phrased in — a corner, an edge, the middle —
        so a stated approach and a reading can be compared without either side
        knowing what the thing is.
        """
        if cell is None or not self.rows or not self.columns:
            return ""
        vertical = "top" if cell.row == 0 else "bottom" if cell.row == self.rows - 1 else ""
        across = "left" if cell.column == 0 else "right" if cell.column == self.columns - 1 else ""
        if vertical and across:
            return f"{vertical}-{across}"
        return vertical or across or "middle"

    # ── how it is said ───────────────────────────────────────────────────

    def places_of(self, said: str) -> set[str]:
        """Every place something occupies, named the way a person names places.

        What makes a plan about a corner a plan that can be checked. Asked of
        the thing itself rather than of prose about it, so "the 64 is still in
        the corner" is a question with an answer.
        """
        wanted = " ".join(str(said or "").split()).lower()
        if not wanted:
            return set()
        return {
            self.place_of(cell) for cell in self.cells if cell.says.strip().lower() == wanted
        }

    def without(self, places: set[tuple[int, int]]) -> "Arrangement":
        """This thing with some places dropped, and any row or column they
        were the whole of dropped with them.

        What is left is the part that behaves like one thing. A score beside a
        board answers to her the same way the board does, so it sits inside the
        part of the screen that responds — and it holds a row nothing ever
        moves into. Cropped out, what remains is the board.
        """
        if not places:
            return self
        kept = [cell for cell in self.cells if (cell.row, cell.column) not in places]
        if not kept:
            return Arrangement(rows=0, columns=0, cells=())
        rows = sorted({cell.row for cell in kept})
        columns = sorted({cell.column for cell in kept})
        row_of = {row: index for index, row in enumerate(rows)}
        column_of = {column: index for index, column in enumerate(columns)}
        return Arrangement(
            rows=len(rows),
            columns=len(columns),
            cells=tuple(
                Cell(
                    row=row_of[cell.row],
                    column=column_of[cell.column],
                    says=cell.says,
                    at=cell.at,
                )
                for cell in kept
            ),
            down_at=tuple(self.down_at[row] for row in rows if row < len(self.down_at)),
            across_at=tuple(
                self.across_at[column] for column in columns if column < len(self.across_at)
            ),
        )

    def as_text(self) -> str:
        """The rendering she reads, with gaps kept so a column stays a column."""
        if not self.rows or not self.columns:
            return ""
        lines: list[str] = []
        for row in range(self.rows):
            said = [
                (cell.says if cell is not None else EMPTY_CELL) for cell in self.row_at(row)
            ]
            lines.append(" ".join(said))
        return "\n".join(lines)

    def as_shape(self) -> str:
        """What kind of position this is, rather than which one.

        Two readings that would be approached the same way should look the
        same here, and two that would not should differ. What a record of
        consequences needs to key on, because keying it on the reading itself
        means no two situations are ever alike and nothing carries over.
        """
        if not self.cells:
            return "empty"
        biggest = self.largest()
        parts = [f"{self.rows}x{self.columns}", f"filled:{self.occupied()}/{self.places()}"]
        if biggest is not None:
            parts.append(f"largest:{biggest.number():g}@{self.place_of(biggest)}")
            ranked = sorted(self.numbers(), reverse=True)[:3]
            parts.append("top:" + ",".join(f"{value:g}" for value in ranked))
        return " ".join(parts)


def arranged(
    cells: Iterable[tuple[float, float, str]], like: "Arrangement | None" = None
) -> Arrangement:
    """Work out the rows and columns that are really there.

    Both are found from the spacing present rather than from a fixed
    tolerance: whatever the thing is, the gaps within one of its rows are
    smaller than the gap to the next row, and a reading of a four-column board
    and a reading of a twelve-column timetable both say so themselves.

    ``like`` is the last reading of the same thing. A shape is a property of
    the thing and not of one glance at it: a four-by-four board whose top row
    happens to be empty reads as four-by-three, and two readings that disagree
    about the shape cannot be compared at all. LIVE 2026-08-26: five of twelve
    readings unusable, and she could work out nothing about how the board
    moved. Given the last one, a reading that fits inside it is placed in it.
    """
    placed = sorted((y, x, said) for y, x, said in cells if str(said or "").strip())
    if not placed:
        return Arrangement(rows=0, columns=0, cells=())

    kept = _placed_in(placed, like)
    if kept is not None:
        return kept

    tolerance = _typical_gap([y for y, _x, _said in placed]) * 0.5
    banded: list[list[tuple[float, str]]] = []
    row_at: float | None = None
    for y, x, said in placed:
        if row_at is None or (y - row_at) > tolerance:
            banded.append([])
            row_at = y
        banded[-1].append((x, str(said).strip()))
    banded = [row for row in banded if row]

    # Rows are still banded from what sits on them, where columns are read
    # from the pitch. A board whose EDGE row is empty therefore reads a row
    # short, and cannot do otherwise: an edge that nothing ever sat on leaves
    # no trace to infer from. What covers it is the reading before — given the
    # shape she already has, a short glance is placed inside it — which is why
    # `like` matters more here than any inference could.
    columns = the_places_nothing_sits_in(
        _column_edges([x for row in banded for x, _said in row])
    )
    if not columns:
        # Nothing repeats across rows, so every row stands alone. Said as one
        # column per thing rather than pretending to a grid that is not there.
        widest = max(len(row) for row in banded)
        found: list[Cell] = []
        for index, row in enumerate(banded):
            for column, (x, said) in enumerate(sorted(row)):
                found.append(Cell(row=index, column=column, says=said, at=(x, 0.0)))
        return Arrangement(rows=len(banded), columns=widest, cells=tuple(found))

    found = []
    for index, row in enumerate(banded):
        for x, said in sorted(row):
            found.append(
                Cell(row=index, column=_nearest(x, columns), says=said, at=(x, 0.0))
            )
    return Arrangement(
        rows=len(banded),
        columns=len(columns),
        cells=tuple(found),
        down_at=tuple(row[0][0] for row in [] ) or _row_edges(banded, placed),
        across_at=columns,
    )


def _typical_gap(values: Sequence[float]) -> float:
    """The gap between neighbours that is usual here, so an outlier sets nothing."""
    gaps = sorted(b - a for a, b in zip(values, values[1:]) if b - a > 0.0)
    typical = gaps[len(gaps) // 2] if gaps else 0.0
    return max(_SAME_ROW, typical)


def _column_edges(xs: Sequence[float]) -> tuple[float, ...]:
    """Where the columns are, from where things actually sit across the rows."""
    if not xs:
        return ()
    ordered = sorted(xs)
    tolerance = _typical_gap(ordered) * 0.5
    edges: list[list[float]] = [[ordered[0]]]
    for x in ordered[1:]:
        if x - edges[-1][-1] > tolerance:
            edges.append([])
        edges[-1].append(x)
    return tuple(sum(group) / len(group) for group in edges)


def the_places_nothing_sits_in(edges: Sequence[float]) -> tuple[float, ...]:
    """The lattice these positions are on, including the ones nothing occupies.

    A grid is defined by its pitch, not by what happens to be on it. Six tiles
    on a four-by-four board occupy three columns, and the fourth leaves no
    trace at all — so a reading built from occupied positions alone is three
    columns wide, and it is a different width on the next glance. Nothing
    downstream can model a thing that changes shape.

    Where the gaps between neighbours are whole multiples of the smallest one,
    the missing places are put back. Where they are not — prose, a heading
    above a board, anything not laid out — nothing is added, because inventing
    a lattice is worse than reading a short one.

    LIVE 2026-08-30 on play2048.co: columns at 0.184, 0.596 and 0.811, gaps of
    0.412 and 0.215, and 0.412 is twice 0.215. There is a column at 0.399 with
    nothing in it.
    """
    ordered = sorted(float(edge) for edge in edges)
    if len(ordered) < 2:
        return tuple(ordered)
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b - a > 0.0]
    if not gaps:
        return tuple(ordered)
    pitch = min(gaps)
    if pitch <= 0.0:
        return tuple(ordered)
    steps: list[int] = []
    for gap in gaps:
        many = round(gap / pitch)
        if many < 1 or abs(gap - many * pitch) > _SPACING_WOBBLE * pitch:
            # Not a multiple of the pitch, so these positions are not one
            # lattice and nothing may be inferred between them.
            return tuple(ordered)
        steps.append(many)
    if max(steps) < 2:
        return tuple(ordered)
    filled: list[float] = [ordered[0]]
    for many, edge in zip(steps, ordered[1:]):
        start = filled[-1]
        for step in range(1, many):
            filled.append(start + step * pitch)
        filled.append(edge)
    return tuple(filled)


#: How far a gap may be from a whole number of pitches and still be one. Text
#: rendering wobbles and a reading measures from the middle of a glyph, so
#: exact multiples never happen.
_SPACING_WOBBLE = 0.25


def _nearest(value: float, edges: Sequence[float]) -> int:
    return min(range(len(edges)), key=lambda index: abs(edges[index] - value))


def holds_in(
    arrangement: Arrangement,
    *,
    contains: Sequence[str] = (),
    absent: Sequence[str] = (),
    at_place: str = "",
    keeping: Sequence[str] = (),
) -> tuple[bool, str]:
    """Whether a claim about a laid-out thing is true of it, and why not.

    A claim with content can be interestingly wrong, which is the whole value
    of making one. "The view will be different" is satisfied by almost any
    keystroke on almost any screen: measured live 2026-08-26, seventeen of
    twenty such predictions held and holding told her nothing, while the
    length of her plans and the record of what her moves lead to were both
    reading that verdict as if it were confidence.

    Every part is optional, so a claim says as much as she said and no more.
    """
    said_here = {cell.says.strip().lower() for cell in arrangement.cells}
    missing = [want for want in contains if str(want).strip().lower() not in said_here]
    if missing:
        return False, f"{', '.join(missing)} did not appear"
    lingering = [gone for gone in absent if str(gone).strip().lower() in said_here]
    if lingering:
        return False, f"{', '.join(lingering)} is still there"
    if at_place and keeping:
        for want in keeping:
            where = arrangement.places_of(want)
            if not where:
                return False, f"{want} is not there at all"
            if at_place not in where:
                return False, f"{want} is {' and '.join(sorted(where))}, not {at_place}"
    return True, ""


def _row_edges(
    banded: list[list[tuple[float, str]]], placed: list[tuple[float, float, str]]
) -> tuple[float, ...]:
    """Where each row was found to be, in the order they run."""
    edges: list[float] = []
    seen = 0
    for row in banded:
        ys = [y for y, _x, _said in placed[seen : seen + len(row)]]
        seen += len(row)
        if ys:
            edges.append(sum(ys) / len(ys))
    return tuple(edges)


def _placed_in(
    placed: list[tuple[float, float, str]], like: Arrangement | None
) -> Arrangement | None:
    """This reading laid into the grid the last one found, when it fits.

    Only when everything in it lands on a row and a column the last reading
    knew about, and no two things land in the same place. Anything else is a
    different thing, or the same thing rearranged, and inferring its own shape
    is the honest answer.
    """
    if like is None or not like.down_at or not like.across_at:
        return None
    rows, columns = like.down_at, like.across_at
    room = _typical_gap(sorted(rows)) if len(rows) > 1 else 1.0
    reach = _typical_gap(sorted(columns)) if len(columns) > 1 else 1.0
    found: dict[tuple[int, int], Cell] = {}
    for y, x, said in placed:
        row, column = _nearest(y, rows), _nearest(x, columns)
        if abs(rows[row] - y) > room or abs(columns[column] - x) > reach:
            return None
        where = (row, column)
        if where in found:
            return None
        found[where] = Cell(row=row, column=column, says=str(said).strip(), at=(x, y))
    return Arrangement(
        rows=len(rows),
        columns=len(columns),
        cells=tuple(found.values()),
        down_at=rows,
        across_at=columns,
    )


def _lattices(values: Sequence[float]) -> list[tuple[float, ...]]:
    """Every even run these positions could be on, longest first.

    A pitch is not given and must not be guessed: each gap that occurs between
    neighbours is tried as one, and what comes back is every unbroken run of
    three or more places at that pitch. Three is the least that shows a rhythm
    — two positions have a gap and no rhythm.
    """
    seen = sorted(set(values))
    if len(seen) < 3:
        return [tuple(seen)]
    found: dict[tuple[float, ...], None] = {}
    for gap in sorted({round(b - a, 3) for a, b in zip(seen, seen[1:]) if b - a > 0}):
        if gap <= 0:
            continue
        for begin in seen:
            steps: dict[int, float] = {}
            for one in seen:
                away = (one - begin) / gap
                if abs(away - round(away)) <= _SPACING_WOBBLE:
                    steps[int(round(away))] = one
            order = sorted(steps)
            run, ends = 1, [order[0]] if order else []
            for before, after in zip(order, order[1:]):
                if after - before == 1:
                    run += 1
                    if run >= 3:
                        found[tuple(steps[at] for at in range(after - run + 1, after + 1))] = None
                else:
                    run = 1
    return sorted(found, key=len, reverse=True)[:24] or [tuple(seen)]


def the_part_laid_out_regularly(
    cells: Sequence[tuple[float, float, str]],
    like: "Arrangement | None" = None,
) -> tuple[tuple[float, float, str], ...]:
    """The part of a reading that is on a lattice, out of everything else.

    A thing she acts in is laid out: its places sit at an even pitch across and
    down, and it is FULL of them. A page around it is not — headings, links,
    prose and adverts fall where they fall, and any rhythm among them is a
    coincidence between two of them rather than a grid.

    Nothing here knows what a board is. What it knows is that being laid out
    means an even pitch in both directions at once, and that a thing is dense
    in its own grid where a coincidence is not. Neither axis decides alone:
    measured on a real page, the strongest rhythm among the x positions was a
    spurious ninety-three shared by three pieces of furniture, beating the
    board's own hundred and thirteen. It is the pair of axes, scored by how
    much of the block is actually occupied, that picks the board out.

    LIVE 2026-08-31 on 2048game.com: she read the whole page as the thing —
    forty-four columns by thirty-seven rows — so of thirty moves only five were
    comparable, the rule that governs the board scored nought out of five, and
    every sentence she said about the position was narration over a reading
    that was not of the board. She was playing correctly and learning nothing.
    """
    placed = [(y, x, said) for y, x, said in cells if str(said or "").strip()]
    if len(placed) < 4:
        return tuple(placed)

    # The shape it had last time, which it still has.
    #
    # A thing does not change size between glances. Cropping each glance on its
    # own merits found the board and then found a different block next time —
    # 44x37, then 21x20, 14x13, 9x9, 19x12 — and two readings that disagree
    # about the shape cannot be compared at all, so half the moves became
    # unreadable and nothing could be learned from them. The furniture around a
    # thing changes; the thing does not.
    was = (int(getattr(like, "rows", 0) or 0), int(getattr(like, "columns", 0) or 0))
    knows_the_shape = was[0] >= 2 and was[1] >= 2

    best: tuple[tuple[float, float, str], ...] = ()
    best_score = 0.0
    for across in _lattices([x for _y, x, _said in placed]):
        wide = set(across)
        for down in _lattices([y for y, _x, _said in placed]):
            tall = set(down)
            on_it = tuple(
                (y, x, said) for y, x, said in placed if x in wide and y in tall
            )
            room = len(wide) * len(tall)
            if room < 4 or len(on_it) < 4:
                continue
            # How much of the block it fills, times how much of it there is.
            # A thing is most of its own grid; a coincidence among furniture is
            # a handful of places in a large one.
            #
            # Two other scorings were tried against uniformly random furniture
            # and both were worse on the real page, which is what this is for.
            # A page is not random noise: its text sits in a few columns and
            # its rhythms are short. Tuning against noise it will never see
            # made it keep seventeen-by-seventeen blocks of nothing.
            score = (len(on_it) / room) * len(on_it)
            if knows_the_shape and (len(down), len(wide)) == was:
                # It is the shape she already knows it is, which is worth
                # something and not everything. Made decisive, the first
                # glance became the only glance: one furniture column crept
                # into it and every reading afterwards kept the column,
                # because nothing was allowed to beat the shape she had.
                #
                # A preference, so a block that is plainly fuller still wins
                # and a tie goes to being comparable with the reading before.
                score *= 1.5
            if score > best_score:
                best, best_score = on_it, score
    # Cropping to nothing is not a reading. Where no block stands out, the
    # whole of it is the honest answer.
    return best if len(best) >= 4 else tuple(placed)
