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

__all__ = ["Arrangement", "Cell", "arranged", "holds_in", "EMPTY_CELL"]

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

    columns = _column_edges([x for row in banded for x, _said in row])
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
