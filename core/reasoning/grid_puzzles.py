"""A grid puzzle someone pastes, worked out rather than narrated.

A Sudoku asked of a language model is answered by a language model: it writes
a filled grid that looks right, and nothing checks that every row, column and
box holds each digit once. The same was true of seating puzzles until
core/reasoning/positional_constraints.py, and the fix is the same: where the
answer follows from the constraints, the runtime computes it.

Nothing here solves Sudoku. It reads a square grid of a size whose side is a
square — four or nine — into variables, values and constraints, and hands them
to core/reasoning/settled_by_neighbours.py, which knows nothing about grids.
What comes back says how it is known: proven the only solution, proven to have
none, more than one found, or found without the uniqueness being settled.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from core.reasoning.settled_by_neighbours import Settled, a_problem, settle

__all__ = ["GridPuzzle", "describe_grid_answer", "parse_grid_puzzle", "solve_grid_puzzle"]

#: A blank is any of these; a given is a digit from one to the side.
_BLANKS = frozenset(".0_*?x")
_CELL = re.compile(r"[0-9._*?xX]")
_ASKS_FOR_A_PUZZLE = re.compile(r"\b(?:sudoku|puzzle|grid|solve|fill)\b", re.IGNORECASE)


@dataclass(frozen=True)
class GridPuzzle:
    side: int
    #: Row by row, 0 for a blank.
    cells: tuple[int, ...]

    @property
    def box(self) -> int:
        return math.isqrt(self.side)


def _rows_of_cells(text: str) -> list[list[str]]:
    """Lines that are nothing but cells and separators, as their cells."""
    rows = []
    for line in str(text or "").splitlines():
        stripped = re.sub(r"[\s|+\-,]", "", line)
        if stripped and all(_CELL.fullmatch(char) for char in stripped):
            rows.append(list(stripped))
    return rows


def parse_grid_puzzle(text: object) -> GridPuzzle | None:
    """A square grid with at least one blank, read from lines or from one run of cells.

    Four by four is common enough as plain digits that it is read only where
    the message also asks for a puzzle to be solved; nine by nine is not.
    """
    raw = str(text or "")
    for side in (9, 4):
        rows = [row for row in _rows_of_cells(raw) if len(row) == side]
        cells: list[str] = []
        if len(rows) >= side:
            cells = [char for row in rows[:side] for char in row]
        else:
            run = re.search(rf"(?<![0-9.])[0-9.]{{{side * side}}}(?![0-9.])", re.sub(r"[ \t]", "", raw))
            if run:
                cells = list(run.group(0))
        if len(cells) != side * side:
            continue
        values = []
        for char in cells:
            if char.lower() in _BLANKS:
                values.append(0)
            elif char.isdigit() and 1 <= int(char) <= side:
                values.append(int(char))
            else:
                values = []
                break
        if not values or 0 not in values:
            continue
        if side == 4 and not _ASKS_FOR_A_PUZZLE.search(raw):
            continue
        return GridPuzzle(side, tuple(values))
    return None


def _groups(side: int) -> list[list[str]]:
    box = math.isqrt(side)
    rows = [[f"r{r}c{c}" for c in range(side)] for r in range(side)]
    columns = [[f"r{r}c{c}" for r in range(side)] for c in range(side)]
    boxes = [
        [f"r{r}c{c}" for r in range(top, top + box) for c in range(left, left + box)]
        for top in range(0, side, box)
        for left in range(0, side, box)
    ]
    return rows + columns + boxes


def solve_grid_puzzle(puzzle: GridPuzzle, *, budget_s: float) -> Settled:
    side = puzzle.side
    domains = {
        f"r{r}c{c}": (puzzle.cells[r * side + c],) if puzzle.cells[r * side + c] else tuple(range(1, side + 1))
        for r in range(side)
        for c in range(side)
    }
    groups = _groups(side)
    pairs = {tuple(sorted((a, b))) for group in groups for a in group for b in group if a < b}
    return settle(
        a_problem(domains, [(pair, lambda x, y: x != y) for pair in sorted(pairs)], groups),
        budget_s=budget_s,
        find_every=False,
    )


def describe_grid_answer(puzzle: GridPuzzle, settled: Settled) -> str:
    """The filled grid and how it is known, or why there is none."""
    if settled.solution is None:
        return f"No filled grid satisfies it: {settled.why}."
    side, box = puzzle.side, puzzle.box
    lines = []
    for r in range(side):
        row = [str(settled.solution[f"r{r}c{c}"]) for c in range(side)]
        lines.append(" | ".join(" ".join(row[at:at + box]) for at in range(0, side, box)))
        if (r + 1) % box == 0 and r + 1 < side:
            lines.append("-+-".join("-" * (2 * box - 1) for _ in range(side // box)))
    if settled.proven and len(settled.solutions) == 1:
        how = "Every row, column and box checked; proven the only solution."
    elif len(settled.solutions) > 1:
        how = "Every row, column and box checked, but this is one of several solutions: the puzzle as given is not unique."
    else:
        how = "Every row, column and box checked; whether it is the only solution was not settled in the time allowed."
    return "\n".join(lines) + "\n" + how
