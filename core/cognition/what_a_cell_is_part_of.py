"""core/cognition/what_a_cell_is_part_of.py — what is true of a cell besides its value.

The index language says where a cell comes from. The value language says what
a value becomes. Between them sits the thing neither can mention: that a cell
belongs to something — a run, a region, a row, an edge — and that what happens
to it depends on which.

``primitive_invention`` says this in its own words and then does not act on it:
"the basis had order, symmetry and adjacency — geometry and number, in the
core-knowledge sense — and nothing for OBJECTHOOD: no way to say that some
cells belong together and travel as a set". Grouping by residue is called the
smallest form of it there, and it is: residue classes are groups a cell falls
into by arithmetic on its position alone, and they are the same groups whatever
the cells hold.

What is here is the other half. Every property is computed from the position,
the length, and the state — nothing is read from a task, a family or a name —
and each one is a number per cell, so a rule over them is the same kind of
object as a rule over values: a table, fitted from the observations, refused
when it turns out to be a transcript.

Regions need a shape, and a shape comes from the divisors of the length, the
same way ``IndexProgram`` kind "shaped" gets one. A sequence of twelve might be
three rows of four; whether it is, is decided by whether a rule stated in those
terms holds.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "THE_PROPERTIES",
    "shapes_of",
    "what_is_true_of_each_cell",
]

#: What can be said about a cell besides what it holds. Each is one number per
#: cell, computed from the position, the length and the state.
THE_PROPERTIES: tuple[str, ...] = (
    "value",
    "row",
    "column",
    "steps to the nearest edge",
    "how many cells hold the same",
    "the size of the region it is in",
    "the rank of its region by size",
)


def shapes_of(size: int) -> list[tuple[int, int]]:
    """Every way this length reads as a grid, from its divisors.

    A row of one and a column of one are not shapes: they say the sequence is
    itself, which every property already knows.
    """

    found: list[tuple[int, int]] = []
    for rows in range(2, size):
        if rows * rows > size:
            break
        if size % rows:
            continue
        for high, across in ((rows, size // rows), (size // rows, rows)):
            if high >= 2 and across >= 2 and (high, across) not in found:
                found.append((high, across))
    return found


def _regions(state: Sequence[Any], rows: int, cols: int) -> tuple[list[int], list[int]]:
    """Which connected region of like cells each position belongs to, and how big.

    Four-connected, because a diagonal touch is a different claim about what
    belongs together and this is the weaker one. Flood filled iteratively: a
    thousand-cell region recurses a thousand deep, and the interpreter stops
    long before the sequence does.
    """

    size = rows * cols
    where: list[int] = [-1] * size
    sizes: list[int] = []
    for start in range(size):
        if where[start] >= 0:
            continue
        mark = len(sizes)
        held = state[start]
        stack = [start]
        where[start] = mark
        count = 0
        while stack:
            at = stack.pop()
            count += 1
            row, col = divmod(at, cols)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, c = row + dr, col + dc
                if not (0 <= r < rows and 0 <= c < cols):
                    continue
                there = r * cols + c
                if where[there] >= 0 or state[there] != held:
                    continue
                where[there] = mark
                stack.append(there)
        sizes.append(count)
    return where, sizes


def what_is_true_of_each_cell(
    state: Sequence[Any], *, rows: int = 0, cols: int = 0
) -> dict[str, tuple[Any, ...]]:
    """Every property, one value per cell, for this state at this shape.

    Without a shape only the properties that need none are returned, which is
    what a sequence whose length is prime gets and is the right answer for it.
    """

    size = len(state)
    if size == 0:
        return {}
    same = {}
    for one in state:
        try:
            same[one] = same.get(one, 0) + 1
        except TypeError:  # unhashable cell: it belongs to no class of like cells
            same = {}
            break
    found: dict[str, tuple[Any, ...]] = {"value": tuple(state)}
    if same:
        found["how many cells hold the same"] = tuple(same[one] for one in state)
    if rows < 2 or cols < 2 or rows * cols != size:
        return found
    found["row"] = tuple(index // cols for index in range(size))
    found["column"] = tuple(index % cols for index in range(size))
    found["steps to the nearest edge"] = tuple(
        min(index // cols, rows - 1 - index // cols, index % cols, cols - 1 - index % cols)
        for index in range(size)
    )
    where, sizes = _regions(state, rows, cols)
    found["the size of the region it is in"] = tuple(sizes[mark] for mark in where)
    order = sorted(range(len(sizes)), key=lambda mark: (-sizes[mark], mark))
    rank = {mark: at for at, mark in enumerate(order)}
    found["the rank of its region by size"] = tuple(rank[mark] for mark in where)
    return found
