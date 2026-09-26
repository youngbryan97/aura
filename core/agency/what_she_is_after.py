"""What she was asked to reach, read so that any situation can be measured against it.

Her search compares futures by how near each is to the goal, and until now a
goal was near or far only if it was a number: "2048" could be measured against
a board and nothing else could. Asked to put a puzzle in order, every future
was equally far from done, and the search ranked them by everything except
the thing she was asked for.

Two kinds of goal can be measured without knowing anything about the world
they are in.

A thing to make, named by its value. Near in doublings of the largest thing
there, because in anything built by combining a step is a doubling.

A layout to make: rows of what goes where, written the way a person writes a
grid, with "/" between rows and "_" for a place left empty —
"1 2 3 / 4 5 6 / 7 8 _". Near by how far each thing is from where it belongs,
counted in steps along the grid, because a thing one step from its place is
nearly there and one across the board is not, and both are equally "wrong" to
a count of what is already in place.

Anything else names nothing she can measure, and this says so rather than
guessing.
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass
from typing import Any

from core.utils.written_layout import EMPTY_MARKS as _EMPTY
from core.utils.written_layout import written_layout

__all__ = ["Goal", "goal_in"]

#: A target written as a number.
_A_NUMBER = re.compile(r"^\d[\d,]*(?:\.\d+)?$")


@dataclass(frozen=True)
class Goal:
    """What she is after, in a form a situation can be measured against."""

    said: str = ""
    #: A thing to make, by value. Nought where the goal is not one.
    number: float = 0.0
    #: A layout to make, row by row; "" is a place left empty.
    layout: tuple[tuple[str, ...], ...] = ()

    def names_something(self) -> bool:
        return bool(self.number) or bool(self.layout)

    def reached(self, state: Any) -> bool:
        if self.number:
            return _biggest(state) >= self.number
        if self.layout:
            return self.nearness(state) >= 1.0
        return False

    def nearness(self, state: Any) -> float:
        """How near this situation is to the goal, where reaching it is one."""
        if self.number:
            biggest = _biggest(state)
            if biggest <= 0:
                return 0.0
            if biggest >= self.number:
                return 1.0
            if self.number <= 1.0:
                return 0.0
            return max(0.0, min(1.0, math.log2(max(1.0, biggest)) / math.log2(self.number)))
        if self.layout:
            return _how_near_the_layout(state, self.layout)
        return 0.0


def _biggest(state: Any) -> float:
    numbers = getattr(state, "numbers", None)
    values = [value for value in (numbers() if callable(numbers) else ()) if value is not None]
    return float(max(values)) if values else 0.0


def _how_near_the_layout(state: Any, layout: tuple[tuple[str, ...], ...]) -> float:
    """The share of the way each wanted thing is to its place, averaged.

    Each thing the layout names is measured from the nearest one like it on
    the board, in steps along rows and columns, against the furthest two
    places can be apart. A thing that is not there at all is as far as it can
    be. One exactly in place, everywhere, is the goal.
    """
    rows = int(getattr(state, "rows", 0) or 0)
    columns = int(getattr(state, "columns", 0) or 0)
    if (rows, columns) != (len(layout), len(layout[0]) if layout else 0):
        return 0.0
    where: dict[str, list[tuple[int, int]]] = {}
    for cell in getattr(state, "cells", ()) or ():
        where.setdefault(str(cell.says).strip(), []).append((int(cell.row), int(cell.column)))
    furthest = max(1, rows - 1 + columns - 1)
    shares: list[float] = []
    for row, line in enumerate(layout):
        for column, wanted in enumerate(line):
            if not wanted:
                continue
            found = where.get(wanted)
            if not found:
                shares.append(0.0)
                continue
            apart = min(abs(row - r) + abs(column - c) for r, c in found)
            shares.append(1.0 - apart / furthest)
    if not shares:
        return 0.0
    exact = all(share == 1.0 for share in shares) and _nothing_else_in_the_way(where, layout)
    near = sum(shares) / len(shares)
    # In place everywhere is the goal; anything short of it stays short of
    # one, so a board with every piece a step away is never mistaken for done.
    return 1.0 if exact else min(near, 1.0 - 1.0 / (2.0 * len(shares) * furthest))


def _nothing_else_in_the_way(
    where: dict[str, list[tuple[int, int]]], layout: tuple[tuple[str, ...], ...]
) -> bool:
    """Every place the layout names is held by what it names, and every empty one is empty."""
    held = {place: said for said, places in where.items() for place in places}
    for row, line in enumerate(layout):
        for column, wanted in enumerate(line):
            if held.get((row, column), "") != wanted:
                return False
    return True


def _layout_in(said: str) -> tuple[tuple[str, ...], ...]:
    written = written_layout(said)
    if not written:
        return ()
    rows = [row.split() for row in written.split(" / ")]
    return tuple(tuple("" if token in _EMPTY else token for token in row) for row in rows)


@functools.lru_cache(maxsize=64)
def goal_in(toward: str) -> Goal:
    """What a goal says, read once."""
    said = " ".join(str(toward or "").replace("\n", " / ").split())
    if not said:
        return Goal()
    plain = said.replace(",", "")
    if _A_NUMBER.match(plain):
        try:
            return Goal(said=said, number=float(plain))
        # not a failure: a value that is not a number is not one this can read.
        except ValueError:
            return Goal(said=said)
    return Goal(said=said, layout=_layout_in(said))
