"""Measures of a situation she can put together herself.

Everything she judges a situation by was written down by somebody. Nearness to
what was asked for, room left to act in, how much runs in order, whether her
own stated line still holds — and, added by hand on 2026-08-29 after a sweep,
how near neighbouring things are in value. That last one doubled her play, and
a person found it, wrote its arithmetic, ran the experiment and admitted it.

The loop underneath could already learn weights, rules, models and strategies.
It could not look at what it fails to explain and say: my measures cannot tell
these two situations apart, here is a NEW property, here is how to compute it,
and here is whether including it actually helps.

This is the space it would have to search to say that. Not a list of measures
anybody wrote — a small algebra whose closure contains them, so a measure is
composed rather than chosen:

    what to look at     every thing · neighbouring pairs · things along a line
                        · things at an edge · the free places
    what to take of it  its value · its size in doublings · the gap between a
                        pair · whether a pair is in order · how far it sits
                        from an edge
    how to sum up       the mean · the largest · the smallest · the share that
                        hold

The measure a person added is one point in that space: neighbouring pairs, the
gap between them in doublings, averaged, and read the other way up. So is
monotonic order: pairs along a line, whether each is in order, the share that
hold. Neither is privileged here, and neither was put in by name.

Nothing in this file knows what a board, a tile or a game is. It knows that a
situation is a laid-out set of numbers, which is as true of a price list, a
seating plan, a heat map or a schedule.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterator, Sequence

__all__ = [
    "AT",
    "OF",
    "SUMMED",
    "Measure",
    "every_measure",
    "measure_named",
]

logger = logging.getLogger("Aura.InventingAMeasure")

#: Anything smaller than this is nothing, and dividing by it is a lie.
_ALMOST_NOTHING = 1e-9


# ── what to look at ──────────────────────────────────────────────────────
#
# Each returns a list of *observations*. An observation is one or two numbers
# and where they sat, which is everything the takes below need.


def _each_thing(state: Any) -> list[tuple[float, float, int, int]]:
    """Every thing in it, with its place."""
    cells = getattr(state, "cells", ()) or ()
    seen: list[tuple[float, float, int, int]] = []
    for cell in cells:
        value = cell.number() if hasattr(cell, "number") else None
        if value is None:
            continue
        seen.append((float(value), float(value), int(cell.row), int(cell.column)))
    return seen


def _neighbouring_pairs(state: Any) -> list[tuple[float, float, int, int]]:
    """Every pair of things that sit next to each other."""
    held = {
        (int(cell.row), int(cell.column)): cell.number()
        for cell in (getattr(state, "cells", ()) or ())
        if hasattr(cell, "number") and cell.number() is not None
    }
    pairs: list[tuple[float, float, int, int]] = []
    for (row, column), value in held.items():
        for beside in ((row, column + 1), (row + 1, column)):
            other = held.get(beside)
            if other is not None:
                pairs.append((float(value), float(other), row, column))
    return pairs


def _pairs_along_a_line(state: Any) -> list[tuple[float, float, int, int]]:
    """Every consecutive pair of things reading along a row or a column.

    Different from neighbouring pairs: a gap between them does not break the
    pair, so this is about the order things come in rather than about what
    touches what.
    """
    from core.agency.how_good_is_this import _lines_of

    pairs: list[tuple[float, float, int, int]] = []
    for index, line in enumerate(_lines_of(state)):
        for position, (one, other) in enumerate(zip(line, line[1:])):
            pairs.append((float(one), float(other), index, position))
    return pairs


def _things_at_an_edge(state: Any) -> list[tuple[float, float, int, int]]:
    """Every thing sitting on the outside of it."""
    rows = int(getattr(state, "rows", 0) or 0)
    columns = int(getattr(state, "columns", 0) or 0)
    if rows <= 0 or columns <= 0:
        return []
    return [
        seen
        for seen in _each_thing(state)
        if seen[2] in (0, rows - 1) or seen[3] in (0, columns - 1)
    ]


#: Where to look. Nothing here is about any particular kind of thing.
AT: dict[str, Callable[[Any], list[tuple[float, float, int, int]]]] = {
    "everything": _each_thing,
    "neighbours": _neighbouring_pairs,
    "along a line": _pairs_along_a_line,
    "at an edge": _things_at_an_edge,
}


# ── what to take of each observation ─────────────────────────────────────


def _its_size(seen: tuple[float, float, int, int], state: Any) -> float:
    """How big it is, against the biggest thing there is."""
    from core.agency.how_good_is_this import _biggest

    most = _biggest(state)
    return seen[0] / most if most > _ALMOST_NOTHING else 0.0


def _in_doublings(seen: tuple[float, float, int, int], state: Any) -> float:
    """How big it is counted in doublings, which is how combining grows."""
    from core.agency.how_good_is_this import _biggest

    most = _biggest(state)
    if seen[0] <= 0.0 or most <= 1.0:
        return 0.0
    return math.log2(seen[0]) / math.log2(most)


def _the_gap(seen: tuple[float, float, int, int], state: Any) -> float:
    """How far apart the two are, in doublings, as a share of the whole range."""
    from core.agency.how_good_is_this import _biggest

    one, other = seen[0], seen[1]
    most = _biggest(state)
    if one <= 0.0 or other <= 0.0 or most <= 1.0:
        return 0.0
    return min(1.0, abs(math.log2(one) - math.log2(other)) / math.log2(most))


def _is_in_order(seen: tuple[float, float, int, int], _state: Any) -> float:
    """Whether the pair reads the same way round as a rising line."""
    return 1.0 if seen[1] >= seen[0] else 0.0


def _how_far_from_an_edge(seen: tuple[float, float, int, int], state: Any) -> float:
    """How far into the middle it sits, where nought is against the outside."""
    rows = int(getattr(state, "rows", 0) or 0)
    columns = int(getattr(state, "columns", 0) or 0)
    if rows <= 1 or columns <= 1:
        return 0.0
    down = min(seen[2], rows - 1 - seen[2]) / ((rows - 1) / 2.0)
    across = min(seen[3], columns - 1 - seen[3]) / ((columns - 1) / 2.0)
    return (down + across) / 2.0


#: What to take of what she looked at.
OF: dict[str, Callable[[tuple[float, float, int, int], Any], float]] = {
    "how big it is": _its_size,
    "its size in doublings": _in_doublings,
    "the gap between them": _the_gap,
    "whether it is in order": _is_in_order,
    "how far from an edge": _how_far_from_an_edge,
}


# ── how to sum the takings up ────────────────────────────────────────────

SUMMED: dict[str, Callable[[Sequence[float]], float]] = {
    "on average": lambda taken: sum(taken) / len(taken),
    "at most": max,
    "at least": min,
    "how many hold": lambda taken: sum(1 for one in taken if one >= 0.5) / len(taken),
}


@dataclass(frozen=True)
class Measure:
    """One property of a situation, composed rather than chosen.

    ``the_other_way_up`` because half the useful properties are the absence of
    something: a good situation has SMALL gaps between neighbours, and a
    measure that returns the gap is the same fact read upside down. Both
    readings are in the space, and which one helps is a question for evidence.
    """

    at: str
    of: str
    summed: str
    the_other_way_up: bool = False

    @property
    def name(self) -> str:
        said = f"{self.of} {self.at}, {self.summed}"
        return f"how little — {said}" if self.the_other_way_up else said

    def read(self, state: Any) -> float:
        """This property of that situation, between nought and one."""
        try:
            seen = AT[self.at](state)
        except (KeyError, AttributeError, TypeError, ValueError):
            return 0.0
        if not seen:
            return 0.0
        try:
            taken = [OF[self.of](one, state) for one in seen]
            summed = float(SUMMED[self.summed](taken))
        except (KeyError, AttributeError, TypeError, ValueError, ZeroDivisionError):
            return 0.0
        summed = max(0.0, min(1.0, summed))
        return (1.0 - summed) if self.the_other_way_up else summed


def every_measure() -> Iterator[Measure]:
    """The whole space, so nothing in it had to be thought of in advance."""
    for at, of, summed, flipped in product(AT, OF, SUMMED, (False, True)):
        yield Measure(at=at, of=of, summed=summed, the_other_way_up=flipped)


def measure_named(name: str) -> Measure | None:
    """The one measure whose name is that, for talking about one out loud."""
    for measure in every_measure():
        if measure.name == name:
            return measure
    return None
