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
    how to combine them the mean · the largest · the smallest · the share
                        that hold

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


def _what_it_offers(state: Any) -> list[tuple[float, float, int, int]] | None:
    """What a situation says about itself, when it can say anything.

    Every reader below reaches for ``cells`` with a row and a column, which
    is a laid-out arrangement and nothing else. So the space of measures she
    can compose is a space of measures about boards, and in a domain that is
    not a board she can invent nothing — not because the algebra is too
    small, but because the only thing that can feed it is one kind of world.

    That is the shape of the whole criticism one level up: a developmental
    language can be universal in expression while the agent stays narrow in
    what evidence reaches it. Here it was narrow in what a situation is
    allowed to BE.

    A situation that offers its own observations is read as it offers them.
    Nothing about a board is assumed, and nothing about a board is lost: an
    arrangement that does not offer them is read the way it always was.
    """

    offers = getattr(state, "observations", None)
    if not callable(offers):
        return None
    try:
        given = offers()
    except (AttributeError, TypeError, ValueError):
        return None
    seen: list[tuple[float, float, int, int]] = []
    for one in given or ():
        try:
            value, other, row, column = one
            seen.append((float(value), float(other), int(row), int(column)))
        except (TypeError, ValueError):
            continue
    return seen


def _each_thing(state: Any) -> list[tuple[float, float, int, int]]:
    """Every thing in it, with its place."""
    offered = _what_it_offers(state)
    if offered is not None:
        return offered
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
    offered = _what_it_offers(state)
    if offered is not None:
        held = {(row, column): value for value, _o, row, column in offered}
        pairs: list[tuple[float, float, int, int]] = []
        for (row, column), value in held.items():
            for beside in ((row, column + 1), (row + 1, column)):
                other = held.get(beside)
                if other is not None:
                    pairs.append((float(value), float(other), row, column))
        return pairs
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

    offered = _what_it_offers(state)
    if offered is not None:
        # Along a line means along a row, for a situation that says where its
        # observations sit and nothing more.
        rows: dict[int, list[tuple[int, float]]] = {}
        for value, _other, row, column in offered:
            rows.setdefault(row, []).append((column, float(value)))
        pairs = []
        for row, held in sorted(rows.items()):
            held.sort()
            for position, ((_a, one), (_b, other)) in enumerate(zip(held, held[1:])):
                pairs.append((one, other, row, position))
        return pairs

    pairs: list[tuple[float, float, int, int]] = []
    for index, line in enumerate(_lines_of(state)):
        for position, (one, other) in enumerate(zip(line, line[1:])):
            pairs.append((float(one), float(other), index, position))
    return pairs


def _things_at_an_edge(state: Any) -> list[tuple[float, float, int, int]]:
    """Every thing sitting on the outside of it."""
    offered = _what_it_offers(state)
    if offered is not None:
        if not offered:
            return []
        least_row = min(one[2] for one in offered)
        most_row = max(one[2] for one in offered)
        least_column = min(one[3] for one in offered)
        most_column = max(one[3] for one in offered)
        return [
            one
            for one in offered
            if one[2] in (least_row, most_row) or one[3] in (least_column, most_column)
        ]
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


# ── how to combine what was taken ────────────────────────────────────────

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
