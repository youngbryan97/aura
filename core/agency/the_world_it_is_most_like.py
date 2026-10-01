"""Which world she has already worked out is shaped most like the one in front of her.

What she learns is kept per world, and carried to another world only when the
two share a name for their kind: the same size, the same number of acts, both
countable. A five-by-five board that moves exactly like the four-by-four she has
mastered shares none of that, so she met it as ignorant as she met the first.
The words on a world are no guide either: a board of 3s, 6s and 12s moves like
a board of 2s, 4s and 8s, and a board of letters laid out the same way does not.

What carries is the shape. Which things on it are made of which, which sit
beside which, which are there more than once, what her acts are. That is a
relation graph, and `core.cognition.structure_mapping` aligns two of them on
their relations rather than their names. The world she has solved whose shape
aligns best with this one's, by more than a scrambled copy of it does, lends
what it taught her about how things move: no more than a fresh start is
worth, scaled by how far the match clears the scrambled control. Her own moves
decide from there, the way they decide anything carried over.

An external review asked for exactly this on 2026-10-01: the structure mapper
was the best generality idea in the repository and nothing she played used it.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.cognition.structure_mapping import Graph, Relation, map_structures, scrambled

__all__ = [
    "Likeness",
    "graph_from_memory",
    "graph_to_memory",
    "lend_to",
    "lent",
    "most_like",
    "shape_of",
    "solved_worlds",
]

logger = logging.getLogger("Aura.TheWorldItIsMostLike")

#: Scrambled copies a match is held against. The control the mapper's own
#: module uses, at the count it uses.
SCRAMBLED_COPIES = 20

#: Where among the scrambled copies a match has to sit to count: above the
#: ninety-fifth percentile of them, not their mean. A distractor scored 0.67
#: against a null mean of 0.50 in the review, which clears a mean and says
#: nothing; the same comparison against the upper tail is the one that rules
#: something out.
CLEARS = 0.95


@dataclass(frozen=True)
class Likeness:
    """How one solved world lines up with this one, and what it is worth."""

    world: str
    #: Shared structure, as a share of both graphs' relations.
    score: float
    #: The same measure on scrambled copies of the solved world, at `CLEARS`.
    scrambled_at: float
    #: This world's things and acts, and what each was in the solved one.
    mapping: Mapping[str, str] = field(default_factory=dict)

    @property
    def clears(self) -> bool:
        return self.score > self.scrambled_at

    @property
    def margin(self) -> float:
        """How far above the scrambled control, as a share of the room above it."""
        if not self.clears or self.scrambled_at >= 1.0:
            return 0.0
        return (self.score - self.scrambled_at) / (1.0 - self.scrambled_at)

    def says(self) -> str:
        return (
            f"{self.world}: {self.score:.2f} of the shape in common, "
            f"against {self.scrambled_at:.2f} for the same world scrambled"
        )


# ── a world's shape, as relations ────────────────────────────────────────


def _thing(said: str) -> str:
    return f"thing {said}"


def shape_of(state: Any, acts: Sequence[str] = ()) -> Graph:
    """What a world shows on a reading, as relations between its things.

    Nothing here knows any world. A thing is whatever a place says; a relation
    is something true of the things as read: that one is there more than once,
    that one is the next size up from another, that one is two of another put
    together, that two sit side by side.

    Not here: that a thing is there at all, and what her acts are. Every world
    has things and acts, so those are an inventory and not a shape, and
    counted as shape they matched every world to every other. Measured on the
    first version: a held-out board's first look, two equal tiles, matched a
    board of letters ten times in ten and cleared its scrambled control,
    because four "an act" facts lined up with four and a scrambled copy put
    them on things. ``acts`` is accepted so a caller can pass what it has; the
    shape does not use it until something is known about what the acts do.
    """
    relations: list[Relation] = []
    cells = tuple(getattr(state, "cells", ()) or ())
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell.says] = counts.get(cell.says, 0) + 1
    for said, count in sorted(counts.items()):
        if count > 1:
            relations.append(Relation("more than once", (_thing(said),)))
    valued = sorted(
        {
            (number, cell.says)
            for cell in cells
            if (number := cell.number()) is not None
        }
    )
    for (low, low_said), (high, high_said) in zip(valued, valued[1:], strict=False):
        if high > low:
            relations.append(Relation("next size up", (_thing(high_said), _thing(low_said))))
    values = {said: number for number, said in valued}
    for high_said, high in values.items():
        for low_said, low in values.items():
            if low > 0 and math.isclose(high, low + low):
                relations.append(Relation("two of", (_thing(high_said), _thing(low_said))))
    beside: set[tuple[str, str]] = set()
    for cell in cells:
        for row, column in ((cell.row, cell.column + 1), (cell.row + 1, cell.column)):
            other = state.at(row, column)
            if other is None:
                continue
            pair = tuple(sorted((cell.says, other.says)))
            beside.add(pair)  # type: ignore[arg-type]
    for first, second in sorted(beside):
        if first == second:
            relations.append(Relation("beside its own kind", (_thing(first),)))
        else:
            relations.append(Relation("beside", (_thing(first), _thing(second))))
    return Graph(name="a look", relations=tuple(relations))


def graph_to_memory(graph: Graph) -> list[list[Any]]:
    """A graph in a form that survives the process."""
    return [[relation.predicate, list(relation.args)] for relation in graph.relations]


def graph_from_memory(held: Any, name: str = "") -> Graph | None:
    """A graph kept by `graph_to_memory`, or None for anything else."""
    if not isinstance(held, list):
        return None
    relations: list[Relation] = []
    for item in held:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], (list, tuple))
        ):
            relations.append(Relation(item[0], tuple(str(arg) for arg in item[1])))
    return Graph(name=name, relations=tuple(relations)) if relations else None


# ── finding the closest ──────────────────────────────────────────────────


def _shared(here: Graph, there: Graph) -> tuple[float, dict[str, str]]:
    """Shared structure, as a share of both graphs, and the mapping that found it.

    Both graphs, not one: measured over this world alone, a look with two
    things in it is wholly contained in any solved world with more, and every
    world scores one.
    """
    total = len(here.relations) + len(there.relations)
    if not total:
        return 0.0, {}
    alignment = map_structures(here, there, same_vocabulary=True)
    if alignment is None:
        return 0.0, {}
    return 2.0 * len(alignment.matched) / total, dict(alignment.mapping)


def _upper_tail(scores: Sequence[float], at: float) -> float:
    ordered = sorted(scores)
    if not ordered:
        return 1.0
    return ordered[min(len(ordered) - 1, max(0, math.ceil(at * len(ordered)) - 1))]


def most_like(
    here: Graph,
    solved: Mapping[str, Graph],
    *,
    copies: int = SCRAMBLED_COPIES,
    teaches: Mapping[str, str] | None = None,
) -> Likeness | None:
    """The solved world shaped most like this one, if it clears its control.

    The closest world is the one sharing the most structure. Only then is it
    held against scrambled copies of itself: the same things and the same
    relation words, with which thing fills which place in each relation drawn
    afresh. A closest world that matches no better than its own scrambled
    copies matched by arithmetic, and nothing is lent.

    The control checks the choice and does not make it. Chosen the other way
    round — the best margin among the worlds that clear — a small world that
    matched perfectly lost to a bigger one that matched worse, because a small
    graph's scrambled copies are often the graph itself. Measured: a letters
    board matched the letters world it was like at 1.0, could not clear that
    world's control, and was lent a sliding number world's rule instead.

    Two worlds level at the top lend nothing either, where they taught
    different things: which one this is like is exactly what is not known.
    ``teaches`` names what each world taught — the rule she settled on there —
    so a tie between two that taught the same is no tie at all.
    """
    if not here.relations:
        return None
    scored: list[tuple[float, str, dict[str, str]]] = []
    for world, there in sorted(solved.items()):
        if not there.relations:
            continue
        score, mapping = _shared(here, there)
        scored.append((score, world, mapping))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    score, world, mapping = scored[0]
    if score <= 0.0:
        return None
    lessons = teaches or {}
    level = [other for other_score, other, _m in scored[1:] if math.isclose(other_score, score)]
    if any(not lessons.get(other) or lessons.get(other) != lessons.get(world) for other in level):
        return None
    there = solved[world]
    rng = random.Random(f"{world}:{len(here.relations)}:{len(there.relations)}")
    controls = [_shared(here, scrambled(there, rng))[0] for _ in range(copies)]
    likeness = Likeness(
        world=world,
        score=score,
        scrambled_at=_upper_tail(controls, CLEARS),
        mapping=mapping,
    )
    logger.debug("shaped like %s", likeness.says())
    return likeness if likeness.clears else None


# ── what is lent ─────────────────────────────────────────────────────────


def lent(knew: Mapping[str, Any], likeness: Likeness) -> dict[str, Any]:
    """What a solved world lends this one, in this world's own terms.

    How things move: the rule she worked out there and how often each rule was
    right, which is about rows and columns and not about any name. What turns
    up on its own, renamed through the alignment from that world's things to
    this one's, and dropped where a thing has no counterpart here.

    Not lent: which of her acts pushes which way, which acts do anything, what
    she got good at. Those are keyed by the names of acts, and the shape pairs
    this world's acts with that world's arbitrarily — nothing she has
    seen yet tells one act from another. She finds those out in a few moves.

    The counts come at no more than a fresh start is worth, and the margin
    over the scrambled control scales them again: a world that only just
    clears its control lends a little, one far above it lends a fresh start.
    """
    from core.perception.how_it_moves import no_more_than_a_fresh_start

    carried = no_more_than_a_fresh_start(knew.get("moves"))
    share = max(0.0, min(1.0, carried * likeness.margin))
    out: dict[str, Any] = {"trust": share}
    moves = knew.get("moves")
    if isinstance(moves, dict):
        out["moves"] = {
            key: value
            for key, value in moves.items()
            if key not in {"pushes", "read_through"}
        }
    world = knew.get("world")
    arrives = world.get("arrives") if isinstance(world, dict) else None
    if isinstance(arrives, dict):
        theirs_to_ours = {
            there: here
            for here, there in likeness.mapping.items()
            if here.startswith("thing ") and there.startswith("thing ")
        }
        renamed = {
            theirs_to_ours[_thing(said)][len("thing "):]: times
            for said, times in arrives.items()
            if _thing(said) in theirs_to_ours
        }
        if renamed:
            out["world"] = {**world, "arrives": renamed}
    return out


def lend_to(rules: Any, given: Mapping[str, Any], world: Any = None) -> bool:
    """Add what was lent to what she has, rather than putting it in its place.

    She may already have watched a few moves here when the closest world is
    found, and those are the best evidence there is about here. The lent counts
    are added to hers, at the share `lent` set.
    """
    from core.perception.how_it_moves import HowItMoves
    from core.perception.what_the_world_does import WhatTheWorldDoes

    share = float(given.get("trust") or 0.0)
    if share <= 0.0 or not isinstance(given.get("moves"), dict):
        return False
    borrowed = HowItMoves.from_memory(given["moves"], share)
    for part in ("right", "tried", "right_when_it_moved", "tried_when_it_moved"):
        mine: dict[str, int] = getattr(rules, part)
        for name, count in getattr(borrowed, part).items():
            mine[name] = mine.get(name, 0) + count
    rules.seen += borrowed.seen
    rules.moved += borrowed.moved
    if world is not None and isinstance(given.get("world"), dict):
        theirs = WhatTheWorldDoes.from_memory(given["world"], share)
        for said, count in theirs.arrives.items():
            world.arrives[said] = world.arrives.get(said, 0) + count
        world.acts += theirs.acts
        world.acts_with_arrivals += theirs.acts_with_arrivals
    return True


def solved_worlds(names: Iterable[str], recall: Any) -> dict[str, tuple[Graph, dict[str, Any]]]:
    """Every remembered world with a shape kept and a way of moving worked out."""
    from core.perception.how_it_moves import HowItMoves

    found: dict[str, tuple[Graph, dict[str, Any]]] = {}
    for name in names:
        knew = recall(name)
        if not isinstance(knew, dict):
            continue
        graph = graph_from_memory(knew.get("shape"), name)
        if graph is None:
            continue
        rule = HowItMoves.from_memory(knew.get("moves") or {}).rule()
        if rule is None:
            continue
        found[name] = (graph, {**knew, "_teaches": rule.name})
    return found
