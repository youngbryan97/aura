"""Working out the order the cells were put in, rather than guessing a key.

The index language says where a cell comes from using its position and the
length, and never what the cell holds. Composing such rules only makes another
one, so ordering cells by a property of their values sits outside it however
long you search — which core.cognition.language_limits now proves rather than
reporting as a search that gave up.

This is the other side of that proof. When the proof fires, there is something
to do.

The move that makes it tractable is refusing to search for a key at all. A
stable ordering fixes the correspondence between before and after even when
values repeat: equal keys keep the order they were in. So each transition hands
over n-1 facts about the key directly —

    key(after[k]) <= key(after[k+1])

and strictly less whenever the sources ran backwards, because a stable ordering
would have kept them in order if their keys had been equal.

Those facts are a graph on the values. Its strongly connected components are
the sets that must share a key; a strict edge inside a component is a
contradiction and the answer is refusal. The components form a chain, and the
longest path to each one is its level. Nothing was searched: the key is read
off.

What the levels turn out to be says which family this was, and the three that
were listed separately are one thing seen at three settings:

    as many levels as distinct values   the cells were sorted
    two levels                          the cells were split by a property
    two levels, and the length free     the cells were filtered

Where it stays quiet: a value it has never seen has no level, and no amount of
cleverness gives it one. That is the honest end of this mechanism rather than a
gap in it — the ordering was learned from the cells that were shown, and a cell
that was not shown was not in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["Composed", "Ordering", "solve_ordering", "solve_ordering_then_move"]


@dataclass(frozen=True)
class Ordering:
    """A level for each value seen, and what kind of rule that turned out to be."""

    level: dict[Any, int]
    levels: int
    kind: str
    drops: frozenset[Any] = frozenset()
    #: Which group of values each one fell into, and which groups the
    #: observations actually put in an order relative to each other.
    #:
    #: Levels are only comparable inside a chain the evidence built. Two chains
    #: that were never related have no order between them, and ranking them
    #: anyway put 6 before 4 on values whose relative order had never been
    #: shown.
    group: dict[Any, int] = None  # type: ignore[assignment]
    ranked: frozenset = frozenset()
    #: "ascending", "descending", or None when the levels are a table.
    #:
    #: The levels are learned from the cells that were shown, and a cell never
    #: shown has no level. Reaching past them needs an assumption — that the
    #: values carry an order of their own — and that assumption is recorded
    #: here rather than applied quietly, because it is the one thing in this
    #: module that was not read off the observations.
    natural: str | None = None

    def _named(self, cell: Any) -> Any:
        """The cell as this ordering names it.

        A written ordering keys its cells by repr, because a cell is anything
        hashable and JSON keys are strings. A restored one therefore knows
        "'blue'" and is asked about "blue", and every lookup misses — the
        ordering survives the restart and stops working at it, which is the
        same as not surviving.
        """

        if cell in self.level or cell in self.drops:
            return cell
        written = repr(cell)
        if written in self.level or written in self.drops:
            return written
        return cell

    def to_json(self) -> dict[str, Any]:
        """A structured value, for the same reason index programs are.

        An ordering worked out on Tuesday and thrown away answers Tuesday's
        question. Written down, it is a piece of language the next world can be
        read in — which is the whole difference between using a thing and
        having learned it.

        Keys are repr'd because the values themselves are anything hashable and
        JSON keys are not.
        """

        return {
            "level": {repr(value): rank for value, rank in self.level.items()},
            "levels": self.levels,
            "kind": self.kind,
            "drops": sorted(repr(value) for value in self.drops),
            "group": {repr(value): where for value, where in (self.group or {}).items()},
            "ranked": sorted([one, other] for one, other in self.ranked),
            "natural": self.natural,
        }

    @classmethod
    def from_json(cls, raw: Any) -> "Ordering | None":
        if not isinstance(raw, dict):
            return None
        level = raw.get("level")
        if not isinstance(level, dict) or not level:
            return None
        try:
            return cls(
                level={str(key): int(rank) for key, rank in level.items()},
                levels=int(raw.get("levels") or 0),
                kind=str(raw.get("kind") or ""),
                drops=frozenset(str(item) for item in (raw.get("drops") or ())),
                group={
                    str(key): int(where)
                    for key, where in (raw.get("group") or {}).items()
                },
                ranked=frozenset(
                    (int(pair[0]), int(pair[1]))
                    for pair in (raw.get("ranked") or ())
                    if isinstance(pair, (list, tuple)) and len(pair) == 2
                ),
                natural=raw.get("natural") or None,
            )
        except (TypeError, ValueError):
            return None

    def keyed_by_repr(self) -> "Ordering":
        """The same ordering with cells named the way the written form names them.

        A restored ordering has repr'd keys and the cells it is asked about do
        not, so a level lookup that worked before a restart misses after one.
        """

        return Ordering(
            level={repr(value): rank for value, rank in self.level.items()},
            levels=self.levels,
            kind=self.kind,
            drops=frozenset(repr(value) for value in self.drops),
            group={repr(value): where for value, where in (self.group or {}).items()},
            ranked=self.ranked,
            natural=self.natural,
        )

    def explains(self, transitions: Sequence[Any]) -> bool:
        """Whether this ordering accounts for every one of these transitions."""

        for item in transitions:
            got = self.apply(tuple(item.before))
            if got is None or tuple(got) != tuple(item.after):
                return False
        return True

    def describe(self) -> str:
        """What was worked out, at the strength it was worked out to.

        Naming this a sort when only a table was learned would be the claim
        doing the work rather than the evidence. The order is only called an
        order when the values were checked to carry one.
        """

        if self.kind == "filtered":
            # Which cells survive is the claim here. Whether the survivors also
            # came out in the order the values carry is true and is not what
            # was asked, and naming it read as part of the rule.
            return "cells are kept or dropped by a property of their values"
        if self.natural is not None:
            return f"the cells are put in {self.natural} order of their values"
        if self.levels >= 3:
            return (
                f"the cells are put in an order worked out from the examples, "
                f"in {self.levels} steps"
            )
        return (
            f"the cells are split into {self.levels} groups by a property of "
            "their values, keeping the order inside each group"
        )

    def apply(self, state: Sequence[Any]) -> tuple[Any, ...] | None:
        """The state reordered, or None when a cell has no level.

        Refusing is the point. A value never seen has no place in an ordering
        learned from values that were, and putting it somewhere would be
        inventing the part that was not observed.
        """

        state = tuple(state)
        if self.drops:
            # Whether a cell is dropped was learned as a list of the cells that
            # were, so a cell never shown has no answer. Filtering by "not in
            # the dropped list" reads unseen as KEEP and quietly returns the
            # whole state — an answer, confidently wrong, where a refusal was
            # the only honest output.
            known = set(self.drops) | set(self.level)
            if any(self._named(cell) not in known for cell in state):
                return None
        named = tuple(self._named(cell) for cell in state)
        # Answer with the cells the caller handed over, never with their names.
        original = [
            cell
            for cell, key in zip(state, named, strict=True)
            if key not in self.drops
        ]
        state = named
        kept = [cell for cell in state if cell not in self.drops]
        if self.natural is not None:
            # The checked order wins over the table.
            #
            # The table has a level for every value seen, so it answered first
            # and answered from levels that are only meaningful inside a chain
            # the evidence built: 4 and 6 were never seen in an order relative
            # to each other, and the table put 6 first. The natural order was
            # tested against every constraint before it was believed, and it
            # covers the pairs the observations left open.
            try:
                ranked = sorted(
                    enumerate(original),
                    key=lambda pair: (pair[1], pair[0]),
                    reverse=self.natural == "descending",
                )
            except TypeError:
                return None
            return tuple(cell for _place, cell in ranked)
        if self.group and all(cell in self.group for cell in kept):
            groups = {self.group[cell] for cell in kept}
            for one in groups:
                for other in groups:
                    if one == other:
                        continue
                    if (one, other) not in self.ranked and (
                        other,
                        one,
                    ) not in self.ranked:
                        # These were never seen in an order relative to each
                        # other. Putting one first would be inventing the fact
                        # that decides the answer.
                        return None
        if all(cell in self.level for cell in kept):
            ranked = sorted(
                range(len(kept)), key=lambda place: (self.level[kept[place]], place)
            )
            return tuple(original[place] for place in ranked)
        # A learned table says nothing about a value it never saw, and putting
        # one somewhere would be inventing the part that decides the answer.
        return None


def _stable_sources(before: Sequence[Any], after: Sequence[Any]) -> list[int] | None:
    """Where each output cell came from, assuming the ordering was stable.

    Ambiguity from repeated values is what forced a tie-break everywhere else.
    Here it is settled by the assumption being tested: a stable ordering keeps
    equal cells in the order it found them, so the k-th copy of a value in the
    output is the k-th copy in the input.
    """

    taken: dict[Any, int] = {}
    places: dict[Any, list[int]] = {}
    for index, cell in enumerate(before):
        places.setdefault(cell, []).append(index)
    sources: list[int] = []
    for cell in after:
        seen = taken.get(cell, 0)
        where = places.get(cell)
        if where is None or seen >= len(where):
            return None
        sources.append(where[seen])
        taken[cell] = seen + 1
    return sources


def _components(
    values: list[Any], atmost: set[tuple[Any, Any]]
) -> tuple[dict[Any, int], list[list[Any]]]:
    """Which values must share a level, by strongly connected component."""

    forward: dict[Any, set[Any]] = {value: set() for value in values}
    back: dict[Any, set[Any]] = {value: set() for value in values}
    for low, high in atmost:
        forward[low].add(high)
        back[high].add(low)

    def reach(start: Any, edges: dict[Any, set[Any]]) -> set[Any]:
        seen = {start}
        stack = [start]
        while stack:
            here = stack.pop()
            for step in edges[here]:
                if step not in seen:
                    seen.add(step)
                    stack.append(step)
        return seen

    owner: dict[Any, int] = {}
    groups: list[list[Any]] = []
    for value in values:
        if value in owner:
            continue
        together = reach(value, forward) & reach(value, back)
        index = len(groups)
        groups.append(sorted(together, key=repr))
        for member in together:
            owner[member] = index
    return owner, groups


def _one_per_value(level: dict[Any, int], atmost: set[tuple[Any, Any]]) -> bool:
    """Whether every pair the observations actually related came out apart.

    Levels collapse when two values were never seen next to each other, which
    is a shortage of evidence rather than a claim that they are equal. Counting
    distinct levels against every value calls a genuine sort a grouping the
    moment two of its values never met.
    """

    related = {value for pair in atmost for value in pair}
    return all(
        level[low] != level[high]
        for low, high in atmost
        if low != high
    ) and bool(related)


def solve_ordering(transitions: Sequence[Any]) -> Ordering | None:
    """The ordering these transitions were made by, or None.

    None for anything this cannot account for. A refusal here is the mechanism
    working: the alternative is a key that reproduces what it was shown and
    means nothing about what it was not.
    """

    observed = [
        (tuple(item.before), tuple(item.after))
        for item in transitions
        if item is not None
    ]
    if not observed:
        return None

    dropped: set[Any] = set()
    for before, after in observed:
        if len(after) > len(before):
            return None
        if len(after) < len(before):
            kept: list[Any] = []
            spare = list(before)
            for cell in after:
                if cell not in spare:
                    return None
                spare.remove(cell)
                kept.append(cell)
            dropped |= set(spare)
    kept_values = {cell for _b, after in observed for cell in after}
    if dropped & kept_values:
        # A value both kept and dropped is not decided by the value.
        return None

    atmost: set[tuple[Any, Any]] = set()
    strict: set[tuple[Any, Any]] = set()
    for before, after in observed:
        surviving = tuple(cell for cell in before if cell not in dropped)
        sources = _stable_sources(surviving, after)
        if sources is None or len(sources) != len(after):
            return None
        for place in range(len(after) - 1):
            low, high = after[place], after[place + 1]
            atmost.add((low, high))
            if sources[place] > sources[place + 1]:
                # They came out of order, so a stable ordering would only have
                # done that if the key genuinely rose.
                strict.add((low, high))

    values = sorted(kept_values, key=repr)
    if len(values) < 2:
        return None
    owner, groups = _components(values, atmost)

    for low, high in strict:
        if owner[low] == owner[high]:
            # It has to rise here and it has to be equal here. No ordering of
            # the values does both, and saying one anyway is the whole failure
            # this is built to avoid.
            return None

    above: dict[int, set[int]] = {index: set() for index in range(len(groups))}
    for low, high in atmost:
        if owner[low] != owner[high]:
            above[owner[low]].add(owner[high])

    depth: dict[int, int] = {}

    def level_of(index: int, seen: frozenset[int] = frozenset()) -> int:
        if index in depth:
            return depth[index]
        if index in seen:
            return 0
        found = 0
        for step in above[index]:
            found = max(found, level_of(step, seen | {index}) + 1)
        depth[index] = found
        return found

    for index in range(len(groups)):
        level_of(index)
    deepest = max(depth.values(), default=0)
    # Deeper means later, so turn the longest path into a rank.
    level = {value: deepest - depth[owner[value]] for value in values}
    distinct = len(set(level.values()))

    if dropped:
        kind = "filtered"
    elif distinct >= len(values) or _one_per_value(level, atmost):
        kind = "sorted"
    elif distinct >= 2:
        kind = "grouped"
    else:
        return None

    # Do the levels the observations forced agree with the order the values
    # already carry? If they do, the ordering reaches values never shown; if
    # they do not, it is a table and it stops where the evidence stopped.
    #
    # Checked, never assumed. A secret ordering over values that never recur
    # comes out as a table and refuses, which is the control that makes this an
    # assumption rather than an answer smuggled in.
    natural = None
    for name, backwards in (("ascending", False), ("descending", True)):
        def rises(low: Any, high: Any, _back: bool = backwards) -> bool | None:
            try:
                return (high < low) if _back else (low < high)
            except TypeError:
                return None

        answers = [rises(low, high) for low, high in atmost]
        if any(answer is None for answer in answers):
            break
        # Every fact the observations gave has to hold under this order: a
        # non-strict pair may be equal, a strict one may not.
        if all(
            answer or low == high
            for (low, high), answer in zip(atmost, answers, strict=True)
        ) and all(rises(low, high) for low, high in strict):
            natural = name
            break

    # Which components the observations actually put in an order, transitively.
    ranked: set[tuple[int, int]] = set()
    for start in range(len(groups)):
        stack = list(above[start])
        seen: set[int] = set()
        while stack:
            here = stack.pop()
            if here in seen:
                continue
            seen.add(here)
            ranked.add((start, here))
            stack.extend(above[here])

    return Ordering(
        level=dict(level),
        levels=distinct,
        kind=kind,
        drops=frozenset(dropped),
        group={value: owner[value] for value in values},
        ranked=frozenset(ranked),
        natural=natural,
    )


@dataclass(frozen=True)
class Composed:
    """An ordering of the cells, and then a move of the positions."""

    ordering: Ordering
    move: Any
    move_said: str

    def describe(self) -> str:
        return f"{self.ordering.describe()}, then {self.move_said}"

    def apply(self, state: Sequence[Any]) -> tuple[Any, ...] | None:
        ordered = self.ordering.apply(tuple(state))
        if ordered is None:
            return None
        size = len(ordered)
        try:
            return tuple(ordered[self.move(place, size)] for place in range(size))
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            return None

    def explains(self, transitions: Sequence[Any]) -> bool:
        for item in transitions:
            got = self.apply(tuple(item.before))
            if got is None or tuple(got) != tuple(item.after):
                return False
        return True


def solve_ordering_then_move(
    transitions: Sequence[Any], forms: Sequence[tuple[str, str, Any]]
) -> Composed | None:
    """An ordering of the cells followed by a rearrangement of the positions.

    The two axes were solved separately and could not meet. "Sorted, then
    rotated" is proved outside the positional language — correctly, the sources
    contradict — and the ordering alone cannot say it either, because the cells
    do not come out in the order the values carry. Between them they say it
    exactly, and neither of them alone says anything.

    Undoing the move is what makes this cheap. If ``after[i] = mid[f(i, n)]``
    then ``mid`` is determined by ``after`` and ``f``, so for each candidate
    move there is exactly one intermediate state to solve the ordering of. The
    search is over the moves already known, not over pairs.

    Only an ordering that extrapolates counts. A table would fit whatever
    intermediate a move happened to produce, and would then be a table of that
    move's arithmetic rather than a claim about the cells.
    """

    observed = [
        (tuple(item.before), tuple(item.after))
        for item in transitions
        if item is not None
    ]
    if not observed or any(len(b) != len(a) for b, a in observed):
        return None

    for _family, said, move in forms:
        rebuilt: list[Any] = []
        for before, after in observed:
            size = len(after)
            middle: list[Any] = [None] * size
            try:
                places = [move(place, size) for place in range(size)]
            except (IndexError, TypeError, ValueError, ZeroDivisionError):
                break
            if sorted(places) != list(range(size)):
                break
            for place, source in enumerate(places):
                middle[source] = after[place]
            rebuilt.append(_Pair(before, tuple(middle)))
        else:
            ordering = solve_ordering(rebuilt)
            if ordering is None or ordering.natural is None:
                continue
            found = Composed(ordering=ordering, move=move, move_said=said)
            if found.explains([_Pair(b, a) for b, a in observed]):
                return found
    return None


@dataclass(frozen=True)
class _Pair:
    """A transition, for handing rebuilt states back to the solver."""

    before: tuple[Any, ...]
    after: tuple[Any, ...]
