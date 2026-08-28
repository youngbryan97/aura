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

__all__ = ["Ordering", "solve_ordering"]


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

        if self.drops:
            # Whether a cell is dropped was learned as a list of the cells that
            # were, so a cell never shown has no answer. Filtering by "not in
            # the dropped list" reads unseen as KEEP and quietly returns the
            # whole state — an answer, confidently wrong, where a refusal was
            # the only honest output.
            known = set(self.drops) | set(self.level)
            if any(cell not in known for cell in state):
                return None
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
                    enumerate(kept),
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
                enumerate(kept), key=lambda pair: (self.level[pair[1]], pair[0])
            )
            return tuple(cell for _place, cell in ranked)
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
