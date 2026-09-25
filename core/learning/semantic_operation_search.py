"""Lazy complete search over bounded, nonoverlapping operation interpretations."""

from __future__ import annotations

import heapq
import math
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator, Sequence
from itertools import count
from typing import Any

from core.learning.dyadic_scores import DyadicScores
from core.verify.invariants import invariant


class OperationSearchIncompleteError(RuntimeError):
    """The caller's work allowance ended before the candidate grammar was exhausted."""


class OperationChartSearch(Iterator):
    """Best-bound enumeration without a top-k prefix or completed-chart cutoff.

    The supplied node inventory, maximum step count and feasibility predicate
    define the grammar. This class makes no completeness claim outside it.
    Suffix dynamic programming bounds only operation scores. Argument scores
    must still be considered by the full graph selector.
    """

    def __init__(
        self,
        nodes: Sequence,
        *,
        max_steps: int,
        length_penalty: float,
        feasible: Callable | None=None,
        max_expansions: int | None=None,
    ) -> None:
        if type(max_steps) is not int or max_steps < 1 or not math.isfinite(length_penalty):
            raise ValueError("operation search needs a positive step bound and finite penalty")
        if max_expansions is not None and (type(max_expansions) is not int or max_expansions < 1):
            raise ValueError("operation search allowance must be a positive integer")
        unique = {}
        for node in nodes:
            if (not math.isfinite(node.score) or node.span.start < 0
                    or node.span.end <= node.span.start or not node.operation):
                raise ValueError("operation search node is invalid")
            key = (node.span.start, node.span.end, node.operation)
            if key not in unique or node.score > unique[key].score:
                unique[key] = node
        self.nodes = tuple(sorted(unique.values(), key=lambda n: (n.span.start, n.span.end, n.operation)))
        self.max_steps = max_steps
        self.length_penalty = float(length_penalty)
        self._scale = DyadicScores.from_values((*[node.score for node in self.nodes], self.length_penalty))
        self._scores = self._scale.integers[:-1]
        self._penalty = self._scale.integers[-1]
        self.max_expansions = max_expansions
        self.feasible = feasible
        self.expanded = 0
        self.yielded = 0
        self.complete = False
        self._serial = count()
        self._heap = []
        starts = [node.span.start for node in self.nodes]
        self._successor = tuple(bisect_left(starts, node.span.end) for node in self.nodes)
        size = len(self.nodes)
        self._suffix = [[0] * (size + 1)] + [[None] * (size + 1) for _ in range(max_steps)]
        for remaining in range(1, max_steps + 1):
            for index in range(size - 1, -1, -1):
                tail = self._suffix[remaining - 1][self._successor[index]]
                take = self._scores[index] + tail if tail is not None else None
                skip = self._suffix[remaining][index + 1]
                self._suffix[remaining][index] = take if skip is None else skip if take is None else max(skip, take)
            self._push(0, remaining, (), remaining)

    def _push(self, index: int, remaining: Any, selected: tuple[Any, ...], size: Any) -> None:
        suffix = self._suffix[remaining][index]
        if suffix is None:
            return
        # Exact binary64 values keep cancellation and penalty rounding from
        # lowering a bound below a reachable completion.
        upper = sum((self._scores[i] for i in selected), suffix) - self._penalty * size
        heapq.heappush(self._heap, (-upper, next(self._serial), index, remaining, selected, size))

    @property
    def remaining_operation_score_upper_bound(self) -> float:
        return self._scale.upper_float(-self._heap[0][0]) if self._heap else -math.inf

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> tuple[Any, ...]:
        while self._heap:
            if self.max_expansions is not None and self.expanded >= self.max_expansions:
                raise OperationSearchIncompleteError(
                    f"operation_search_incomplete:expanded={self.expanded}:frontier={len(self._heap)}"
                )
            _, _, index, remaining, selected, size = heapq.heappop(self._heap)
            self.expanded += 1
            if remaining == 0:
                chart = tuple(self.nodes[i] for i in selected)
                if self.feasible is None or self.feasible(chart):
                    self.yielded += 1
                    return chart
                continue
            self._push(index + 1, remaining, selected, size)
            self._push(self._successor[index], remaining - 1, (*selected, index), size)
        self.complete = True
        raise StopIteration


def best_charts_by_operation_sequence(
    nodes: Sequence,
    *,
    max_steps: int,
    length_penalty: float,
    feasible: Callable | None = None,
    max_table_entries: int = 1_000_000,
) -> tuple[tuple, ...]:
    """Exact best nonoverlapping chart for each ordered operation sequence.

    This inventories signatures, not all graphs: a lower-scored span placement
    with the same operations may bind differently.
    """
    if (type(max_steps) is not int or max_steps < 1 or not math.isfinite(length_penalty)
            or type(max_table_entries) is not int or max_table_entries < 1):
        raise ValueError("operation-sequence inventory needs finite positive bounds")
    ordered = tuple(sorted(nodes, key=lambda node: (
        node.span.end, node.span.start, node.operation, -node.score)))
    if any(not math.isfinite(node.score) or node.span.start < 0
           or node.span.end <= node.span.start or not node.operation for node in ordered):
        raise ValueError("operation-sequence inventory node is invalid")
    ends = [node.span.end for node in ordered]
    table = [[{} for _ in range(max_steps + 1)] for _ in range(len(ordered) + 1)]
    table[0][0][()] = (0.0, ())
    retained = 1
    for index, node in enumerate(ordered, 1):
        prior = bisect_right(ends, node.span.start, hi=index - 1)
        for size in range(max_steps + 1):
            current = table[index - 1][size].copy()
            if size:
                for signature, (score, chart) in table[prior][size - 1].items():
                    next_signature = (*signature, node.operation)
                    proposal = (score + node.score, (*chart, node))
                    incumbent = current.get(next_signature)
                    if (incumbent is None or proposal[0] > incumbent[0]
                            or (proposal[0] == incumbent[0] and
                                tuple((part.span.start, part.span.end) for part in proposal[1])
                                < tuple((part.span.start, part.span.end) for part in incumbent[1]))):
                        current[next_signature] = proposal
            retained += len(current)
            if retained > max_table_entries:
                raise OperationSearchIncompleteError(
                    f"operation_sequence_inventory_incomplete:table_entries={retained}")
            table[index][size] = current
    ranked = []
    for size in range(1, max_steps + 1):
        for signature, (score, chart) in table[-1][size].items():
            if feasible is None or feasible(chart):
                ranked.append((score - length_penalty * size, signature, chart))
    ranked.sort(key=lambda row: (-row[0], row[1],
                tuple((node.span.start, node.span.end) for node in row[2])))
    return tuple(chart for _, _, chart in ranked)


@invariant("learning.operation_search_does_not_drop_low_ranked_charts", scope="learning",
           owner="core/learning/semantic_operation_search.py", observational=False)
def _operation_search_coverage() -> tuple:
    from types import SimpleNamespace

    from core.learning.semantic_program_ir import TokenSpan

    nodes = [SimpleNamespace(span=TokenSpan(i, i + 1), operation="add", score=float(-i)) for i in range(20)]
    search = OperationChartSearch(nodes, max_steps=1, length_penalty=0)
    assert len(list(search)) == 20 and search.complete
    return ()
