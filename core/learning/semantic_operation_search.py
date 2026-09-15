"""Lazy complete search over bounded, nonoverlapping operation interpretations."""

from __future__ import annotations

import heapq
import math
from bisect import bisect_left
from collections.abc import Callable, Iterator, Sequence
from fractions import Fraction
from itertools import count

from core.verify.invariants import invariant


class OperationSearchIncompleteError(RuntimeError):
    """The caller's work allowance ended before the candidate grammar was exhausted."""


def _float_upper(value: Fraction) -> float:
    try:
        rounded = float(value)
    except OverflowError:
        return math.inf if value > 0 else -float.fromhex("0x1.fffffffffffffp+1023")
    return math.nextafter(rounded, math.inf) if Fraction(rounded) < value else rounded


class OperationChartSearch(Iterator):
    """Best-bound enumeration without a top-k prefix or completed-chart cutoff.

    The supplied node inventory, maximum step count and feasibility predicate
    define the grammar. This class makes no completeness claim outside it.
    Suffix dynamic programming bounds only operation scores. Argument scores
    must still be considered by the full graph selector.
    """

    def __init__(self, nodes: Sequence, *, max_steps: int, length_penalty: float,
                 feasible: Callable | None = None, max_expansions: int | None = None):
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
        self._penalty = Fraction(self.length_penalty)
        self._scores = tuple(Fraction(float(node.score)) for node in self.nodes)
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
        self._suffix = [[Fraction(0)] * (size + 1)] + [[None] * (size + 1) for _ in range(max_steps)]
        for remaining in range(1, max_steps + 1):
            for index in range(size - 1, -1, -1):
                tail = self._suffix[remaining - 1][self._successor[index]]
                take = self._scores[index] + tail if tail is not None else None
                skip = self._suffix[remaining][index + 1]
                self._suffix[remaining][index] = take if skip is None else skip if take is None else max(skip, take)
            self._push(0, remaining, (), remaining)

    def _push(self, index, remaining, selected, size):
        suffix = self._suffix[remaining][index]
        if suffix is None:
            return
        # Exact binary64 values keep cancellation and penalty rounding from
        # lowering a bound below a reachable completion.
        upper = sum((self._scores[i] for i in selected), suffix) - self._penalty * size
        heapq.heappush(self._heap, (-upper, next(self._serial), index, remaining, selected, size))

    @property
    def remaining_operation_score_upper_bound(self) -> float:
        return _float_upper(-self._heap[0][0]) if self._heap else -math.inf

    def __iter__(self):
        return self

    def __next__(self):
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


@invariant("learning.operation_search_does_not_drop_low_ranked_charts", scope="learning",
           owner="core/learning/semantic_operation_search.py", observational=False)
def _operation_search_coverage() -> tuple:
    from types import SimpleNamespace

    from core.learning.semantic_program_ir import TokenSpan

    nodes = [SimpleNamespace(span=TokenSpan(i, i + 1), operation="add", score=float(-i)) for i in range(20)]
    search = OperationChartSearch(nodes, max_steps=1, length_penalty=0)
    assert len(list(search)) == 20 and search.complete
    return ()
