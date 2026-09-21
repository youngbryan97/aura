"""The existing complete chart search ranks the span-set training support."""

from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from core.learning.semantic_operation_search import OperationChartSearch
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_span_set_learning import span_set_partition


@pytest.mark.parametrize("n,width,count", [(1, 1, 1), (4, 1, 4), (5, 3, 2), (5, 7, 9)])
@pytest.mark.parametrize("penalty", [0.0, 0.75, -0.25])
def test_runtime_enumeration_matches_partition_support_and_map(n, width, count, penalty):
    scores = np.random.default_rng(91).normal(size=(n, width))
    scores[np.arange(n)[:, None] + np.arange(1, width + 1) > n] = -np.inf
    nodes = [SimpleNamespace(span=TokenSpan(int(s), int(s + column + 1)), operation="add", score=float(scores[s, column]))
             for s, column in zip(*np.nonzero(np.isfinite(scores)), strict=True)]
    expected = []
    for size in range(1, min(count, n) + 1):
        expected.extend(chart for chart in combinations(nodes, size)
                        if all(a.span.end <= b.span.start for a, b in zip(chart, chart[1:], strict=False)))
    search = OperationChartSearch(nodes, max_steps=count, length_penalty=penalty)
    actual = list(search)
    def key(chart):
        return tuple((node.span.start, node.span.end) for node in chart)
    assert {key(chart) for chart in actual} == {key(chart) for chart in expected}
    assert len(actual) == len(expected) and search.complete
    values = [sum(node.score - penalty for node in chart) for chart in actual]
    assert values == pytest.approx(sorted(values, reverse=True), abs=1e-12)
    partition, _ = span_set_partition(scores - penalty, count)
    # Training includes the empty set once. Runtime tasks require an operation.
    assert logsumexp([0.0, *values]) == pytest.approx(partition, abs=1e-12)
    assert max([0.0, *values]) == pytest.approx(max(0.0, values[0]), abs=1e-12)


def test_runtime_search_recovers_disjoint_set_despite_overlapping_top_nodes():
    nodes = [SimpleNamespace(span=TokenSpan(5, 12), operation="add", score=4.0),
             SimpleNamespace(span=TokenSpan(5, 7), operation="add", score=3.0),
             SimpleNamespace(span=TokenSpan(24, 27), operation="add", score=2.0)]
    result = next(OperationChartSearch(nodes, max_steps=2, length_penalty=0))
    assert tuple(node.span for node in result) == (TokenSpan(5, 12), TokenSpan(24, 27))


def test_runtime_nonempty_requirement_is_explicit_not_a_duplicate_background_path():
    nodes = [SimpleNamespace(span=TokenSpan(0, 1), operation="add", score=-5.0)]
    result = list(OperationChartSearch(nodes, max_steps=1, length_penalty=0))
    assert len(result) == 1 and result[0] == tuple(nodes)
    partition, _ = span_set_partition(np.array([[-5.0]]), 1)
    assert partition == pytest.approx(np.logaddexp(0, -5))
