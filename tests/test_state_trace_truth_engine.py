from __future__ import annotations

import pytest

from core.brain.verifiers.state_trace_engine import StateTraceTruthEngine


BAD_DIJKSTRA_TRACE = """Dijkstra's invariant says a closed distance never changes.

1. Initialize:
- Distances: `A=0`, B=`∞`, C=`∞`, D=`∞`
2. Close A and examine its neighbors:
- Set `dist[B] = 5`
- Set `dist[C] = 2`
3. Close the vertex with the smallest known distance (B=5):
- Set `dist[D] = 9`
4. Close the vertex with the smallest known distance (C=2):
- Set `dist[B] = 3`
5. Close B (now with distance=3).
6. Close the vertex with the smallest known distance (D=9).
"""


GOOD_DIJKSTRA_TRACE = """1. Initialize distances: `A=0`, B=`∞`, C=`∞`, D=`∞`.
2. Close A, then set `dist[B] = 5` and set `dist[C] = 2`.
3. Close the vertex with the smallest known distance (C=2), then set `dist[B] = 3` and set `dist[D] = 8`.
4. Close the vertex with the smallest known distance (B=3), then set `dist[D] = 7`.
5. Close the vertex with the smallest known distance (D=7).
"""


@pytest.mark.asyncio
async def test_rejects_extremum_violation_and_finalized_state_mutation() -> None:
    result = await StateTraceTruthEngine().verify(BAD_DIJKSTRA_TRACE)

    assert result.checked is True
    assert result.ok is False
    codes = {item["code"] for item in result.detail["issues"]}
    assert "extremum_selection_violation" in codes
    assert "finalized_state_mutation" in codes
    assert "duplicate_finalization" in codes


@pytest.mark.asyncio
async def test_accepts_consistent_ordered_state_trace() -> None:
    result = await StateTraceTruthEngine().verify(GOOD_DIJKSTRA_TRACE)

    assert result.checked is True
    assert result.ok is True
    assert result.detail["selections"] == 3


@pytest.mark.asyncio
async def test_generalizes_to_priority_processing_not_algorithm_name() -> None:
    trace = """1. Initialize priorities: `red=7`, `blue=2`, `green=4`.
2. Finalize the item with the lowest priority (blue=2).
3. Update `green=3`.
4. Finalize the item with the lowest priority (green=3).
5. Finalize the item with the lowest priority (red=7).
"""
    result = await StateTraceTruthEngine().verify(trace)

    assert result.checked is True
    assert result.ok is True


@pytest.mark.asyncio
async def test_ordinary_numbered_prose_remains_unchecked() -> None:
    result = await StateTraceTruthEngine().verify(
        "1. Gather the requirements.\n2. Discuss the tradeoffs.\n3. Write the report."
    )

    assert result.checked is False
    assert result.ok is True


@pytest.mark.asyncio
async def test_trace_under_critique_is_not_treated_as_authored_execution() -> None:
    result = await StateTraceTruthEngine().verify(
        BAD_DIJKSTRA_TRACE,
        context={"objective": "Find the error in this invalid execution trace."},
    )

    assert result.checked is False
    assert result.ok is True
