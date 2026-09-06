"""One trace id, and the boundaries that drop it."""
from __future__ import annotations

import asyncio
import queue

from core.observability.does_one_trace_reach_the_end import (
    a_carrier_for_a_queue,
    carrying_the_trace,
    how_far_a_trace_reaches,
    in_a_thread_that_carries_it,
    on_an_executor_that_carries_it,
    under_a_carrier,
    where_the_trace_is_lost,
)
from core.runtime.causal_trace import current_trace_id, new_trace, trace_scope


def test_the_probe_names_every_boundary_and_says_what_crosses_it() -> None:
    boundaries = where_the_trace_is_lost()
    assert len(boundaries) >= 8
    for b in boundaries:
        assert b.crossed_by.strip(), f"{b.name} has a mechanism but no consequence"


def test_a_context_variable_follows_a_call_and_an_awaited_coroutine() -> None:
    kept = {b.name for b in where_the_trace_is_lost() if b.kept}
    assert "a plain call" in kept
    assert "an awaited coroutine" in kept
    assert "a spawned task" in kept, "create_task copies the context by default"


def test_it_does_not_follow_a_bare_thread_or_an_executor() -> None:
    """The measured loss. If this ever passes, the remedy became unnecessary."""
    lost = {b.name for b in where_the_trace_is_lost() if b.lost}
    assert "a bare thread" in lost
    assert "an executor" in lost
    assert "a queue between threads" in lost


def test_a_carried_context_closes_the_thread_loss() -> None:
    span = new_trace("t", origin="test")
    with trace_scope(span):
        seen: list[str] = []
        worker = in_a_thread_that_carries_it(lambda: seen.append(current_trace_id()))
        worker.start()
        worker.join()
    assert seen == [span.trace_id]


def test_a_carried_context_closes_the_executor_loss() -> None:
    span = new_trace("t", origin="test")

    async def go() -> str:
        return await on_an_executor_that_carries_it(None, current_trace_id)

    with trace_scope(span):
        assert asyncio.run(go()) == span.trace_id


def test_a_queue_carries_the_trace_as_data_because_no_context_can_cross_it() -> None:
    span = new_trace("t", origin="test")
    box: queue.Queue = queue.Queue()
    with trace_scope(span):
        box.put(a_carrier_for_a_queue())
    with under_a_carrier(box.get()) as trace_id:
        assert trace_id == span.trace_id
        assert current_trace_id() == span.trace_id


def test_an_absent_carrier_yields_nothing_rather_than_inventing_continuity() -> None:
    with under_a_carrier(None) as trace_id:
        assert trace_id == ""
    with under_a_carrier({}) as trace_id:
        assert trace_id == ""


def test_the_capture_happens_where_the_id_still_exists() -> None:
    """Wrapping outside the scope must not smuggle in a later trace."""
    outside = carrying_the_trace(current_trace_id)
    span = new_trace("t", origin="test")
    with trace_scope(span):
        assert outside() != span.trace_id, "captured before the scope, so it has none"
        inside = carrying_the_trace(current_trace_id)
    assert inside() == span.trace_id, "captured inside, so it keeps it after"


def test_the_report_counts_kept_and_lost_and_they_add_up() -> None:
    seen = how_far_a_trace_reaches()
    assert seen["kept"] + seen["lost"] == seen["boundaries"]
    assert seen["lost"] >= 3
    assert len(seen["lost_at"]) == seen["lost"]
