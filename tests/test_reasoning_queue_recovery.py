from __future__ import annotations

import asyncio

import pytest

from core.brain.reasoning_queue import BackgroundReasoningQueue, ReasoningPriority
from core.runtime.errors import get_degradation_tracker


def _disable_registry_updates(queue: BackgroundReasoningQueue) -> None:
    queue._schedule_registry_size_update = lambda *, reason: None


async def _stop_queue(queue: BackgroundReasoningQueue) -> None:
    workers = list(queue._worker_tasks)
    queue.stop()
    if workers:
        await asyncio.gather(*workers, return_exceptions=True)


@pytest.mark.asyncio
async def test_reasoning_queue_uses_configured_concurrency():
    queue = BackgroundReasoningQueue(max_concurrent=2)
    _disable_registry_updates(queue)
    started = 0
    all_started = asyncio.Event()
    release = asyncio.Event()

    async def reasoning_task():
        nonlocal started
        started += 1
        if started == 2:
            all_started.set()
        await release.wait()
        return started

    await queue.start()
    first_id = await queue.submit(reasoning_task, description="first")
    second_id = await queue.submit(reasoning_task, description="second")

    await asyncio.wait_for(all_started.wait(), timeout=1.0)
    release.set()
    await asyncio.wait_for(queue._queue.join(), timeout=1.0)

    assert len(queue._worker_tasks) == 2
    assert queue.get_result(first_id) == 2
    assert queue.get_result(second_id) == 2

    await _stop_queue(queue)


@pytest.mark.asyncio
async def test_reasoning_queue_preserves_result_when_callback_fails():
    tracker = get_degradation_tracker()
    tracker.reset()
    queue = BackgroundReasoningQueue()
    _disable_registry_updates(queue)

    def callback(_result):
        raise RuntimeError("callback broke")

    await queue.start()
    task_id = await queue.submit(lambda: "complete", callback=callback, description="callback-test")
    await asyncio.wait_for(queue._queue.join(), timeout=1.0)

    assert queue.get_result(task_id) == "complete"
    assert tracker.count("reasoning_queue", "warning") >= 1

    await _stop_queue(queue)


@pytest.mark.asyncio
async def test_reasoning_queue_stores_failure_envelope_and_continues():
    tracker = get_degradation_tracker()
    tracker.reset()
    queue = BackgroundReasoningQueue()
    _disable_registry_updates(queue)

    def broken_task():
        raise ValueError("bad premise")

    await queue.start()
    failed_id = await queue.submit(broken_task, description="broken")
    ok_id = await queue.submit(lambda: "ok", description="ok")
    await asyncio.wait_for(queue._queue.join(), timeout=1.0)

    failure = queue.get_result(failed_id)
    assert failure["status"] == "failed"
    assert failure["error_type"] == "ValueError"
    assert queue.get_result(ok_id) == "ok"
    assert tracker.count("reasoning_queue", "degraded") >= 1

    await _stop_queue(queue)


@pytest.mark.asyncio
async def test_prune_low_priority_keeps_higher_priority_work():
    queue = BackgroundReasoningQueue()
    _disable_registry_updates(queue)

    keep_id = await queue.submit(lambda: "keep", priority=ReasoningPriority.HIGH)
    drop_id = await queue.submit(lambda: "drop", priority=ReasoningPriority.LOW)

    dropped = await queue.prune_low_priority(threshold_priority=ReasoningPriority.HIGH.value)

    assert dropped == 1
    retained = queue._queue.get_nowait()
    assert retained.task_id == keep_id
    assert retained.task_id != drop_id
