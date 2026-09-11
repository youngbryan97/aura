from __future__ import annotations

import asyncio
import multiprocessing
import time

import pytest

from core.brain.llm.worker_progress import activity_age, create_channel, publish


def _publish_from_child(channel, ready, request_seq):
    ready.set()
    for count in range(1, 21):
        publish(channel, request_seq, count)
        time.sleep(0.01)


def test_progress_is_request_bound_and_ages_without_new_activity():
    channel = create_channel(multiprocessing.get_context("spawn"))
    assert activity_age(channel, 1) is None
    assert publish(channel, 1, 1)
    first = activity_age(channel, 1)
    assert first is not None and first >= 0
    assert activity_age(channel, 2) is None
    assert activity_age(channel, 0) is None
    assert activity_age(channel, 1) >= first
    assert activity_age(create_channel(multiprocessing.get_context("spawn")), 1) is None


def test_reader_never_waits_on_a_dead_writers_lock():
    context = multiprocessing.get_context("spawn")
    channel = context.Array("Q", 3, lock=context.Lock())
    assert publish(channel, 1, 1)
    lock = channel.get_lock()
    assert lock.acquire(False)
    try:
        assert activity_age(channel, 1) is None
        assert publish(channel, 1, 2) is False
    finally:
        lock.release()
    assert activity_age(channel, 1) is not None


@pytest.mark.asyncio
async def test_child_progress_remains_visible_after_parent_loop_was_blocked():
    context = multiprocessing.get_context("spawn")
    channel = create_channel(context)
    ready = context.Event()
    child = context.Process(target=_publish_from_child, args=(channel, ready, 7))
    child.start()
    try:
        assert await asyncio.to_thread(ready.wait, 10.0)
        # Deliberately block this event loop. No listener or heartbeat consumer
        # runs while the child continues publishing inference activity.
        time.sleep(0.12)
        age = activity_age(channel, 7)
        assert age is not None and age < 0.1
        assert activity_age(channel, 8) is None
        await asyncio.to_thread(child.join, 10.0)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            await asyncio.to_thread(child.join, 10.0)
        child.close()


def test_watchdog_publishes_inference_and_completion_without_heartbeat():
    from core.brain.llm.mlx_worker import JobWatchdog

    channel = create_channel(multiprocessing.get_context("spawn"))
    watchdog = JobWatchdog(progress_channel=channel)
    watchdog.start_job("job", "latent_reason", request_seq=9)
    assert activity_age(channel, 9) is not None
    watchdog.activity()
    assert int(channel[2]) == 2
    watchdog.stop_job()
    assert int(channel[2]) == 3
    assert activity_age(channel, 9) is not None


def test_first_token_watchdog_renews_while_worker_is_advancing(monkeypatch):
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient("/models/Qwen2.5-7B-Instruct-4bit")
    monkeypatch.setattr(client, "_is_primary_or_deep_lane", lambda: True)
    cancelled = []
    monkeypatch.setattr(
        client, "_first_token_watchdog_soft_cancel",
        lambda request: cancelled.append(request) or True,
    )
    client._current_request_id = "job"
    client._current_request_seq = 9
    client._current_request_started_at = time.time() - 120
    client._worker_progress_channel = create_channel(client._mp_context)
    publish(client._worker_progress_channel, 9, 2)
    timer = client._start_foreground_first_token_watchdog(
        "job", foreground_request=True, hard_ceiling_s=10,
    )
    assert timer is not None
    try:
        timer.cancel()
        timer.function()
        assert cancelled == []
        assert client._foreground_generation_watchdog is not timer
        # A later check with no fresh evidence still enforces the same stall
        # budget; renewal cannot permanently disable the watchdog.
        client._worker_progress_channel = None
        client._last_worker_job_activity_at = time.time() - 600
        client._foreground_generation_watchdog.cancel()
        client._foreground_generation_watchdog.function()
        assert cancelled == ["job"]
    finally:
        timer.cancel()
        client._foreground_generation_watchdog.cancel()
