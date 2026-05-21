from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.cognitive_loop import CognitiveLoop


def close_loop(loop: CognitiveLoop) -> None:
    loop._deliberation_pool.shutdown(wait=False)


@pytest.mark.asyncio
async def test_dispatch_failure_is_contained():
    class Orchestrator:
        async def process_user_input_priority(self, content, *, origin):
            if content and origin:
                raise RuntimeError("dispatch offline")

    loop = CognitiveLoop(Orchestrator())
    try:
        await loop._dispatch_message({"content": "hello", "origin": "user"})
    finally:
        close_loop(loop)


@pytest.mark.asyncio
async def test_queue_failure_returns_no_message():
    class Queue:
        def empty(self):
            return False

        async def get(self):
            if self.empty() is False:
                raise RuntimeError("queue unavailable")
            return None

    loop = CognitiveLoop(SimpleNamespace(message_queue=Queue()))
    try:
        assert await loop._acquire_next_message() is None
    finally:
        close_loop(loop)


@pytest.mark.asyncio
async def test_autonomous_action_failure_stays_inside_deliberation_task():
    loop = CognitiveLoop(SimpleNamespace())

    async def fail_action(fe_state):
        if fe_state:
            raise RuntimeError("autonomous action failed")
        return None

    loop._autonomous_action = fail_action
    try:
        await loop._autonomous_action_async(SimpleNamespace(dominant_action="explore"))
    finally:
        close_loop(loop)


@pytest.mark.asyncio
async def test_stop_cancels_background_and_deliberation_tasks():
    async def sleeper():
        await asyncio.sleep(60.0)

    loop = CognitiveLoop(SimpleNamespace())
    loop.is_running = True
    loop._task = asyncio.create_task(sleeper())
    loop._active_deliberation_task = asyncio.create_task(sleeper())

    await loop.stop()

    assert loop.is_running is False
    assert loop._task is None
    assert loop._active_deliberation_task is None


@pytest.mark.asyncio
async def test_run_recovers_stalled_cycle():
    recovered = asyncio.Event()
    loop = CognitiveLoop(SimpleNamespace())
    loop.stall_threshold = 0.01
    loop.is_running = True

    async def stalled_cycle():
        await asyncio.sleep(60.0)

    async def recover_from_stall():
        recovered.set()
        loop.is_running = False

    loop._process_cycle = stalled_cycle
    loop._recover_from_stall = recover_from_stall

    try:
        await asyncio.wait_for(loop.run(), timeout=1.0)
    finally:
        close_loop(loop)

    assert recovered.is_set()
