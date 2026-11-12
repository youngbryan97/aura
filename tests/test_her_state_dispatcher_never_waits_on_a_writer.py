"""Her state's notification dispatcher runs on her event loop and never waits for a writer's state lock.

LIVE 11 November 2026, lockdep: the dispatcher held her event loop up to 508 ms, ten times in a day, waiting for
`state_registry.state` while a writer applied an update and built its snapshot under it.
"""
from __future__ import annotations

import asyncio
import threading
import time

from core.state.state_registry import UnifiedStateRegistry


def test_a_queued_state_reaches_its_listener_while_a_writer_holds_the_lock():
    async def scenario():
        registry = UnifiedStateRegistry()
        delivered = asyncio.Event()

        async def listener(_state):
            delivered.set()

        registry.subscribe(listener)
        registry.ensure_dispatcher()
        await asyncio.sleep(0.05)
        released = threading.Event()

        def writer():
            with registry._sync_lock:
                released.wait(2.0)

        holding = threading.Thread(target=writer)
        holding.start()
        try:
            await asyncio.sleep(0.05)
            registry._notify_queue.put_nowait(registry._state)
            registry._notify_ready.set()
            began = time.monotonic()
            await asyncio.wait_for(delivered.wait(), timeout=1.0)
            assert time.monotonic() - began < 0.5, "delivered without waiting for the writer"
            registry.ensure_dispatcher()  # running already: returns without the writer's lock
        finally:
            released.set()
            holding.join()
            await registry.shutdown_dispatcher()

    asyncio.run(scenario())
