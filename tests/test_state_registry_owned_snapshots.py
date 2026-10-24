"""State updates remain coherent under contention and slow listeners."""
from __future__ import annotations

import asyncio
import threading

import pytest

from core.state.state_registry import UnifiedStateRegistry


@pytest.mark.asyncio
async def test_async_and_thread_writers_share_the_snapshot_lock():
    registry = UnifiedStateRegistry()
    entered, release = threading.Event(), threading.Event()
    reads = []

    class HeldCopy(dict):
        def __deepcopy__(self, memo):
            entered.set()
            assert release.wait(2)
            return {"paired": 1.0}

    writer = asyncio.create_task(registry.update(valence=1.0, resonance=HeldCopy(paired=1.0)))
    await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=1)
    reader = asyncio.create_task(asyncio.to_thread(lambda: reads.append(registry.get_state())))
    sync = asyncio.create_task(asyncio.to_thread(registry.sync_update, valence=2.0, resonance={"paired": 2.0}))
    try:
        await asyncio.sleep(0.01)
        assert not reader.done() and not sync.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(writer, reader, sync), timeout=2)
        assert reads[0].valence == reads[0].resonance["paired"]
        assert registry.get_state().version == 2
    finally:
        release.set()
        await asyncio.gather(writer, reader, sync, return_exceptions=True)
        await registry.shutdown_dispatcher()


@pytest.mark.asyncio
async def test_slow_listener_retains_latest_state_with_bounded_backlog():
    registry = UnifiedStateRegistry(notification_capacity=3)
    entered, release = asyncio.Event(), asyncio.Event()
    versions = []

    async def listener(snapshot):
        versions.append(snapshot.version)
        if len(versions) == 1:
            entered.set()
            await release.wait()

    registry.subscribe(listener)
    await registry.update(valence=1)
    await asyncio.wait_for(entered.wait(), timeout=1)
    try:
        for value in range(2, 21):
            registry.sync_update(valence=value)
        assert registry._notify_queue.qsize() == 3
        assert registry.coalesced_notifications == 16
        release.set()
        await asyncio.wait_for(asyncio.to_thread(registry._notify_queue.join), timeout=1)
        assert versions == [1, 18, 19, 20]
    finally:
        release.set()
        await registry.shutdown_dispatcher()
    assert registry._notify_queue.unfinished_tasks == 0


@pytest.mark.asyncio
async def test_inputs_and_each_listener_own_their_nested_state():
    registry = UnifiedStateRegistry()
    readings = []

    async def mutator(snapshot):
        snapshot.resonance["a"] = 99

    async def observer(snapshot):
        readings.append(snapshot.resonance["a"])

    registry.subscribe(mutator)
    registry.subscribe(observer)
    incoming = {"a": 1.0}
    await registry.update(resonance=incoming)
    incoming["a"] = 100
    try:
        await asyncio.wait_for(asyncio.to_thread(registry._notify_queue.join), timeout=1)
        assert readings == [1.0]
        assert registry.get_state().resonance == {"a": 1.0}
    finally:
        await registry.shutdown_dispatcher()


@pytest.mark.asyncio
async def test_cancelling_dispatcher_accounts_for_inflight_and_queued_snapshots():
    registry = UnifiedStateRegistry(notification_capacity=2)
    entered = asyncio.Event()

    async def blocked(snapshot):
        entered.set()
        await asyncio.Future()

    registry.subscribe(blocked)
    await registry.update(phi=0.1)
    await asyncio.wait_for(entered.wait(), timeout=1)
    registry.sync_update(phi=0.2)
    registry.sync_update(phi=0.3)
    await registry.shutdown_dispatcher(timeout_s=0.01)
    assert registry._dispatcher_task is None
    assert registry._notify_queue.unfinished_tasks == 0
    assert registry._notify_queue.empty()
