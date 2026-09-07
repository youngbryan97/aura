import asyncio
import threading
from types import SimpleNamespace

from core.consciousness.heartbeat import CognitiveHeartbeat


def test_sync_persistence_runs_off_loop_and_is_awaited():
    async def scenario():
        heartbeat = object.__new__(CognitiveHeartbeat)
        loop_thread = threading.get_ident()
        calls = []

        def save():
            calls.append(threading.get_ident())

        await heartbeat._sync_mind_model(
            SimpleNamespace(save=save), heartbeat._NARRATIVE_EMIT_TICKS
        )
        assert len(calls) == 1
        assert calls[0] != loop_thread

    asyncio.run(scenario())


def test_async_persistence_remains_on_its_owner_loop():
    async def scenario():
        heartbeat = object.__new__(CognitiveHeartbeat)
        loop = asyncio.get_running_loop()
        calls = []

        async def save():
            calls.append(asyncio.get_running_loop())

        await heartbeat._sync_mind_model(
            SimpleNamespace(save=save), heartbeat._NARRATIVE_EMIT_TICKS
        )
        assert calls == [loop]

    asyncio.run(scenario())
