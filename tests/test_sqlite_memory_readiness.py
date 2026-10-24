"""First use, failed setup and synchronous entry obey the memory lifecycle."""
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from core.memory.sqlite_storage import SQLiteMemory


@pytest.mark.asyncio
async def test_concurrent_first_writes_wait_for_schema(tmp_path, monkeypatch):
    memory = SQLiteMemory(str(tmp_path / "memory.db"))
    entered, resume = asyncio.Event(), asyncio.Event()
    ensure = memory._ensure_schema

    async def paused(conn=None):
        entered.set()
        await resume.wait()
        await ensure(conn)

    monkeypatch.setattr(memory, "_ensure_schema", paused)
    first = asyncio.create_task(memory.log_event_async({"event_type": "first"}))
    await asyncio.wait_for(entered.wait(), timeout=1)
    second = asyncio.create_task(memory.log_event_async({"event_type": "second"}))
    try:
        await asyncio.sleep(0)
        assert memory._conn is None
        assert not second.done()
        resume.set()
        assert await asyncio.wait_for(asyncio.gather(first, second), timeout=2) == [True, True]
        events = await memory.get_recent_events_async()
        assert {event["event_type"] for event in events} == {"first", "second"}
    finally:
        resume.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await memory.on_stop_async()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_or_cancelled_setup_closes_unpublished_connection(tmp_path, monkeypatch, cancel):
    from core.memory import db_config

    memory = SQLiteMemory(str(tmp_path / "memory.db"))
    configure = db_config.configure_connection_async
    connections = []

    async def opened(path):
        conn = await configure(path)
        connections.append(conn)
        return conn

    async def fail(conn=None):
        if cancel:
            raise asyncio.CancelledError()
        raise sqlite3.OperationalError("schema failed")

    monkeypatch.setattr(db_config, "configure_connection_async", opened)
    monkeypatch.setattr(memory, "_ensure_schema", fail)
    with pytest.raises(asyncio.CancelledError if cancel else sqlite3.OperationalError):
        await memory._get_conn()
    assert memory._conn is None and not memory._initialized
    with pytest.raises(ValueError, match="no active connection"):
        await connections[0].execute("SELECT 1")
    monkeypatch.undo()
    try:
        assert await memory.log_event_async({"event_type": "retry"})
    finally:
        await memory.on_stop_async()


@pytest.mark.asyncio
async def test_sync_calls_on_a_running_loop_refuse_without_hanging(tmp_path):
    memory = SQLiteMemory(str(tmp_path / "memory.db"))
    try:
        await memory._get_conn()
        for call in (memory.close, memory.get_recent_events):
            with pytest.raises(RuntimeError, match="await the async method"):
                call()
        assert await memory.log_event_async({"event_type": "still_open"})
    finally:
        await memory.on_stop_async()


@pytest.mark.asyncio
async def test_sync_close_from_a_worker_thread_finishes_on_owner_loop(tmp_path):
    memory = SQLiteMemory(str(tmp_path / "memory.db"))
    await memory._get_conn()
    await asyncio.wait_for(asyncio.to_thread(memory.close), timeout=2)
    assert memory._conn is None and not memory._initialized


def test_sync_entry_can_reopen_after_close(tmp_path):
    memory = SQLiteMemory(str(tmp_path / "memory.db"))
    try:
        assert memory.log_event({"event_type": "before_close"})
        memory.close()
        assert memory.log_event({"event_type": "after_close"})
        assert len(memory.get_recent_events()) == 2
    finally:
        memory.close()
