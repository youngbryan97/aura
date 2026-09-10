import asyncio
import json
from types import SimpleNamespace

import pytest

from core import continuity


@pytest.mark.asyncio
async def test_flush_waits_for_storage_without_blocking_loop(monkeypatch, tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()
    written = []

    async def append(path, payload, **kwargs):
        started.set()
        await release.wait()
        written.append(payload)

    monkeypatch.setattr(continuity, "_signed_record_payload", lambda record: {"value": record})
    monkeypatch.setattr(continuity, "get_file_write_gateway", lambda: SimpleNamespace(write_text_async=append))
    continuity._persist_continuity_record(tmp_path / "continuity.json", 7, "continuity.shutdown_record")
    await asyncio.wait_for(started.wait(), 2)
    flush = asyncio.create_task(continuity.flush_continuity_writes())
    try:
        await asyncio.sleep(0)
        assert not flush.done()
        assert written == []
    finally:
        release.set()
        await asyncio.wait_for(flush, 2)
    assert [json.loads(payload) for payload in written] == [{"value": 7}]


@pytest.mark.asyncio
async def test_snapshots_reach_storage_in_submission_order(monkeypatch, tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()
    written = []

    async def write(path, payload, **kwargs):
        value = json.loads(payload)["value"]
        if value == 1:
            started.set()
            await release.wait()
        written.append(value)

    monkeypatch.setattr(continuity, "_signed_record_payload", lambda record: {"value": record})
    monkeypatch.setattr(continuity, "get_file_write_gateway", lambda: SimpleNamespace(write_text_async=write))
    path = tmp_path / "continuity.json"
    continuity._persist_continuity_record(path, 1, "continuity.shutdown_record")
    await asyncio.wait_for(started.wait(), 2)
    continuity._persist_continuity_record(path, 2, "continuity.shutdown_record")
    await asyncio.sleep(0)
    assert written == []
    release.set()
    await asyncio.wait_for(continuity.flush_continuity_writes(), 2)
    assert written == [1, 2]


@pytest.mark.asyncio
async def test_flush_reports_already_completed_storage_failure(monkeypatch, tmp_path):
    async def write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(continuity, "_signed_record_payload", lambda record: {"value": record})
    monkeypatch.setattr(continuity, "get_file_write_gateway", lambda: SimpleNamespace(write_text_async=write))
    continuity._persist_continuity_record(tmp_path / "continuity.json", 1, "continuity.shutdown_record")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="Continuity persistence failed") as raised:
        await continuity.flush_continuity_writes()
    assert isinstance(raised.value.__cause__, OSError)
