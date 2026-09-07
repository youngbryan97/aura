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
