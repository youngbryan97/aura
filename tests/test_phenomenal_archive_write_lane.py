"""Narrative persistence must release the event loop while storage is busy."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from core.consciousness.phenomenological_experiencer import PhenomenologicalExperiencer
from core.runtime import atomic_writer


@pytest.mark.asyncio
async def test_narrative_archive_runs_off_loop_and_is_awaited(monkeypatch, tmp_path):
    experiencer = object.__new__(PhenomenologicalExperiencer)
    experiencer.save_dir = tmp_path
    experiencer._current_schema = SimpleNamespace(focal_object="test")
    experiencer._current_qualia = []
    experiencer._current_emotion = "curiosity"
    experiencer._dominant_motivation = "understand"
    experiencer.continuity = SimpleNamespace(current_thread="test-thread")
    async def narrative(**_):
        return "archived report"
    experiencer.psm = SimpleNamespace(run_deep_narrative_update=narrative)
    experiencer._rebuild_context_string = lambda: None
    loop_thread = threading.get_ident()
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = atomic_writer.atomic_append_text

    def slow_append(path, text, **kwargs):
        assert threading.get_ident() != loop_thread
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "test did not release archive writer"
        original(path, text, **kwargs)

    monkeypatch.setattr(atomic_writer, "atomic_append_text", slow_append)
    task = asyncio.create_task(experiencer._run_deep_narrative())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert not task.done()
        assert not (tmp_path / "phenomenal_archive.jsonl").exists()
    finally:
        release.set()
        await asyncio.wait_for(task, 5)
    record = json.loads((tmp_path / "phenomenal_archive.jsonl").read_text())
    assert record["report"] == "archived report"
    assert record["thread"] == "test-thread"
