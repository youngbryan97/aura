"""The two per-turn trace writers append and save off the event loop.

Live census (2026-09-15): ``append_text:thought_tracer.log_cycle`` ran on
the loop thread 34 times and ``write_text:cognitive_trace.save`` 14 times
in one boot. Both were sync gateway calls made from inside coroutines —
the kernel tick and the two response finalizers. The coroutine callers
now await the async twins; the sync methods stay for bare-thread callers.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from core.introspection import thought_tracer
from core.meta import cognitive_trace

ROOT = Path(__file__).resolve().parent.parent


class _Gateway:
    def __init__(self) -> None:
        self.sync: list[str] = []
        self.async_: list[str] = []
        self.written: dict[str, str] = {}

    def append_text(self, path, text, *, encoding="utf-8", source):
        self.sync.append(source)
        self.written[str(path)] = self.written.get(str(path), "") + text

    async def append_text_async(self, path, text, *, encoding="utf-8", source):
        self.async_.append(source)
        self.written[str(path)] = self.written.get(str(path), "") + text

    def write_text(self, path, text, *, source, **_):
        self.sync.append(source)
        self.written[str(path)] = text

    async def write_text_async(self, path, text, *, source, **_):
        self.async_.append(source)
        self.written[str(path)] = text


@pytest.mark.asyncio
async def test_the_cycle_log_awaits_the_async_lane(monkeypatch, tmp_path):
    gateway = _Gateway()
    tracer = thought_tracer.ThoughtTracer(log_dir=str(tmp_path))
    monkeypatch.setattr(thought_tracer, "get_file_write_gateway", lambda: gateway)

    await tracer.log_cycle_async("objective", {"k": 1}, {"answer": "x"}, "ok")

    assert gateway.sync == []
    assert gateway.async_ == ["thought_tracer.log_cycle"]
    line = gateway.written[str(tracer.current_trace_file)]
    assert json.loads(line)["objective"] == "objective"


def test_the_sync_cycle_log_still_serves_bare_threads(monkeypatch, tmp_path):
    gateway = _Gateway()
    tracer = thought_tracer.ThoughtTracer(log_dir=str(tmp_path))
    monkeypatch.setattr(thought_tracer, "get_file_write_gateway", lambda: gateway)

    tracer.log_cycle("objective", {"k": 1}, {"answer": "x"}, "ok")

    assert gateway.sync == ["thought_tracer.log_cycle"]
    assert gateway.async_ == []


@pytest.mark.asyncio
async def test_the_trace_save_awaits_the_async_lane(monkeypatch, tmp_path):
    gateway = _Gateway()
    monkeypatch.setattr(
        "core.runtime.file_write_gateway.get_file_write_gateway", lambda: gateway
    )
    trace = cognitive_trace.CognitiveTrace(trace_id="t1")
    trace.log_dir = str(tmp_path)
    trace.record_step("end", {"response": "done"})

    await trace.save_async()

    assert gateway.sync == []
    assert gateway.async_ == ["cognitive_trace.save"]
    payload = json.loads(gateway.written[str(tmp_path / "trace_t1.json")])
    assert payload["steps"][0]["content"] == {"response": "done"}


def test_an_unencodable_entry_is_a_degradation_not_a_crash(monkeypatch, tmp_path):
    gateway = _Gateway()
    tracer = thought_tracer.ThoughtTracer(log_dir=str(tmp_path))
    monkeypatch.setattr(thought_tracer, "get_file_write_gateway", lambda: gateway)

    tracer.log_cycle("objective", {}, {"answer": object()}, "ok")
    asyncio.run(tracer.log_cycle_async("objective", {}, {"answer": object()}, "ok"))

    assert gateway.sync == [] and gateway.async_ == []


@pytest.mark.parametrize(
    ("path", "sync_call", "async_call"),
    [
        ("core/kernel/aura_kernel.py", "tracer.log_cycle(", "await tracer.log_cycle_async("),
        ("core/orchestrator/mixins/response_processing.py", "trace.save()", "await trace.save_async()"),
        ("core/coordinators/cognitive_coordinator.py", "trace.save()", "await trace.save_async()"),
    ],
)
def test_every_coroutine_caller_awaits_the_async_twin(path, sync_call, async_call):
    source = (ROOT / path).read_text(encoding="utf-8")
    assert async_call in source
    assert not re.search(r"(?<!await )" + re.escape(sync_call).replace(r"\(", r"\(") + r"(?!_async)", source.replace(async_call, "")), (
        f"{path} still makes the sync trace write from a coroutine"
    )
