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


@pytest.mark.asyncio
async def test_the_unified_self_saves_off_the_loop(monkeypatch, tmp_path):
    """write_text:unified_self.save_to_disk ran on the loop thread twice per
    boot (2026-09-15). Every caller is a coroutine."""
    from core.consciousness import unified_self as us

    gateway = _Gateway()

    async def ensure_directory_async(path, *, source):
        gateway.async_.append(f"mkdir:{source}")
        return str(path)

    gateway.ensure_directory_async = ensure_directory_async
    monkeypatch.setattr(
        "core.runtime.file_write_gateway.get_file_write_gateway", lambda: gateway
    )
    self_ = us.UnifiedSelf.__new__(us.UnifiedSelf)
    self_._storage_path = tmp_path / "self" / "unified_self.json"
    self_._state = us.UnifiedSelfState()

    await self_.interact()

    assert gateway.sync == []
    assert gateway.async_ == ["mkdir:unified_self.save_to_disk", "unified_self.save_to_disk"]
    assert json.loads(gateway.written[str(self_._storage_path)])["interaction_count"] == 1


def test_the_belief_graph_writes_off_the_lock_and_off_the_loop(tmp_path):
    """write_text:belief_graph.save_graph ran on the loop thread at boot
    (2026-09-16), under the graph lock."""
    import threading

    from core.world_model.belief_graph import BeliefGraph

    graph = BeliefGraph(
        persist_path=str(tmp_path / "world_model.json"),
        causal_path=str(tmp_path / "causal.json"),
    )
    writers: list[tuple[str, bool]] = []
    real = graph._write_graph_payload

    def spy(payload):
        # Not under the graph lock: a second acquire from another thread
        # would block if it were held; an RLock lets this thread through,
        # so record whether the lock is currently held by asking a helper.
        writers.append((threading.current_thread().name, _held_by_other(graph._graph_lock)))
        real(payload)

    def _held_by_other(lock):
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
            return False
        return True

    graph._write_graph_payload = spy
    graph._state_writer._write = spy

    async def on_loop():
        graph.graph.add_node("test.belief", value=1.0)
        graph._save(force=True)
        assert threading.current_thread().name not in [w[0] for w in writers]

    asyncio.run(on_loop())
    assert graph.flush(5.0)
    assert writers
    assert all(name != "MainThread" for name, _ in writers)
    assert all(not held for _, held in writers)
    assert (tmp_path / "world_model.json").exists()
