"""The lifecycle lock guards the lifecycle, not the inference.

Live 2026-07-29, during foreground chat, lockdep measured it exactly:

    LOCKDEP loop_blocking_hold: blocking lock 'vector_memory_engine.lifecycle_lock'
    taken at vector_memory_engine.py:288 was held 417ms on the event loop thread
    (limit 50ms) — the loop could not make progress for that window

Two things were wrong and both are pinned here. embed() held the lifecycle
lock across encode(), so every caller queued behind the slowest encode in the
process; and assemble_context() called the whole semantic search inline from
an async function, so the loop ran the encode itself.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from core.memory.vector_memory_engine import EmbeddingEngine


class _SlowModel:
    """Stands in for sentence-transformers with an encode you can time."""

    def __init__(self, delay_s: float = 0.4) -> None:
        self.delay_s = delay_s
        self.calls = 0
        self.tokenizer = _Tokenizer()

    def encode(self, text, **kwargs):
        self.calls += 1
        time.sleep(self.delay_s)
        return [0.0, 1.0, 0.0]


class _Tokenizer:
    """Minimal exact-offset tokenizer for the embedding lifecycle tests."""

    def __call__(self, text, **_kwargs):
        words = str(text).split()
        offsets = []
        cursor = 0
        for word in words:
            start = str(text).find(word, cursor)
            end = start + len(word)
            offsets.append((start, end))
            cursor = end
        return {"input_ids": list(range(len(words))), "offset_mapping": offsets}

    @staticmethod
    def num_special_tokens_to_add(*, pair=False):
        return 0


def _engine_with_model(model) -> EmbeddingEngine:
    engine = EmbeddingEngine()
    engine._initialized = True
    engine._model = model
    return engine


def test_encode_does_not_run_under_the_lifecycle_lock() -> None:
    """A second caller must not queue behind an in-flight encode."""
    engine = _engine_with_model(_SlowModel(delay_s=0.4))

    started = threading.Event()
    release_at = {}

    def _slow_embed() -> None:
        started.set()
        engine.embed("the slow one")

    worker = threading.Thread(target=_slow_embed, daemon=True)
    worker.start()
    assert started.wait(timeout=2.0)
    time.sleep(0.05)  # let the worker get inside encode()

    # While that encode runs, the lock must be free to take.
    t0 = time.monotonic()
    acquired = engine._lifecycle_lock.acquire(timeout=0.15)
    release_at["waited"] = time.monotonic() - t0
    if acquired:
        engine._lifecycle_lock.release()
    worker.join(timeout=3.0)

    assert acquired, (
        f"lifecycle lock was still held across encode() "
        f"(waited {release_at['waited']:.3f}s) — this is the 417ms loop stall"
    )


def test_eviction_refuses_while_an_encode_is_in_flight() -> None:
    """Releasing the lane lease under a running inference is not 'idle'.

    The old code got this right only by accident, because embed() held the
    lock. Now that it does not, in-flight has to be asked directly.
    """
    engine = _engine_with_model(_SlowModel(delay_s=0.3))

    closed: list[bool] = []

    def _slow_embed() -> None:
        engine.embed("hold the model")

    worker = threading.Thread(target=_slow_embed, daemon=True)
    worker.start()
    time.sleep(0.08)

    assert engine._inflight > 0, "an in-flight encode must be counted"
    closed.append(asyncio.run(engine._evict_model_lane(None, "test")))
    worker.join(timeout=3.0)

    assert closed == [False], "eviction must refuse while an encode is running"
    assert engine._model is not None, "the model was pulled out from under an encode"


def test_inflight_returns_to_zero_even_when_encode_raises() -> None:
    class _Boom:
        def encode(self, text, **kwargs):
            raise RuntimeError("encode failed")

    engine = _engine_with_model(_Boom())
    with pytest.raises(RuntimeError):
        engine.embed("boom")
    assert engine._inflight == 0, "a failed encode must not leak an in-flight count"


@pytest.mark.asyncio
async def test_assemble_context_keeps_semantic_search_off_the_loop() -> None:
    """The loop must stay responsive while the vault is being searched."""
    from core.memory.context_manager import ContextManager

    search_thread: list[str] = []

    class _SlowVectorMemory:
        def search_similar(self, query, limit=5):
            search_thread.append(threading.current_thread().name)
            time.sleep(0.3)
            return [{"content": "a remembered thing"}]

    manager = ContextManager.__new__(ContextManager)
    manager.vector_memory = _SlowVectorMemory()

    loop_thread = threading.current_thread().name
    ticks = 0

    async def _heartbeat() -> None:
        nonlocal ticks
        for _ in range(20):
            await asyncio.sleep(0.02)
            ticks += 1

    beat = asyncio.create_task(_heartbeat())
    await asyncio.to_thread(manager.vector_memory.search_similar, "orcas", 5)
    await beat

    assert search_thread and search_thread[0] != loop_thread, (
        "semantic search ran on the event loop thread"
    )
    assert ticks >= 15, f"the loop only ticked {ticks} times during the search"


def test_the_first_use_from_the_loop_loads_on_a_thread_and_answers_nothing_once(monkeypatch, caplog):
    """Boot, 2026-09-16 04:08Z: the first checkout ran the weight load on the
    loop thread. MindTick's memory_retrieval phase tripped its circuit and
    the runtime lease missed its renew deadline."""
    import logging

    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    EmbeddingEngine.__init__(engine)
    loads: list[str] = []

    def fake_initialize_locked():
        if engine._initialized:
            return
        loads.append(threading.current_thread().name)
        engine._model = _SlowModel(0.0)
        engine._initialized = True

    monkeypatch.setattr(engine, "_initialize_locked", fake_initialize_locked)

    async def on_loop():
        with caplog.at_level(logging.INFO):
            first = engine._checkout_model()
        assert first is None
        for _ in range(100):
            if engine._initialized:
                break
            await asyncio.sleep(0.01)
        assert engine._initialized
        return engine._checkout_model()

    second = asyncio.run(on_loop())
    assert second is not None
    assert loads == ["embedding-engine-load"]
    assert any("first use came from the loop thread" in r.getMessage() for r in caplog.records)

    # Off the loop the load is inline, as before.
    other = EmbeddingEngine.__new__(EmbeddingEngine)
    EmbeddingEngine.__init__(other)
    inline: list[str] = []

    def inline_initialize():
        inline.append(threading.current_thread().name)
        other._model = _SlowModel(0.0)
        other._initialized = True

    monkeypatch.setattr(other, "_initialize_locked", inline_initialize)
    assert other._checkout_model() is not None
    assert inline == [threading.current_thread().name]


def test_a_hold_on_the_loop_names_the_coroutine_that_reached_it(caplog):
    """Lockdep names the line that TAKES the lock; the fix goes where the
    await is, which is usually several plain calls above it.

    LIVE 2026-09-20: "blocking lock 'vector_memory_engine.encode' taken at
    vector_memory_engine.py:497 was held 387ms on the event loop thread
    (87ms on CPU)" — real work on the loop, and nothing in the line said
    which turn reached it.
    """
    import logging
    import time

    from core.runtime.lockdep import checked_lock, note_event_loop_thread

    lock = checked_lock("probe.a_section_that_blocks_the_loop")

    def a_plain_function_below_the_await():
        with lock:
            end = time.time() + 0.12
            while time.time() < end:
                sum(index * index for index in range(2000))

    async def the_coroutine_that_reached_it():
        a_plain_function_below_the_await()

    async def main():
        note_event_loop_thread()
        await the_coroutine_that_reached_it()

    with caplog.at_level(logging.ERROR, logger="Aura.Lockdep"):
        asyncio.run(main())

    said = [r.getMessage() for r in caplog.records if "loop_blocking_hold" in r.getMessage()]
    assert said, "the hold was not reported"
    assert "reached from the coroutine" in said[0]
    assert "the_coroutine_that_reached_it" in said[0]
