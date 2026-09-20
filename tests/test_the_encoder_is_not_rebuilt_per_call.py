"""A burst of acquires must not rebuild the encoder each time.

The main runtime holds a process-owned lease from
`prewarm_shared_embedding_runtime`, so its engine stays resident. The MLX
worker has no such lease and every consumer there acquires and releases
around one call. Live on 2026-09-20 the worker loaded Qwen3-Embedding-0.6B
five times in six seconds — 03:55:27 to 03:55:33 — each load followed within
a second by "closed engine after final owner release".

The engine is held for exactly as long as building it took. That is the
trade in its own terms: at worst the memory is held for the time it would
have cost to get it back, and at best a burst reuses one engine. A process
that stops embedding still gives the memory back.
"""
from __future__ import annotations

import time

from core.memory.embedding_runtime import SharedEmbeddingRuntime


class _Engine:
    def __init__(self, built: list[int], closed: list[int], ordinal: int) -> None:
        self.ordinal = ordinal
        self._closed = closed
        built.append(ordinal)

    def close(self) -> None:
        self._closed.append(self.ordinal)


def _runtime(build_seconds: float = 0.0):
    built: list[int] = []
    closed: list[int] = []
    made = {"n": 0}

    def factory():
        made["n"] += 1
        if build_seconds:
            time.sleep(build_seconds)
        return _Engine(built, closed, made["n"])

    return SharedEmbeddingRuntime(factory), built, closed


def _wait_until(predicate, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_burst_of_acquires_builds_one_engine() -> None:
    """Five acquire/release rounds, which is what the worker does per turn."""
    runtime, built, closed = _runtime(build_seconds=0.20)

    for _ in range(5):
        runtime.acquire("worker").close()

    assert built == [1], f"rebuilt {len(built)} times"
    assert closed == [], "closed while the burst was still going"


def test_the_engine_goes_when_the_process_stops_asking() -> None:
    runtime, built, closed = _runtime(build_seconds=0.05)

    runtime.acquire("worker").close()

    assert _wait_until(lambda: closed == [1]), "the engine was never released"
    assert runtime.snapshot()["engine_live"] is False


def test_a_build_that_cost_nothing_is_held_for_nothing() -> None:
    """The window is what the build cost, so a free build is freed at once."""
    runtime, built, closed = _runtime(build_seconds=0.0)

    runtime.acquire("worker").close()

    assert _wait_until(lambda: closed == [1], seconds=1.0)
    assert runtime.snapshot()["engine_live"] is False


def test_a_second_owner_keeps_it_alive() -> None:
    runtime, built, closed = _runtime(build_seconds=0.05)

    first = runtime.acquire("one")
    second = runtime.acquire("two")
    first.close()

    assert closed == []
    assert runtime.snapshot()["lease_count"] == 1
    second.close()
    assert _wait_until(lambda: closed == [1])


def test_close_cancels_the_pending_release() -> None:
    runtime, built, closed = _runtime(build_seconds=5.0)

    runtime.acquire("worker").close()
    runtime.close()

    assert closed == [1]
    assert runtime.snapshot()["engine_live"] is False
