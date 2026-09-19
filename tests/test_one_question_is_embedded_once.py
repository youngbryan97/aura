"""One recall asks one question of several sources, and it is embedded once.

LIVE 2026-09-19: "embedding one query took 13.3s ... 1434 texts, 0 embedded
now". Every document was cached; the sources of one recall each embedded the
same question, in threads, one behind another on the encode lock.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from core.memory.vector_memory_engine import EmbeddingEngine


def _engine(calls: list[str]) -> EmbeddingEngine:
    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    EmbeddingEngine.__init__(engine)

    def uncached(text: str, task: str) -> np.ndarray:
        calls.append(text)
        time.sleep(0.05)
        return np.ones(4, dtype=np.float32) * len(text)

    engine._embed_query_uncached = uncached
    return engine


def test_the_same_question_from_five_sources_is_embedded_once():
    calls: list[str] = []
    engine = _engine(calls)
    got: list[np.ndarray] = []
    threads = [
        threading.Thread(target=lambda: got.append(engine.embed_query("what did I say"))) for _ in range(5)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert calls == ["what did I say"]
    assert len(got) == 5 and all(np.array_equal(v, got[0]) for v in got)


def test_a_different_task_is_a_different_question():
    calls: list[str] = []
    engine = _engine(calls)
    engine.embed_query("same words", task="retrieve")
    engine.embed_query("same words", task="classify")
    assert len(calls) == 2


def test_a_caller_cannot_change_what_the_next_one_gets():
    engine = _engine([])
    first = engine.embed_query("q")
    first[:] = 0
    assert engine.embed_query("q").sum() > 0
