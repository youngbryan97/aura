"""A background warm must not hold the encoder through a foreground recall.

Live, 2026-09-20: "embedding one query took 37.5s (SharedEmbeddingLease);
1430 texts, 0 embedded now", inside an episodic recall that took 39.2s,
inside a CognitiveRoutingPhase that took 49.6s. That is the event-loop lag
that killed `state_vault` and `SensoryGate`, and RAM was at 37% — nothing
was short of memory. The query had every document cached and did no work at
all: it waited for the encode lock while a 1,430-text warm held it.

The yield between batches existed already and asked the wrong question. It
read `primary_inference_active()`, which is whether the model is GENERATING,
and a recall runs before generation starts. Somebody waiting for the lock is
the fact that matters, and it needs no notion of foreground.
"""
from __future__ import annotations

import threading
import time

import pytest

from core.memory.vector_memory_engine import EmbeddingEngine, EmbeddingWorkDeferredError


@pytest.fixture
def engine() -> EmbeddingEngine:
    return EmbeddingEngine.__new__(EmbeddingEngine)


def _bare(engine: EmbeddingEngine) -> EmbeddingEngine:
    """Only the two locks and the counter: no model, no checkout, no threads."""
    from core.runtime.lockdep import LockRank, checked_lock

    engine._encode_lock = checked_lock("test.encode", rank=LockRank.LEAF)
    engine._encode_waiting_lock = checked_lock("test.encode_waiting", rank=LockRank.LEAF)
    engine._encode_waiting = 0
    return engine


def test_a_waiter_makes_a_background_batch_defer(engine: EmbeddingEngine, monkeypatch) -> None:
    subject = _bare(engine)
    monkeypatch.setattr(
        EmbeddingEngine, "_primary_inference_active", staticmethod(lambda: False)
    )

    assert subject._background_should_defer() is False

    holding = threading.Event()
    release = threading.Event()
    saw = {}

    def _warm() -> None:
        with subject._encoding():
            holding.set()
            release.wait(5.0)
            saw["while_waited_on"] = subject._background_should_defer()

    warm = threading.Thread(target=_warm)
    warm.start()
    assert holding.wait(5.0)
    assert subject._background_should_defer() is False, "nobody is waiting yet"

    recall = threading.Thread(target=lambda: subject._encoding().__enter__())
    recall.start()
    deadline = time.monotonic() + 5.0
    while subject._encode_waiting == 0 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert subject._encode_waiting == 1
    release.set()
    warm.join(5.0)
    recall.join(5.0)

    assert saw["while_waited_on"] is True


def test_generation_still_makes_it_defer(engine: EmbeddingEngine, monkeypatch) -> None:
    """The reason that was already there has to keep working."""
    subject = _bare(engine)
    monkeypatch.setattr(
        EmbeddingEngine, "_primary_inference_active", staticmethod(lambda: True)
    )

    assert subject._background_should_defer() is True


def test_the_count_is_waiters_not_holders(engine: EmbeddingEngine, monkeypatch) -> None:
    """A holder that counted itself would defer to itself and never finish."""
    subject = _bare(engine)
    monkeypatch.setattr(
        EmbeddingEngine, "_primary_inference_active", staticmethod(lambda: False)
    )

    with subject._encoding():
        assert subject._encode_waiting == 0
        assert subject._background_should_defer() is False

    assert subject._encode_waiting == 0


def test_a_failed_acquisition_does_not_leak_a_waiter(engine: EmbeddingEngine) -> None:
    subject = _bare(engine)

    with pytest.raises(EmbeddingWorkDeferredError):
        with subject._encoding():
            raise EmbeddingWorkDeferredError("boom")

    assert subject._encode_waiting == 0
