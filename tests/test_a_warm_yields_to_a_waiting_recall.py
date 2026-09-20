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
    engine._encode_queue_lock = checked_lock("test.encode_queue", rank=LockRank.LEAF)
    engine._encode_queue = 0
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
    while subject._encode_queue < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert subject._encode_queue == 2
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


def test_a_lone_holder_does_not_defer_to_itself(engine: EmbeddingEngine, monkeypatch) -> None:
    """The holder is in the queue, so the test is two, not one."""
    subject = _bare(engine)
    monkeypatch.setattr(
        EmbeddingEngine, "_primary_inference_active", staticmethod(lambda: False)
    )

    with subject._encoding():
        assert subject._encode_queue == 1
        assert subject._background_should_defer() is False

    assert subject._encode_queue == 0


def test_a_raised_body_does_not_leak_a_queue_slot(engine: EmbeddingEngine) -> None:
    subject = _bare(engine)

    with pytest.raises(EmbeddingWorkDeferredError):
        with subject._encoding():
            raise EmbeddingWorkDeferredError("boom")

    assert subject._encode_queue == 0


def test_the_two_locks_are_never_nested() -> None:
    """Both are LEAF, and lockdep requires strictly increasing rank.

    Counting only the WAITERS meant decrementing after acquiring the encode
    lock, which nests one LEAF inside another. The first live boot after
    that change recorded a rank_inversion splat naming both lines.
    """
    import ast
    import inspect
    import textwrap

    from core.memory.vector_memory_engine import EmbeddingEngine

    body = ast.parse(textwrap.dedent(inspect.getsource(EmbeddingEngine._encoding)))
    for node in ast.walk(body):
        if not isinstance(node, ast.With):
            continue
        held = {ast.unparse(item.context_expr) for item in node.items}
        if not any("_encode_lock" in name for name in held):
            continue
        inner = {
            ast.unparse(item.context_expr)
            for child in ast.walk(node)
            if isinstance(child, ast.With) and child is not node
            for item in child.items
        }
        assert not [name for name in inner if "_encode_queue_lock" in name], (
            "the queue lock is taken inside the encode lock again"
        )
