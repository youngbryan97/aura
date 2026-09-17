"""Deciding what a turn needs must not load a model to decide it.

Whether a turn needs a camera reading ends in a semantic routing question, and
asking it called `semantic_routing_available`, which checks the model out —
which loads it. The first turn after a restart paid for that load in the
foreground before anything was said back.

(The preflight timings are milliseconds. An earlier note here read them as
seconds and claimed a 234-second sight step; the cold load was real, the
figure was not.)

The readiness question now loads nothing, a cold turn falls to the lexical
floor and asks for a warm in the background, and the warm happens on the
orchestrator's own background task.
"""

from __future__ import annotations

import time
from pathlib import Path

from core.cognition.evidence_relevance import (
    PHYSICAL_PERCEPTION,
    semantic_routing_ready,
    wants_evidence,
)


def test_asking_whether_routing_is_ready_loads_nothing():
    """A readiness question must not do the thing it is asking about."""
    started = time.perf_counter()
    semantic_routing_ready()
    assert time.perf_counter() - started < 1.0


def test_a_cold_turn_is_answered_without_waiting():
    started = time.perf_counter()
    wants_evidence("what is 2 + 2", PHYSICAL_PERCEPTION)
    assert time.perf_counter() - started < 1.0


def test_the_lexical_floor_still_decides_when_the_model_is_cold():
    """Falling back must not mean falling silent: the floor can still add."""
    assert wants_evidence("x", PHYSICAL_PERCEPTION, lexical_floor=lambda _text: True) is True


def test_the_warm_is_owned_by_a_background_task():
    source = Path("core/orchestrator/handlers/status_manager.py").read_text(encoding="utf-8")
    assert "warm_semantic_routing" in source
    assert "semantic_routing_ready" in source


def test_routing_no_longer_checks_out_the_model_to_answer_wants_evidence():
    source = Path("core/cognition/evidence_relevance.py").read_text(encoding="utf-8")
    body = source[source.index("def wants_evidence") :]
    assert "semantic_routing_ready()" in body
    assert "semantic_routing_available()" not in body


def test_a_resident_model_with_cold_anchors_is_not_ready(monkeypatch):
    """LIVE 2026-09-16: the boot warmup was skipped on a loaded host, the
    model was resident, and the first turn encoded every anchor sentence of
    every evidence kind through it, synchronously, on the API server's loop.
    The server answered nothing for fifty minutes. The anchors are part of
    the model as far as a turn is concerned."""
    import core.cognition.evidence_relevance as m

    class _Embedder:
        _model = object()

    monkeypatch.setattr(m, "_embedder", lambda: _Embedder())
    monkeypatch.setattr(m, "_bind_cache_to_engine", lambda engine: None)
    with m._LOCK:
        saved = dict(m._ANCHOR_CACHE)
        m._ANCHOR_CACHE.clear()
    try:
        assert m.anchors_warm() is False
        assert semantic_routing_ready() is False
        encoded: list[str] = []
        monkeypatch.setattr(m, "relevance", lambda text, kind: encoded.append(kind) or 1.0)
        m._WARMING["asked"] = True  # the warm was asked; the turn does not encode
        assert wants_evidence("what do you see on my desk", PHYSICAL_PERCEPTION) is False
        assert encoded == [], "a turn encoded anchors on the request path"
        with m._LOCK:
            for kind in m._ANCHORS:
                m._ANCHOR_CACHE[kind] = [[1.0]]
                m._ANCHOR_CACHE[f"__baseline__:{kind}"] = [[1.0]]
        assert m.anchors_warm() is True
        assert semantic_routing_ready() is True
    finally:
        with m._LOCK:
            m._ANCHOR_CACHE.clear()
            m._ANCHOR_CACHE.update(saved)


def test_the_warm_encodes_the_anchors_too():
    source = Path("core/cognition/evidence_relevance.py").read_text(encoding="utf-8")
    body = source[source.index("def warm_semantic_routing") :]
    body = body[: body.index("def _ask_for_a_warm")]
    assert "_prewarm_anchor_vectors()" in body
    assert "anchors_warm()" in body


def test_the_preflight_classifies_sight_off_the_server_loop():
    source = Path("interface/routes/chat_preflight.py").read_text(encoding="utf-8")
    assert "await asyncio.to_thread(_classify_sight, _original_user_message)" in source


def test_the_warmup_waits_for_the_cortex_rather_than_giving_up():
    source = Path("interface/server.py").read_text(encoding="utf-8")
    body = source[source.index("async def _prewarm_chat_dependencies_after_cortex_ready") :]
    body = body[: body.index("from core.cognition.evidence_relevance import prewarm_evidence_relevance")]
    assert "Chat dependency warmup skipped" not in body
    assert "still waiting" in body
