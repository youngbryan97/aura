"""Deciding what a turn needs must not load a model to decide it.

LIVE, 2026-08-21. A chat preflight logged sight at 234.321 seconds. Whether a
turn needs a camera reading ends in a semantic routing question, and asking it
called `semantic_routing_available`, which checks the model out — which loads
it. The first turn after a restart paid for that load in the foreground before
anything was said back.

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
