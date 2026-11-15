"""A mind-tick phase that runs out of time says where it was.

LIVE 14 November 2026, her app's boot: "MindTick: Phase 'cognitive_integration' STALLED (timeout)", a phase that runs
a dozen organs one after another, and nothing said which of them held it.
"""
from __future__ import annotations

import asyncio
import time

from core.mind_tick import _what_the_phase_was_doing
from core.phases.cognitive_integration_phase import CognitiveIntegrationPhase


def test_a_phase_that_says_where_it_is_is_named_there():
    class Phase:
        in_progress = ("_run_autopoiesis", time.monotonic() - 9.5)

    said = _what_the_phase_was_doing(Phase())
    assert said.startswith(" in _run_autopoiesis, ") and said.endswith("s into it")
    assert _what_the_phase_was_doing(object()) == ""
    # A phase registered as a bound method is read through the object it belongs to.
    assert _what_the_phase_was_doing(Phase().__init__).startswith(" in _run_autopoiesis")
    assert _what_the_phase_was_doing(object().__init__) == ""


def test_cognitive_integration_keeps_the_organ_it_is_in_and_clears_it():
    phase = CognitiveIntegrationPhase.__new__(CognitiveIntegrationPhase)
    phase.in_progress = None
    seen = []

    async def organ():
        seen.append(phase.in_progress)
        return 7

    async def failing():
        raise ValueError("an organ that raises")

    assert asyncio.run(phase._organ(None, "_run_strange_loop", organ())) == 7
    assert seen[0][0] == "_run_strange_loop" and phase.in_progress is None

    import core.phases.cognitive_integration_phase as module

    original = module._record_cognitive_degradation
    module._record_cognitive_degradation = lambda *a, **k: None
    try:
        assert asyncio.run(phase._organ(None, "_run_autopoiesis", failing())) is None
    finally:
        module._record_cognitive_degradation = original
    assert phase.in_progress is None
