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
    phase.in_progress, phase.taken = None, []
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


def test_an_organ_the_tick_ran_out_of_time_on_is_still_named_after_the_cancel():
    """35 stalls from 12 to 15 November named no organ: the `finally` that cleared it ran on the cancel, before the
    tick read it."""
    phase = CognitiveIntegrationPhase.__new__(CognitiveIntegrationPhase)
    phase.in_progress, phase.taken = None, []

    async def a_tick_that_runs_out_of_time():
        try:
            await asyncio.wait_for(phase._organ(None, "_run_topology_evolution", asyncio.sleep(10)), timeout=0.05)
        except TimeoutError:
            return _what_the_phase_was_doing(phase)
        return ""

    said = asyncio.run(a_tick_that_runs_out_of_time())
    assert said.startswith(" in _run_topology_evolution, "), said


def test_a_stall_names_where_the_rest_of_the_time_went():
    """LIVE 15 November 2026: "STALLED (timeout of 5s) in _run_autopoiesis, 0.4s into it"; the other 4.6 s were the
    organs that had already finished."""
    phase = CognitiveIntegrationPhase.__new__(CognitiveIntegrationPhase)
    phase.in_progress, phase.taken = None, []

    async def slow(seconds):
        await asyncio.sleep(seconds)

    async def a_tick_that_runs_out_of_time():
        async def organs():
            await phase._organ(None, "_run_strange_loop", slow(0.01))
            await phase._organ(None, "_run_adaptive_immunity", slow(0.12))
            await phase._organ(None, "_run_autopoiesis", slow(10))

        try:
            await asyncio.wait_for(organs(), timeout=0.3)
        except TimeoutError:
            return _what_the_phase_was_doing(phase)
        return ""

    said = asyncio.run(a_tick_that_runs_out_of_time())
    assert said.startswith(" in _run_autopoiesis, "), said
    before = said.split("; before it ", 1)[1]
    assert before.startswith("_run_adaptive_immunity 0.1s"), "the costliest finished organ comes first"
    assert "_run_strange_loop" in before


def test_a_stall_says_how_many_jobs_were_queued_for_a_worker_thread():
    """At boot her organs' thread work can queue behind boot's: the stall line counts the queue."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from core.mind_tick import _what_the_worker_threads_were_doing

    release = threading.Event()

    async def a_loop_whose_one_worker_is_busy():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        busy = [loop.run_in_executor(None, release.wait) for _ in range(3)]
        await asyncio.sleep(0.05)
        said = _what_the_worker_threads_were_doing()
        release.set()
        await asyncio.gather(*busy)
        return said, _what_the_worker_threads_were_doing()

    queued, after = asyncio.run(a_loop_whose_one_worker_is_busy())
    assert queued.startswith("; 2 job(s) waiting for the loop's 1 worker threads, held 1 at "), queued
    assert " wait" in queued, "the held worker is named by where it is: in threading.Event.wait"
    assert after == ""
