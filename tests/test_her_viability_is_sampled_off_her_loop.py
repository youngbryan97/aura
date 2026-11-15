"""Her viability sample reads the filesystem's free space and other organs' counts, and her event loop waits on neither.

Master Todo R06: two of twelve stall dumps of 11 November 2026 held her loop thread in this sample's disk read. The
sample is taken on a worker thread; classifying it and moving between states stay on the loop.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from core.organism import viability

pytestmark = pytest.mark.unit


def test_the_sample_is_taken_off_the_loop_and_the_state_moves_on_it():
    sampled_on, moved_on = [], []
    engine = viability.ViabilityEngine()
    real_sampler = engine._sampler

    def sampler():
        sampled_on.append(threading.get_ident())
        return real_sampler()

    engine._sampler = sampler
    real_tick = engine.tick

    def tick(sample=None):
        moved_on.append(threading.get_ident())
        return real_tick(sample)

    engine.tick = tick

    async def run():
        loop_thread = threading.get_ident()
        await engine.start(interval=0.01)
        while not moved_on:
            await asyncio.sleep(0.01)
        await engine.stop()
        return loop_thread

    loop_thread = asyncio.run(run())
    assert sampled_on and all(thread != loop_thread for thread in sampled_on)
    assert moved_on and all(thread == loop_thread for thread in moved_on)
