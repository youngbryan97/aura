"""A background kernel tick bounded by its caller ends at that bound, and runs no phase after it.

Her mind tick runs a background kernel tick under ``asyncio.wait_for(kernel.tick(...), timeout=45)``. On Python 3.12
that timeout cancels the mind tick's own task, and the kernel's phase loop caught the cancellation as "a background
phase was cancelled" and went on to the next phase, so the bound never ended the tick. LIVE 20 November 2026 her log
said "Background phase 'CognitiveRoutingPhase' was cancelled (no reason given); continuing tick." after 30 s.
"""
from __future__ import annotations

import asyncio
import time

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


class _Slow:
    async def execute(self, state, objective="", **_kwargs):
        await asyncio.sleep(5.0)
        return state


class _After:
    ran = False

    async def execute(self, state, objective="", **_kwargs):
        type(self).ran = True
        return state


@pytest.mark.asyncio
async def test_the_callers_timeout_ends_the_tick_and_no_later_phase_runs(tmp_path):
    from tests.test_integration_pipeline import _boot_kernel

    kernel = await _boot_kernel(str(tmp_path))
    try:
        kernel._phases = [_Slow(), _After()]
        began = time.monotonic()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(kernel.tick("a background thought", priority=False), timeout=0.5)
        assert time.monotonic() - began < 4.0, "the bound ended the tick, not the slow phase"
        await asyncio.sleep(0.2)
        assert _After.ran is False, "no phase runs after the tick's caller stopped it"
    finally:
        await kernel.shutdown()
