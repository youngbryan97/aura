"""No thought is given more time than the run it belongs to has left.

Measured live on 2026-08-26: twenty-nine narrated moves, a 64 built into the
corner, and "Operation took too long. Completed 0/0 steps." A cycle checked
the clock at its top and then went away to think for longer than the run had.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.skills.screen_pursuit import _within_the_run


async def _slow(_objective, _evidence):
    await asyncio.sleep(5.0)
    return "left"


async def _quick(_objective, _evidence):
    return "left"


@pytest.mark.asyncio
async def test_a_thought_that_would_outlast_the_run_is_cut_short():
    bounded = _within_the_run(_slow, time.monotonic() + 1.2)
    with pytest.raises(asyncio.TimeoutError):
        await bounded("which way", [])


@pytest.mark.asyncio
async def test_a_thought_that_fits_is_left_alone():
    bounded = _within_the_run(_quick, time.monotonic() + 30.0)
    assert await bounded("which way", []) == "left"


@pytest.mark.asyncio
async def test_a_run_already_out_of_time_does_not_start_thinking():
    bounded = _within_the_run(_slow, time.monotonic() - 1.0)
    with pytest.raises(TimeoutError):
        await bounded("which way", [])


def test_with_no_deadline_her_thinking_is_untouched():
    assert _within_the_run(_quick, 0.0) is _quick
    assert _within_the_run(None, 100.0) is None
