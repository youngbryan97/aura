"""A deadline asks how long; what matters about long work is whether it stopped.

LIVE 2026-09-20 at 09:27: a game she had been asked to play until the 2048
tile was killed an hour in — "Task computer_use timed out" — while it was
landing a move every two seconds and had climbed 32, 64, 128 and 256.
"""
from __future__ import annotations

import asyncio

import pytest

from core.resilience.cognitive_governor import CognitiveGovernor
from core.runtime.still_getting_somewhere import it_got_somewhere


@pytest.mark.asyncio
async def test_work_that_reports_nothing_still_times_out():
    async def wedged():
        await asyncio.sleep(5)
        return {"ok": True}

    result = await CognitiveGovernor().execute_safely(
        "wedged", wedged, timeout_seconds=0.05
    )
    assert result["status"] == "timeout"


@pytest.mark.asyncio
async def test_work_that_keeps_moving_is_given_as_long_again():
    async def playing():
        for _ in range(10):
            await asyncio.sleep(0.02)
            it_got_somewhere("a key landed")
        return {"ok": True, "moves": 10}

    result = await CognitiveGovernor().execute_safely(
        "playing", playing, timeout_seconds=0.05
    )
    assert result == {"ok": True, "moves": 10}


@pytest.mark.asyncio
async def test_work_that_moved_and_then_stopped_is_cancelled():
    cancelled: list[bool] = []

    async def stalls():
        it_got_somewhere("one key landed")
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return {"ok": True}

    result = await CognitiveGovernor().execute_safely(
        "stalls", stalls, timeout_seconds=0.05
    )
    assert result["status"] == "timeout"
    await asyncio.sleep(0.01)
    assert cancelled == [True], "the work it gave up on is stopped"


@pytest.mark.asyncio
async def test_a_report_from_nowhere_is_not_an_error():
    assert it_got_somewhere("nobody is listening") is False
