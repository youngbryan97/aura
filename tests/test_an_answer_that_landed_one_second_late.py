"""A wait ends when the generation stops producing, not when a guess expires.

LIVE, 2026-09-21. Asked to measure 4 litres with a 3-litre and a 5-litre
jug, Aura answered "That answer took too long to finish cleanly." The log
from the same turn:

    🧠 [ANSWER CLOCK] 256 tokens ... decode in about 108s and the prompt
       takes about 11s to read ... deadline 103s → 122s
    ⏳ Kernel soft deadline missed but cortex is alive and generating.
       Waiting 30s for kernel to finish (no competing request).
    KernelInterface chat timed out after its budget; the fallback ladder
       takes this turn (TimeoutError).
    ⏱️ [WORKER] prefill=1422 tokens/32.53s (43.7 tok/s), decode=225
       tokens/39.90s
    ✅ Brainstem response received (len=667)

The prompt took 32.5 seconds to read, not 11. Everything downstream was
priced off that estimate, the wait ran out one second before the answer
arrived, and 667 characters of real work were thrown away.

``still_producing`` already exists for this. The cognitive engine renews its
cycle from it and the mlx client tells a slow decode from a wedged one with
it. The outermost wait — the one that decides whether the person gets the
answer — could not see it.
"""

from __future__ import annotations

import asyncio

import pytest

from interface.routes.chat_foreground_lane import _wait_while_it_is_still_answering


@pytest.fixture
def a_quiet_gap(monkeypatch):
    """Pin what "still normal" means so the test does not read the host."""
    import core.runtime.turn_progress as progress

    monkeypatch.setattr(progress, "normal_gap_between_tokens", lambda *_a, **_k: 2.0)
    return 2.0


def _answers_after(delay: float, text: str):
    async def _work():
        await asyncio.sleep(delay)
        return text

    return _work


def test_an_answer_one_second_past_the_budget_is_still_served(monkeypatch, a_quiet_gap):
    import core.runtime.turn_progress as progress

    monkeypatch.setattr(progress, "still_producing", lambda **_k: True)

    async def _run():
        task = asyncio.ensure_future(_answers_after(0.45, "four litres")())
        return await _wait_while_it_is_still_answering(task, 0.2)

    assert asyncio.run(_run()) == "four litres"


def test_a_wedged_turn_still_ends(monkeypatch, a_quiet_gap):
    """No token within a normal gap, and the wait is over."""
    import core.runtime.turn_progress as progress

    monkeypatch.setattr(progress, "still_producing", lambda **_k: False)

    async def _run():
        task = asyncio.ensure_future(_answers_after(30.0, "never arrives")())
        try:
            await _wait_while_it_is_still_answering(task, 0.2)
        finally:
            task.cancel()

    with pytest.raises(TimeoutError):
        asyncio.run(_run())


def test_it_does_not_wait_at_all_when_the_answer_is_already_there(a_quiet_gap):
    async def _run():
        task = asyncio.ensure_future(_answers_after(0.0, "here")())
        return await _wait_while_it_is_still_answering(task, 60.0)

    assert asyncio.run(_run()) == "here"


def test_production_stopping_ends_a_renewed_wait(monkeypatch, a_quiet_gap):
    """It renews while tokens arrive and stops the moment they stop."""
    import core.runtime.turn_progress as progress

    arrivals = iter([True, True, False])
    monkeypatch.setattr(
        progress, "still_producing", lambda **_k: next(arrivals, False)
    )

    async def _run():
        task = asyncio.ensure_future(_answers_after(30.0, "never arrives")())
        try:
            await _wait_while_it_is_still_answering(task, 0.1)
        finally:
            task.cancel()

    with pytest.raises(TimeoutError):
        asyncio.run(_run())
