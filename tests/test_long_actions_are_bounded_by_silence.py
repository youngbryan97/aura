"""A ceiling on total duration is a work budget in disguise.

Every action used to be capped at ten minutes regardless of what it had
declared. Measured live: a sixty-question form was killed partway through and
every answer it had landed was thrown away, because how long a questionnaire
takes is not knowable in advance — question counts, page loads and a remote
site's latency are all outside the caller's control.

Silence is the claim the constant can actually make. Ten minutes with nothing
happening is a wedge whatever the work is; an action still reporting progress
is not wedged, and is bounded by the total its caller declared.
"""

from __future__ import annotations

import asyncio

import pytest

from core.runtime import action_executor
from core.runtime.action_executor import (
    SILENCE_CEILING_S,
    _coerce_execution_timeout,
    _invoke_effect_handler,
)


def test_a_declared_total_is_no_longer_clipped():
    assert _coerce_execution_timeout(3195.0) == 3195.0, (
        "the declared total is derived from the work requested; clipping it "
        "replaces that with a constant chosen here"
    )
    assert _coerce_execution_timeout(None) == 60.0


@pytest.mark.asyncio
async def test_a_silent_handler_still_dies_at_the_ceiling(monkeypatch):
    """Nothing changes for handlers that never report."""
    monkeypatch.setattr(action_executor, "SILENCE_CEILING_S", 0.15)

    async def wedged(_context):
        await asyncio.sleep(30)
        return {"ok": True}

    with pytest.raises(TimeoutError, match="silence ceiling"):
        await _invoke_effect_handler(wedged, {}, timeout_s=30.0)


@pytest.mark.asyncio
async def test_a_handler_that_reports_progress_outlives_the_ceiling(monkeypatch):
    monkeypatch.setattr(action_executor, "SILENCE_CEILING_S", 0.15)
    rounds = 0

    async def working(context):
        nonlocal rounds
        report = context["report_progress"]
        for _ in range(6):
            await asyncio.sleep(0.1)
            rounds += 1
            report("still going")
        return {"ok": True, "rounds": rounds}

    result = await _invoke_effect_handler(working, {}, timeout_s=30.0)
    assert result["rounds"] == 6, "progress must extend the window, not the total"


@pytest.mark.asyncio
async def test_the_declared_total_is_still_a_hard_stop(monkeypatch):
    """Reporting progress forever must not mean running forever."""
    monkeypatch.setattr(action_executor, "SILENCE_CEILING_S", 5.0)

    async def forever(context):
        report = context["report_progress"]
        while True:
            await asyncio.sleep(0.05)
            report("busy")

    with pytest.raises(TimeoutError, match="declared total"):
        await _invoke_effect_handler(forever, {}, timeout_s=0.3)


@pytest.mark.asyncio
async def test_the_callback_is_optional():
    """Handlers that ignore it are unaffected, including sync ones."""

    def quick(_context):
        return {"ok": True}

    assert (await _invoke_effect_handler(quick, {}, timeout_s=5.0))["ok"] is True
