"""A pointer thrown into a corner of the screen means stop, not crash.

PyAutoGUI raises its fail-safe when the pointer is in a corner. That is how a
person stops automation that is running away with their machine, and it came
through uncaught: a scroll interrupted that way ended her whole run
(2026-09-18). It is a refusal of that one act, said as the person's stop.
"""

from __future__ import annotations

import sys

import pytest


@pytest.mark.asyncio
async def test_a_scroll_stopped_by_the_person_is_refused_not_raised(monkeypatch):
    from core.capabilities.host_automation import get_host_automation

    stand_in = sys.modules["pyautogui"]

    def thrown_into_a_corner(*_args, **_kwargs):
        raise stand_in.FailSafeException("fail-safe triggered")

    monkeypatch.setattr(stand_in, "scroll", thrown_into_a_corner)
    receipt = await get_host_automation().scroll(dy=-5)
    assert receipt.success is False
    assert "stopped_by_person" in receipt.error


@pytest.mark.asyncio
async def test_no_test_scrolls_the_machine_it_runs_on():
    from core.capabilities.host_automation import get_host_automation

    stand_in = sys.modules["pyautogui"]
    await get_host_automation().scroll(dy=-5)
    assert ("scroll", (-5,), {"_pause": False}) in stand_in.asked
