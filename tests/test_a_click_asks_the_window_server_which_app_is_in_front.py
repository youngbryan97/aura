"""A click or key asks the window server which application is in front before it asks System Events.

Her focus guard asked System Events before every click and key: 143 ms each, measured 22 November 2026, where the
window server's list of windows, front to back, answers in about two. In a game on a three-minute clock that was two
clicks a move and one for every piece picked up. A plain yes from the window server lets the action go; anything else
is still asked of System Events, so nothing is refused that was not refused before.
"""
from __future__ import annotations

import asyncio

import core.capabilities.host_automation as host_automation
from core.capabilities.host_automation import AutomationReceipt, get_host_automation


def _guard(monkeypatch, *, shown: str, frontmost: str | None):
    asked: list[str] = []

    async def system_events():
        asked.append("System Events")
        if frontmost is None:
            raise AssertionError("System Events asked though the window server said yes")
        return AutomationReceipt(action="get_frontmost_window_context", target="frontmost_window",
                                 adapter="applescript", success=True, result=f"{frontmost}|a window")

    host = get_host_automation()
    monkeypatch.setattr(host_automation, "_owner_of_the_front_window", lambda: shown)
    monkeypatch.setattr(host, "get_frontmost_window_context", system_events)
    return asyncio.run(host._refuse_if_not_frontmost("en-croissant", "click_at")), asked


def test_a_yes_from_the_window_server_is_enough(monkeypatch) -> None:
    refusal, asked = _guard(monkeypatch, shown="en-croissant", frontmost=None)
    assert refusal is None
    assert asked == []


def test_anything_else_is_still_asked_of_system_events(monkeypatch) -> None:
    refusal, asked = _guard(monkeypatch, shown="Aura", frontmost="en-croissant")
    assert refusal is None and asked == ["System Events"]
    refusal, asked = _guard(monkeypatch, shown="", frontmost="Finder")
    assert refusal is not None and asked == ["System Events"]
    assert "not frontmost" in refusal.error
