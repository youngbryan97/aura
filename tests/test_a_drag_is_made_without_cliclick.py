"""A drag is made without cliclick, and a click keeps the moment after it.

The machine she runs on has no cliclick. Her drags called it alone and could not be made at all; they fall back to
PyAutoGUI. Her clicks fall back to PyAutoGUI too, and keep its pause after each click: LIVE 23 November 2026, game 24
against Stockfish, without it her moves took 3.4 s to be seen made instead of 1, and one was not taken.
"""
from __future__ import annotations

import asyncio
import sys
import types

import core.capabilities.host_automation as host_automation
import core.skills.screen_pursuit_surface as surface


class _NoCliclick:
    async def spawn_async(self, *args, **kwargs):
        raise FileNotFoundError("cliclick")


def _a_pointer(monkeypatch) -> list[tuple[str, tuple, dict]]:
    calls: list[tuple[str, tuple, dict]] = []
    fake = types.ModuleType("pyautogui")
    for name in ("click", "rightClick", "mouseDown", "mouseUp", "moveTo"):
        setattr(fake, name, lambda *a, _n=name, **k: calls.append((_n, a, k)))
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    return calls


def test_a_click_without_cliclick_keeps_the_moment_after_it(monkeypatch) -> None:
    # LIVE 23 November 2026, game 24: without the pause after each click her moves took 3.4 s to be seen made, and one
    # was not taken. The pause is part of every wait tuned after a click.
    calls = _a_pointer(monkeypatch)
    monkeypatch.setattr(host_automation, "get_subprocess_gateway", lambda: _NoCliclick())
    host = host_automation.HostAutomationProvider()
    receipt = asyncio.run(host.click_at(10, 20))
    assert receipt.success and receipt.adapter == "pyautogui"
    assert calls == [("click", (10, 20), {})]


def test_a_drag_without_cliclick_is_made_by_pyautogui(monkeypatch) -> None:
    calls = _a_pointer(monkeypatch)
    import core.runtime.subprocess_gateway as gateway

    monkeypatch.setattr(gateway, "get_subprocess_gateway", lambda: _NoCliclick())
    monkeypatch.setattr("core.runtime.proof_policy.proof_run_active", lambda: False)
    monkeypatch.setattr(host_automation.HostAutomationProvider, "_mark_her_hand", lambda self, receipt: None)
    assert asyncio.run(surface._drag_on_the_host((0, 0), (100, 50)))
    assert [name for name, _a, _k in calls] == ["mouseDown", "moveTo", "moveTo", "mouseUp"]
    assert all(k.get("_pause") is False for _n, _a, k in calls)
