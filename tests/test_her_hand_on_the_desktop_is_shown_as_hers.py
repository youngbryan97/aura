"""Her actions on the desktop are shown as hers: her neuron where her pointer is, her hue at the screen's edges, and
her name with what she is using.

Bryan, 22 November 2026, during her chess against Stockfish: "why isnt her icon the mouse clicker? Was the case for
the YouTube demo. Should expand to all computer use. Same with the 'Aura is using the computer' tag ... And the
highlighted borders? ... helps denote what she is doing vs what im doing". The launcher showed them only for actions
it made itself through its bridge; her clicks in an application were made by another tool and were not shown.
"""
from __future__ import annotations

import asyncio

import core.capabilities.host_automation as host_automation
from core.capabilities.host_automation import AutomationReceipt, HostAutomationProvider


def _told(monkeypatch) -> list[str]:
    told: list[str] = []
    monkeypatch.setattr(host_automation, "_tell_the_launcher", told.append)
    return told


def _logged(host: HostAutomationProvider, *receipts: AutomationReceipt) -> None:
    async def go() -> None:
        for receipt in receipts:
            host._log_receipt(receipt)
        # The launcher is told in the background; let it run.
        await asyncio.sleep(0.05)

    asyncio.run(go())


def test_a_click_she_made_is_shown_with_the_application_she_is_using(monkeypatch) -> None:
    told = _told(monkeypatch)
    host = HostAutomationProvider()
    host._acting_in = "En Croissant"
    _logged(host, AutomationReceipt(action="click", target="10,10", adapter="cliclick", success=True))
    assert told == ["using En Croissant"]


def test_what_failed_what_is_not_her_hand_and_what_the_launcher_made_itself_are_not_told(monkeypatch) -> None:
    told = _told(monkeypatch)
    host = HostAutomationProvider()
    _logged(host,
            AutomationReceipt(action="click", target="10,10", adapter="cliclick", success=False),
            AutomationReceipt(action="get_frontmost_window_context", target="x", adapter="applescript", success=True),
            AutomationReceipt(action="click", target="10,10", adapter="resident_native_bridge", success=True))
    assert told == []


def test_keys_typing_and_scrolling_are_her_hand_too(monkeypatch) -> None:
    told = _told(monkeypatch)
    host = HostAutomationProvider()
    _logged(host, *(AutomationReceipt(action=action, target="", adapter="applescript", success=True)
                    for action in ("hotkey", "hotkeys", "type_text", "scroll")))
    assert told == [""] * 4


def test_the_application_named_for_an_action_is_the_one_she_is_using(monkeypatch) -> None:
    monkeypatch.setattr(host_automation, "_owner_of_the_front_window", lambda: "Some App")
    host = HostAutomationProvider()
    assert asyncio.run(host._refuse_if_not_frontmost("Some App", "click_at")) is None
    assert host._acting_in == "Some App"
