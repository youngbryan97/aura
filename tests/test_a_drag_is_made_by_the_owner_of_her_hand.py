"""A drag made without cliclick goes through the owner of her hand on the desktop, as her clicks do.

The drag fallback (core/skills/screen_pursuit_surface.py) called PyAutoGUI itself, outside the one place her hand on
the desktop is made and recorded (core/capabilities/host_automation.py), and the governance gate on main was red for
it from 23 November 2026. The pointer here is a stand-in: no test moves the real one.
"""
from __future__ import annotations

import sys
import types

from core.capabilities.host_automation import get_host_automation
from core.skills import screen_pursuit_surface


def _a_pointer(made: list, fails: bool = False) -> types.ModuleType:
    pointer = types.ModuleType("pyautogui")

    def record(name):
        def call(*args, **kwargs):
            if fails:
                raise OSError("no pointer here")
            made.append((name, args, kwargs.get("_pause")))
        return call

    pointer.mouseDown, pointer.moveTo, pointer.mouseUp = record("down"), record("move"), record("up")
    return pointer


def test_the_drag_is_down_through_the_middle_and_up_without_pauses(monkeypatch) -> None:
    made: list = []
    monkeypatch.setitem(sys.modules, "pyautogui", _a_pointer(made))
    receipt = get_host_automation().drag_with_the_pointer((10, 20), (30, 60))
    assert receipt.success and receipt.action == "drag"
    assert [(name, args) for name, args, _pause in made] == [("down", (10, 20)), ("move", (20, 40)), ("move", (30, 60)),
                                                              ("up", (30, 60))]
    assert all(pause is False for _name, _args, pause in made)


def test_the_pursuits_drag_goes_through_the_owner(monkeypatch) -> None:
    made: list = []
    monkeypatch.setitem(sys.modules, "pyautogui", _a_pointer(made))
    assert screen_pursuit_surface._drag_with_pyautogui((1, 2), (3, 4))
    assert made and made[0][0] == "down"
    monkeypatch.setitem(sys.modules, "pyautogui", _a_pointer([], fails=True))
    assert not screen_pursuit_surface._drag_with_pyautogui((1, 2), (3, 4))
