"""The page she is reading and the window she is acting in are one thing.

They were allowed to be two. A run could confirm it was in a game by the
browser's address while anchoring itself to whatever happened to be in front
— and then type into that. Measured live 2026-08-26: thirty-five moves into a
terminal, with the game open one window back and never brought forward.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from core.capabilities.browser_controller import BrowserController

SOURCE = Path("core/skills/screen_pursuit.py").read_text()


def test_the_page_says_which_application_holds_it():
    said = inspect.getsource(BrowserController.current_page)
    assert '"app": browser' in said


def test_a_run_about_a_page_anchors_to_the_application_holding_it():
    body = SOURCE[SOURCE.index("async def observe") :]
    assert "about_a_page = bool(open_page or expect_page)" in body
    assert 'holder = str(page.get("app") or "") if about_a_page else ""' in body


def test_a_run_about_something_else_anchors_to_what_is_in_front():
    body = SOURCE[SOURCE.index("async def observe") :]
    assert 'anchor["app"] = (holder or await _frontmost() or "").strip()' in body


def test_and_that_window_is_the_one_brought_forward():
    body = SOURCE[SOURCE.index("async def observe") :]
    assert 'mine = target_app or anchor["app"]' in body
    assert "await _ensure_frontmost(mine)" in body


def test_and_every_act_is_bound_to_it():
    assert 'expect_app=target_app or anchor["app"]' in SOURCE
    assert "expect_app=target_app)" not in SOURCE
