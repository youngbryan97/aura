"""The page she is reading and the window she is acting in are one thing.

They were allowed to be two. A run could confirm it was in a game by the
browser's address while anchoring itself to whatever happened to be in front
— and then type into that. Measured live 2026-08-26: thirty-five moves into a
terminal, with the game open one window back and never brought forward.
"""

from __future__ import annotations

from screen_pursuit_support import pursuit_source

import ast
import inspect
from pathlib import Path

from core.capabilities.browser_controller import BrowserController

SOURCE = pursuit_source()


def _assignment(name: str) -> str:
    """The right-hand side of ``name = ...`` inside ``observe``, as source.

    Read through the syntax tree rather than matched as a line of text. The
    literal-string form of this check broke the moment the expression was
    rewrapped across three lines and given another clause -- the coupling it
    guards was intact and widened, and the test reported it gone. A check that
    fails when the code is reworded is a check that will be deleted the next
    time somebody reformats, which is how the thing it guards goes quiet.
    """
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "observe":
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in inner.targets
            ):
                return ast.unparse(inner.value)
    raise AssertionError(f"observe() no longer assigns {name}")


def test_the_page_says_which_application_holds_it():
    said = inspect.getsource(BrowserController.current_page)
    assert '"app": browser' in said


def test_a_run_about_a_page_anchors_to_the_application_holding_it():
    # What has to hold: a run is about a page when a page was asked for, and
    # when it is, the anchor comes from whichever application holds that page
    # rather than from whatever is in front.
    about = _assignment("about_a_page")
    assert "open_page" in about, about
    assert "expect_page" in about, about

    holder = _assignment("holder")
    assert "about_a_page" in holder, holder
    assert "app" in holder, holder
    assert "page" in holder, holder


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
