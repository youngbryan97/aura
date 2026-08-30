"""A stopwatch cannot tell a generation that is writing from one that stopped.

LIVE 2026-08-29: asked what she could work out about herself from what she can
measure, the desktop turn ran 185 seconds and ended in "TimeoutError: <no
message; raised in asyncio.timeouts:__aexit__>". The empty message is what a
stopwatch has to say about work it did not watch. The person got the canned
apology.

Every other clock a desktop turn passes through had already been taught the
difference. This was the last hard one on the compact path.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ENGINE = Path("core/brain/cognitive_engine.py")
CHAT = Path("interface/routes/chat.py")


def _the_quick_reply() -> ast.AsyncFunctionDef:
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_direct_desktop_quick_reply"
        ):
            return node
    raise AssertionError("the compact desktop path is gone")


def test_it_waits_while_the_generation_is_working() -> None:
    body = ast.unparse(_the_quick_reply())
    assert "_await_while_it_is_working" in body
    assert "asyncio.wait_for(router.think" not in body.replace(" ", "")


def test_it_asks_whether_a_person_is_waiting() -> None:
    body = ast.unparse(_the_quick_reply())
    assert "a_person_is_waiting" in body
    assert "person_is_waiting=" in body


def test_the_origin_is_judged_as_itself() -> None:
    """Every "desktop_quick_*" matches the foreground prefix, initiatives too."""

    from core.runtime.turn_origin import a_person_is_waiting

    assert a_person_is_waiting("user") is True
    assert a_person_is_waiting("autonomous_initiative_loop") is False
    assert a_person_is_waiting("desktop_quick_autonomous_initiative_loop") is True, (
        "the decorated name is why the bare one is what gets asked"
    )

    body = ast.unparse(_the_quick_reply())
    assert 'a_person_is_waiting(origin' in body.replace("\n", " ").replace("  ", " ")


def test_the_budget_is_still_bounded() -> None:
    body = ast.unparse(_the_quick_reply())
    assert "budget_s=request_timeout + 3.0" in body


def test_the_desktop_transaction_identifies_the_waiting_person() -> None:
    tree = ast.parse(CHAT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        if node.name != "_run_cognitive_engine_chat_turn":
            continue
        body = ast.unparse(node)
        assert "person_is_waiting=bool(require_engine)" in body
        return
    raise AssertionError("desktop cognitive transaction is gone")
