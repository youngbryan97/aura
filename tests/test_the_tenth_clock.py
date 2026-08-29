"""The phase that composes what a person reads was cancelled on a stopwatch.

LIVE 2026-08-29: "ResponseGeneration Phase TIMEOUT (476s). Logic took too
long." on a turn that was running a tool and composing an answer from it. The
person got the canned apology.

Nine clocks before this one learned that a generation still emitting tokens is
working and that silence is what a deadline stands in for. This was the tenth.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PHASE = Path("core/phases/response_generation.py")


def test_the_generation_wait_is_not_a_stopwatch() -> None:
    source = PHASE.read_text(encoding="utf-8")
    assert "_await_while_it_is_working(" in source
    assert "asyncio.wait_for(\n                        think_coro," not in source


def test_it_asks_whether_a_person_is_waiting() -> None:
    source = PHASE.read_text(encoding="utf-8")
    assert "person_is_waiting=(" in source
    assert "not is_background and a_person_is_waiting(origin)" in source


def test_background_generation_has_nobody_waiting_whatever_its_origin() -> None:
    """Otherwise a background pass takes the turn ceiling for itself."""

    tree = ast.parse(PHASE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "person_is_waiting":
                continue
            said = ast.unparse(keyword.value)
            assert "not is_background" in said
            return
    raise AssertionError("nothing passes person_is_waiting here any more")


def test_the_budget_is_still_bounded() -> None:
    source = PHASE.read_text(encoding="utf-8")
    assert "budget_s=ordinary_timeout + 2.0" in source
