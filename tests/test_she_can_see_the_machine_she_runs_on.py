"""Load is a fact about her condition, and it was measured where she could not read it.

LIVE 2026-08-29, asked whether slow turns were the machine or the code: "tell
me if there's a system-stats command or monitoring tool available in this
environment and I'll run it — otherwise those numbers are genuinely invisible
to me." Her own feed was printing "processor 5%, memory 62%" every few seconds
at the time.

There is a reader for it and a matcher that decides when to staple its answer
on, and the matcher recognises "how hard is the machine working" and not "why
are you slow". Adding a phrase fixes that question and not the next one, so the
reading goes where she reasons from instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_the_reading_is_in_her_state_snapshot() -> None:
    from interface.routes.chat import _host_condition

    condition = _host_condition()
    assert isinstance(condition, dict)
    for name, value in condition.items():
        assert isinstance(value, float), name


def test_a_reading_that_did_not_answer_is_absent_not_zero() -> None:
    """A load reported as 0% because nothing answered is worse than a gap."""

    source = Path("interface/routes/chat.py").read_text(encoding="utf-8")
    assert "Absent rather than zero" in source
    assert "if load is not None and load.present:" in source


def test_both_compactions_carry_it() -> None:
    """Which branch a turn takes must not decide whether she sees her body."""

    source = Path("core/brain/cognitive_engine.py").read_text(encoding="utf-8")
    assert source.count('"host": live_mind_context.get("host"),') == 2


def test_the_payload_offers_it() -> None:
    source = Path("interface/routes/chat.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "_build_live_mind_context_payload":
            continue
        assert '"host": _host_condition()' in ast.unparse(node).replace("'", '"')
        return
    raise AssertionError("the live mind payload is gone")
