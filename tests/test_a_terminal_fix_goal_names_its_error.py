"""The fix goal the terminal monitor hands on names the error it is for.

On her day runs the metabolic coordinator passed "Terminal Command Failure: Unknown" to the self-modifier eight times:
the goal carried an objective and an id, and no `error`, so the repair it asked for had nothing to repair.
"""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit


def test_the_goal_carries_the_error(monkeypatch, tmp_path):
    from core import terminal_monitor as module

    monkeypatch.setattr(module.TerminalMonitor, "_load_blacklist", lambda self: set())
    monitor = module.TerminalMonitor()
    monitor._error_buffer.append(
        module.ErrorEntry(message="Traceback: ModuleNotFoundError: No module named 'x'", level="ERROR", source="stderr")
    )
    monkeypatch.setattr(monitor, "_classify_error", lambda entry: "Fix the missing module")
    goal = asyncio.run(monitor.check_for_errors())
    assert goal is not None
    assert "ModuleNotFoundError" in goal["error"]
    assert goal["source"] == "stderr"
