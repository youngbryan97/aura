"""A task that met its governor's bound is not code to repair (third day run, 6 November 2026).

At 03:15:10 one read of her email met its 45 s bound; the next, two seconds later, took 2 s. The governor said it at
ERROR, her terminal monitor read every ERROR as code to repair, and the silent self-modification turn it started is the
turn whose origin stuck and stopped her state log.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from core.resilience.cognitive_governor import CognitiveGovernor


@pytest.mark.asyncio
async def test_a_held_bound_is_said_as_a_warning(caplog):
    async def hangs():
        await asyncio.sleep(10)

    with caplog.at_level(logging.WARNING, logger="core.resilience.cognitive_governor"):
        result = await CognitiveGovernor().execute_safely("email_adapter", hangs, timeout_seconds=0.05)
    assert result == {"status": "timeout", "error": "Operation took too long"}
    said = [r for r in caplog.records if "email_adapter timed out" in r.getMessage()]
    assert said and all(r.levelno == logging.WARNING for r in said)


def test_the_terminal_monitor_does_not_take_a_held_bound_for_a_fault(monkeypatch):
    from core import terminal_monitor as module

    monkeypatch.setattr(module.TerminalMonitor, "_load_blacklist", lambda self: set())
    monitor = module.TerminalMonitor()
    monitor._ingest_error(module.ErrorEntry(
        message="core.resilience.cognitive_governor | Task email_adapter timed out.", level="ERROR", source="log"))
    assert not monitor._error_buffer
    # A timeout that is not a held bound is still for it to look at.
    monitor._ingest_error(module.ErrorEntry(
        message="TimeoutError: the vault lock was not released", level="ERROR", source="log"))
    assert len(monitor._error_buffer) == 1
