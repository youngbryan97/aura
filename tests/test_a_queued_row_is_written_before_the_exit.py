"""Rows queued for a writer thread reach disk before the process leaves.

The outcome ledger and the ontogeny authority ledger hand writes made on the
event loop to a daemon writer thread. Every exit path leaves through the
finalizer's `os._exit`, which does not wait for daemon threads, and nothing
flushed either queue first — so whatever was written in the last moments
before a shutdown was gone. Each registers its flush in the coordinator's
`memory_commit` phase the first time it starts a writer.
"""
from __future__ import annotations

import asyncio

import pytest

from core.runtime import shutdown_coordinator as sc
from core.runtime.sqlite_support import connecting


@pytest.fixture
def coordinator():
    sc.reset_shutdown_coordinator()
    yield sc.get_shutdown_coordinator()
    sc.reset_shutdown_coordinator()


@pytest.mark.asyncio
async def test_the_outcome_ledger_registers_its_flush_when_it_starts_a_writer(tmp_path, coordinator):
    from core.cognition.outcome_ledger import OutcomeLedger

    ledger = OutcomeLedger(db_path=str(tmp_path / "outcomes.db"))
    assert "outcome_ledger" not in coordinator.handler_names("memory_commit")
    ledger.open("an action written from the loop", 0.5)   # on a running loop: queued
    assert coordinator.handler_names("memory_commit").count("outcome_ledger") == 1
    ledger.open("a second one", 0.5)
    assert coordinator.handler_names("memory_commit").count("outcome_ledger") == 1


@pytest.mark.asyncio
async def test_the_flush_writes_what_was_queued(tmp_path, coordinator):
    from core.cognition.outcome_ledger import OutcomeLedger

    ledger = OutcomeLedger(db_path=str(tmp_path / "outcomes.db"))
    ledger.open("queued at the last moment", 0.5)
    await asyncio.to_thread(ledger._flush_writes)
    import sqlite3

    with connecting(sqlite3.connect(str(tmp_path / "outcomes.db"))) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM outcome_receipts").fetchone()
    assert count >= 1


def test_both_ledgers_name_the_memory_commit_phase(monkeypatch):
    """Each ledger hands its flush to the shutdown coordinator for the phase that runs before state is vaulted."""
    from types import SimpleNamespace

    from core.cognition import outcome_ledger
    from core.ontogeny import authority
    from core.runtime import shutdown_coordinator

    for module in (outcome_ledger, authority):
        asked: list[tuple[object, dict]] = []
        coordinator = SimpleNamespace(register=lambda flush, **kwargs: asked.append((flush, kwargs)))
        monkeypatch.setattr(shutdown_coordinator, "get_shutdown_coordinator", lambda: coordinator)

        def flush():
            return None

        assert module._flush_at_shutdown(flush, name="a ledger") is True
        assert asked == [(flush, {"phase": "memory_commit", "name": "a ledger"})], module.__name__
