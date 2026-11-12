"""R06: every garbage collection is timed, and one that holds a thread past lockdep's loop budget is said.

LIVE 2026-11-12, 09:31: lockdep reported her loop held 363 ms (356 on the CPU) inside a section that sets one flag.
A collection started by an allocation inside it is charged to whatever lock is held; nothing measured the collector.
"""
from __future__ import annotations

import gc
import logging

import pytest

from core.runtime import collector_pauses

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _timing():
    collector_pauses.uninstall_collector_timing()
    yield
    collector_pauses.uninstall_collector_timing()


def test_a_collection_is_counted_by_its_generation():
    assert collector_pauses.install_collector_timing() is True
    assert collector_pauses.install_collector_timing() is False  # once
    gc.collect(2)
    report = collector_pauses.collector_pauses()
    assert report["generations"]["2"]["collections"] >= 1
    assert report["generations"]["2"]["worst_s"] >= 0.0


def test_a_pause_past_the_loop_budget_is_said(monkeypatch, caplog):
    monkeypatch.setattr(collector_pauses, "LOOP_BLOCKING_HOLD_S", -1.0)  # every collection is past it
    collector_pauses.install_collector_timing()
    with caplog.at_level(logging.INFO, logger="Aura.CollectorPauses"):
        gc.collect(1)
    said = [r.getMessage() for r in caplog.records if "collection held" in r.getMessage()]
    assert said and "generation-1" in said[0]
    assert collector_pauses.collector_pauses()["generations"]["1"]["over_budget"] >= 1


def test_uninstalling_stops_timing():
    collector_pauses.install_collector_timing()
    collector_pauses.uninstall_collector_timing()
    gc.collect(0)
    assert collector_pauses.collector_pauses()["generations"] == {}


def test_what_boot_made_is_frozen_out_of_later_collections():
    """LIVE 2026-11-12: 269 full collections in three hours, median 0.4 s, worst 2.7 s, each holding the GIL."""
    keep = [object() for _ in range(10)]  # noqa: F841 - lives through the freeze
    try:
        frozen = collector_pauses.freeze_what_boot_made()
        assert frozen == gc.get_freeze_count() > 0
    finally:
        gc.unfreeze()
