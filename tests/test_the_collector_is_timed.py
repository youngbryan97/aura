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
        collector_pauses.say_what_was_measured()
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
    keep = [[] for _ in range(10)]  # containers, which the collector tracks; they live through the freeze
    try:
        frozen = collector_pauses.freeze_what_boot_made()
        # A frozen object can still be freed by its reference count, so in a busy process the count read a moment
        # later can be lower (suite of 14 November: 1,561,137 returned, 1,561,134 read). What the freeze promises is
        # that what lives is held: these containers are out of every later collection.
        assert frozen > 0 and 0 < gc.get_freeze_count() <= frozen
        collected = {id(o) for o in gc.get_objects()}  # what later collections walk, by identity
        assert not any(id(k) in collected for k in keep)
    finally:
        gc.unfreeze()


def test_a_long_collection_says_whether_a_line_asked_for_it_or_an_allocation_set_it_off(monkeypatch, caplog):
    """LIVE 2026-11-16: 180 full collections past the budget in 40 minutes; whether her timers asked for them or the
    collector started them decides what to change."""
    monkeypatch.setattr(collector_pauses, "LOOP_BLOCKING_HOLD_S", -1.0)
    collector_pauses.install_collector_timing()
    with caplog.at_level(logging.INFO, logger="Aura.CollectorPauses"):
        gc.collect(2)
        collector_pauses.uninstall_collector_timing()
    said = [r.getMessage() for r in caplog.records if "collection held" in r.getMessage()]
    assert said and "asked for at tests/test_the_collector_is_timed.py:" in said[0]

    caplog.clear()
    previous = gc.get_threshold()
    gc.set_threshold(1)
    collector_pauses.install_collector_timing()
    try:
        with caplog.at_level(logging.INFO, logger="Aura.CollectorPauses"):
            made = [[] for _ in range(50)]
            collector_pauses.say_what_was_measured()
    finally:
        gc.set_threshold(*previous)
        collector_pauses.uninstall_collector_timing()
    said = [r.getMessage() for r in caplog.records if "collection held" in r.getMessage()]
    assert made and said and all("set off by an allocation at" in line for line in said)


def test_each_line_that_asks_for_collections_is_measured_over_a_window():
    """What a line that asks for collections costs and frees, every collection it asks for, so that a change to what it
    asks for is measured on the same terms before and after (Master Todo F16, part 3)."""
    import gc
    import time

    from core.runtime import collector_pauses

    collector_pauses.install_collector_timing()
    try:
        began = time.time()
        for _ in range(3):
            gc.collect()
        gc.collect(1)
        sites = collector_pauses.asked_for_collections(since=began)
    finally:
        collector_pauses.uninstall_collector_timing()
    here = [site for site in sites if site.startswith("tests/test_the_collector_is_timed.py:")]
    assert len(here) == 2, sites
    full = next(sites[s] for s in here if sites[s]["generations"] == [2])
    young = next(sites[s] for s in here if sites[s]["generations"] == [1])
    assert full["collections"] == 3 and young["collections"] == 1
    assert full["total_s"] >= full["median_s"] > 0
    assert collector_pauses.asked_for_collections(since=time.time() + 60) == {}


def test_a_collection_inside_a_section_whose_lock_logging_needs_does_not_wedge_the_thread(monkeypatch):
    """LIVE 2026-11-17, 14:07 local: a full collection set off inside lockdep's bookkeeping logged from its callback, the
    log handler took a lock lockdep was already holding on that thread, and her event loop waited behind it for good."""
    import threading

    held = threading.Lock()  # not reentrant: what lockdep's registry lock is

    class _NeedsTheLock(logging.Handler):
        def emit(self, record):
            with held:
                pass

    handler = _NeedsTheLock()
    log = logging.getLogger("Aura.CollectorPauses")
    log.addHandler(handler)
    previous_level = log.level
    log.setLevel(logging.INFO)
    monkeypatch.setattr(collector_pauses, "LOOP_BLOCKING_HOLD_S", -1.0)
    collector_pauses.install_collector_timing()

    def collect_inside_the_section():
        with held:
            gc.collect(2)

    worker = threading.Thread(target=collect_inside_the_section, daemon=True)
    try:
        worker.start()
        worker.join(timeout=10.0)
        assert not worker.is_alive(), "the collecting thread waited on its own lock"
        collector_pauses.say_what_was_measured()
    finally:
        collector_pauses.uninstall_collector_timing()
        log.removeHandler(handler)
        log.setLevel(previous_level)
