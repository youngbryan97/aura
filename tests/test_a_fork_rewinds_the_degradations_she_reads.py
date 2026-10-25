"""Every arm of a paired trial was reading the degradations of the arms before it.

Her existential stakes read the degradation tracker as threat: a sixty-second
window, weighted by severity and age, with a discount for a signature that keeps
coming back. The tracker lives in `core.runtime`, which a snapshot never carries,
and a restore rewinds the experiment's clock, so a record written by one arm sat
inside the window of every later arm.

MEASURED 25 October on the seed-23 anchors: an untouched fork rolled after sixty
cuts read a degradation weight of 0 where the same fork rolled before them read
0.0387. The tracker's records now fork with her, as the consequence bus's history
does (tests/test_a_fork_rewinds_the_consequences_she_reads.py).
"""
from __future__ import annotations

import pytest

from core.runtime import errors
from core.subject import snapshot as sn

pytestmark = pytest.mark.unit

KEY = "core.runtime.errors:_RECORDS"


def _record(n: int, *, at: float = 100.0) -> errors.DegradationRecord:
    return errors.DegradationRecord(
        subsystem=f"organ_{n}", severity="degraded", error_type="RuntimeError",
        error_message=f"fault {n}", action="continued", timestamp=at,
    )


@pytest.fixture
def tracker():
    made = errors.get_degradation_tracker()
    made.reset()
    yield made
    made.reset()


def test_the_singleton_keeps_its_records_where_a_fork_can_reach_them(tracker):
    assert tracker._records is errors._RECORDS
    tracker.record(_record(1))
    assert errors._RECORDS[-1].subsystem == "organ_1"


def test_trimming_keeps_the_records_it_shares(tracker):
    small = errors.DegradationTracker(max_records=4, records=errors._RECORDS)
    for n in range(10):
        small.record(_record(n))
    assert small._records is errors._RECORDS, "a rebinding trim leaves what a fork restores behind"
    assert [r.subsystem for r in errors._RECORDS] == [f"organ_{n}" for n in range(6, 10)]


def test_a_tracker_made_on_its_own_keeps_its_own(tracker):
    alone = errors.DegradationTracker()
    alone.record(_record(9))
    assert alone._records is not errors._RECORDS
    assert not any(r.subsystem == "organ_9" for r in errors._RECORDS)


def test_the_snapshot_carries_the_records(tracker):
    assert KEY in sn._module_holdings()


def test_a_restore_rewinds_the_degradations_an_arm_recorded(tracker):
    tracker.record(_record(0))
    saved = sn._module_state({KEY})
    for n in range(1, 30):
        tracker.record(_record(n))
    sn._restore_module_state(saved)
    assert tracker._records is errors._RECORDS, "restored in place, so the live tracker sees it"
    assert [r.subsystem for r in tracker.recent(limit=50)] == ["organ_0"]


def test_her_stakes_read_the_same_threat_after_a_restore(tracker, monkeypatch):
    from core.consciousness import existential_stakes as es

    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(es.time, "time", lambda: clock["now"])
    tracker.record(_record(0, at=clock["now"] - 5.0))
    saved = sn._module_state({KEY})
    first = es.ExistentialStakes()
    first.update()
    before = first._recent_degradation_weight
    # Another arm runs from the same instant and records faults of its own.
    for n in range(1, 8):
        tracker.record(_record(n, at=clock["now"] + 2.0))
    sn._restore_module_state(saved)
    again = es.ExistentialStakes()
    again.update()
    assert before > 0.0
    assert again._recent_degradation_weight == before
