"""What an arm left behind that a restore did not put back, one cause at a time.

Measured on 22 September with a probe that takes a snapshot, runs a memory arm,
restores and takes a second snapshot, then compares every captured field
exactly: 123 fields differed on 44f56ca0b. Most were the comparison failing to
read a generator, a slotted object or a bound method, which also cost a copy on
every snapshot. The rest were real, and each has a check here:

- the effort ledger's lifetime grew by what was pending, on every restore;
- the being runtime's self field stepped itself on the wall clock, on a thread;
- the ontogeny core retrained and swept outcomes on its own timers;
- the affect engine's last receipt holds a read-only mapping, which would not
  copy, so the frozen receipt was taken apart and could not be written back;
- the scientific engine lost its table when a restore removed the database an
  arm had made, and the failure it then recorded reached the substrate on the
  next arm's last frame: 0.025 in C's valence columns, the order test's failure.
"""

from __future__ import annotations

import copy
import sqlite3
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from core.subject.snapshot import identical

pytestmark = pytest.mark.unit


def test_restoring_the_effort_ledger_sets_both_totals_and_counts_nothing_twice() -> None:
    from core.soma.effort import EffortLedger

    ledger = EffortLedger()
    ledger.note("recall", 3.0)
    pending, lifetime = ledger.peek(), ledger.lifetime()
    ledger.note("recall", 5.0)
    ledger.restore(pending, lifetime)
    ledger.restore(pending, lifetime)
    assert ledger.peek() == {"recall": 3.0}
    assert ledger.lifetime() == {"recall": 3.0}


def test_the_fork_carries_the_effort_lifetime(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.soma import effort
    from core.subject.snapshot import _effort_state, _restore_effort

    ledger = effort.EffortLedger()
    monkeypatch.setattr(effort, "get_effort_ledger", lambda: ledger)
    ledger.note("phases", 2.0)
    saved = _effort_state()
    ledger.note("phases", 7.0)
    _restore_effort(saved)
    assert ledger.lifetime() == {"phases": 2.0}
    assert ledger.peek() == {"phases": 2.0}


def test_the_self_field_integrates_over_the_clock_it_is_given() -> None:
    from core.being.continuous_substrate import ContinuousSelfField

    now = [100.0]
    runs = []
    for _ in range(2):
        field = ContinuousSelfField(dim=8)
        field.use_clock(lambda: now[0])
        now[0] += 0.5
        runs.append(field.step({"body_pressure": 0.4}, (0.2,)).state)
        now[0] = 100.0
    assert runs[0] == runs[1]


def test_switching_the_clock_keeps_the_fields_age() -> None:
    from core.being.continuous_substrate import ContinuousSelfField

    field = ContinuousSelfField(dim=8)
    field._created -= 7.0
    field.use_clock(lambda: 1000.0)
    assert field.get_phenomenal_packet()["elapsed_since_start_s"] == pytest.approx(7.0, abs=0.05)


def test_the_wind_down_stops_the_self_field_and_the_ontogeny_timers(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.being.continuous_substrate import ContinuousSelfField
    from core.ontogeny import service
    from core.subject import organism

    field = ContinuousSelfField(dim=8)
    field.start(hz=50.0)
    runtime = types.SimpleNamespace(field=field)
    monkeypatch.setattr(
        organism,
        "optional_service",
        lambda name, default=None: runtime if name == "being_runtime" else default,
    )
    stopped_event = threading.Event()
    swept = []
    core = types.SimpleNamespace(_stopped=stopped_event, _sweeper=types.SimpleNamespace(stop=lambda: swept.append(1)))
    monkeypatch.setattr(service, "_core", core)
    names = organism._stop_the_self_field() + organism._stop_ontogeny_loops()
    assert names == ["being_runtime.self_field", "ontogeny.maintenance", "ontogeny.sweeper"]
    assert not field._running and field._thread is None
    assert stopped_event.is_set() and swept == [1]


def test_a_frozen_receipt_holding_a_read_only_mapping_copies_whole() -> None:
    from core.affect.damasio_v2 import AffectStimulusReceipt

    receipt = AffectStimulusReceipt(
        event_id="e1", source="s", evidence_status="ok", applied=True, duplicate=False,
        intensity=0.3, appraisal=types.MappingProxyType({"v": 0.1}), timestamp=1.0,
    )
    copied = copy.deepcopy(receipt)
    assert copied is not receipt
    assert isinstance(copied.appraisal, types.MappingProxyType)
    assert identical(receipt, copied)


def test_the_scientific_engine_makes_its_table_again_when_the_file_is_gone(tmp_path: Path) -> None:
    from core.cognition.scientific_engine import ScientificEngine

    db = tmp_path / "scientific_engine.db"
    engine = ScientificEngine(db_path=str(db))
    db.unlink()
    hypothesis_id = engine.form_hypothesis("x rises", predicted_observable="x", expected=1.0)
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT hypothesis_id FROM hypotheses").fetchall()
    assert [r[0] for r in rows] == [hypothesis_id]


class _Slotted:
    __slots__ = ("value", "items")

    def __init__(self, value: float) -> None:
        self.value = value
        self.items = [value]


class _Holder:
    def __init__(self) -> None:
        self.table = {"a": 1}

    def read(self) -> int:
        return 1


@pytest.mark.parametrize(
    "value",
    [
        np.random.default_rng(5),
        np.random.RandomState(5),
        _Slotted(0.5),
        _Holder().read,
        {"a": 1}.items,
        types.MappingProxyType({"k": [1, 2]}),
        types.SimpleNamespace(a=1, b=[2]),
    ],
)
def test_the_comparison_reads_what_organs_hold(value) -> None:
    assert identical(value, copy.deepcopy(value))


def test_a_condition_is_a_handle_with_nothing_to_restore() -> None:
    """It will not deep-copy at all; two of them hold nothing a restore puts back."""
    assert identical(threading.Condition(), threading.Condition())


def test_a_generator_that_drew_is_not_the_one_that_did_not() -> None:
    first = np.random.default_rng(5)
    second = copy.deepcopy(first)
    second.random()
    assert not identical(first, second)


def test_a_slotted_object_that_changed_is_not_the_same() -> None:
    first = _Slotted(0.5)
    second = copy.deepcopy(first)
    second.items.append(2.0)
    assert not identical(first, second)
