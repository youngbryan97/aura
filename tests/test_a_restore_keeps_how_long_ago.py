"""A restore hands back each reading of `time.monotonic` at the age it had.

The machine's monotonic clock runs on while a snapshot waits, and a restore
cannot rewind it. Before this, a stamp restored an hour later came back an hour
older: seed 71's untouched anchors, rolled an hour into the process, read idle
and cooldown timers she never lived and parted from their own first rolls at
three anchors of eight (26 October).
"""

from __future__ import annotations

import copy
import pickle
import sys
import time
import types
from collections import deque, namedtuple
from types import SimpleNamespace

import numpy as np
import pytest

from core.subject import monotonic_instants
from core.subject.copies import identical
from core.subject.monotonic_instants import (
    HORIZON_S,
    MonotonicInstant,
    rebased,
    tag_instants,
    tag_snapshot,
)
from core.subject.snapshot import _module_state, _place, _restore_module_state

pytestmark = pytest.mark.unit

PROBE = "core._monotonic_probe_holdings"
Mark = namedtuple("Mark", "kind at")


@pytest.fixture(autouse=True)
def _a_process_ten_minutes_old(monkeypatch: pytest.MonkeyPatch) -> None:
    # A pytest process is seconds old, and the probes stamp a minute back.
    monkeypatch.setattr(monotonic_instants, "PROCESS_START", time.monotonic() - 600.0)


def _window() -> tuple[float, float]:
    return monotonic_instants.PROCESS_START, time.monotonic() + HORIZON_S


def test_only_readings_of_this_process_clock_are_tagged() -> None:
    now = time.monotonic()
    held = {
        "stamp": now - 2.0,
        "deadline": now + 30.0,
        "duration": 0.3,
        "count": 7,
        "epoch": time.time(),
        "flag": True,
        "array": np.array([now, now]),
        "before_the_process": monotonic_instants.PROCESS_START - 100.0,
    }
    low, high = _window()
    tag_instants(held, low, high, set())
    assert type(held["stamp"]) is MonotonicInstant
    assert type(held["deadline"]) is MonotonicInstant
    for name in ("duration", "count", "epoch", "flag", "before_the_process"):
        assert type(held[name]) is not MonotonicInstant, name
    assert held["array"].dtype == np.float64


def _probe_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType(PROBE)

    class Engine:
        def __init__(self) -> None:
            now = time.monotonic()
            self.last_activity = now - 5.0
            self.recent = deque([now - 3.0, now - 1.0], maxlen=8)
            self.cooldowns = {"walk": now - 60.0}
            self.mark = Mark("seen", now - 2.0)
            self.decay = 0.25

    Engine.__module__ = PROBE
    module.Engine = Engine
    module._engine = Engine()
    monkeypatch.setitem(sys.modules, PROBE, module)
    return module


def test_a_restore_an_hour_later_keeps_every_age(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _probe_module(monkeypatch)
    engine = module._engine
    taken_at = time.monotonic()
    ages = {
        "last_activity": taken_at - engine.last_activity,
        "recent": [taken_at - t for t in engine.recent],
        "walk": taken_at - engine.cooldowns["walk"],
        "mark": taken_at - engine.mark.at,
    }
    snapshot = SimpleNamespace(module_state=_module_state({f"{PROBE}:_engine"}))
    tag_snapshot(snapshot, taken_at)

    real = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real() + 3600.0)
    engine.last_activity = time.monotonic()
    engine.recent.append(time.monotonic())
    engine.decay = 0.9

    with rebased(taken_at):
        now = time.monotonic()
        _restore_module_state(snapshot.module_state)

    assert module._engine is engine
    assert now - engine.last_activity == pytest.approx(ages["last_activity"], abs=0.05)
    assert [now - t for t in engine.recent] == pytest.approx(ages["recent"], abs=0.05)
    assert now - engine.cooldowns["walk"] == pytest.approx(ages["walk"], abs=0.05)
    assert now - engine.mark.at == pytest.approx(ages["mark"], abs=0.05)
    assert engine.decay == 0.25
    assert type(engine.last_activity) is float
    assert all(type(t) is float for t in engine.recent)


def test_without_a_reading_of_the_clock_a_restore_is_what_it_was(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _probe_module(monkeypatch)
    engine = module._engine
    before = engine.last_activity
    snapshot = SimpleNamespace(module_state=_module_state({f"{PROBE}:_engine"}))
    tag_snapshot(snapshot, time.monotonic())
    engine.last_activity = 1.0
    with rebased(None):
        _restore_module_state(snapshot.module_state)
    assert engine.last_activity == before


def test_the_live_object_is_never_tagged(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _probe_module(monkeypatch)
    snapshot = SimpleNamespace(module_state=_module_state({f"{PROBE}:_engine"}))
    tag_snapshot(snapshot, time.monotonic())
    live, copied = snapshot.module_state[f"{PROBE}:_engine"]
    assert live is module._engine
    assert type(live.last_activity) is float
    assert type(copied["last_activity"]) is MonotonicInstant


def test_a_tagged_copy_is_still_the_same_copy() -> None:
    now = time.monotonic()
    low, high = _window()
    tagged = tag_instants({"at": now, "trail": [now - 1.0]}, low, high, set())
    assert identical(tagged, {"at": now, "trail": [now - 1.0]})
    assert not identical(tagged, {"at": now + 1e-6, "trail": [now - 1.0]})


def test_a_tag_survives_pickling_and_copying_outside_a_restore() -> None:
    instant = MonotonicInstant(time.monotonic())
    assert type(pickle.loads(pickle.dumps(instant))) is MonotonicInstant
    assert copy.deepcopy([instant])[0] is instant
    assert _place(instant) == float(instant)
    assert type(_place(instant)) is float


def test_a_restore_moves_a_tagged_value_however_it_is_copied(monkeypatch: pytest.MonkeyPatch) -> None:
    taken_at = time.monotonic()
    instant = MonotonicInstant(taken_at - 4.0)
    monkeypatch.setattr(monotonic_instants.time, "monotonic", lambda: taken_at + 100.0)
    with rebased(taken_at):
        assert _place(instant) == pytest.approx(taken_at + 96.0)
        assert copy.deepcopy({"x": (instant, 1)})["x"][0] == pytest.approx(taken_at + 96.0)
        assert _place((instant, "a")) [0] == pytest.approx(taken_at + 96.0)
    assert monotonic_instants._SHIFT is None


def test_the_event_graph_travels_to_another_process() -> None:
    from core.cognition.cognitive_event import EventGraph
    from core.subject.anchor_travel import LiveFunction, dumps, loads

    graph = loads(dumps(EventGraph()))
    assert not isinstance(graph._clock, LiveFunction)
    assert isinstance(graph._clock(), float)


def test_a_handle_inside_a_copy_is_left_alone() -> None:
    from core.runtime.descriptor_owner import OwnsDescriptors

    class Handle(OwnsDescriptors):
        __slots__ = ("opened_at", "__dict__")

    handle = Handle()
    handle.opened_at = time.monotonic()
    handle.last_used = time.monotonic()
    snapshot = SimpleNamespace(services={"store": {"handle": copy.deepcopy(handle)}})
    assert snapshot.services["store"]["handle"] is handle
    tag_snapshot(snapshot, time.monotonic())
    assert type(handle.last_used) is float
