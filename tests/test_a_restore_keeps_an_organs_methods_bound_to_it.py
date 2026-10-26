"""A restore left an organ's table of methods bound to a copy of the organ.

`_organ_state` skipped a callable field but deep-copied a dict of them, and a
deep copy of a bound method copies the object it is bound to. Restoring that
dict put back methods bound to the organ as it stood at the snapshot. MEASURED
25 October: after one restore, all six of affect grounding's conditions were
bound to a frozen copy and read the anchor's buffer, so no arm of any paired
trial could move them.
"""
from __future__ import annotations

from collections import deque

import pytest

from core.subject.snapshot import _organ_state, _restore_organ

pytestmark = pytest.mark.unit


class Organ:
    def __init__(self) -> None:
        self.buf = deque([0.5], maxlen=8)
        self.table = {"mean": self.mean, "last": self.last}

    def mean(self) -> float:
        return sum(self.buf) / len(self.buf)

    def last(self) -> float:
        return self.buf[-1]


def test_a_table_of_methods_is_wiring_and_is_not_captured():
    assert "table" not in _organ_state(Organ())
    assert "buf" in _organ_state(Organ())


def test_after_a_restore_the_methods_still_read_the_live_organ():
    organ = Organ()
    saved = _organ_state(organ)
    organ.buf.append(1.0)
    _restore_organ(organ, saved)
    assert all(method.__self__ is organ for method in organ.table.values())
    organ.buf.append(0.9)
    assert organ.table["last"]() == 0.9


def test_her_affect_grounding_keeps_its_conditions():
    import importlib

    module = None
    for name in ("core.affect.affect_grounding", "core.consciousness.affect_grounding"):
        try:
            module = importlib.import_module(name)
            break
        except ImportError:
            continue
    if module is None or not hasattr(module, "AffectGroundingEngine"):
        pytest.skip("affect grounding engine not importable here")
    engine = module.AffectGroundingEngine()
    _restore_organ(engine, _organ_state(engine))
    assert all(getattr(method, "__self__", None) is engine for method in engine._conditions.values())
