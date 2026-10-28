"""The units planted outside K* to validate ISC-v7 behave as their answers say.

In a live organism (27 October, seeds 101 and 103), the interventional test read
the planted cause as moving K at 12 of 12 anchor-donor pairs, the AND gate at 6
of 12 swapped one flag at a time and 12 of 12 swapped all at once, and the record
at none; every unit outside K* swapped at once, with only the record planted,
moved nothing at 12 of 12.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.subject import planted

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean():
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()


def _her(curiosity: float = 0.3) -> SimpleNamespace:
    return SimpleNamespace(affect=SimpleNamespace(emotions={"curiosity": curiosity}))


def test_the_cause_moves_her_curiosity_by_its_level() -> None:
    cause = planted.PlantedCause()
    cause.level, cause.count = 0.5, 10
    ServiceContainer.register_instance("planted_cause", cause, required=False)
    her = _her()
    planted.act(her)
    assert her.affect.emotions["curiosity"] == pytest.approx(0.3 + planted.CAUSE_GAIN * cause.level)


def test_the_gate_acts_only_with_both_flags_on() -> None:
    a, b = planted.PlantedGate(), planted.PlantedGate()
    ServiceContainer.register_instance("planted_gate_a", a, required=False)
    ServiceContainer.register_instance("planted_gate_b", b, required=False)
    her = _her()
    planted.act(her)
    assert her.affect.emotions["curiosity"] == pytest.approx(0.3 + planted.GATE_GAIN)
    b.on = 0
    b.count = 1
    her = _her()
    planted.act(her)
    assert her.affect.emotions["curiosity"] == pytest.approx(0.3)


def test_the_record_copies_her_and_changes_nothing() -> None:
    record = planted.PlantedRecord()
    ServiceContainer.register_instance("planted_record", record, required=False)
    her = _her(0.42)
    planted.act(her)
    assert record.value == pytest.approx(0.42)
    assert her.affect.emotions["curiosity"] == pytest.approx(0.42)


def test_a_planted_unit_keeps_its_own_clock_so_a_swap_carries_over() -> None:
    gate = planted.PlantedGate()
    for _ in range(planted.FLIP_EVERY):
        gate.step()
    assert gate.on == 0 and gate.count == planted.FLIP_EVERY
