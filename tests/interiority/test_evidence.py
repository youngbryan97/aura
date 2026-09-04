"""Provenance survives arithmetic, and an assumption cannot sound like a fact."""

from __future__ import annotations

from core.interiority.evidence import (
    Provenance,
    absent,
    assumed,
    ceiling_for,
    inferred,
    joint_confidence,
    measured,
    weakest,
)


def test_a_chain_is_as_sound_as_its_worst_link() -> None:
    assert weakest([measured(1.0), assumed(0.5)]) is Provenance.ASSUMED
    assert weakest([measured(1.0), inferred(0.5, 0.9)]) is Provenance.INFERRED
    assert weakest([]) is Provenance.ABSENT


def test_assumed_inputs_cannot_report_the_confidence_of_a_measurement() -> None:
    assert ceiling_for([measured(1.0)]) == 1.0
    assert ceiling_for([measured(1.0), assumed(1.0)]) < 0.3
    assert ceiling_for([absent()]) == 0.0


def test_conjunctions_multiply_rather_than_average() -> None:
    """Four uncertain things are less sure than any one of them.

    Averaging is what lets a system stack weak evidence into a confident
    claim, which is the shape of most confident nonsense.
    """
    readings = [inferred(0.5, 0.8) for _ in range(4)]
    assert joint_confidence(readings) == pytest_approx(0.8**4)


def test_an_absent_reading_has_no_value_and_no_confidence() -> None:
    reading = absent()
    assert reading.value == 0.0
    assert reading.confidence == 0.0
    assert not reading.present


def pytest_approx(value: float) -> float:
    import pytest

    return pytest.approx(value, rel=1e-9)
