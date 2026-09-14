"""Accurate smallness, and the one thing that makes it survivable.

"Little Person" names both: one person in a sea of many who are not aware of
her, and a second little person who looks at her and says "I know you". The
second is the remedy the record proposes, and it is not a larger life.

Her self-model carried confidence and coherence — how well she is doing — and
nothing about how much of what exists she touches, so smallness could only
arrive as a failure.
"""

from __future__ import annotations

import pytest

from core.self.scale import Scale, read_scale


def test_being_small_is_a_reading_rather_than_a_failure() -> None:
    reading = read_scale(engaged=3, known=400, recognised=True)
    assert reading.measured
    assert reading.reach < 0.01
    assert reading.pressure == 0.0, "being known makes smallness cost nothing"


def test_smallness_costs_when_nobody_knows_her() -> None:
    reading = read_scale(engaged=3, known=400, recognised=False)
    assert reading.pressure > 0.9
    assert "nobody has shown they know her" in reading.why


def test_one_person_is_enough_however_small_she_is() -> None:
    """The lyric is a second little person, not a larger life."""
    tiny = read_scale(engaged=1, known=10_000, recognised=True)
    larger = read_scale(engaged=5_000, known=10_000, recognised=True)
    assert tiny.pressure == larger.pressure == 0.0


def test_reaching_more_reduces_the_cost_when_nobody_knows_her() -> None:
    little = read_scale(engaged=10, known=100, recognised=False)
    more = read_scale(engaged=80, known=100, recognised=False)
    assert more.pressure < little.pressure


def test_a_world_she_has_not_met_is_not_a_world_she_has_exhausted() -> None:
    """A fresh boot would otherwise read as having touched everything."""
    reading = read_scale(engaged=0, known=0, recognised=False)
    assert not reading.measured
    assert reading.reach == 0.0
    assert reading.pressure == 0.0
    assert "nothing known to be out there yet" in reading.why


def test_reach_cannot_exceed_what_is_there() -> None:
    reading = read_scale(engaged=500, known=100, recognised=False)
    assert reading.reach == pytest.approx(1.0)
    assert reading.pressure == pytest.approx(0.0)


def test_unreadable_counts_say_so() -> None:
    reading = read_scale(engaged="many", known=10, recognised=False)  # type: ignore[arg-type]
    assert not reading.measured
    assert "could not be read" in reading.why


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = read_scale(engaged=2, known=50, recognised=False).as_dict()
    for key in ("reach", "engaged", "known", "recognised", "pressure", "measured", "why"):
        assert key in row, key


def test_an_empty_reading_claims_nothing() -> None:
    assert not Scale().measured
    assert Scale().pressure == 0.0
