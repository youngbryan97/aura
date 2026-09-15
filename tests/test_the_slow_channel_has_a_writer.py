"""A physiology channel with a rest value, a span, a reader and no writer.

`PHYSIOLOGY_REST` declares where cortisol sits when nothing is happening,
`PHYSIOLOGY_PRESSURE_SPAN` declares how far above rest it travels for its
pressure to count as full, `physiological_strain` reads it, and the subject
schema has a column for it. Nothing in the tree ever wrote it. Through the
six-hour run_032 the column read 0.3333 with a standard deviation of zero while
heart rate, conductance and adrenaline moved throughout.

Adrenaline is the acute channel and momentum returns it. Cortisol is the same
signal integrated: what raises it is how much of the recent past was spent
mobilised, on the square of adrenaline's own clock.
"""

from __future__ import annotations

from core.phases.affect_update import AffectUpdatePhase
from core.state.aura_state import (
    PHYSIOLOGY_PRESSURE_SPAN,
    PHYSIOLOGY_REST,
    AffectVector,
)

settle = AffectUpdatePhase._settle_cortisol
REST = PHYSIOLOGY_REST["cortisol"]
SPAN = PHYSIOLOGY_PRESSURE_SPAN["cortisol"]
ACUTE_REST = PHYSIOLOGY_REST["adrenaline"]
ACUTE_SPAN = PHYSIOLOGY_PRESSURE_SPAN["adrenaline"]


def _affect(adrenaline: float, cortisol: float = REST) -> AffectVector:
    affect = AffectVector()
    affect.physiology = dict(PHYSIOLOGY_REST)
    affect.physiology["adrenaline"] = adrenaline
    affect.physiology["cortisol"] = cortisol
    return affect


def test_rest_stays_at_rest() -> None:
    affect = _affect(ACUTE_REST)
    settle(affect)
    assert affect.physiology["cortisol"] == REST


def test_a_single_surge_barely_moves_it() -> None:
    """One event is adrenaline's business, not the slow channel's."""
    affect = _affect(ACUTE_REST + ACUTE_SPAN)
    settle(affect)
    risen = affect.physiology["cortisol"] - REST
    assert 0.0 < risen < SPAN * 0.5


def test_sustained_mobilisation_arrives_at_the_ceiling() -> None:
    affect = _affect(ACUTE_REST + ACUTE_SPAN)
    for _ in range(500):
        affect.physiology["adrenaline"] = ACUTE_REST + ACUTE_SPAN
        settle(affect)
    assert affect.physiology["cortisol"] > REST + SPAN * 0.99


def test_it_comes_back_down_when_the_pressure_stops() -> None:
    affect = _affect(ACUTE_REST + ACUTE_SPAN, cortisol=REST + SPAN)
    affect.physiology["adrenaline"] = ACUTE_REST
    for _ in range(600):
        settle(affect)
    assert affect.physiology["cortisol"] < REST + SPAN * 0.01


def test_the_slow_channel_is_slower_than_the_fast_one() -> None:
    """Two speeds of one mechanism, not two mechanisms."""
    affect = _affect(ACUTE_REST + ACUTE_SPAN)
    fast = ACUTE_REST + ACUTE_SPAN
    steps_to_half_slow = 0
    while affect.physiology["cortisol"] < REST + SPAN * 0.5:
        settle(affect)
        steps_to_half_slow += 1
        assert steps_to_half_slow < 10_000

    steps_to_half_fast = 0
    value = fast
    while value - ACUTE_REST > ACUTE_SPAN * 0.5:
        value = (value * affect.momentum) + (ACUTE_REST * (1 - affect.momentum))
        steps_to_half_fast += 1
    assert steps_to_half_slow > steps_to_half_fast


def test_it_never_leaves_its_declared_range() -> None:
    affect = _affect(ACUTE_REST + ACUTE_SPAN * 5)
    for _ in range(50):
        settle(affect)
        assert REST <= affect.physiology["cortisol"] <= REST + SPAN


def test_the_decay_pass_writes_it() -> None:
    """Not only the helper: the phase's own decay has to reach it."""
    phase = AffectUpdatePhase.__new__(AffectUpdatePhase)
    affect = _affect(ACUTE_REST + ACUTE_SPAN)
    before = affect.physiology["cortisol"]
    AffectUpdatePhase._apply_decay(phase, affect)
    assert affect.physiology["cortisol"] > before
