"""The despair surge had no way back down.

`_check_resilience_surges` writes `affect.physiology["adrenaline"]` when it
finds a despair spiral, and nothing else in the tree writes that channel.
Nothing decayed it either, so the first spiral of a run raised
`physiological_strain` and left it raised until the process ended — through
memory salience, the affect signature in her system prompt and every
consolidation record written afterwards.

The value was also off the scale the strain formula read it on. The other
three channels are divided by the distance they travel; adrenaline was taken
raw, so a surge of 5.0 arrived as five times full pressure and held the whole
reading at its ceiling whatever heart rate, conductance and cortisol were
doing.
"""

from __future__ import annotations

import asyncio

import pytest

from core.phases.affect_update import AffectUpdatePhase
from core.state.aura_state import (
    PHYSIOLOGY_PRESSURE_SPAN,
    PHYSIOLOGY_REST,
    AffectVector,
    AuraState,
)

#: What `_check_resilience_surges` writes: half of full mobilization.
SURGE = PHYSIOLOGY_REST["adrenaline"] + (PHYSIOLOGY_PRESSURE_SPAN["adrenaline"] * 0.5)


def _surging() -> AffectVector:
    affect = AffectVector()
    affect.physiology["adrenaline"] = SURGE
    return affect


def _despairing() -> AffectVector:
    affect = AffectVector()
    affect.emotions["sadness"] = 0.95
    affect.emotions["fear"] = 0.9
    affect.emotions["joy"] = 0.0
    return affect


def test_the_surge_returns_to_rest():
    affect = _surging()
    phase = AffectUpdatePhase(None)

    readings = []
    for _ in range(30):
        phase._apply_decay(affect)
        readings.append(affect.physiology["adrenaline"])

    assert readings == sorted(readings, reverse=True)
    assert readings[0] < SURGE
    assert readings[-1] < SURGE * 0.01
    assert readings[-1] >= PHYSIOLOGY_REST["adrenaline"]


def test_it_fades_at_the_rate_the_despair_emotions_do():
    """The half-life is `affect.momentum`, the rate that returns sadness and
    fear to their baselines, rather than a number chosen for adrenaline."""
    for momentum in (0.85, 0.5):
        affect = _surging()
        affect.momentum = momentum
        phase = AffectUpdatePhase(None)
        for cycle in range(1, 6):
            phase._apply_decay(affect)
            assert affect.physiology["adrenaline"] == pytest.approx(SURGE * momentum**cycle)


def test_the_strain_reading_follows_it_down():
    affect = _surging()
    affect.physiology["heart_rate"] = 90.0
    phase = AffectUpdatePhase(None)

    at_rest = AffectVector()
    at_rest.physiology["heart_rate"] = 90.0
    load_alone = at_rest.physiological_strain()

    assert affect.physiological_strain() > load_alone
    for _ in range(60):
        phase._apply_decay(affect)
    assert affect.physiological_strain() == pytest.approx(load_alone, abs=1e-4)


def test_full_mobilization_is_a_fifth_of_the_reading():
    """Its weight in the formula, the same share the other three channels get
    at full pressure."""
    affect = AffectVector()
    affect.physiology["adrenaline"] = (
        PHYSIOLOGY_REST["adrenaline"] + PHYSIOLOGY_PRESSURE_SPAN["adrenaline"]
    )
    assert affect.physiological_strain() == pytest.approx(0.2)

    surging = _surging()
    assert surging.physiological_strain() == pytest.approx(0.1)


def test_a_surge_leaves_the_rest_of_the_body_readable():
    """Taken raw it pinned the reading at 1.0, where a machine running hot and
    a machine at rest were the same number."""
    calm = _surging()
    working = _surging()
    working.physiology["heart_rate"] = 100.0
    working.physiology["cortisol"] = 25.0

    assert calm.physiological_strain() < 1.0
    assert working.physiological_strain() > calm.physiological_strain()


def test_the_spiral_writes_half_of_full_mobilization():
    affect = _despairing()
    AffectUpdatePhase(None)._check_resilience_surges(affect)
    assert affect.physiology["adrenaline"] == pytest.approx(SURGE)


def test_the_phase_brings_a_surge_down():
    """The decay is wired into the cycle, not only into the helper."""
    state = AuraState.default()
    state.cognition.working_memory.append({"role": "user", "content": "hello"})
    state.affect.physiology["adrenaline"] = SURGE
    phase = AffectUpdatePhase(None)

    readings = []
    for _ in range(8):
        asyncio.run(phase.execute(state))
        readings.append(state.affect.physiology["adrenaline"])

    assert readings == sorted(readings, reverse=True)
    assert readings[-1] < SURGE * 0.5
