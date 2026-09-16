"""A steady feeling levels off. It used to climb for the life of the process.

`AffectUpdatePhase._apply_decay` moves each emotion toward its mood baseline at
the vector's momentum and moves the baseline toward the emotion at a fixed
rate. As a linear system that pair has an eigenvalue of exactly one, so any
steady input is integrated: on the seed-7 recording fear and frustration rose
and valence fell for 1,200 turns without levelling. The baseline now also
returns toward its declared rest at the learning rate.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from core.phases import affect_update
from core.phases.affect_update import AffectUpdatePhase, bump_emotion
from core.state.aura_state import AffectVector

pytestmark = pytest.mark.unit


def _spectral_radius(momentum: float, learn: float, leak: float) -> float:
    transition = np.array([[momentum, 1.0 - momentum], [learn, 1.0 - learn - leak]])
    return float(np.max(np.abs(np.linalg.eigvals(transition))))


def test_learning_alone_is_a_unit_root_and_the_return_to_rest_removes_it() -> None:
    momentum = AffectVector().momentum
    rate = affect_update._BASELINE_RATE
    assert _spectral_radius(momentum, rate, 0.0) == pytest.approx(1.0, abs=1e-12)
    assert _spectral_radius(momentum, rate, rate) < 1.0


def _fear_after(turns: int) -> tuple[float, float]:
    random.seed(7)
    affect = AffectVector()
    phase = object.__new__(AffectUpdatePhase)
    for _ in range(turns):
        bump_emotion(affect.emotions, "fear", 0.02)
        AffectUpdatePhase._apply_decay(phase, affect)
    return float(affect.emotions["fear"]), float(affect.mood_baselines["fear"])


def test_a_steady_push_on_a_feeling_levels_off() -> None:
    fear_4000, baseline_4000 = _fear_after(4000)
    fear_6000, baseline_6000 = _fear_after(6000)
    assert abs(baseline_6000 - baseline_4000) < 0.005
    assert abs(fear_6000 - fear_4000) < 0.02
    assert fear_6000 < 0.5


def test_with_nothing_pressing_a_baseline_stays_at_rest() -> None:
    random.seed(11)
    affect = AffectVector()
    phase = object.__new__(AffectUpdatePhase)
    rest = dict(affect.mood_baselines)
    for _ in range(2000):
        for name in list(affect.emotions):
            affect.emotions[name] = rest.get(name, 0.05)
        AffectUpdatePhase._apply_decay(phase, affect)
    drift = max(abs(affect.mood_baselines[name] - rest[name]) for name in rest if name in affect.mood_baselines)
    assert drift < 0.01
