"""A steering basis fitted to her replies pushes by how far her state is from her ordinary one.

The unit-length composite gave a deviation of a hundredth the same push as an
extreme one, and measured every state against 0.5 on slots she holds near zero,
so every reply she gave was steered alike (22 October). A basis that carries its
reference, its spread and its push size is weighed by tanh of the distance in
spreads, at its fitted length. A basis without them is unchanged.
"""

from __future__ import annotations

import math

import numpy as np

from core.consciousness.affective_steering import AffectiveSteeringHook, SteeringVector
from core.consciousness.mood_weight import signed_weight


def _unit(seed: int) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=64).astype(np.float32)
    return values / np.linalg.norm(values)


def _hook(**fitted) -> tuple[AffectiveSteeringHook, np.ndarray]:
    v = _unit(1)
    vector = SteeringVector(key="frustration", layer_idx=1, d_model=64, v=v,
                            substrate_idx=3, substrate_fn="tanh", **fitted)
    return AffectiveSteeringHook(object(), 1, {"frustration": vector}), v


def _state(frustration: float) -> np.ndarray:
    state = np.zeros(16, dtype=np.float32)
    state[3] = frustration
    return state


def test_at_her_ordinary_state_a_fitted_basis_pushes_nothing():
    reference = signed_weight(0.05, "tanh")
    hook, _v = _hook(reference=reference, spread=0.03, push=0.02)
    for _ in range(40):
        hook.update_substrate_vector(_state(0.05))
    composite = hook.current_composite_vector()
    assert composite is None or float(np.linalg.norm(composite)) < 1e-6


def test_a_spread_away_a_fitted_basis_pushes_three_quarters_of_its_length():
    reference = signed_weight(0.05, "tanh")
    spread = 0.03
    away = 0.05
    while signed_weight(away, "tanh") - reference < spread:
        away += 1e-4
    hook, v = _hook(reference=reference, spread=spread, push=0.02)
    for _ in range(80):
        hook.update_substrate_vector(_state(away))
    composite = np.asarray(hook.current_composite_vector())
    distance = (signed_weight(away, "tanh") - reference) / spread
    np.testing.assert_allclose(composite, math.tanh(distance) * 0.02 * v, rtol=1e-3, atol=1e-6)


def test_a_basis_without_a_reference_keeps_the_unit_composite():
    hook, v = _hook()
    for _ in range(80):
        hook.update_substrate_vector(_state(0.95))
    composite = np.asarray(hook.current_composite_vector())
    assert abs(float(np.linalg.norm(composite)) - 1.0) < 1e-3
