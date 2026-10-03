"""A loop that must return to rest is certified to, from its equations alone."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.verify import settling
from core.verify.settling import (
    SettlingCertificate,
    leaky_recurrent_certificate,
    linear_certificate,
)


def test_the_sixteenth_of_september_mood_pair_had_rate_one() -> None:
    """Baseline learning the feeling, feeling decaying to the baseline, no rest: eigenvalue one."""
    m, r = 0.85, 0.001
    without_rest = linear_certificate("before", [[m, 1 - m], [r, 1 - r]], must_settle=True)
    assert math.isclose(without_rest.rate, 1.0, abs_tol=1e-12) and not without_rest.certified
    with_rest = linear_certificate("after", [[m, 1 - m], [r, 1 - 2 * r]], must_settle=True)
    assert with_rest.certified and 0.99 < with_rest.rate < 1.0
    assert 500 < with_rest.half_life_steps < 1000


def test_a_leaky_recurrent_loop_contracts_while_its_row_mass_is_under_the_leak() -> None:
    weights = np.array([[0.2, -0.1], [0.05, 0.3]])
    under = leaky_recurrent_certificate("under", weights, leak=0.5, dt=0.1, must_settle=True)
    assert under.certified and math.isclose(under.rate, 1 - 0.1 * (0.5 - 0.35))
    over = leaky_recurrent_certificate("over", 4 * weights, leak=0.5, dt=0.1, must_settle=False)
    assert not over.certified
    # The last step's movement bounds what is left, and a warm start has a step budget.
    assert math.isclose(under.remaining_distance(0.01), 0.01 * under.rate / (1 - under.rate))
    steps = under.steps_within(1.0, 1e-3)
    assert under.rate**steps <= 1e-3 < under.rate ** (steps - 1)
    with pytest.raises(ValueError, match="not certified"):
        over.steps_within(1.0, 1e-3)


def test_the_certificate_holds_on_the_simulated_loop() -> None:
    rng = np.random.default_rng(3)
    weights = rng.normal(scale=0.05, size=(6, 6))
    certificate = leaky_recurrent_certificate("loop", weights, leak=1.0, dt=0.2, must_settle=True)
    assert certificate.certified
    drive = rng.normal(size=6)
    states = [rng.uniform(-1, 1, size=6), rng.uniform(-1, 1, size=6)]
    for _ in range(40):
        before = max(abs(states[0] - states[1]))
        states = [np.clip((1 - 0.2) * x + 0.2 * np.tanh(weights @ x + drive), -1, 1) for x in states]
        assert max(abs(states[0] - states[1])) <= certificate.rate * before + 1e-12


def test_only_a_loop_that_must_settle_and_does_not_is_a_violation(monkeypatch) -> None:
    monkeypatch.setattr(settling, "_providers", {})
    settling.register_settling("reservoir", lambda: SettlingCertificate("reservoir", 1.3, "row_mass", False))
    settling.register_settling("feeling", lambda: SettlingCertificate("feeling", 1.0, "spectral", True))
    settling.register_settling("calm", lambda: SettlingCertificate("calm", 0.5, "spectral", True))
    settling.register_settling("broken", lambda: 1 / 0)
    violations = list(settling._declared_loops_settle())
    assert sorted(v.subject for v in violations) == ["broken", "feeling"]


def test_the_mood_pair_registers_itself_and_settles_at_either_momentum() -> None:
    from core.phases import affect_update

    certificates = {c.name: c for c in settling.settling_report()}
    assert certificates["affect.mood_baseline"].certified
    previous = affect_update._momentum_seen[0]
    try:
        affect_update._momentum_seen[0] = 0.95
        assert affect_update._mood_settling().certified
    finally:
        affect_update._momentum_seen[0] = previous
