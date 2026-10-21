"""Phi_R is positive where the whole carries what neither part carries and not where the parts are apart."""
from __future__ import annotations

import numpy as np

from core.subject.phi_id import phi_r


def _system(coupled: bool, n: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = np.zeros((n + 1, 2))
    b = np.zeros((n + 1, 2))
    for t in range(n):
        if coupled:
            a[t + 1] = 0.6 * b[t] + rng.normal(0, 0.5, 2)
            b[t + 1] = 0.6 * a[t] + rng.normal(0, 0.5, 2)
        else:
            a[t + 1] = 0.6 * a[t] + rng.normal(0, 0.5, 2)
            b[t + 1] = 0.6 * b[t] + rng.normal(0, 0.5, 2)
    return a[:-1], b[:-1], a[1:], b[1:]


def test_two_parts_that_drive_each_other_lose_information_when_cut():
    assert phi_r(*_system(coupled=True))["phi_r"] > 0.1


def test_two_parts_that_never_meet_lose_nothing():
    assert abs(phi_r(*_system(coupled=False))["phi_r"]) < 0.05
