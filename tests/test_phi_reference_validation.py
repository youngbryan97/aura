from __future__ import annotations

import numpy as np

from core.consciousness.phi_core import PhiCore


def _row_normalize(tpm: np.ndarray) -> np.ndarray:
    row_sums = tpm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return tpm / row_sums


def test_phi_reference_independent_bit_memory_is_zero():
    """Two independent one-bit memories are decomposable across the cut."""
    pc = PhiCore()
    tpm = np.eye(4, dtype=np.float64)
    p = np.full(4, 0.25, dtype=np.float64)

    phi = pc._phi_for_subset_bipartition(tpm, p, (0,), (1,), 2)

    assert phi < 1e-8


def test_phi_reference_constant_sink_is_zero():
    """A degenerate constant sink must not be mistaken for integration."""
    pc = PhiCore()
    tpm = np.zeros((4, 4), dtype=np.float64)
    tpm[:, 0] = 1.0
    p = np.full(4, 0.25, dtype=np.float64)

    phi = pc._phi_for_subset_bipartition(tpm, p, (0,), (1,), 2)

    assert phi < 1e-8


def test_phi_reference_coupled_xor_dynamics_positive():
    """A coupled system where each next bit depends on the joint state is positive."""
    pc = PhiCore()
    tpm = np.full((4, 4), 1e-4, dtype=np.float64)
    transitions = {
        0b00: 0b00,
        0b01: 0b11,
        0b10: 0b11,
        0b11: 0b00,
    }
    for src, dst in transitions.items():
        tpm[src, dst] += 1.0
    tpm = _row_normalize(tpm)
    p = np.full(4, 0.25, dtype=np.float64)

    phi = pc._phi_for_subset_bipartition(tpm, p, (0,), (1,), 2)

    assert phi > 0.1
