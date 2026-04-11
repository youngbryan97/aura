"""
Tests for IIT 4.0 Exclusion Postulate implementation in phi_core.py.
Validates compute_max_phi_complex() and related methods.
"""

import numpy as np
import pytest


def _make_phi_core_with_history(n_steps=200):
    """Create a PhiCore with enough state history to compute phi."""
    from core.consciousness.phi_core import PhiCore, N_NODES

    pc = PhiCore()
    rng = np.random.RandomState(42)

    # Generate correlated state transitions to produce non-trivial phi
    x = rng.randn(N_NODES).astype(np.float32) * 0.5
    for _ in range(n_steps):
        # Correlated dynamics: each node depends on its neighbors
        noise = rng.randn(N_NODES).astype(np.float32) * 0.2
        x = 0.7 * x + 0.3 * np.roll(x, 1) + noise
        # Record as substrate state (needs at least 8 elements)
        state = np.zeros(16, dtype=np.float32)
        state[:N_NODES] = x
        pc.record_state(state)

    return pc


def test_phi_core_init_exclusion_fields():
    """PhiCore has exclusion postulate tracking fields."""
    from core.consciousness.phi_core import PhiCore

    pc = PhiCore()
    assert pc._max_phi_complex is None
    assert pc._max_phi_value == 0.0
    assert pc._max_phi_complex_names == []


def test_compute_max_phi_complex_returns_result():
    """compute_max_phi_complex returns a (subset, phi) tuple."""
    pc = _make_phi_core_with_history(200)
    result = pc.compute_max_phi_complex()

    assert result is not None, "Should return a result with enough history"
    subset, phi_val = result
    assert isinstance(subset, tuple)
    assert len(subset) >= 2, "Max phi complex must have at least 2 nodes"
    assert phi_val >= 0.0, "Phi must be non-negative"


def test_compute_max_phi_complex_stores_state():
    """After computation, internal state is updated."""
    pc = _make_phi_core_with_history(200)
    pc.compute_max_phi_complex()

    assert pc._max_phi_complex is not None
    assert pc._max_phi_value >= 0.0
    assert len(pc._max_phi_complex_names) == len(pc._max_phi_complex)


def test_max_phi_subset_nodes_valid():
    """The returned subset contains valid node indices."""
    from core.consciousness.phi_core import N_NODES

    pc = _make_phi_core_with_history(200)
    subset, _ = pc.compute_max_phi_complex()

    for idx in subset:
        assert 0 <= idx < N_NODES, f"Invalid node index {idx}"


def test_compute_phi_for_subset_2nodes():
    """_compute_phi_for_subset works for a 2-node subset."""
    pc = _make_phi_core_with_history(200)
    tpm = pc.build_tpm()
    p = pc._get_stationary_distribution()

    phi = pc._compute_phi_for_subset(tpm, p, (0, 1))
    assert phi >= 0.0


def test_compute_phi_for_subset_full():
    """Phi for the full 8-node subset should match compute_phi result closely."""
    from core.consciousness.phi_core import N_NODES

    pc = _make_phi_core_with_history(200)
    full_result = pc.compute_phi()
    assert full_result is not None

    tpm = pc.build_tpm()
    p = pc._get_stationary_distribution()
    full_subset = tuple(range(N_NODES))
    subset_phi = pc._compute_phi_for_subset(tpm, p, full_subset)

    # Should be close to the full phi_s (same computation, different code path)
    assert abs(subset_phi - full_result.phi_s) < 0.01, (
        f"Full subset phi ({subset_phi:.5f}) should match compute_phi ({full_result.phi_s:.5f})"
    )


def test_max_phi_geq_full_phi():
    """The max-phi complex must have phi >= the full system's phi."""
    pc = _make_phi_core_with_history(200)
    full_result = pc.compute_phi()
    assert full_result is not None

    exclusion_result = pc.compute_max_phi_complex()
    assert exclusion_result is not None
    _, max_phi = exclusion_result

    assert max_phi >= full_result.phi_s - 1e-6, (
        f"Max phi ({max_phi:.5f}) must be >= full system phi ({full_result.phi_s:.5f})"
    )


def test_get_status_includes_exclusion():
    """get_status includes exclusion postulate fields after computation."""
    pc = _make_phi_core_with_history(200)
    pc.compute_phi()  # This triggers compute_max_phi_complex internally

    status = pc.get_status()
    assert "exclusion_max_phi" in status
    assert "exclusion_complex_nodes" in status
    assert "exclusion_complex_names" in status
    assert "exclusion_is_full_system" in status
    assert "exclusion_complex_size" in status


def test_get_phi_statement_includes_exclusion():
    """get_phi_statement mentions exclusion postulate after computation."""
    pc = _make_phi_core_with_history(200)
    pc.compute_phi()

    statement = pc.get_phi_statement()
    assert "EXCLUSION" in statement


def test_exclusion_not_in_status_before_compute():
    """Before any computation, exclusion fields are absent from status."""
    from core.consciousness.phi_core import PhiCore

    pc = PhiCore()
    status = pc.get_status()
    assert "exclusion_max_phi" not in status


def test_compute_max_phi_caching():
    """Repeated calls within interval return cached result."""
    pc = _make_phi_core_with_history(200)
    result1 = pc.compute_max_phi_complex()
    assert result1 is not None

    # Second call should hit cache (interval not elapsed)
    result2 = pc.compute_max_phi_complex()
    assert result2 is not None
    assert result1[0] == result2[0]
    assert result1[1] == result2[1]


def test_insufficient_history_returns_none():
    """compute_max_phi_complex returns None with insufficient history."""
    from core.consciousness.phi_core import PhiCore

    pc = PhiCore()
    # Only record 10 states (below MIN_HISTORY_FOR_TPM)
    for i in range(10):
        state = np.ones(16, dtype=np.float32) * i * 0.1
        pc.record_state(state)

    result = pc.compute_max_phi_complex()
    assert result is None


def test_phi_for_subset_bipartition_basic():
    """_phi_for_subset_bipartition produces non-negative results."""
    pc = _make_phi_core_with_history(200)
    tpm = pc.build_tpm()
    p = pc._get_stationary_distribution()

    # Build a 3-node subset TPM
    subset = (0, 1, 2)
    # We'll call the internal method via _compute_phi_for_subset
    phi = pc._compute_phi_for_subset(tpm, p, subset)
    assert phi >= 0.0
