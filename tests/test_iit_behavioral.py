"""Behavioral tests for the IIT Surrogate (RIIU).

Tests verify that the surrogate integrated information (Φ) scales with
system integration and correlation, respects warmup periods, and
uses balanced partitions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from core.consciousness.iit_surrogate import RIIU, _MIN_SAMPLES


@pytest.fixture
def riiu():
    """Fresh RIIU instance."""
    return RIIU(neuron_count=16, buffer_size=32)


class TestRIIUWarmup:
    """RIIU must not compute Φ until it has enough samples."""

    def test_warmup_period(self, riiu):
        """Φ should be 0.0 and warmup should be True during initial samples."""
        for _ in range(_MIN_SAMPLES - 1):
            state = np.random.rand(16)
            phi = riiu.compute_phi(state)
            assert phi == 0.0
            assert riiu.get_stats()["warmup"] is True

        # At MIN_SAMPLES, it might not compute yet due to the tick % 5 condition
        # but warmup should be False
        state = np.random.rand(16)
        riiu.compute_phi(state)
        assert riiu.get_stats()["warmup"] is False


class TestTickAmortization:
    """Full Φ calculation should only run periodically."""

    def test_amortized_computation(self, riiu):
        """Φ should update every 5 ticks."""
        # Fill buffer past warmup
        for _ in range(_MIN_SAMPLES):
            riiu.compute_phi(np.random.rand(16))
            
        initial_phi = riiu.get_phi()
        
        updates = 0
        for _ in range(10):
            # Pass distinct state to force a new phi value if computed
            state = np.random.rand(16) * 10
            new_phi = riiu.compute_phi(state)
            if new_phi != initial_phi:
                updates += 1
                initial_phi = new_phi
                
        # In 10 ticks, we should see exactly 2 updates (tick % 5 == 0)
        assert updates == 2, f"Expected 2 updates in 10 ticks, got {updates}"


class TestIntegrationMeasure:
    """Φ must measure integration (correlation across partitions)."""

    def test_correlated_system_has_higher_phi(self):
        """A system with strong global correlations should have higher Φ
        than a system of independent noise."""
        riiu_noise = RIIU(neuron_count=16, buffer_size=32)
        riiu_integrated = RIIU(neuron_count=16, buffer_size=32)
        
        # 1. Independent noise
        rng = np.random.RandomState(42)
        for _ in range(32):
            # Align with 5 ticks to force compute
            for _ in range(5):
                riiu_noise.compute_phi(rng.randn(16))
            
        phi_noise = riiu_noise.get_phi()
        
        # 2. Integrated system (strong global mode)
        for t in range(32):
            for _ in range(5):
                # Base signal
                signal = np.sin(t * 0.1)
                # Add signal to all neurons + small noise
                state = np.ones(16) * signal + rng.randn(16) * 0.01
                riiu_integrated.compute_phi(state)
            
        phi_integrated = riiu_integrated.get_phi()
        
        # Phi should be significantly higher for the integrated system
        # because the whole system covariance matrix has high logdet 
        # relative to the partitions (which lose the cross-partition correlations)
        # Note: Depending on the surrogate exact math, perfect correlation might drop logdet.
        # But generally, structured correlation increases phi compared to pure noise.
        assert phi_integrated > phi_noise, (
            f"Integrated Φ ({phi_integrated:.4f}) should be higher than "
            f"noise Φ ({phi_noise:.4f})"
        )


class TestPartitionBalance:
    """Partitions should be strictly balanced."""

    def test_partitions_are_balanced(self, riiu):
        """All bipartitions must be near the midpoint."""
        total = riiu.total_dim
        for part_a, part_b in riiu._partitions:
            assert len(part_a) + len(part_b) == total
            # The split is constrained to [total_dim//2 - 4, total_dim//2 + 5)
            # which is an 8-element range around the middle.
            mid = total // 2
            assert abs(len(part_a) - mid) <= 4, (
                f"Partition unbalanced: {len(part_a)} vs {len(part_b)}"
            )


class TestDegenerateStates:
    """RIIU must not crash on degenerate states."""

    def test_all_zeros_does_not_crash(self, riiu):
        """Feeding all zeros should yield Φ = 0.0 safely."""
        for _ in range(32):
            phi = riiu.compute_phi(np.zeros(16))
            
        assert np.isfinite(phi)
        assert phi < 1e-6

    def test_nan_does_not_crash(self, riiu):
        """Feeding NaNs should be handled or result in 0.0 without crashing."""
        state = np.ones(16)
        state[0] = np.nan
        for _ in range(10):
            phi = riiu.compute_phi(state)
            
        assert np.isfinite(phi)
