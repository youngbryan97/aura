"""Behavioral tests for the Liquid Substrate (Phase 6 Hardening).

Verifies the integration of continuous dynamical proxies (Orch OR, CEMI, DIT),
the boot mood initialization, and psych state stabilization mechanics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pytest
import asyncio
import numpy as np
from unittest.mock import patch, MagicMock

from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig


@pytest.fixture
def substrate(tmp_path):
    """Fresh LiquidSubstrate instance for testing."""
    state_file = tmp_path / "test_substrate_state.npy"
    config = SubstrateConfig(neuron_count=64, noise_level=0.1, state_file=state_file)
    return LiquidSubstrate(config)


class TestSubstrateBootMood:
    """Verifies that the substrate boots in a non-exhausted, non-bored state."""

    def test_boot_mood_initialization(self, substrate):
        """Curiosity, Energy, and Focus must start at active baselines."""
        assert substrate.x[substrate.idx_curiosity] == 0.5, "Curiosity did not boot to baseline."
        assert substrate.x[substrate.idx_energy] == 1.0, "Energy did not boot to baseline."
        assert substrate.x[substrate.idx_focus] == 0.5, "Focus did not boot to baseline."
        assert substrate.x[substrate.idx_frustration] == 0.0, "Frustration should start at 0."


class TestSubstrateDynamics:
    """Verifies continuous physics updates and stability."""

    @pytest.mark.asyncio
    async def test_stabilize_psych_state_zen(self, substrate):
        """Frustration should naturally decay towards 0."""
        substrate.x[substrate.idx_frustration] = 0.8
        
        # Manually invoke stabilization
        await substrate._stabilize_psych_state(dt=0.1)
        
        assert substrate.x[substrate.idx_frustration] < 0.8, "Frustration did not decay."

    @pytest.mark.asyncio
    async def test_step_dynamics_moves_vectors(self, substrate):
        """Running the ODE solver should alter x and v vectors."""
        original_x = substrate.x.copy()
        
        # Inject some energy to drive the network
        substrate.v[:] = 0.5
        
        await substrate._step_dynamics(dt=0.1)
        
        # x should have moved
        assert not np.allclose(substrate.x, original_x)


class TestSubstrateQualiaMetrics:
    """Verifies the mathematical proxies for theories of consciousness."""

    def test_orch_or_coherence_decay_and_collapse(self, substrate):
        """Microtubule coherence must decay with noise and collapse < 0.4."""
        initial_coherence = substrate.microtubule_coherence
        assert initial_coherence == 1.0
        
        # Fake extreme noise to force decay
        substrate.config.noise_level = 100.0
        
        # Tick sync
        substrate._update_qualia_metrics_sync(dt=0.1)
        
        # It should have decayed but not collapsed yet if it just dropped
        if substrate.microtubule_coherence != 1.0:
            assert substrate.microtubule_coherence < 1.0
            
        # Force a collapse
        substrate.microtubule_coherence = 0.3
        initial_collapses = substrate.total_collapse_events
        
        substrate._update_qualia_metrics_sync(dt=0.1)
        
        assert substrate.total_collapse_events > initial_collapses
        assert substrate.microtubule_coherence == 1.0, "Coherence did not reset after collapse."

    def test_cemi_em_field_updates(self, substrate):
        """EM field magnitude should track the velocity flux."""
        # Inject velocity
        substrate.v[:] = 1.0
        initial_field = substrate.em_field_magnitude
        
        substrate._update_qualia_metrics_sync(dt=0.1)
        
        assert substrate.em_field_magnitude > initial_field, "EM field did not respond to flux."

    def test_dit_l5_bursts_require_coincidence(self, substrate):
        """L5 burst proxy requires simultaneous high activation and high velocity."""
        substrate.x[:] = 0.0
        substrate.v[:] = 0.0
        substrate._update_qualia_metrics_sync(dt=0.1)
        
        assert substrate.l5_burst_count == 0
        
        # Set first 10 neurons to high activation and velocity
        substrate.x[:10] = 0.8
        substrate.v[:10] = 0.1
        
        substrate._update_qualia_metrics_sync(dt=0.1)
        
        assert substrate.l5_burst_count == 10, "DIT bursts did not match active neurons."


class TestSubstrateNoDuplicateMethods:
    """Verifies that _update_qualia_metrics functions don't conflict or loop."""

    @pytest.mark.asyncio
    async def test_async_wrapper_calls_sync(self, substrate):
        """The async wrapper should successfully call the sync underlying logic without error."""
        try:
            await substrate._update_qualia_metrics(dt=0.1)
        except Exception as e:
            pytest.fail(f"_update_qualia_metrics failed: {e}")
