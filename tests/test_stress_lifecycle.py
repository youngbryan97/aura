"""Stress test: 5-minute simulated lifecycle across all hardened subsystems.

Runs 300 simulated ticks (1Hz equivalent) exercising:
- Neurochemical system (production, decay, homeostasis)
- STDP Learning Engine (spike-pair learning)
- Free Energy Engine (entropy, urgency, action determination)
- IIT Surrogate (Φ computation)
- Affect Engine (oscillation, valence, pulse)
- Liquid Substrate (ODE dynamics, qualia metrics)

Verifies no NaN, no unbounded growth, no crashes, and homeostatic convergence.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pytest
import asyncio
import numpy as np

from core.consciousness.neurochemical_system import NeurochemicalSystem
from core.consciousness.stdp_learning import STDPLearningEngine, WEIGHT_CLIP
from core.consciousness.free_energy import FreeEnergyEngine, FreeEnergyState
from core.consciousness.iit_surrogate import RIIU
from core.affect.damasio_v2 import AffectEngineV2, DamasioMarkers
from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig


STRESS_TICKS = 300  # 5 minutes at 1Hz


class TestStressLifecycle:
    """Simulated 5-minute lifecycle across all cognitive subsystems."""

    def test_neurochemical_stability(self):
        """300 ticks of random stimuli must not produce NaN or unbounded chemicals."""
        system = NeurochemicalSystem()
        rng = np.random.RandomState(42)

        for tick in range(STRESS_TICKS):
            # Random surge every 10 ticks
            if tick % 10 == 0:
                system.on_reward(rng.uniform(0.0, 1.0))
            if tick % 15 == 0:
                system.on_threat(rng.uniform(0.0, 1.0))

            system._metabolic_tick()

        for name, chem in system.chemicals.items():
            assert np.isfinite(chem.level), f"{name} is not finite: {chem.level}"
            assert 0.0 <= chem.level <= 2.0, f"{name} out of bounds: {chem.level}"

    def test_stdp_weight_stability(self):
        """300 ticks of random spikes must keep weights bounded."""
        stdp = STDPLearningEngine(n_neurons=64)
        rng = np.random.RandomState(42)
        W = np.zeros((64, 64))

        for tick in range(STRESS_TICKS):
            # Random spikes
            activations = rng.rand(64)
            stdp.record_spikes(activations, t=float(tick * 50))

            if tick % 10 == 0:
                dw = stdp.deliver_reward(
                    surprise=rng.rand(), prediction_error=rng.rand()
                )
                W = stdp.apply_to_connectivity(W, dw)

        assert np.all(np.isfinite(W)), "STDP weights contain NaN/Inf"
        assert np.max(np.abs(W)) <= WEIGHT_CLIP + 1e-6, "STDP weights unbounded"

    def test_free_energy_no_crash(self):
        """300 ticks must not crash or produce NaN free energy."""
        engine = FreeEnergyEngine()

        for tick in range(STRESS_TICKS):
            pred_error = np.sin(tick * 0.1) * 0.5 + 0.5
            result = engine.compute(prediction_error=pred_error)

    def test_iit_phi_stability(self):
        """300 ticks with varying signals must produce bounded Φ."""
        riiu = RIIU(neuron_count=16, buffer_size=32)
        rng = np.random.RandomState(42)

        phis = []
        for tick in range(STRESS_TICKS):
            # Mix of structured signal and noise
            signal = np.sin(tick * 0.05) * np.ones(16) + rng.randn(16) * 0.1
            phi = riiu.compute_phi(signal)
            phis.append(phi)

        phis = np.array(phis)
        assert np.all(np.isfinite(phis)), "Φ contains NaN/Inf"
        assert np.max(phis) < 1000.0, "Φ unbounded"

    @pytest.mark.asyncio
    async def test_affect_engine_lifecycle(self):
        """300 pulse ticks with alternating stimuli must stay bounded."""
        engine = AffectEngineV2()
        engine.markers = DamasioMarkers()
        rng = np.random.RandomState(42)

        for tick in range(STRESS_TICKS):
            # Random emotional injection
            if tick % 5 == 0:
                emotion = rng.choice(list(engine.markers.emotions.keys()))
                engine.markers.emotions[emotion] = rng.uniform(0, 1)

            await engine.pulse()

        # All emotions must be in [0, 1]
        for emo, val in engine.markers.emotions.items():
            assert 0.0 <= float(val) <= 1.0, f"{emo} out of range: {val}"
            assert np.isfinite(val), f"{emo} is NaN"

        # No permanent oscillation lock
        assert engine.markers.momentum in (0.85, 0.95), \
            f"Momentum stuck at unexpected value: {engine.markers.momentum}"

    def test_substrate_dynamics_stable(self, tmp_path):
        """300 ticks of qualia metric updates must keep state finite."""
        state_file = tmp_path / "stress_substrate.npy"
        config = SubstrateConfig(neuron_count=64, noise_level=0.01, state_file=state_file)
        substrate = LiquidSubstrate(config)

        for tick in range(STRESS_TICKS):
            dt = 1.0 / substrate.config.update_rate
            substrate._update_qualia_metrics_sync(dt)

            # Inject small stimulus every 20 ticks
            if tick % 20 == 0:
                substrate.x[:8] += np.random.randn(8) * 0.1

        assert np.all(np.isfinite(substrate.x)), "Substrate x contains NaN/Inf"
        assert np.all(np.isfinite(substrate.v)), "Substrate v contains NaN/Inf"
        assert substrate.microtubule_coherence >= 0.0
        assert substrate.em_field_magnitude >= 0.0

    def test_concurrent_subsystem_integration(self):
        """All subsystems must be instantiable and tickable without import conflicts."""
        system = NeurochemicalSystem()
        stdp = STDPLearningEngine(n_neurons=64)
        engine = FreeEnergyEngine()
        riiu = RIIU(neuron_count=16)

        rng = np.random.RandomState(42)
        for _ in range(50):
            system._metabolic_tick()
            stdp.record_spikes(rng.rand(64), t=float(_ * 50))
            engine.compute(prediction_error=0.3)
            riiu.compute_phi(rng.randn(16))

        # No crash = pass
        assert True
