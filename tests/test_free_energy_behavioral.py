"""Behavioral tests for the FreeEnergyEngine.

Verifies the Active Inference dynamics: entropy calculation, hysteresis
in action selection, and trend-weighted urgency.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pytest
import threading

from core.consciousness.free_energy import FreeEnergyEngine, FreeEnergyState


@pytest.fixture
def fe_engine():
    """Fresh FreeEnergyEngine instance."""
    return FreeEnergyEngine()


class TestSystemEntropy:
    """System entropy must use rolling averages and be bounded."""

    def test_entropy_is_bounded(self, fe_engine):
        """System entropy must remain between 0.0 and 1.0."""
        # Seed the history
        for _ in range(5):
            val = fe_engine._compute_system_entropy()
            assert 0.0 <= val <= 1.0

    def test_rolling_average_smooths_noise(self, fe_engine, monkeypatch):
        """Rolling average should mitigate single-tick spikes."""
        import psutil

        # Mock psutil to return stable values, then a spike
        mock_cpu = 10.0
        monkeypatch.setattr(psutil, "cpu_percent", lambda interval=0: mock_cpu)
        monkeypatch.setattr(psutil, "virtual_memory", lambda: type('obj', (object,), {'percent': 50.0}))

        # Build stable history
        for _ in range(10):
            base_entropy = fe_engine._compute_system_entropy()

        # Inject a spike
        mock_cpu = 95.0
        spike_entropy = fe_engine._compute_system_entropy()

        # The spike should not drastically alter the entropy in one tick due to the rolling average
        # If it was instantaneous, it would jump significantly.
        # It's hard to assert exact bounds without knowing the math exactly,
        # but we can ensure it doesn't jump immediately to the target value.
        assert abs(spike_entropy - base_entropy) < 0.2, (
            "Rolling average failed to smooth the CPU spike."
        )


class TestActionHysteresis:
    """Action selection must not flap wildly."""

    def test_hysteresis_prevents_flapping(self, fe_engine):
        """Minor fluctuations should not cause immediate action switches."""
        # Stabilize on "rest"
        for _ in range(5):
            fe_engine.compute(prediction_error=0.0)
            
        assert fe_engine._current_action == "rest"

        # Minor increase in FE should not switch action immediately if under threshold
        fe_engine.compute(prediction_error=0.1)
        assert fe_engine._current_action == "rest", "Flapped on minor FE change."

    def test_large_delta_forces_switch(self, fe_engine):
        """A large FE delta should force an immediate switch, bypassing hold time."""
        for _ in range(5):
            fe_engine.compute(prediction_error=0.0)
            
        assert fe_engine._current_action == "rest"

        # Massive surprise
        fe_engine.compute(prediction_error=1.0)
        assert fe_engine._current_action == "update_beliefs", "Did not switch on large delta."


class TestTrendWeightedUrgency:
    """Rising free energy should increase urgency."""

    def test_rising_trend_boosts_urgency(self, fe_engine):
        """A rising FE trend must produce higher urgency than stable FE of the same value."""
        # Create a stable baseline
        for _ in range(10):
            fe_engine.compute(prediction_error=0.2)
        
        stable_urgency = fe_engine.get_action_urgency()
        
        # Create a rising trend
        for val in [0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]:
            fe_engine.compute(prediction_error=val)
            
        rising_urgency = fe_engine.get_action_urgency()
        
        assert fe_engine.get_trend() == "rising"
        assert rising_urgency > stable_urgency, "Rising trend did not boost urgency."


class TestThreadSafety:
    """Engine must be thread-safe."""

    def test_concurrent_computes(self, fe_engine):
        """Concurrent computes should not raise exceptions or corrupt state."""
        def worker():
            for _ in range(100):
                fe_engine.compute(prediction_error=0.1)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # If it didn't crash, the lock worked.
        assert fe_engine._total_computes == 1000
        assert fe_engine.smoothed_fe > 0.0
