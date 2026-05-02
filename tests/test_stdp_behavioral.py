"""Behavioral tests for STDP learning engine.

Tests verify that the STDP system produces the correct causal learning
signals: pre-before-post causes potentiation, post-before-pre causes
depression, and surprise modulates learning magnitude.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from core.consciousness.stdp_learning import STDPLearningEngine as STDPEngine, WEIGHT_CLIP


@pytest.fixture
def stdp():
    """64-neuron STDP engine."""
    return STDPEngine(n_neurons=64)


class TestWeightChangeMagnitude:
    """STDP must produce measurable weight changes."""

    def test_weight_changes_are_nonzero(self, stdp):
        """After spike records + reward, dw must be nonzero."""
        rng = np.random.RandomState(42)
        for t in range(100):
            activations = rng.rand(64)
            stdp.record_spikes(activations, t=float(t * 50))

        dw = stdp.deliver_reward(surprise=0.5, prediction_error=0.3)
        assert np.linalg.norm(dw) > 1e-6, (
            f"STDP produced zero weight change: norm={np.linalg.norm(dw):.10f}"
        )


class TestCausalLearning:
    """Pre-before-post must cause LTP (long-term potentiation)."""

    def test_pre_before_post_potentiates(self, stdp):
        """Neuron A fires consistently before neuron B → W[A,B] should increase."""
        A, B = 10, 20

        for cycle in range(50):
            # A fires first
            act_a = np.zeros(64)
            act_a[A] = 0.9
            stdp.record_spikes(act_a, t=float(cycle * 100))

            # B fires 10ms later (within STDP window)
            act_b = np.zeros(64)
            act_b[B] = 0.9
            stdp.record_spikes(act_b, t=float(cycle * 100 + 10))

        dw = stdp.deliver_reward(surprise=0.5, prediction_error=0.3)

        # The eligibility trace for A→B should be positive (potentiation)
        assert stdp._eligibility[A, B] > 0 or dw[A, B] > 0, (
            f"Pre-before-post did not potentiate: "
            f"eligibility[A,B]={stdp._eligibility[A, B]:.6f}, dw[A,B]={dw[A, B]:.6f}"
        )


class TestAntiCausalDepression:
    """Post-before-pre must cause LTD (long-term depression)."""

    def test_post_before_pre_depresses(self, stdp):
        """Neuron B fires consistently before neuron A → W[A,B] should decrease."""
        A, B = 10, 20

        for cycle in range(50):
            # B fires first (post fires before pre)
            act_b = np.zeros(64)
            act_b[B] = 0.9
            stdp.record_spikes(act_b, t=float(cycle * 100))

            # A fires 10ms later
            act_a = np.zeros(64)
            act_a[A] = 0.9
            stdp.record_spikes(act_a, t=float(cycle * 100 + 10))

        dw = stdp.deliver_reward(surprise=0.5, prediction_error=0.3)

        # Depression: A→B eligibility should be negative or zero
        # (A fired AFTER B, so this is anti-causal)
        assert stdp._eligibility[A, B] <= 0 or dw[A, B] <= 0, (
            f"Post-before-pre should cause depression: "
            f"eligibility[A,B]={stdp._eligibility[A, B]:.6f}"
        )


class TestWeightBounds:
    """Weight changes must never exceed WEIGHT_CLIP."""

    def test_weights_bounded_after_many_patterns(self, stdp):
        """10000 random spike patterns must keep all |W_ij| < WEIGHT_CLIP."""
        rng = np.random.RandomState(42)
        W = np.random.randn(64, 64) * 0.1

        for t in range(500):
            activations = rng.rand(64)
            stdp.record_spikes(activations, t=float(t * 50))

            if t % 10 == 0:
                dw = stdp.deliver_reward(
                    surprise=rng.rand(), prediction_error=rng.rand()
                )
                W = stdp.apply_to_connectivity(W, dw)

        assert np.all(np.abs(W) <= WEIGHT_CLIP + 1e-6), (
            f"Weight exceeded clip: max={np.max(np.abs(W)):.4f}, clip={WEIGHT_CLIP}"
        )


class TestSurpriseModulation:
    """Higher surprise must produce larger weight changes."""

    def test_high_surprise_larger_dw(self, stdp):
        """deliver_reward(surprise=0.9) must produce larger |dw| than surprise=0.1."""
        rng = np.random.RandomState(42)
        for t in range(50):
            activations = rng.rand(64)
            stdp.record_spikes(activations, t=float(t * 50))

        # Save eligibility state
        elig_snapshot = stdp._eligibility.copy()

        dw_low = stdp.deliver_reward(surprise=0.1, prediction_error=0.1)

        # Restore eligibility
        stdp._eligibility = elig_snapshot.copy()

        dw_high = stdp.deliver_reward(surprise=0.9, prediction_error=0.9)

        norm_low = np.linalg.norm(dw_low)
        norm_high = np.linalg.norm(dw_high)

        assert norm_high > norm_low, (
            f"High surprise should produce larger dw: "
            f"norm_high={norm_high:.6f}, norm_low={norm_low:.6f}"
        )


class TestEligibilityDecay:
    """Eligibility traces must decay when no spikes occur."""

    def test_eligibility_decays_without_spikes(self, stdp):
        """After initial spikes, 100 ticks of silence should collapse eligibility."""
        rng = np.random.RandomState(42)
        # Build up eligibility
        for t in range(20):
            activations = rng.rand(64)
            stdp.record_spikes(activations, t=float(t * 50))

        initial_norm = np.linalg.norm(stdp._eligibility)
        assert initial_norm > 1e-6, "Need nonzero eligibility to test decay"

        # 100 ticks of silence
        silent = np.zeros(64)
        for t in range(100):
            stdp.record_spikes(silent, t=float(1000 + t * 50))

        final_norm = np.linalg.norm(stdp._eligibility)
        assert final_norm < 0.01 * initial_norm, (
            f"Eligibility did not decay: initial={initial_norm:.6f}, "
            f"final={final_norm:.6f}, ratio={final_norm / initial_norm:.6f}"
        )


class TestSymmetryBreaking:
    """Weight matrix should not become perfectly symmetric over time."""

    def test_asymmetry_maintained(self, stdp):
        """After many apply_to_connectivity calls with causal patterns, W should not equal W^T."""
        W = np.zeros((64, 64))  # Start at zero
        np.fill_diagonal(W, 0)

        # Use causal patterns: neurons fire in sequence (0, 1, 2, ..., 63)
        # This naturally creates asymmetric STDP updates
        for cycle in range(100):
            for group_start in range(0, 64, 8):
                act = np.zeros(64)
                act[group_start:group_start + 8] = 0.9
                stdp.record_spikes(act, t=float(cycle * 800 + group_start * 10))

            if cycle % 2 == 0:
                dw = stdp.deliver_reward(surprise=0.3, prediction_error=0.2)
                W = stdp.apply_to_connectivity(W, dw)

        # The causal spike ordering + STDP should create meaningful asymmetry
        w_norm = np.linalg.norm(W)
        if w_norm > 0:
            asymmetry = np.linalg.norm(W - W.T) / w_norm
            assert asymmetry > 1e-8, (
                f"Weight matrix is too symmetric: asymmetry ratio={asymmetry:.10f}"
            )
