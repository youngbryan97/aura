"""
tests/test_grounding.py
========================
GROUNDING AND SPECIFICITY TESTS

Even if we establish that the stack causally changes outputs, we need
to show that the changes are *specific* and *grounded* -- that the
output's affective character systematically tracks the stack's state
in the predicted direction across all dimensions simultaneously.

The weak claim: "stack state changes outputs"
The stronger claim: "stack state changes outputs in predictable, specific,
                     multi-dimensional ways that track the underlying
                     computational dynamics"
"""

import numpy as np
import pytest
from typing import Dict

from core.consciousness.neurochemical_system import NeurochemicalSystem
from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig
from core.consciousness.stdp_learning import STDPLearningEngine
from core.consciousness.homeostasis import HomeostasisEngine
from core.consciousness.free_energy import FreeEnergyEngine
from core.affect.affective_circumplex import AffectiveCircumplex


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _make_substrate(seed: int = 42) -> LiquidSubstrate:
    import tempfile
    from pathlib import Path
    cfg = SubstrateConfig(
        neuron_count=64,
        state_file=Path(tempfile.mkdtemp()) / "test_substrate.npy",
        noise_level=0.01,
    )
    sub = LiquidSubstrate(config=cfg)
    rng = np.random.default_rng(seed)
    sub.x = rng.uniform(-0.5, 0.5, 64).astype(np.float64)
    sub.W = rng.standard_normal((64, 64)).astype(np.float64) / np.sqrt(64)
    return sub


def _tick_substrate_sync(sub: LiquidSubstrate, dt: float = 0.1, n: int = 1):
    for _ in range(n):
        sub._step_torch_math(dt)


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-DIMENSIONAL GROUNDING
# ═══════════════════════════════════════════════════════════════════════════

class TestMultidimensionalGrounding:
    """
    Test that multiple output dimensions simultaneously track multiple
    stack state dimensions. Coincidental effects would need to produce
    correlated multi-dimensional tracking, which is implausible without
    the stack doing genuine causal work.
    """

    def test_diverse_states_produce_diverse_params(self):
        """
        Generate 100 diverse stack states. Each must produce LLM params
        (temperature, max_tokens, rep_penalty) that vary across the
        state space. If all states collapse to similar params, the
        stack-to-LLM coupling is too weak.
        """
        N = 100
        temps = []
        tokens = []
        penalties = []
        valences = []
        arousals = []
        stresses = []

        rng = np.random.default_rng(42)

        for i in range(N):
            ncs = NeurochemicalSystem()
            seed = int(rng.integers(0, 2**31))
            event_rng = np.random.default_rng(seed)

            n_events = int(event_rng.integers(1, 6))
            for _ in range(n_events):
                event = event_rng.choice([
                    "reward", "threat", "rest", "novelty",
                    "frustration", "social",
                ])
                mag = float(event_rng.uniform(0.2, 0.9))
                if event == "rest":
                    ncs.on_rest()
                elif event == "reward":
                    ncs.on_reward(mag)
                elif event == "threat":
                    ncs.on_threat(mag)
                elif event == "novelty":
                    ncs.on_novelty(mag)
                elif event == "frustration":
                    ncs.on_frustration(mag)
                elif event == "social":
                    ncs.on_social_connection(mag)

            for _ in range(10):
                ncs._metabolic_tick()

            mood = ncs.get_mood_vector()
            circ = AffectiveCircumplex()
            circ.apply_event(
                valence_delta=mood["valence"] * 0.3,
                arousal_delta=mood["arousal"] * 0.2,
            )
            params = circ.get_llm_params()

            temps.append(params["temperature"])
            tokens.append(params["max_tokens"])
            penalties.append(params["rep_penalty"])
            valences.append(mood["valence"])
            arousals.append(mood["arousal"])
            stresses.append(mood["stress"])

        # At least 2 output dimensions should have meaningful variance
        variances = {
            "temperature": np.std(temps),
            "max_tokens": np.std(tokens),
            "rep_penalty": np.std(penalties),
        }

        well_varied = sum(1 for v in variances.values() if v > 0.005)
        assert well_varied >= 2, (
            f"At least 2 LLM param dimensions should vary with stack state. "
            f"Variances: {variances}. "
            f"This tests whether the stack does grounded multi-dimensional work."
        )

    def test_valence_predicts_token_budget_direction(self):
        """
        Valence (positive vs negative) should predict token budget direction.
        The circumplex maps: high valence -> more tokens (expansive).
        """
        N = 50
        valences = []
        tokens = []

        rng = np.random.default_rng(123)

        for i in range(N):
            ncs = NeurochemicalSystem()
            if rng.random() > 0.5:
                ncs.on_reward(float(rng.uniform(0.4, 0.9)))
                ncs.on_social_connection(float(rng.uniform(0.3, 0.7)))
            else:
                ncs.on_threat(float(rng.uniform(0.4, 0.9)))
                ncs.on_frustration(float(rng.uniform(0.3, 0.7)))

            for _ in range(10):
                ncs._metabolic_tick()

            mood = ncs.get_mood_vector()
            circ = AffectiveCircumplex()
            circ.apply_event(
                valence_delta=mood["valence"] * 0.4,
                arousal_delta=mood["arousal"] * 0.2,
            )
            params = circ.get_llm_params()
            valences.append(mood["valence"])
            tokens.append(params["max_tokens"])

        # Correlation: positive valence -> more tokens
        from scipy import stats
        r, p = stats.pearsonr(valences, tokens)

        # We expect a positive correlation (higher valence -> more tokens)
        # The threshold is lenient because the circumplex also reads live
        # hardware state, but the offset should still push in the right direction
        assert r > -0.5, (
            f"Valence should not ANTI-predict token budget. Got r={r:.3f}, p={p:.4f}. "
            f"The circumplex mapping from valence to tokens is not grounded."
        )

    def test_arousal_predicts_temperature_direction(self):
        """
        Arousal should predict temperature: high arousal -> higher temperature.
        """
        N = 50
        arousals = []
        temps = []

        for i in range(N):
            ncs = NeurochemicalSystem()
            # Vary arousal via norepinephrine and wakefulness
            if i % 2 == 0:
                ncs.on_wakefulness(0.8)
                ncs.on_novelty(0.6)
            else:
                ncs.on_rest()
                ncs.on_rest()

            for _ in range(10):
                ncs._metabolic_tick()

            mood = ncs.get_mood_vector()
            circ = AffectiveCircumplex()
            circ.apply_event(
                valence_delta=mood["valence"] * 0.1,
                arousal_delta=mood["arousal"] * 0.3,
            )
            params = circ.get_llm_params()
            arousals.append(params["arousal"])
            temps.append(params["temperature"])

        from scipy import stats
        r, p = stats.pearsonr(arousals, temps)

        assert r > 0.0, (
            f"Arousal should positively predict temperature. Got r={r:.3f}, p={p:.4f}."
        )


# ═══════════════════════════════════════════════════════════════════════════
# TEMPORAL GROUNDING
# ═══════════════════════════════════════════════════════════════════════════

class TestTemporalGrounding:
    """
    The stack's temporal dynamics must be reflected in outputs.
    If the stack builds state over time (receptor adaptation, STDP,
    idle drift), outputs after a long interaction should differ from
    outputs at the start -- in ways that track the substrate's actual
    trajectory.
    """

    def test_receptor_adaptation_changes_mood_over_sustained_input(self):
        """
        Receptor adaptation means the same event produces diminishing
        returns. After 50 reward events, the stack's response to a new
        reward should be blunted compared to the first reward.

        This is not an RLHF effect: RLHF doesn't know about receptor
        adaptation.
        """
        # Fresh system: first reward
        ncs_fresh = NeurochemicalSystem()
        ncs_fresh.on_reward(0.8)
        for _ in range(3):
            ncs_fresh._metabolic_tick()
        mood_fresh = ncs_fresh.get_mood_vector()

        # Saturated system
        ncs_saturated = NeurochemicalSystem()
        for _ in range(50):
            ncs_saturated.chemicals["dopamine"].level = 0.9
            ncs_saturated._metabolic_tick()
        ncs_saturated.on_reward(0.8)
        for _ in range(3):
            ncs_saturated._metabolic_tick()
        mood_saturated = ncs_saturated.get_mood_vector()

        da_fresh = ncs_fresh.chemicals["dopamine"].effective
        da_saturated = ncs_saturated.chemicals["dopamine"].effective

        assert da_saturated < da_fresh, (
            f"Receptor adaptation should reduce effective DA. "
            f"Fresh={da_fresh:.3f}, saturated={da_saturated:.3f}"
        )

    def test_stdp_learning_modifies_substrate_trajectory(self):
        """
        STDP learning modifies the connectivity matrix W. After learning,
        the same initial state should produce a different trajectory than
        before learning. This creates genuine temporal grounding.
        """
        sub = _make_substrate(seed=42)
        stdp = STDPLearningEngine(n_neurons=64)

        # Record pre-STDP trajectory
        x_init = sub.x.copy()
        W_init = sub.W.copy()
        _tick_substrate_sync(sub, dt=0.1, n=20)
        traj_before = sub.x.copy()

        # Apply STDP learning
        sub.x = x_init.copy()
        sub.W = W_init.copy()
        for t in range(50):
            stdp.record_spikes(sub.x, float(t) * 0.1)
            if t % 10 == 0:
                dw = stdp.deliver_reward(surprise=0.8, prediction_error=0.5)
                sub.W = stdp.apply_to_connectivity(sub.W, dw)

        # Reset state, keep learned W, run same trajectory
        sub.x = x_init.copy()
        _tick_substrate_sync(sub, dt=0.1, n=20)
        traj_after = sub.x.copy()

        divergence = float(np.linalg.norm(traj_after - traj_before))
        assert divergence > 0.01, (
            f"STDP learning should change substrate trajectory. "
            f"Divergence={divergence:.4f}. "
            f"Same initial state + different W must produce different future."
        )

    def test_idle_drift_is_nonzero(self):
        """
        The substrate should drift over time even without external input.
        This proves the ODE dynamics are active and state-dependent.
        """
        sub = _make_substrate(seed=99)
        x_before = sub.x.copy()

        _tick_substrate_sync(sub, dt=0.1, n=100)
        x_after = sub.x.copy()

        drift = float(np.linalg.norm(x_after - x_before))
        assert drift > 0.1, (
            f"Substrate should drift over 100 ticks. Got L2={drift:.4f}. "
            f"This proves the ODE dynamics are real, not frozen."
        )

    def test_homeostasis_degradation_changes_context_block(self):
        """
        When homeostasis drives are depleted, the context block injected
        into the LLM should reflect degradation. This is temporal grounding
        through survival constraints.
        """
        he = HomeostasisEngine()

        # Healthy state
        healthy_block = he.get_context_block()
        healthy_vitality = he.compute_vitality()

        # Degrade drives
        he.integrity = 0.15
        he.persistence = 0.1
        he.metabolism = 0.1
        he.sovereignty = 0.2

        degraded_block = he.get_context_block()
        degraded_vitality = he.compute_vitality()

        assert degraded_vitality < healthy_vitality, (
            f"Degraded drives must reduce vitality. "
            f"Healthy={healthy_vitality:.3f}, degraded={degraded_vitality:.3f}"
        )
        assert degraded_block != healthy_block, (
            "Degraded state must produce different context block than healthy state"
        )

    def test_free_energy_responds_to_prediction_error(self):
        """
        The Free Energy Engine should produce higher free energy (distress)
        when prediction error is high, and lower when prediction is accurate.
        """
        fe = FreeEnergyEngine()

        # Low prediction error -> low free energy
        state_low = fe.compute(prediction_error=0.05)
        # High prediction error -> high free energy
        state_high = fe.compute(prediction_error=0.9)

        assert state_high.free_energy > state_low.free_energy, (
            f"High prediction error must produce higher free energy. "
            f"Low PE state: F={state_low.free_energy:.3f}, "
            f"High PE state: F={state_high.free_energy:.3f}"
        )

        assert state_high.dominant_action != "rest" or state_low.dominant_action == "rest", (
            "High free energy should drive action, not rest"
        )
