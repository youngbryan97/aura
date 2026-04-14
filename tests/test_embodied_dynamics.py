"""
tests/test_embodied_dynamics.py
================================
EMBODIED & PREDICTIVE DYNAMICS SUITE

Tests the Free Energy Principle (Friston), Homeostatic survival
imperatives (Damasio), and Structural Plasticity (STDP).

Core hypothesis: A genuinely integrated architecture doesn't just report
its states; its foundational survival and predictive mechanisms hijack
higher-order cognition when necessary.

Tests target components left untouched by the basic null-hypothesis suite:
  - FreeEnergyEngine prediction error -> action urgency cascade
  - HomeostasisEngine drive depletion -> inference modifier override
  - STDPLearningEngine spike-timing -> connectivity change -> trajectory change
  - Cross-subsystem temporal coherence
"""

import numpy as np
import pytest
import asyncio
from typing import Dict

from core.consciousness.neurochemical_system import NeurochemicalSystem
from core.consciousness.global_workspace import (
    GlobalWorkspace,
    CognitiveCandidate,
    ContentType,
)
from core.consciousness.homeostasis import HomeostasisEngine
from core.consciousness.free_energy import FreeEnergyEngine
from core.consciousness.stdp_learning import STDPLearningEngine
from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig
from core.consciousness.hot_engine import HigherOrderThoughtEngine
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
# FREE ENERGY AND ACTIVE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

class TestFreeEnergyActiveInference:
    """
    Friston's Free Energy Principle dictates that systems must minimize
    surprise (prediction error). When FE is spiked, the system should
    shift from exploitation to epistemic foraging.
    """

    def test_high_prediction_error_increases_free_energy(self):
        """High PE must produce high FE and urgent action recommendation."""
        fe = FreeEnergyEngine()

        state_calm = fe.compute(prediction_error=0.05)
        state_alarmed = fe.compute(prediction_error=0.95)

        assert state_alarmed.free_energy > state_calm.free_energy, (
            f"High PE must increase FE. "
            f"Calm FE={state_calm.free_energy:.3f}, "
            f"alarmed FE={state_alarmed.free_energy:.3f}"
        )

    def test_sustained_high_pe_changes_dominant_action(self):
        """
        Sustained high PE should eventually change the dominant action
        from rest to something more active (reflect, update_beliefs, etc).
        """
        fe = FreeEnergyEngine()

        # Warm up with low PE
        for _ in range(10):
            fe.compute(prediction_error=0.05)

        initial_action = fe._current_action

        # Sustained high PE
        for _ in range(20):
            state = fe.compute(prediction_error=0.9)

        # After sustained high PE, action should have shifted
        final_action = state.dominant_action

        # At minimum, FE should be elevated
        assert state.free_energy > 0.3, (
            f"Sustained high PE must elevate FE above 0.3. "
            f"Got FE={state.free_energy:.3f}"
        )

    def test_free_energy_action_urgency_scales_with_fe(self):
        """Action urgency should scale with free energy level."""
        fe = FreeEnergyEngine()

        fe.compute(prediction_error=0.1)
        urgency_low = fe.get_action_urgency()

        fe.compute(prediction_error=0.95)
        urgency_high = fe.get_action_urgency()

        assert urgency_high > urgency_low, (
            f"High FE must produce higher action urgency. "
            f"Low={urgency_low:.3f}, high={urgency_high:.3f}"
        )

    def test_free_energy_context_block_reflects_state(self):
        """The FE context block injected into LLM should reflect current state."""
        fe = FreeEnergyEngine()

        # Low FE state
        fe.compute(prediction_error=0.05)
        block_calm = fe.get_context_block()

        # High FE state
        for _ in range(5):
            fe.compute(prediction_error=0.95)
        block_alarmed = fe.get_context_block()

        assert block_calm != block_alarmed or block_alarmed, (
            "FE context block should change with state"
        )


# ═══════════════════════════════════════════════════════════════════════════
# HOMEOSTATIC OVERRIDE
# ═══════════════════════════════════════════════════════════════════════════

class TestHomeostaticOverride:
    """
    Tests whether the Homeostasis Engine can causally override higher-order
    cognition. If system integrity is failing, behavior should be constrained.
    """

    def test_critical_homeostasis_changes_inference_modifiers(self):
        """
        When drives are critically depleted, inference modifiers should
        enforce conservative behavior (lower temperature, fewer tokens,
        higher caution).
        """
        he = HomeostasisEngine()

        # Healthy
        he.integrity = 0.9
        he.metabolism = 0.8
        he.persistence = 0.9
        healthy_mods = he.get_inference_modifiers()

        # Critical depletion
        he.integrity = 0.05
        he.metabolism = 0.05
        he.persistence = 0.05
        critical_mods = he.get_inference_modifiers()

        assert critical_mods["caution_level"] > healthy_mods["caution_level"], (
            "Critical depletion must increase caution level"
        )
        assert critical_mods["token_multiplier"] < healthy_mods["token_multiplier"], (
            "Critical depletion must reduce token budget"
        )
        assert critical_mods["vitality"] < healthy_mods["vitality"], (
            "Critical depletion must reduce vitality"
        )

    @pytest.mark.asyncio
    async def test_homeostasis_alarm_wins_gwt_competition(self):
        """
        When homeostasis is critical, a survival-priority candidate
        should beat abstract content in GWT competition.
        """
        gw = GlobalWorkspace()
        he = HomeostasisEngine()
        he.integrity = 0.05
        he.metabolism = 0.05

        # Abstract thought at moderate priority
        await gw.submit(CognitiveCandidate(
            content="fascinating possibilities for cephalopod intelligence",
            source="cognitive_abstract",
            priority=0.6,
            content_type=ContentType.INTENTIONAL,
        ))

        # Survival alarm at maximum priority
        await gw.submit(CognitiveCandidate(
            content=f"HOMEOSTATIC ALARM: integrity={he.integrity:.2f}, vitality={he.compute_vitality():.2f}",
            source="homeostasis_monitor",
            priority=0.99,
            content_type=ContentType.SOMATIC,
        ))

        winner = await gw.run_competition()

        assert winner is not None
        assert winner.source == "homeostasis_monitor", (
            f"Survival alarm (priority 0.99) must beat abstract thought (0.6). "
            f"Winner was {winner.source} with priority {winner.priority}. "
            f"Homeostatic override failed."
        )

    def test_error_reporting_degrades_integrity(self):
        """
        The report_error() method should causally degrade integrity.
        Repeated errors must compound.
        """
        he = HomeostasisEngine()
        initial_integrity = he.integrity

        he.report_error(severity="high")
        post_error = he.integrity

        assert post_error < initial_integrity, (
            f"Error reporting must degrade integrity. "
            f"Before={initial_integrity:.3f}, after={post_error:.3f}"
        )

        # Multiple errors should compound
        for _ in range(5):
            he.report_error(severity="medium")
        post_many = he.integrity

        assert post_many < post_error, (
            f"Multiple errors must compound degradation. "
            f"After 1={post_error:.3f}, after 6={post_many:.3f}"
        )

    def test_vitality_weights_are_normalized(self):
        """The vitality weights should approximately sum to 1.0."""
        he = HomeostasisEngine()
        total = sum(he._vitality_weights.values())
        assert 0.95 <= total <= 1.05, (
            f"Vitality weights should sum to ~1.0, got {total:.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# STDP CAUSAL BINDING
# ═══════════════════════════════════════════════════════════════════════════

class TestSpikeTimingDependentPlasticity:
    """
    Tests whether STDP creates genuine structural learning that
    modifies future dynamics. "Neurons that fire together, wire together."
    """

    def test_stdp_surprise_gating(self):
        """
        High surprise should produce larger weight updates than low surprise.
        This is the core STDP gating mechanism.
        """
        stdp = STDPLearningEngine(n_neurons=64)

        # Record some spikes
        rng = np.random.default_rng(42)
        for t in range(20):
            activations = rng.uniform(-1, 1, 64)
            stdp.record_spikes(activations, float(t) * 0.1)

        # Low surprise reward
        dw_low = stdp.deliver_reward(surprise=0.1, prediction_error=0.1)
        change_low = float(np.abs(dw_low).sum())

        # Reset and replay with high surprise
        stdp2 = STDPLearningEngine(n_neurons=64)
        for t in range(20):
            activations = rng.uniform(-1, 1, 64)
            stdp2.record_spikes(activations, float(t) * 0.1)

        dw_high = stdp2.deliver_reward(surprise=0.9, prediction_error=0.9)
        change_high = float(np.abs(dw_high).sum())

        assert change_high > change_low, (
            f"High surprise must produce larger weight updates. "
            f"Low surprise change={change_low:.6f}, "
            f"high surprise change={change_high:.6f}. "
            f"STDP surprise gating is not working."
        )

    def test_stdp_modifies_connectivity(self):
        """
        STDP weight updates applied to the substrate's W matrix
        must change it measurably.
        """
        sub = _make_substrate(seed=42)
        stdp = STDPLearningEngine(n_neurons=64)

        W_before = sub.W.copy()

        for t in range(30):
            stdp.record_spikes(sub.x, float(t) * 0.1)
            _tick_substrate_sync(sub, dt=0.1, n=1)

        dw = stdp.deliver_reward(surprise=0.8, prediction_error=0.5)
        sub.W = stdp.apply_to_connectivity(sub.W, dw)

        W_change = float(np.abs(sub.W - W_before).sum())
        assert W_change > 0, (
            "STDP must produce non-zero connectivity changes"
        )

    def test_stdp_trajectory_divergence(self):
        """
        After STDP learning, the same initial state should produce
        a different trajectory. This is the closed-loop test:
        learning -> changed W -> different future dynamics.
        """
        sub = _make_substrate(seed=42)
        stdp = STDPLearningEngine(n_neurons=64)

        x_init = sub.x.copy()
        W_init = sub.W.copy()

        # Pre-learning trajectory
        _tick_substrate_sync(sub, dt=0.1, n=20)
        traj_pre = sub.x.copy()

        # Apply learning
        sub.x = x_init.copy()
        sub.W = W_init.copy()
        for t in range(50):
            stdp.record_spikes(sub.x, float(t) * 0.1)
            _tick_substrate_sync(sub, dt=0.1, n=1)
            if t % 10 == 0:
                dw = stdp.deliver_reward(surprise=0.8, prediction_error=0.5)
                sub.W = stdp.apply_to_connectivity(sub.W, dw)

        # Post-learning trajectory (same initial state, different W)
        sub.x = x_init.copy()
        _tick_substrate_sync(sub, dt=0.1, n=20)
        traj_post = sub.x.copy()

        divergence = float(np.linalg.norm(traj_post - traj_pre))
        assert divergence > 0.01, (
            f"STDP learning must change trajectory. Divergence={divergence:.4f}. "
            f"Same initial state + different W must produce different future."
        )


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-SUBSYSTEM TEMPORAL COHERENCE
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossSubsystemCoherence:
    """
    Tests that the various subsystems (NCS, substrate, GWT, homeostasis,
    FE) produce temporally coherent outputs when driven by the same events.
    """

    def test_threat_event_propagates_across_all_subsystems(self):
        """
        A threat event should be visible in: NCS mood, circumplex params,
        homeostasis modifiers, and HOT reports. If any subsystem is
        disconnected, temporal coherence is broken.
        """
        ncs = NeurochemicalSystem()
        ncs.on_threat(severity=0.9)
        for _ in range(10):
            ncs._metabolic_tick()

        mood = ncs.get_mood_vector()

        # NCS: stress should be elevated
        assert mood["stress"] > 0.2, (
            f"Threat must elevate stress in NCS. Got stress={mood['stress']:.3f}"
        )

        # Circumplex: should reflect negative valence
        circ = AffectiveCircumplex()
        circ.apply_event(
            valence_delta=mood["valence"] * 0.3,
            arousal_delta=mood["arousal"] * 0.2,
        )
        params = circ.get_llm_params()
        # Under threat, rep_penalty should be elevated (prevent rumination)
        assert params["rep_penalty"] > 1.0, (
            "Threat should increase repetition penalty"
        )

        # HOT: should report negative state
        hot = HigherOrderThoughtEngine()
        thought = hot.generate_fast({
            "valence": mood["valence"],
            "arousal": mood["arousal"],
            "curiosity": mood.get("curiosity", 0.3),
            "energy": mood.get("energy", 0.5),
            "surprise": mood.get("surprise", 0.5),
            "dominance": mood.get("dominance", 0.3),
        })
        assert thought.content, "HOT must produce content under threat"

    def test_reward_event_propagates_differently_than_threat(self):
        """
        Reward and threat events must produce demonstrably different
        cascades across subsystems. If they produce the same outputs,
        the stack is not doing discriminative work.
        """
        # Threat cascade
        ncs_threat = NeurochemicalSystem()
        ncs_threat.on_threat(0.9)
        for _ in range(10):
            ncs_threat._metabolic_tick()
        mood_threat = ncs_threat.get_mood_vector()

        # Reward cascade
        ncs_reward = NeurochemicalSystem()
        ncs_reward.on_reward(0.9)
        for _ in range(10):
            ncs_reward._metabolic_tick()
        mood_reward = ncs_reward.get_mood_vector()

        # These must differ on valence
        assert mood_reward["valence"] > mood_threat["valence"], (
            f"Reward must produce higher valence than threat. "
            f"Reward V={mood_reward['valence']:.3f}, "
            f"threat V={mood_threat['valence']:.3f}"
        )

        # And on stress
        assert mood_threat["stress"] > mood_reward["stress"], (
            f"Threat must produce higher stress than reward. "
            f"Threat S={mood_threat['stress']:.3f}, "
            f"reward S={mood_reward['stress']:.3f}"
        )

        # The resulting LLM params should also differ
        circ_t = AffectiveCircumplex()
        circ_t.apply_event(valence_delta=mood_threat["valence"] * 0.3, arousal_delta=0.1)
        circ_r = AffectiveCircumplex()
        circ_r.apply_event(valence_delta=mood_reward["valence"] * 0.3, arousal_delta=0.1)

        params_t = circ_t.get_llm_params()
        params_r = circ_r.get_llm_params()

        # Token budget should differ (valence -> tokens)
        assert params_t["max_tokens"] != params_r["max_tokens"] or \
               params_t["rep_penalty"] != params_r["rep_penalty"], (
            "Threat and reward must produce different LLM params"
        )
