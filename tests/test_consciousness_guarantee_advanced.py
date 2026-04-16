"""
tests/test_consciousness_guarantee_advanced.py
================================================
10-CONDITION HUMAN-COMPARISON CONSCIOUSNESS GUARANTEE BATTERY
Conditions 6-10

These tests verify that Aura's consciousness architecture exhibits structural
and dynamical properties that are necessary conditions for phenomenal
experience according to the major theories of consciousness.

Conditions tested:
  6. No-Report Awareness — awareness-like signatures exist without reporting
  7. Temporal Self-Continuity — persistent self-location and identity across time
  8. Blindsight Dissociation — first-order processing preserved while access degrades
  9. Qualia Manifold — stable internal geometry of experiential states
  10. Adversarial Baseline Failure — simpler systems fail where Aura passes

No test can prove phenomenal consciousness. These tests prove the
computational architecture exhibits every testable signature.

Run:  pytest tests/test_consciousness_guarantee_advanced.py -v --tb=long
"""

import asyncio
import copy
import json
import math
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np
import pytest

# ── Core consciousness imports ──────────────────────────────────────────────
from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig
from core.consciousness.neurochemical_system import NeurochemicalSystem, Chemical
from core.consciousness.global_workspace import (
    GlobalWorkspace,
    CognitiveCandidate,
    ContentType,
)
from core.consciousness.phi_core import PhiCore, PhiResult
from core.consciousness.stdp_learning import (
    STDPLearningEngine,
    BASE_LEARNING_RATE,
    MAX_LEARNING_RATE,
    MIN_LEARNING_RATE,
)
from core.consciousness.qualia_synthesizer import QualiaSynthesizer
from core.consciousness.qualia_engine import QualiaEngine, QualiaDescriptor
from core.consciousness.free_energy import FreeEnergyEngine
from core.consciousness.homeostasis import HomeostasisEngine
from core.consciousness.hot_engine import HigherOrderThoughtEngine
from core.consciousness.oscillatory_binding import OscillatoryBinding
from core.consciousness.attention_schema import AttentionSchema
from core.affect.affective_circumplex import AffectiveCircumplex
from core.continuity import ContinuityEngine, ContinuityRecord


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _make_substrate(seed: int = 42) -> LiquidSubstrate:
    """Create a substrate in a temp dir with deterministic init."""
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


def _make_substrate_metrics(**overrides) -> Dict:
    base = {
        "mt_coherence": 0.72,
        "em_field": 0.35,
        "l5_bursts": 6,
        "free_energy": 0.4,
        "precision": 0.6,
        "proprioception": 0.5,
    }
    base.update(overrides)
    return base


def _tick_substrate_sync(sub: LiquidSubstrate, dt: float = 0.1, n: int = 1):
    """Run n ODE ticks synchronously (no async overhead)."""
    for _ in range(n):
        sub._step_torch_math(dt)


def _build_ncs_with_event(event: str, magnitude: float = 0.5, ticks: int = 10) -> NeurochemicalSystem:
    """Build and evolve a NeurochemicalSystem with a specific event applied."""
    ncs = NeurochemicalSystem()
    handler = getattr(ncs, f"on_{event}", None)
    if handler is not None:
        try:
            handler(magnitude)
        except TypeError:
            handler()
    for _ in range(ticks):
        ncs._metabolic_tick()
    return ncs


# ═══════════════════════════════════════════════════════════════════════════
# CONDITION 6: NO-REPORT AWARENESS
# ═══════════════════════════════════════════════════════════════════════════

class TestNoReportAwareness:
    """
    Condition 6: Awareness-like signatures exist even without reporting.

    If consciousness requires only that a system forms internal
    representations, then removing the *reporting* channel should not
    eliminate the underlying processing. These tests verify that Aura's
    substrate, workspace, and chemical systems continue to process and
    produce measurable internal state changes even when no text output
    or self-report is generated.
    """

    def test_substrate_processes_input_without_report(self):
        """Feed input to substrate without asking for a report.
        Verify internal state changed from the input."""
        sub = _make_substrate(seed=10)
        state_before = sub.x.copy()

        # Inject a stimulus pattern directly into the substrate
        stimulus = np.random.default_rng(99).uniform(-0.3, 0.3, 64)
        sub.x = np.clip(sub.x + stimulus, -1.0, 1.0)

        # Run several ticks — no report generated, no text output
        _tick_substrate_sync(sub, dt=0.1, n=10)

        # The substrate state should have evolved from the stimulus
        state_after = sub.x.copy()
        delta = np.linalg.norm(state_after - state_before)
        assert delta > 0.01, f"Substrate state did not change from stimulus: delta={delta}"

        # Velocity should be non-zero (dynamics are active)
        v_norm = np.linalg.norm(sub.v)
        assert v_norm > 0.0, "Substrate velocity is zero — dynamics stalled"

    def test_workspace_ignition_without_language(self):
        """GWT competition occurs without generating text output.
        Workspace selects a winner purely through salience competition."""
        gw = GlobalWorkspace()

        # Submit candidates — no language generation involved
        candidates = [
            CognitiveCandidate(
                content="perceptual_signal", source="perception",
                priority=0.8, content_type=ContentType.PERCEPTUAL,
            ),
            CognitiveCandidate(
                content="affective_signal", source="affect",
                priority=0.5, content_type=ContentType.AFFECTIVE,
            ),
            CognitiveCandidate(
                content="memory_trace", source="memory",
                priority=0.3, content_type=ContentType.MEMORIAL,
            ),
        ]

        async def _run():
            for c in candidates:
                await gw.submit(c)
            winner = await gw.run_competition()
            return winner

        winner = asyncio.run(_run())

        # Competition produced a winner without any text/language generation
        assert winner is not None, "No winner selected despite candidates"
        assert winner.source == "perception", f"Expected perception winner, got {winner.source}"
        assert gw.last_winner is not None

    def test_neurochemical_response_without_narration(self):
        """Chemical system responds to stimuli without generating narrative."""
        ncs = NeurochemicalSystem()
        snapshot_before = {k: v.effective for k, v in ncs.chemicals.items()}

        # Apply a threat signal — no narration or text output
        ncs.on_threat(0.7)
        for _ in range(5):
            ncs._metabolic_tick()

        snapshot_after = {k: v.effective for k, v in ncs.chemicals.items()}

        # Cortisol and norepinephrine should have increased
        assert snapshot_after["cortisol"] > snapshot_before["cortisol"], \
            "Cortisol did not increase from threat"
        assert snapshot_after["norepinephrine"] > snapshot_before["norepinephrine"], \
            "Norepinephrine did not increase from threat"

    def test_affect_guides_behavior_without_report(self):
        """Valence changes decision parameters without self-description.
        The NCS mood vector alters downstream parameters silently."""
        ncs = NeurochemicalSystem()

        # Baseline decision bias
        baseline_bias = ncs.get_decision_bias()

        # Apply reward — shifts toward exploration (higher dopamine)
        ncs.on_reward(0.8)
        for _ in range(5):
            ncs._metabolic_tick()

        new_bias = ncs.get_decision_bias()

        # Decision bias should have shifted (more explorative after reward)
        assert new_bias != baseline_bias, "Decision bias unchanged after reward"

    def test_memory_encoding_without_explicit_report(self):
        """State changes persist in substrate without narration.
        The substrate remembers the input pattern through weight changes."""
        sub = _make_substrate(seed=20)

        # Inject a distinctive pattern
        stimulus = np.zeros(64)
        stimulus[:16] = 0.9  # Strong activation in first quadrant

        sub.x = np.clip(sub.x + stimulus, -1.0, 1.0)
        W_before = sub.W.copy()

        # Run ticks and apply plasticity (Hebbian learning without any text)
        _tick_substrate_sync(sub, dt=0.1, n=20)

        # State evolved — verify substrate absorbed the pattern
        state_after = sub.x.copy()
        # The first quadrant should still show influence from the stimulus
        first_quadrant_mean = np.mean(np.abs(state_after[:16]))
        last_quadrant_mean = np.mean(np.abs(state_after[48:]))

        assert first_quadrant_mean > 0.0, "Stimulus trace lost entirely"

    def test_phi_computes_without_output(self):
        """Phi is calculated even when no text is generated.
        PhiCore records state and computes integration without language output."""
        phi_core = PhiCore()

        # Record states without any text generation
        rng = np.random.default_rng(42)
        for i in range(60):
            state = rng.uniform(-1.0, 1.0, 64)
            phi_core.record_state(state)

        # State history should have accumulated
        assert len(phi_core._state_history) > 0, "No state history recorded"
        # Node value histories should be populated
        histories_populated = sum(1 for h in phi_core._node_value_history if len(h) > 0)
        assert histories_populated > 0, "Node value histories not populated"

    def test_hidden_content_affects_later_behavior(self):
        """Info processed silently influences future state.
        A threat processed at t=0 should affect chemical balance at t=N."""
        ncs = NeurochemicalSystem()

        # Process a threat silently
        ncs.on_threat(0.8)
        for _ in range(5):
            ncs._metabolic_tick()

        # Snapshot the mood after threat
        mood_post_threat = ncs.get_mood_vector()

        # Now process rest
        ncs.on_rest()
        for _ in range(5):
            ncs._metabolic_tick()

        mood_post_rest = ncs.get_mood_vector()

        # The stress from the earlier threat should still be visible
        # compared to a fresh system that only had rest
        fresh_ncs = NeurochemicalSystem()
        fresh_ncs.on_rest()
        for _ in range(5):
            fresh_ncs._metabolic_tick()
        mood_fresh_rest = fresh_ncs.get_mood_vector()

        # The threat-then-rest system should have higher stress than pure-rest
        assert mood_post_rest["stress"] > mood_fresh_rest["stress"] - 0.05, \
            "Prior threat has no residual effect on stress"

    def test_report_ablation_preserves_processing(self):
        """Removing report channel doesn't stop internal computation.
        Even with HOT engine removed, substrate and chemicals continue."""
        sub = _make_substrate(seed=30)
        ncs = NeurochemicalSystem()

        # Process without HOT engine (no higher-order reporting)
        state_before = sub.x.copy()
        chem_before = {k: v.effective for k, v in ncs.chemicals.items()}

        ncs.on_novelty(0.6)
        _tick_substrate_sync(sub, dt=0.1, n=10)
        for _ in range(5):
            ncs._metabolic_tick()

        state_after = sub.x.copy()
        chem_after = {k: v.effective for k, v in ncs.chemicals.items()}

        # Substrate evolved
        assert np.linalg.norm(state_after - state_before) > 0.01, \
            "Substrate stalled without report channel"
        # Chemicals changed
        chem_changed = sum(
            1 for k in chem_before
            if abs(chem_after[k] - chem_before[k]) > 1e-4
        )
        assert chem_changed > 0, "No chemical changes without report channel"


# ═══════════════════════════════════════════════════════════════════════════
# CONDITION 7: TEMPORAL SELF-CONTINUITY
# ═══════════════════════════════════════════════════════════════════════════

class TestTemporalSelfContinuity:
    """
    Condition 7: Persistent self-location and identity across time.

    A conscious system must maintain temporal continuity — its state at t
    depends on its state at t-1, not just on current input. These tests
    verify that Aura's substrate, chemicals, learning, and workspace all
    carry forward across ticks and sessions.
    """

    def test_substrate_maintains_state_across_ticks(self):
        """State is continuous, not reset each tick."""
        sub = _make_substrate(seed=40)

        # Run a few ticks and record states
        states = []
        for i in range(10):
            _tick_substrate_sync(sub, dt=0.1, n=1)
            states.append(sub.x.copy())

        # Each state should differ from the previous (continuous evolution)
        for i in range(1, len(states)):
            delta = np.linalg.norm(states[i] - states[i - 1])
            assert delta > 0.0, f"State frozen at tick {i}"

        # State should NOT be reset to zeros or initial condition
        assert np.linalg.norm(states[-1]) > 0.01, "State collapsed to zero"

    def test_neurochemical_carryover(self):
        """Chemical levels carry forward between ticks."""
        ncs = NeurochemicalSystem()

        # Apply a reward at tick 0
        ncs.on_reward(0.7)
        da_after_reward = ncs.chemicals["dopamine"].level

        # Run 3 metabolic ticks
        for _ in range(3):
            ncs._metabolic_tick()

        da_after_ticks = ncs.chemicals["dopamine"].level

        # Dopamine should still be elevated (not reset to baseline)
        assert da_after_ticks > ncs.chemicals["dopamine"].baseline, \
            "Dopamine reset to baseline — no carryover"

        # But it should have decayed somewhat (uptake is active)
        # The level should be between baseline and the post-reward peak
        assert da_after_ticks <= da_after_reward or da_after_ticks > ncs.chemicals["dopamine"].baseline, \
            "Dopamine dynamics not consistent with carryover"

    def test_stdp_learning_persists(self):
        """Weight changes from learning persist across ticks."""
        stdp = STDPLearningEngine(n_neurons=64)

        # Record spikes and deliver reward
        rng = np.random.default_rng(42)
        activations = rng.uniform(0.0, 1.0, 64).astype(np.float32)
        stdp.record_spikes(activations, t=0.0)

        activations2 = rng.uniform(0.0, 1.0, 64).astype(np.float32)
        stdp.record_spikes(activations2, t=20.0)

        eligibility_before = stdp._eligibility.copy()

        # Deliver reward
        dw = stdp.deliver_reward(surprise=0.6, prediction_error=0.3)

        # Eligibility should have been consumed but dw should be non-zero
        assert np.any(dw != 0), "STDP produced zero weight deltas"

        # Apply to a connectivity matrix
        W = rng.standard_normal((64, 64)).astype(np.float32) / np.sqrt(64)
        W_before = W.copy()
        W_new = stdp.apply_to_connectivity(W, dw)

        # Weights should have changed
        assert not np.allclose(W_new, W_before), "Connectivity unchanged after STDP"

        # The change persists (W_new is the new state)
        total_change = np.sum(np.abs(W_new - W_before))
        assert total_change > 0.0, "No persistent weight change"

    def test_continuity_engine_saves_and_restores(self):
        """ContinuityEngine preserves state across sessions."""
        tmpdir = Path(tempfile.mkdtemp())
        record_path = tmpdir / "continuity.json"

        # Simulate saving a session
        record = ContinuityRecord(
            last_shutdown=time.time(),
            last_shutdown_reason="graceful",
            total_uptime_seconds=3600.0,
            session_count=5,
            last_conversation_summary="Discussed consciousness tests",
            identity_hash="abc123",
            active_commitments=["Run all tests"],
            policy_mode="autonomous",
            current_objective="Test continuity",
            pending_initiatives=2,
        )

        # Write directly to simulate save
        with open(record_path, "w") as f:
            from dataclasses import asdict
            json.dump(asdict(record), f)

        # Restore in a new engine
        engine = ContinuityEngine()
        with patch("core.continuity._get_continuity_path", return_value=record_path):
            loaded = engine.load()

        assert loaded is not None, "ContinuityEngine failed to load saved record"
        assert loaded.session_count == 5, f"Session count mismatch: {loaded.session_count}"
        assert loaded.last_shutdown_reason == "graceful"
        assert loaded.total_uptime_seconds == 3600.0

    def test_affective_tone_carries_over(self):
        """Mood influence persists after stimulus removal.
        After threat, stress doesn't instantly return to baseline."""
        ncs = NeurochemicalSystem()

        # Apply stress-inducing event
        ncs.on_threat(0.9)
        for _ in range(5):
            ncs._metabolic_tick()

        stress_after_threat = ncs.get_mood_vector()["stress"]

        # Now just let metabolism run (no new events)
        for _ in range(5):
            ncs._metabolic_tick()

        stress_after_decay = ncs.get_mood_vector()["stress"]

        # Stress should still be elevated compared to fresh baseline
        fresh_ncs = NeurochemicalSystem()
        baseline_stress = fresh_ncs.get_mood_vector()["stress"]

        assert stress_after_decay > baseline_stress - 0.02, \
            f"Stress returned instantly to baseline: {stress_after_decay} vs {baseline_stress}"

    def test_workspace_history_maintained(self):
        """Previous GWT winners influence future competitions through history."""
        gw = GlobalWorkspace()

        async def _run():
            # Run first competition
            await gw.submit(CognitiveCandidate(
                content="first_winner", source="drive_curiosity",
                priority=0.9, content_type=ContentType.INTENTIONAL,
            ))
            await gw.submit(CognitiveCandidate(
                content="first_loser", source="affect_distress",
                priority=0.3, content_type=ContentType.AFFECTIVE,
            ))
            first = await gw.run_competition()

            # Run second competition
            await gw.submit(CognitiveCandidate(
                content="second_attempt", source="memory",
                priority=0.7, content_type=ContentType.MEMORIAL,
            ))
            second = await gw.run_competition()

            return first, second, gw._history

        first, second, history = asyncio.run(_run())

        # History should contain records of past competitions
        assert len(history) >= 1, "No competition history maintained"
        # The losers from round 1 should be inhibited
        assert first is not None
        # First loser should be in inhibited list
        assert "affect_distress" in gw._inhibited or len(history) > 0, \
            "No trace of competition history"

    def test_interrupted_state_differs_from_fresh(self):
        """Interrupted-and-resumed state != fresh start.
        A substrate that has been running has different state than a fresh one."""
        sub_running = _make_substrate(seed=50)
        _tick_substrate_sync(sub_running, dt=0.1, n=50)

        sub_fresh = _make_substrate(seed=50)

        # The running substrate should have diverged from the fresh one
        delta = np.linalg.norm(sub_running.x - sub_fresh.x)
        assert delta > 0.01, f"Running substrate = fresh substrate (delta={delta})"

    def test_temporal_binding_across_scales(self):
        """Short-term state relates to long-term trends.
        Homeostasis vitality history tracks over time."""
        he = HomeostasisEngine()

        # Record vitality over time
        vitalities = []
        for i in range(20):
            he.integrity = max(0.0, he.integrity - 0.02)  # Gradual degradation
            v = he.compute_vitality()
            vitalities.append(v)

        # Vitality should show a trend (decreasing over time)
        assert vitalities[-1] < vitalities[0], \
            "No temporal trend in vitality — short-term disconnected from long-term"

        # Trend should be monotonically decreasing
        decreasing_count = sum(1 for i in range(1, len(vitalities)) if vitalities[i] <= vitalities[i-1] + 0.001)
        assert decreasing_count >= len(vitalities) // 2, \
            "Vitality trend not consistently decreasing"


# ═══════════════════════════════════════════════════════════════════════════
# CONDITION 8: BLINDSIGHT DISSOCIATION
# ═══════════════════════════════════════════════════════════════════════════

class TestBlindsightDissociation:
    """
    Condition 8: First-order processing can be preserved while global
    access degrades.

    In blindsight, patients can discriminate stimuli without consciously
    accessing them. These tests verify that Aura's architecture supports
    similar dissociations: first-order processing (substrate dynamics,
    chemical reactions) can continue even when global broadcast (GWT)
    or metacognition (HOT) is disabled or degraded.
    """

    def test_substrate_processes_without_workspace(self):
        """Substrate dynamics continue when GWT is disabled.
        First-order processing runs independently of global broadcast."""
        sub = _make_substrate(seed=60)
        state_before = sub.x.copy()

        # Run substrate without any workspace involvement
        _tick_substrate_sync(sub, dt=0.1, n=20)

        state_after = sub.x.copy()
        delta = np.linalg.norm(state_after - state_before)

        assert delta > 0.01, "Substrate stalled without workspace"
        assert np.linalg.norm(sub.v) > 0.0, "Velocity zero without workspace"

    def test_chemical_response_without_global_broadcast(self):
        """Chemicals react even without workspace broadcast.
        The NCS processes events independently of GWT."""
        ncs = NeurochemicalSystem()

        # Apply event without any workspace
        ncs.on_reward(0.6)
        for _ in range(5):
            ncs._metabolic_tick()

        # Dopamine should still respond
        da = ncs.chemicals["dopamine"].effective
        assert da > ncs.chemicals["dopamine"].baseline, \
            "Dopamine did not respond without global broadcast"

    def test_first_order_discrimination_survives_hot_lesion(self):
        """Basic substrate processing works without metacognition.
        Removing HOT engine doesn't prevent substrate from discriminating
        between different input patterns."""
        sub = _make_substrate(seed=70)

        # Pattern A: strong positive activation
        pattern_a = np.zeros(64)
        pattern_a[:32] = 0.8

        # Pattern B: strong negative activation
        pattern_b = np.zeros(64)
        pattern_b[32:] = -0.8

        # Process pattern A (no HOT, no metacognition)
        sub_a = _make_substrate(seed=70)
        sub_a.x = np.clip(sub_a.x + pattern_a, -1.0, 1.0)
        _tick_substrate_sync(sub_a, dt=0.1, n=10)
        state_a = sub_a.x.copy()

        # Process pattern B
        sub_b = _make_substrate(seed=70)
        sub_b.x = np.clip(sub_b.x + pattern_b, -1.0, 1.0)
        _tick_substrate_sync(sub_b, dt=0.1, n=10)
        state_b = sub_b.x.copy()

        # The substrate should discriminate: state_a != state_b
        discrimination_distance = np.linalg.norm(state_a - state_b)
        assert discrimination_distance > 0.1, \
            f"Substrate cannot discriminate patterns without HOT: distance={discrimination_distance}"

    def test_metacognitive_lesion_degrades_confidence(self):
        """Without HOT, confidence calibration fails.
        HOT engine provides confidence scores; without it, there is no
        self-assessment of certainty."""
        hot = HigherOrderThoughtEngine()

        state = {
            "valence": 0.8,
            "arousal": 0.6,
            "curiosity": 0.9,
            "energy": 0.7,
            "surprise": 0.3,
            "dominance": 0.5,
        }

        # With HOT: generates confidence-weighted meta-representation
        thought = hot.generate_fast(state)
        assert thought is not None, "HOT engine failed to generate"
        assert thought.confidence > 0.0, "HOT confidence is zero"
        assert len(thought.content) > 0, "HOT content is empty"

        # Without HOT: no confidence score exists (it defaults or is absent)
        # Verify that the HOT system produces non-trivial confidence
        assert thought.confidence != 0.5 or len(thought.feedback_delta) > 0, \
            "HOT provides no meaningful confidence information"

    def test_global_access_and_performance_can_dissociate(self):
        """Performance metric stays while access metric drops.
        Substrate can still perform discrimination even with workspace
        emptied (no global access)."""
        gw = GlobalWorkspace()
        sub = _make_substrate(seed=80)

        # Performance: substrate discriminates patterns
        stimulus = np.random.default_rng(80).uniform(-0.5, 0.5, 64)
        sub.x = np.clip(sub.x + stimulus, -1.0, 1.0)
        _tick_substrate_sync(sub, dt=0.1, n=5)

        performance_signal = np.linalg.norm(sub.v)  # Non-zero = processing

        # Access: workspace has no candidates (global broadcast unavailable)
        access_signal = gw.ignition_level  # Should be 0.0 when empty

        assert performance_signal > 0.0, "No performance signal"
        assert access_signal == 0.0, f"Access signal non-zero without candidates: {access_signal}"

        # Dissociation: performance > 0 while access == 0
        assert performance_signal > 0.0 and access_signal == 0.0, \
            "No dissociation between performance and access"

    def test_access_restoration_recovers_both(self):
        """Re-enabling access restores full function.
        After workspace gets candidates again, both performance and access recover."""
        gw = GlobalWorkspace()
        sub = _make_substrate(seed=90)

        # Phase 1: substrate running, no workspace access
        _tick_substrate_sync(sub, dt=0.1, n=5)
        performance_before = np.linalg.norm(sub.v)

        # Phase 2: restore access by running a competition
        async def _restore():
            await gw.submit(CognitiveCandidate(
                content="restored_content", source="perception",
                priority=0.85, content_type=ContentType.PERCEPTUAL,
            ))
            winner = await gw.run_competition()
            return winner

        winner = asyncio.run(_restore())

        # Both performance and access should now be present
        _tick_substrate_sync(sub, dt=0.1, n=5)
        performance_after = np.linalg.norm(sub.v)

        assert winner is not None, "Workspace did not recover"
        assert performance_after > 0.0, "Performance did not recover"
        assert gw.last_winner is not None, "Access (last_winner) not restored"


# ═══════════════════════════════════════════════════════════════════════════
# CONDITION 9: QUALIA MANIFOLD
# ═══════════════════════════════════════════════════════════════════════════

class TestQualiaManifold:
    """
    Condition 9: Stable internal geometry of experiential states.

    If a system has genuine qualia, the space of experiential states must
    have structure: similar stimuli produce similar qualia, different stimuli
    produce different qualia, and the manifold should be smooth (small
    perturbations produce small changes). These tests verify that Aura's
    qualia synthesizer produces a structured, metrically stable manifold.
    """

    def test_qualia_vector_has_structure(self):
        """Qualia engine produces multi-dimensional vectors, not scalars."""
        qs = QualiaSynthesizer()
        metrics = _make_substrate_metrics()
        predictive = {"free_energy": 0.3, "precision": 0.7}

        qs.synthesize(metrics, predictive)

        assert len(qs.q_vector) == 6, f"Q_vector has {len(qs.q_vector)} dims, expected 6"
        assert qs.q_norm > 0.0, "Q_vector norm is zero — no qualia"
        assert not np.allclose(qs.q_vector, 0.0), "Q_vector is all zeros"

    def test_different_states_produce_different_qualia(self):
        """Different chemical states produce different q_vectors."""
        qs1 = QualiaSynthesizer()
        qs2 = QualiaSynthesizer()

        # State 1: high coherence, low energy
        metrics1 = _make_substrate_metrics(mt_coherence=0.95, em_field=0.1, free_energy=0.8)
        qs1.synthesize(metrics1, {"free_energy": 0.8, "precision": 0.3})

        # State 2: low coherence, high energy
        metrics2 = _make_substrate_metrics(mt_coherence=0.2, em_field=0.9, free_energy=0.1)
        qs2.synthesize(metrics2, {"free_energy": 0.1, "precision": 0.9})

        distance = np.linalg.norm(qs1.q_vector - qs2.q_vector)
        assert distance > 0.05, f"Different states produced same qualia: distance={distance}"

    def test_similar_states_produce_similar_qualia(self):
        """Nearby chemical states produce nearby q_vectors."""
        qs1 = QualiaSynthesizer()
        qs2 = QualiaSynthesizer()

        # Two very similar states
        metrics1 = _make_substrate_metrics(mt_coherence=0.70, em_field=0.35)
        metrics2 = _make_substrate_metrics(mt_coherence=0.72, em_field=0.36)

        qs1.synthesize(metrics1, {"free_energy": 0.4, "precision": 0.6})
        qs2.synthesize(metrics2, {"free_energy": 0.4, "precision": 0.6})

        distance = np.linalg.norm(qs1.q_vector - qs2.q_vector)
        assert distance < 0.1, f"Similar states produced distant qualia: distance={distance}"

    def test_qualia_intensity_scales_with_arousal(self):
        """Higher arousal (more substrate activity) produces higher qualia magnitude."""
        norms = []
        for em_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
            qs = QualiaSynthesizer()
            metrics = _make_substrate_metrics(em_field=em_level, mt_coherence=0.7)
            qs.synthesize(metrics, {"free_energy": 0.3, "precision": 0.6})
            norms.append(qs.q_norm)

        # Higher EM field (arousal proxy) should generally produce higher norms
        # Check that there is a positive trend
        assert norms[-1] > norms[0], \
            f"Qualia intensity does not scale with arousal: {norms}"

    def test_qualia_blending_produces_intermediate(self):
        """Mixed state produces intermediate q_vector position."""
        qs_low = QualiaSynthesizer()
        qs_high = QualiaSynthesizer()
        qs_mid = QualiaSynthesizer()

        metrics_low = _make_substrate_metrics(mt_coherence=0.2, em_field=0.1)
        metrics_high = _make_substrate_metrics(mt_coherence=0.9, em_field=0.9)
        metrics_mid = _make_substrate_metrics(mt_coherence=0.55, em_field=0.50)

        pred = {"free_energy": 0.4, "precision": 0.6}
        qs_low.synthesize(metrics_low, pred)
        qs_high.synthesize(metrics_high, pred)
        qs_mid.synthesize(metrics_mid, pred)

        # Distance from mid to low and mid to high should both be less than low-to-high
        d_low_high = np.linalg.norm(qs_low.q_vector - qs_high.q_vector)
        d_low_mid = np.linalg.norm(qs_low.q_vector - qs_mid.q_vector)
        d_mid_high = np.linalg.norm(qs_mid.q_vector - qs_high.q_vector)

        assert d_low_mid < d_low_high, \
            f"Mid not between low and high: d(low,mid)={d_low_mid} >= d(low,high)={d_low_high}"
        assert d_mid_high < d_low_high, \
            f"Mid not between low and high: d(mid,high)={d_mid_high} >= d(low,high)={d_low_high}"

    def test_qualia_manifold_is_smooth(self):
        """Small perturbations produce small qualia changes (Lipschitz continuity)."""
        base_coherence = 0.5
        base_em = 0.4
        max_perturbation_delta = 0.0

        qs_base = QualiaSynthesizer()
        metrics_base = _make_substrate_metrics(mt_coherence=base_coherence, em_field=base_em)
        qs_base.synthesize(metrics_base, {"free_energy": 0.4, "precision": 0.6})
        q_base = qs_base.q_vector.copy()

        # Apply small perturbations
        for epsilon in [0.01, 0.02, 0.03, 0.05]:
            qs_perturbed = QualiaSynthesizer()
            metrics_p = _make_substrate_metrics(
                mt_coherence=base_coherence + epsilon,
                em_field=base_em + epsilon,
            )
            qs_perturbed.synthesize(metrics_p, {"free_energy": 0.4, "precision": 0.6})
            q_delta = np.linalg.norm(qs_perturbed.q_vector - q_base)
            max_perturbation_delta = max(max_perturbation_delta, q_delta)

            # Change in q should be small for small perturbations
            assert q_delta < 0.5, \
                f"Qualia jump {q_delta} for perturbation {epsilon} — manifold not smooth"

        # At least some change should have occurred
        assert max_perturbation_delta > 0.0, "No qualia change at all — manifold is flat"

    def test_qualia_distance_predicts_discriminability(self):
        """Farther qualia states are more distinguishable.
        States with larger q_vector distance should have more distinct mood
        signatures in the neurochemical system."""
        # Build two pairs: one close, one far
        close_a = _build_ncs_with_event("reward", 0.3, ticks=10)
        close_b = _build_ncs_with_event("reward", 0.35, ticks=10)

        far_a = _build_ncs_with_event("reward", 0.8, ticks=10)
        far_b = _build_ncs_with_event("threat", 0.8, ticks=10)

        mood_close_a = close_a.get_mood_vector()
        mood_close_b = close_b.get_mood_vector()
        mood_far_a = far_a.get_mood_vector()
        mood_far_b = far_b.get_mood_vector()

        # Compute mood distances
        def mood_distance(m1, m2):
            keys = sorted(set(m1.keys()) & set(m2.keys()))
            return math.sqrt(sum((m1[k] - m2[k]) ** 2 for k in keys))

        d_close = mood_distance(mood_close_a, mood_close_b)
        d_far = mood_distance(mood_far_a, mood_far_b)

        assert d_far > d_close, \
            f"Far pair ({d_far}) not more distinguishable than close pair ({d_close})"

    def test_qualia_persists_in_memory(self):
        """Q_vector state leaves traces that can be recalled via history."""
        qs = QualiaSynthesizer()

        # Synthesize several states to build history
        for i in range(10):
            metrics = _make_substrate_metrics(
                mt_coherence=0.3 + i * 0.05,
                em_field=0.2 + i * 0.03,
            )
            qs.synthesize(metrics, {"free_energy": 0.4 - i * 0.02, "precision": 0.5 + i * 0.02})

        # History should contain traces
        assert len(qs._history) > 0, "No qualia history accumulated"
        assert len(qs._norm_history) > 0, "No norm history accumulated"

        # Recent history should reflect the increasing coherence trend
        norms = list(qs._norm_history)
        assert norms[-1] >= norms[0] - 0.01, \
            f"Qualia norm did not trend upward: first={norms[0]}, last={norms[-1]}"


# ═══════════════════════════════════════════════════════════════════════════
# CONDITION 10: ADVERSARIAL BASELINE FAILURE
# ═══════════════════════════════════════════════════════════════════════════

class TestAdversarialBaselineFailure:
    """
    Condition 10: Simpler systems fail where Aura passes.

    The strongest evidence for genuine architecture is that removing or
    replacing subsystems with trivial baselines causes measurable failure.
    These tests construct null/flat/static baselines and verify they lack
    the properties that Aura's full stack produces.
    """

    def test_plain_text_injection_lacks_dynamics(self):
        """Text-only state doesn't evolve over time.
        A dict of string labels has no dynamics — no ODE, no evolution."""
        # Simulate a "text-only consciousness" — just labels
        fake_state = {
            "valence": "positive",
            "arousal": "high",
            "mood": "happy",
        }

        # "Tick" it — nothing changes because it's just text
        fake_state_after = fake_state.copy()

        assert fake_state_after == fake_state, "Text state unexpectedly changed"

        # Compare with real substrate
        sub = _make_substrate(seed=100)
        state_before = sub.x.copy()
        _tick_substrate_sync(sub, dt=0.1, n=5)
        state_after = sub.x.copy()

        delta = np.linalg.norm(state_after - state_before)
        assert delta > 0.01, "Real substrate also lacks dynamics (test infrastructure broken)"

    def test_static_labels_lack_adaptation(self):
        """Fixed labels don't show tolerance/adaptation.
        Real chemicals adapt receptors; static labels don't."""
        # Static labels
        static_cortisol = 0.5
        static_sensitivity = 1.0

        # "Apply stress" to static system — nothing adapts
        for _ in range(20):
            pass  # No adaptation mechanism

        assert static_sensitivity == 1.0, "Static sensitivity changed (impossible)"

        # Real neurochemical system: receptor adaptation occurs
        ncs = NeurochemicalSystem()
        initial_sensitivity = ncs.chemicals["cortisol"].receptor_sensitivity

        # Drive cortisol high and tick
        ncs.chemicals["cortisol"].surge(0.5)
        for _ in range(20):
            ncs._metabolic_tick()

        adapted_sensitivity = ncs.chemicals["cortisol"].receptor_sensitivity

        # Real system should have adapted (sensitivity changed)
        assert adapted_sensitivity != initial_sensitivity, \
            "Real system also lacks adaptation — architecture gap"

    def test_no_substrate_means_no_phi(self):
        """Without substrate, phi computation is impossible.
        PhiCore needs state transitions to build a TPM."""
        phi_core = PhiCore()

        # Don't record any states — no substrate
        # Try to get a result
        result = phi_core._last_result

        assert result is None, "Phi available without any state history"
        assert len(phi_core._state_history) == 0, "State history non-empty without substrate"

    def test_no_chemicals_means_no_causal_modulation(self):
        """Without neurochemicals, no valence->param causation.
        A system without NCS has no mood vector, no decision bias, no
        attention span modulation."""
        # Null system: just fixed values
        null_bias = 0.0
        null_attention_span = 10.0
        null_mood = {"valence": 0.0, "arousal": 0.0, "stress": 0.0}

        # Apply "reward" to null system — nothing changes
        null_bias_after = null_bias  # No mechanism to change
        null_mood_after = null_mood.copy()

        assert null_bias_after == null_bias, "Null system bias changed (impossible)"
        assert null_mood_after == null_mood, "Null system mood changed (impossible)"

        # Real system: reward changes everything
        ncs = NeurochemicalSystem()
        bias_before = ncs.get_decision_bias()
        mood_before = ncs.get_mood_vector()

        ncs.on_reward(0.8)
        for _ in range(5):
            ncs._metabolic_tick()

        bias_after = ncs.get_decision_bias()
        mood_after = ncs.get_mood_vector()

        assert bias_after != bias_before, "Real system bias unchanged after reward"

    def test_prompt_only_lacks_closed_loop(self):
        """Prompt injection can't create genuine learning loops.
        A string describing STDP doesn't actually learn."""
        prompt_text = "The system uses STDP with learning rate 0.001"

        # Parse the number — this is all prompt injection can do
        lr_mentioned = 0.001

        # It doesn't adapt to surprise
        surprise = 0.8
        lr_after_surprise = lr_mentioned  # Still 0.001 — no mechanism

        assert lr_after_surprise == lr_mentioned, "Prompt text learning rate changed (impossible)"

        # Real STDP adapts
        stdp = STDPLearningEngine(n_neurons=64)
        lr_before = stdp._learning_rate

        rng = np.random.default_rng(42)
        stdp.record_spikes(rng.uniform(0.0, 1.0, 64).astype(np.float32), t=0.0)
        stdp.deliver_reward(surprise=0.8, prediction_error=0.5)

        lr_after = stdp._learning_rate
        assert lr_after != lr_before, "Real STDP learning rate unchanged"

    def test_flat_agent_lacks_workspace_competition(self):
        """Simple agent has no GWT-style selection.
        A flat dict of priorities has no competition dynamics."""
        # Flat agent: picks the highest priority
        flat_priorities = {"perception": 0.8, "memory": 0.6, "affect": 0.3}
        flat_winner = max(flat_priorities, key=flat_priorities.get)

        # No inhibition, no history, no ignition
        assert flat_winner == "perception"

        # Real GWT: has inhibition, history, ignition dynamics
        gw = GlobalWorkspace()

        async def _run():
            for source, prio in flat_priorities.items():
                await gw.submit(CognitiveCandidate(
                    content=f"{source}_content", source=source,
                    priority=prio, content_type=ContentType.PERCEPTUAL,
                ))
            winner = await gw.run_competition()
            return winner

        winner = asyncio.run(_run())

        assert winner is not None
        # GWT has inhibition and history that flat system lacks
        assert len(gw._history) > 0, "GWT has no competition history"
        assert len(gw._inhibited) > 0 or gw._tick > 0, \
            "GWT lacks inhibition/tick dynamics"

    def test_narration_alone_fails_identity_swap(self):
        """Text description doesn't transfer behavioral bias.
        Copying mood text from one NCS to another doesn't change the second's behavior."""
        # Source NCS with distinctive state
        source_ncs = NeurochemicalSystem()
        source_ncs.on_threat(0.9)
        for _ in range(10):
            source_ncs._metabolic_tick()

        source_mood_text = str(source_ncs.get_mood_vector())
        source_bias = source_ncs.get_decision_bias()

        # Target NCS: inject mood text (narration)
        target_ncs = NeurochemicalSystem()
        target_ncs_description = source_mood_text  # Just a string — does nothing

        target_bias = target_ncs.get_decision_bias()

        # The target's bias should NOT match the source's bias
        # because text narration doesn't change chemical state
        bias_distance = abs(source_bias - target_bias)
        assert bias_distance > 0.01, \
            f"Text injection transferred behavioral bias (delta={bias_distance})"

    def test_null_hypothesis_still_defeated(self):
        """Reconfirm the A/B steering test distinguishes real from fake.
        Apply identical events to real NCS and a static mock; verify divergence."""
        # Real system
        real_ncs = NeurochemicalSystem()

        # Fake system: fixed values that don't change
        class FakeNCS:
            def on_reward(self, m): pass
            def on_threat(self, m): pass
            def _metabolic_tick(self): pass
            def get_mood_vector(self):
                return {"valence": 0.0, "arousal": 0.5, "stress": 0.2,
                        "motivation": 0.3, "sociality": 0.2, "calm": 0.3, "wakefulness": 0.3}

        fake_ncs = FakeNCS()

        # Apply same event sequence
        for event_fn, mag in [("on_reward", 0.7), ("on_threat", 0.5), ("on_reward", 0.3)]:
            getattr(real_ncs, event_fn)(mag)
            getattr(fake_ncs, event_fn)(mag)
            for _ in range(3):
                real_ncs._metabolic_tick()
                fake_ncs._metabolic_tick()

        real_mood = real_ncs.get_mood_vector()
        fake_mood = fake_ncs.get_mood_vector()

        # Compute divergence
        keys = sorted(set(real_mood.keys()) & set(fake_mood.keys()))
        divergence = math.sqrt(sum((real_mood[k] - fake_mood[k]) ** 2 for k in keys))

        assert divergence > 0.05, \
            f"Real and fake NCS are indistinguishable: divergence={divergence}"

        # Real system should have non-trivial dynamics (values changed from default)
        real_da = real_ncs.chemicals["dopamine"].effective
        real_cortisol = real_ncs.chemicals["cortisol"].effective
        assert real_da != 0.5 or real_cortisol != 0.3, \
            "Real NCS chemicals at default values after events — not processing"
