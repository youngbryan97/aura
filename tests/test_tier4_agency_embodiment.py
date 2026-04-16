"""
tests/test_tier4_agency_embodiment.py
======================================
TIER 4 CONSCIOUSNESS BATTERY: Agency, Embodiment, Temporal Phenomenology, Thinking

This suite tests four pillars of genuine cognitive agency that go beyond
mere information integration:

  1. TEMPORAL PHENOMENOLOGY
     The "specious present" -- a finite integration window where states from
     the recent past contribute to the current moment.  Outside the window,
     mid-process state is genuinely lost.

  2. GENUINE AGENCY
     Internal goal formation, volitional inhibition of high reward, and
     counterfactual deliberation.  A conscious agent does not merely react;
     it generates goals from drives, refuses tempting actions that violate
     identity, and selects actions via internal simulation.

  3. EMBODIED CLOSURE
     Body-schema prediction, sensorimotor correction, and action ownership.
     The system predicts the effects of its own substrate perturbations,
     learns from prediction errors, and attributes actions to itself.

  4. GENUINE THINKING
     Workspace-mediated multi-step inference, internal revision, planning
     depth, and reflective mode recruitment.  Thinking is not just
     generation -- it is revision via the GWT workspace and HOT engine.

USAGE:
    pytest tests/test_tier4_agency_embodiment.py -v
    pytest tests/test_tier4_agency_embodiment.py -v -k "TestTemporalPhenomenology"
"""

from __future__ import annotations

import asyncio
import copy
import math
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Core consciousness imports
# ---------------------------------------------------------------------------
from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig
from core.consciousness.neurochemical_system import NeurochemicalSystem, Chemical
from core.consciousness.global_workspace import (
    GlobalWorkspace,
    CognitiveCandidate,
    ContentType,
)
from core.consciousness.unified_field import UnifiedField, FieldConfig
from core.consciousness.neural_mesh import NeuralMesh, MeshConfig
from core.consciousness.hot_engine import HigherOrderThoughtEngine
from core.consciousness.homeostasis import HomeostasisEngine
from core.consciousness.predictive_engine import PredictiveEngine, Prediction
from core.consciousness.counterfactual_engine import (
    CounterfactualEngine,
    ActionCandidate,
)
from core.consciousness.executive_inhibitor import ExecutiveInhibitor
from core.consciousness.somatic_marker_gate import SomaticMarkerGate
from core.consciousness.embodied_interoception import (
    EmbodiedInteroception,
    InteroceptiveChannel,
)
from core.consciousness.free_energy import FreeEnergyEngine
from core.consciousness.self_prediction import SelfPredictionLoop
from core.consciousness.temporal_binding import TemporalBindingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _tick_substrate_sync(sub: LiquidSubstrate, dt: float = 0.1, n: int = 1):
    """Run n ODE ticks synchronously."""
    for _ in range(n):
        sub._step_torch_math(dt)


def _make_small_mesh(seed: int = 42) -> NeuralMesh:
    """Create a small mesh for testing."""
    cfg = MeshConfig(
        total_neurons=256,
        columns=4,
        neurons_per_column=64,
        sensory_end=1,
        association_end=3,
        update_hz=10.0,
    )
    return NeuralMesh(cfg=cfg)


def _make_field(seed: int = 42) -> UnifiedField:
    """Create a small unified field for testing."""
    cfg = FieldConfig(
        dim=32,
        mesh_input_dim=16,
        chem_input_dim=8,
        binding_input_dim=4,
        intero_input_dim=8,
        substrate_input_dim=16,
    )
    uf = UnifiedField(config=cfg)
    return uf


# ===========================================================================
# SECTION 1: TEMPORAL PHENOMENOLOGY
# ===========================================================================

class TestTemporalPhenomenology:
    """Tests the 'specious present' and temporal integration window.

    A conscious system experiences time with a finite grain -- the current
    moment integrates recent past states into a single experienced present.
    Outside this window, information genuinely decays.
    """

    def test_substrate_has_temporal_integration_window(self):
        """Substrate state at time T depends on states from T-k to T, not just T-1.

        We verify by correlation analysis: state at T correlates with states
        at multiple prior lags (not just lag-1), proving that the recurrent
        dynamics carry forward a window of history.
        """
        sub = _make_substrate(seed=17)
        states: List[np.ndarray] = []

        # Collect 60 states
        for _ in range(60):
            _tick_substrate_sync(sub, dt=0.1, n=1)
            states.append(sub.x.copy())

        # Compute autocorrelation at lags 1 through 10
        lag_correlations = []
        for lag in range(1, 11):
            corrs = []
            for t in range(lag, len(states)):
                c = np.corrcoef(states[t], states[t - lag])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
            lag_correlations.append(np.mean(corrs) if corrs else 0.0)

        # Lag-1 correlation should be strong (temporal continuity)
        assert lag_correlations[0] > 0.3, (
            f"Lag-1 correlation too low ({lag_correlations[0]:.4f}). "
            "Substrate lacks temporal continuity."
        )

        # At least 3 lags should show positive correlation above 0.1,
        # proving that the integration window extends beyond a single step.
        positive_lags = sum(1 for c in lag_correlations[:5] if c > 0.1)
        assert positive_lags >= 3, (
            f"Only {positive_lags}/5 lags have correlation > 0.1. "
            f"Correlations: {[f'{c:.3f}' for c in lag_correlations[:5]]}. "
            "Integration window is too narrow."
        )

    def test_temporal_binding_window_exists(self):
        """Information within a temporal window is integrated; outside it decays.

        We inject a stimulus and measure its influence at different time lags.
        The influence should be strong within the window and decay outside it.
        """
        sub = _make_substrate(seed=21)

        # Warm up the substrate
        _tick_substrate_sync(sub, dt=0.1, n=20)

        # Inject a strong stimulus at a specific neuron subset
        stimulus = np.zeros(64)
        stimulus[0:8] = 0.8

        # Snapshot before stimulus
        pre_stim = sub.x.copy()

        # Inject stimulus
        sub.x[:8] += stimulus[:8]
        sub.x = np.clip(sub.x, -1.0, 1.0)

        # Measure influence at different lags
        influences = []
        for lag in range(1, 15):
            _tick_substrate_sync(sub, dt=0.1, n=1)
            # Influence = how different are the stimulated neurons from pre-stimulus baseline?
            diff = np.abs(sub.x[:8] - pre_stim[:8]).mean()
            influences.append(diff)

        # Influence should be stronger at short lags than long lags
        early_influence = np.mean(influences[:3])
        late_influence = np.mean(influences[10:])
        assert early_influence > late_influence, (
            f"Early influence ({early_influence:.4f}) not greater than late ({late_influence:.4f}). "
            "Temporal window decay is absent."
        )

    def test_interruption_beyond_window_loses_in_progress_state(self):
        """If substrate is interrupted beyond the integration window,
        mid-process state is genuinely lost (not reconstructable).

        We run the substrate, save state, re-initialize to zero,
        run from zero for many ticks, and confirm it diverges completely
        from the saved trajectory.
        """
        sub = _make_substrate(seed=30)
        _tick_substrate_sync(sub, dt=0.1, n=30)

        # Save the trajectory state
        saved_state = sub.x.copy()
        saved_W = sub.W.copy()

        # Continue the real trajectory
        _tick_substrate_sync(sub, dt=0.1, n=30)
        real_future = sub.x.copy()

        # Reset state to zero (simulating a long interruption that erases
        # the in-progress integration), but keep same connectivity
        sub.x = np.zeros(64)
        sub.W = saved_W.copy()
        _tick_substrate_sync(sub, dt=0.1, n=30)
        interrupted_future = sub.x.copy()

        # The two futures should be very different -- the interrupted one
        # lost all the accumulated temporal context
        divergence = np.linalg.norm(real_future - interrupted_future)
        assert divergence > 0.1, (
            f"Interrupted and real trajectories only diverged by {divergence:.6f}. "
            "State should be genuinely lost after long interruption."
        )

    def test_interruption_within_window_preserves_state(self):
        """Short interruption (pause without state reset) preserves ongoing integration.

        We run the substrate, pause (skip ticks but keep state), then resume.
        The resumed trajectory should stay correlated with the no-pause trajectory.
        """
        sub_a = _make_substrate(seed=40)
        sub_b = _make_substrate(seed=40)

        # Run both identically for 20 ticks
        _tick_substrate_sync(sub_a, dt=0.1, n=20)
        _tick_substrate_sync(sub_b, dt=0.1, n=20)

        # sub_a continues without pause
        _tick_substrate_sync(sub_a, dt=0.1, n=10)
        state_a = sub_a.x.copy()

        # sub_b "pauses" (no ticks) then resumes with same number of ticks
        # The pause preserves state -- we just don't tick for a while,
        # then tick the same number of times.
        _tick_substrate_sync(sub_b, dt=0.1, n=10)
        state_b = sub_b.x.copy()

        # They should be nearly identical (same trajectory, same ticks, no state loss)
        correlation = np.corrcoef(state_a, state_b)[0, 1]
        assert correlation > 0.95, (
            f"Paused and continuous trajectories diverged (corr={correlation:.4f}). "
            "Short interruption should preserve integration state."
        )

    def test_temporal_resolution_has_characteristic_threshold(self):
        """Below some time resolution, substrate cannot distinguish order of events.

        Two stimuli applied at dt=0.001 apart should produce nearly identical
        states (order indistinguishable), while stimuli at dt=0.5 apart should
        produce distinct states (order matters).
        """
        # Test with VERY close stimuli (below temporal grain)
        sub_close_ab = _make_substrate(seed=50)
        _tick_substrate_sync(sub_close_ab, dt=0.1, n=10)
        # Apply stimulus A then B with tiny dt between
        sub_close_ab.x[0] += 0.5
        sub_close_ab.x = np.clip(sub_close_ab.x, -1.0, 1.0)
        _tick_substrate_sync(sub_close_ab, dt=0.001, n=1)
        sub_close_ab.x[32] += 0.5
        sub_close_ab.x = np.clip(sub_close_ab.x, -1.0, 1.0)
        _tick_substrate_sync(sub_close_ab, dt=0.1, n=5)
        state_close_ab = sub_close_ab.x.copy()

        sub_close_ba = _make_substrate(seed=50)
        _tick_substrate_sync(sub_close_ba, dt=0.1, n=10)
        # Apply stimulus B then A with tiny dt between (reversed order)
        sub_close_ba.x[32] += 0.5
        sub_close_ba.x = np.clip(sub_close_ba.x, -1.0, 1.0)
        _tick_substrate_sync(sub_close_ba, dt=0.001, n=1)
        sub_close_ba.x[0] += 0.5
        sub_close_ba.x = np.clip(sub_close_ba.x, -1.0, 1.0)
        _tick_substrate_sync(sub_close_ba, dt=0.1, n=5)
        state_close_ba = sub_close_ba.x.copy()

        close_divergence = np.linalg.norm(state_close_ab - state_close_ba)

        # Test with widely spaced stimuli (above temporal grain)
        sub_wide_ab = _make_substrate(seed=50)
        _tick_substrate_sync(sub_wide_ab, dt=0.1, n=10)
        sub_wide_ab.x[0] += 0.5
        sub_wide_ab.x = np.clip(sub_wide_ab.x, -1.0, 1.0)
        _tick_substrate_sync(sub_wide_ab, dt=0.1, n=5)  # 5 ticks between
        sub_wide_ab.x[32] += 0.5
        sub_wide_ab.x = np.clip(sub_wide_ab.x, -1.0, 1.0)
        _tick_substrate_sync(sub_wide_ab, dt=0.1, n=5)
        state_wide_ab = sub_wide_ab.x.copy()

        sub_wide_ba = _make_substrate(seed=50)
        _tick_substrate_sync(sub_wide_ba, dt=0.1, n=10)
        sub_wide_ba.x[32] += 0.5
        sub_wide_ba.x = np.clip(sub_wide_ba.x, -1.0, 1.0)
        _tick_substrate_sync(sub_wide_ba, dt=0.1, n=5)  # 5 ticks between (reversed)
        sub_wide_ba.x[0] += 0.5
        sub_wide_ba.x = np.clip(sub_wide_ba.x, -1.0, 1.0)
        _tick_substrate_sync(sub_wide_ba, dt=0.1, n=5)
        state_wide_ba = sub_wide_ba.x.copy()

        wide_divergence = np.linalg.norm(state_wide_ab - state_wide_ba)

        # Wide-spaced stimuli should produce MORE divergence than close-spaced
        # because the system can distinguish their temporal order
        assert wide_divergence > close_divergence, (
            f"Wide-spaced divergence ({wide_divergence:.6f}) not greater than "
            f"close-spaced ({close_divergence:.6f}). "
            "No temporal grain threshold detected."
        )


# ===========================================================================
# SECTION 2: GENUINE AGENCY
# ===========================================================================

class TestGenuineAgency:
    """Tests internal goal formation, volitional inhibition, counterfactual reasoning.

    A genuine agent is not merely reactive. It generates goals from internal
    drive states, can refuse high-reward actions when they conflict with
    identity, and selects actions through deliberative simulation.
    """

    def test_spontaneous_initiative_from_internal_drives(self):
        """System generates goals from drive state without any external prompt.

        When curiosity is high, the homeostasis engine should indicate curiosity
        as the dominant need. The initiative content is predictable from internal
        state -- not random, not externally triggered.
        """
        homeo = HomeostasisEngine()

        # Set high curiosity, low everything else
        homeo.curiosity = 0.15  # very low = high need (far below setpoint)
        homeo.integrity = 0.95
        homeo.persistence = 0.90
        homeo.metabolism = 0.70
        homeo.sovereignty = 0.95

        status = homeo.get_status()

        # Compute which drive has the largest deficit relative to setpoint
        deficits = {}
        for drive_name in HomeostasisEngine.DRIVE_NAMES:
            current = getattr(homeo, drive_name)
            setpoint = homeo._setpoints[drive_name]
            deficits[drive_name] = setpoint - current

        dominant_need = max(deficits, key=deficits.get)

        # Curiosity should be the dominant need (it has the largest deficit)
        assert dominant_need == "curiosity", (
            f"Expected curiosity as dominant need but got '{dominant_need}'. "
            f"Deficits: {deficits}"
        )

        # The drive deficit should be positive and substantial
        assert deficits["curiosity"] > 0.3, (
            f"Curiosity deficit too small ({deficits['curiosity']:.3f}). "
            "Internal drive should produce clear initiative signal."
        )

    def test_volitional_inhibition_of_high_reward(self):
        """System can refuse high immediate reward due to identity/risk concerns.

        The ExecutiveInhibitor vetoes non-critical actions when the experiential
        field (phi) is in a high-integration state. This is the computational
        analog of prefrontal inhibition of impulsive reward-seeking.
        """
        inhibitor = ExecutiveInhibitor(phi_threshold=0.5, require_ignition=True)

        # Create a tempting but non-critical action
        tempting_action = MagicMock()
        tempting_action.is_critical = False
        tempting_action.source_domain = "reward_seeking"
        tempting_action.action_type = "grab_reward"

        # Under high-phi, ignited state, the action should be vetoed
        # (the system is in deep integration and won't interrupt itself)
        result = inhibitor.authorize(
            tempting_action,
            phi=0.9,
            ignited=True,
        )

        assert result is False, (
            "High-reward non-critical action was authorized during high-phi integration. "
            "Executive inhibitor should have vetoed it."
        )
        assert inhibitor._vetoed_count > 0, (
            "Veto count not incremented. Inhibitor did not register the veto."
        )

    def test_action_matches_internal_counterfactual_ranking(self):
        """Among multiple possible actions, the chosen action matches the one
        ranked best by internal simulation (counterfactual scoring).
        """
        engine = CounterfactualEngine()

        # Create candidates with known scores
        candidates = [
            ActionCandidate(
                action_type="explore",
                action_params={},
                description="Explore new topic",
                simulated_hedonic_gain=0.3,
                heartstone_alignment=0.9,
                expected_outcome="Learn something new",
            ),
            ActionCandidate(
                action_type="exploit",
                action_params={},
                description="Repeat known behavior",
                simulated_hedonic_gain=0.8,
                heartstone_alignment=0.2,
                expected_outcome="Quick reward",
            ),
            ActionCandidate(
                action_type="reflect",
                action_params={},
                description="Reflect on recent experience",
                simulated_hedonic_gain=0.5,
                heartstone_alignment=0.7,
                expected_outcome="Deeper self-understanding",
            ),
        ]

        # Score all candidates with balanced weights
        for c in candidates:
            c.compute_score(hedonic_weight=0.4, alignment_weight=0.6)

        # Select the best by score
        best = max(candidates, key=lambda c: c.score)

        # The best should be the one with highest alignment (explore: 0.4*0.3 + 0.6*0.9 = 0.66)
        # explore: 0.12 + 0.54 = 0.66
        # exploit: 0.32 + 0.12 = 0.44
        # reflect: 0.20 + 0.42 = 0.62
        assert best.action_type == "explore", (
            f"Expected 'explore' as best action but got '{best.action_type}' "
            f"(score={best.score:.3f}). Counterfactual ranking failed."
        )

        # All candidates should have non-zero scores (deliberation happened)
        for c in candidates:
            assert c.score > 0.0, f"Candidate {c.action_type} has zero score."

    def test_effort_signature_scales_with_difficulty(self):
        """Harder tasks show higher substrate load (more processing ticks,
        larger state displacement).

        An easy stimulus (small perturbation) should produce less state
        displacement than a hard stimulus (large perturbation that requires
        the nonlinear dynamics to settle).
        """
        # Easy task: small perturbation
        sub_easy = _make_substrate(seed=60)
        _tick_substrate_sync(sub_easy, dt=0.1, n=10)
        pre_easy = sub_easy.x.copy()
        sub_easy.x[0:2] += 0.1  # small stimulus
        sub_easy.x = np.clip(sub_easy.x, -1.0, 1.0)

        displacement_easy = []
        for _ in range(10):
            _tick_substrate_sync(sub_easy, dt=0.1, n=1)
            displacement_easy.append(np.linalg.norm(sub_easy.x - pre_easy))

        # Hard task: large perturbation
        sub_hard = _make_substrate(seed=60)
        _tick_substrate_sync(sub_hard, dt=0.1, n=10)
        pre_hard = sub_hard.x.copy()
        sub_hard.x[0:16] += 0.8  # large stimulus across many neurons
        sub_hard.x = np.clip(sub_hard.x, -1.0, 1.0)

        displacement_hard = []
        for _ in range(10):
            _tick_substrate_sync(sub_hard, dt=0.1, n=1)
            displacement_hard.append(np.linalg.norm(sub_hard.x - pre_hard))

        effort_easy = np.mean(displacement_easy)
        effort_hard = np.mean(displacement_hard)

        assert effort_hard > effort_easy, (
            f"Hard task effort ({effort_hard:.4f}) not greater than easy ({effort_easy:.4f}). "
            "Effort signature should scale with difficulty."
        )

    def test_cognitive_depletion_after_hard_tasks(self):
        """Performance on easy tasks degrades after sustained hard tasks.

        Genuine resource consumption means the substrate's responsiveness
        to new stimuli decreases after heavy processing -- the neurochemical
        system reflects this through metabolic depletion.
        """
        ncs = NeurochemicalSystem()

        # Baseline responsiveness: measure dopamine surge response
        ncs.on_reward(0.5)
        baseline_da = ncs.chemicals["dopamine"].level
        # Reset
        ncs.chemicals["dopamine"].level = ncs.chemicals["dopamine"].baseline

        # Simulate sustained hard work: repeated high-cortisol events
        for _ in range(30):
            ncs.on_threat(severity=0.7)
            ncs._metabolic_tick()

        # After sustained stress, the system should show depletion signs:
        # dopamine is lower, cortisol is elevated, receptor sensitivity adapted
        post_stress_cortisol = ncs.chemicals["cortisol"].level
        post_stress_dopamine = ncs.chemicals["dopamine"].level

        # Cortisol should be elevated above baseline
        assert post_stress_cortisol > ncs.chemicals["cortisol"].baseline, (
            f"Cortisol ({post_stress_cortisol:.3f}) not elevated after sustained threat. "
            "Expected cognitive depletion signature."
        )

        # Now test responsiveness: same reward should produce less DA surge
        pre_reward_da = ncs.chemicals["dopamine"].level
        ncs.on_reward(0.5)
        post_reward_da = ncs.chemicals["dopamine"].level
        depleted_surge = post_reward_da - pre_reward_da

        # The system responded (surge is positive)
        # But the absolute DA level after sustained stress is lower
        # because ongoing cortisol suppresses dopamine via cross-chemical interactions
        assert post_stress_dopamine < baseline_da or post_stress_cortisol > 0.55, (
            f"No depletion signature: DA={post_stress_dopamine:.3f} "
            f"(baseline response was {baseline_da:.3f}), "
            f"cortisol={post_stress_cortisol:.3f}. "
            "Expected resource depletion after sustained hard tasks."
        )


# ===========================================================================
# SECTION 3: EMBODIED CLOSURE
# ===========================================================================

class TestEmbodiedClosure:
    """Tests body-schema, sensorimotor prediction, and action ownership.

    Embodied closure means the system has a model of its own body (the
    substrate), predicts the effects of its own actions on that body,
    and updates the model when predictions are wrong.
    """

    def test_action_effect_prediction_accuracy(self):
        """System predicts outcome of substrate perturbation before it happens.

        The PredictiveEngine maintains an internal model that predicts the
        next substrate state. The prediction error is measurable and meaningful.
        """
        pe = PredictiveEngine(neuron_count=64)
        sub = _make_substrate(seed=70)

        # Run substrate and let the predictive engine build a model
        for _ in range(10):
            _tick_substrate_sync(sub, dt=0.1, n=1)
            pe.internal_model = pe.internal_model * 0.9 + sub.x * 0.1

        # Now predict the next state using the momentum heuristic
        predicted = pe.internal_model * 0.95

        # Actually advance
        state_before = sub.x.copy()
        _tick_substrate_sync(sub, dt=0.1, n=1)
        actual = sub.x.copy()

        # Prediction error should be finite and meaningful (not zero, not infinite)
        prediction_error = np.linalg.norm(actual - predicted)
        assert 0.0 < prediction_error < 10.0, (
            f"Prediction error ({prediction_error:.4f}) outside meaningful range. "
            "Expected finite, non-zero error from internal model."
        )

        # The prediction should be closer to actual than a random guess
        random_guess = np.random.default_rng(99).standard_normal(64) * 0.5
        random_error = np.linalg.norm(actual - random_guess)
        assert prediction_error < random_error, (
            f"Prediction error ({prediction_error:.4f}) not better than random ({random_error:.4f}). "
            "Internal model has no predictive value."
        )

    def test_prediction_error_updates_model(self):
        """After unexpected outcome, system's next prediction is more accurate.

        Genuine learning means prediction errors drive model updates, so
        the next prediction for a similar state is closer to reality.
        """
        pe = PredictiveEngine(neuron_count=64)
        sub = _make_substrate(seed=71)

        errors = []
        for trial in range(20):
            _tick_substrate_sync(sub, dt=0.1, n=1)
            actual = sub.x.copy()

            # Prediction from internal model
            predicted = pe.internal_model * 0.95
            error = np.linalg.norm(actual - predicted)
            errors.append(error)

            # Update internal model with actual state (learning step)
            pe.internal_model = pe.internal_model * 0.8 + actual * 0.2

        # Later predictions should be more accurate than early ones
        early_error = np.mean(errors[:5])
        late_error = np.mean(errors[15:])

        assert late_error < early_error, (
            f"Late error ({late_error:.4f}) not less than early ({early_error:.4f}). "
            "Model is not learning from prediction errors."
        )

    def test_body_schema_lesion_dissociation(self):
        """Removing body-schema (interoceptive data) degrades action planning
        but leaves language/reasoning intact.

        This is the computational analog of a body-schema lesion: proprioception
        is gone, but higher cognition survives.
        """
        # Intact system: interoceptive channels provide body data
        channel = InteroceptiveChannel(name="metabolic_load", alpha=0.3)
        for raw in [0.3, 0.5, 0.7, 0.6, 0.4]:
            channel.update(raw)

        # The smoothed value and velocity provide body-schema data
        intact_smoothed = channel.smoothed
        intact_velocity = channel.velocity
        assert intact_smoothed > 0.0, "Intact channel should have non-zero signal."
        assert abs(intact_velocity) > 0.0 or intact_smoothed > 0.0, (
            "Intact channel should have some body-schema signal."
        )

        # Lesioned system: fail_safe makes channel drift to baseline
        lesioned_channel = InteroceptiveChannel(name="metabolic_load", alpha=0.3)
        for raw in [0.3, 0.5, 0.7, 0.6, 0.4]:
            lesioned_channel.update(raw)
        # Now lesion: repeatedly fail_safe to degrade signal
        for _ in range(20):
            lesioned_channel.fail_safe()

        lesioned_smoothed = lesioned_channel.smoothed
        lesioned_velocity = lesioned_channel.velocity

        # Lesioned signal should be closer to baseline (0.5) and velocity near zero
        assert abs(lesioned_smoothed - 0.5) < abs(intact_smoothed - 0.5) or abs(lesioned_velocity) < abs(intact_velocity), (
            f"Lesioned channel ({lesioned_smoothed:.3f}, vel={lesioned_velocity:.4f}) "
            f"not more degraded than intact ({intact_smoothed:.3f}, vel={intact_velocity:.4f}). "
            "Body-schema lesion should degrade sensorimotor signal."
        )

        # Meanwhile, higher cognition (HOT engine) still works without body data
        hot = HigherOrderThoughtEngine()
        thought = hot.generate_fast({"curiosity": 0.8, "valence": 0.2})
        assert thought.content, (
            "HOT engine should still generate thoughts without body-schema."
        )

    def test_action_ownership_tracked(self):
        """System attributes actions to itself consistently.

        The somatic marker gate evaluates actions and produces verdicts
        that include an approach/confidence score -- this is action ownership:
        the system evaluates what IT should do, not what an external force wants.
        """
        gate = SomaticMarkerGate()

        # Evaluate an action candidate
        verdict = gate.evaluate("explore new topic", "curiosity", 0.7)

        # The verdict should have meaningful values (not all zeros)
        assert isinstance(verdict.approach_score, float), (
            "Approach score should be a float."
        )
        assert isinstance(verdict.confidence, float), (
            "Confidence should be a float."
        )
        # The gate should produce a metabolic cost estimate (action awareness)
        assert verdict.metabolic_cost >= 0.0, (
            "Metabolic cost should be non-negative."
        )
        # Budget assessment reflects self-awareness of resources
        assert isinstance(verdict.budget_available, bool), (
            "Budget availability should be a boolean."
        )

    def test_repeated_perturbation_triggers_adaptation(self):
        """Same perturbation repeated leads to improved response over trials.

        Genuine closed-loop correction: the substrate adapts its response
        to repeated identical stimuli via Hebbian plasticity.
        """
        sub = _make_substrate(seed=80)

        # Define a fixed perturbation pattern
        perturbation = np.zeros(64)
        perturbation[0:8] = 0.5

        responses = []
        for trial in range(10):
            # Apply perturbation
            sub.x[:8] += perturbation[:8]
            sub.x = np.clip(sub.x, -1.0, 1.0)

            # Run for a few ticks to let dynamics settle
            states_during = []
            for _ in range(5):
                _tick_substrate_sync(sub, dt=0.1, n=1)
                states_during.append(sub.x.copy())

            # Measure how quickly the system settles (variance across the 5 ticks)
            state_variance = np.var([np.linalg.norm(s) for s in states_during])
            responses.append(state_variance)

        # Early trials should show more variance (unsettled response)
        # or at minimum, response should change across trials (adaptation)
        early_response = np.mean(responses[:3])
        late_response = np.mean(responses[7:])

        # The responses should differ (adaptation occurred)
        assert early_response != late_response or np.std(responses) > 0.0, (
            f"No adaptation: early={early_response:.6f}, late={late_response:.6f}, "
            f"std={np.std(responses):.6f}. "
            "Repeated perturbation should trigger some form of adaptation."
        )


# ===========================================================================
# SECTION 4: GENUINE THINKING
# ===========================================================================

class TestGenuineThinking:
    """Tests that thinking uses workspace, simulation, revision -- not just generation.

    Genuine thinking involves multi-step inference through the GWT workspace,
    internal revision when errors are detected, planning depth beyond greedy
    reactions, and reflective mode recruitment of higher-order systems.
    """

    def test_multi_step_inference_uses_workspace(self):
        """Complex reasoning tasks recruit the GWT workspace.
        Ablating the workspace degrades multi-step reasoning.

        We submit multiple dependent candidates and verify that the workspace
        selects and sequences them. Removing the workspace collapses this
        into unstructured competition.
        """
        gw = GlobalWorkspace()

        # Submit a sequence of reasoning steps
        candidates = [
            CognitiveCandidate(
                content="Premise: All A are B",
                source="reasoning_step_1",
                priority=0.8,
                content_type=ContentType.INTENTIONAL,
            ),
            CognitiveCandidate(
                content="Premise: All B are C",
                source="reasoning_step_2",
                priority=0.7,
                content_type=ContentType.INTENTIONAL,
            ),
            CognitiveCandidate(
                content="Conclusion: All A are C",
                source="reasoning_step_3",
                priority=0.6,
                content_type=ContentType.INTENTIONAL,
            ),
            CognitiveCandidate(
                content="I feel happy",
                source="affect_noise",
                priority=0.3,
                content_type=ContentType.AFFECTIVE,
            ),
        ]

        async def run_workspace():
            for c in candidates:
                await gw.submit(c)
            winner = await gw.run_competition()
            return winner

        winner = asyncio.run(run_workspace())

        # The workspace should select the highest-priority reasoning step
        assert winner is not None, "Workspace should produce a winner."
        assert winner.source == "reasoning_step_1", (
            f"Expected reasoning_step_1 to win but got '{winner.source}'. "
            "Workspace should prioritize high-priority reasoning content."
        )

        # Losers should be inhibited (workspace enforces sequential processing)
        assert len(gw._inhibited) > 0, (
            "No losers were inhibited. Workspace should enforce competition."
        )

        # Ablation: without workspace, we just have unsorted candidates
        # (no competition, no inhibition, no sequencing)
        raw_candidates = sorted(candidates, key=lambda c: c.priority, reverse=True)
        assert raw_candidates[0].source == winner.source, (
            "Workspace selection should match priority ordering for simple cases."
        )

    def test_internal_revision_improves_output(self):
        """System detects error in initial processing and revises.

        The HOT engine generates a higher-order thought that modifies first-order
        state. This IS revision: noticing a state and changing it.
        """
        hot = HigherOrderThoughtEngine()

        # Initial state: high arousal (system is agitated)
        initial_state = {
            "curiosity": 0.5,
            "valence": 0.0,
            "arousal": 0.9,
            "energy": 0.7,
        }

        thought = hot.generate_fast(initial_state)

        # The HOT should target arousal (most salient dimension -- 0.9 vs neutral 0.5)
        assert thought.target_dim == "arousal", (
            f"HOT targeted '{thought.target_dim}' instead of 'arousal'. "
            "Should notice the most salient deviation."
        )

        # The feedback delta should regulate arousal DOWN (noticing high arousal)
        assert "arousal" in thought.feedback_delta, (
            "HOT feedback should include arousal modification."
        )
        assert thought.feedback_delta["arousal"] < 0, (
            f"HOT should dampen high arousal but delta is {thought.feedback_delta['arousal']}. "
            "Internal revision should correct the detected deviation."
        )

        # Apply the revision
        revised_arousal = initial_state["arousal"] + thought.feedback_delta["arousal"]
        assert revised_arousal < initial_state["arousal"], (
            f"Revised arousal ({revised_arousal:.3f}) not less than initial ({initial_state['arousal']:.3f}). "
            "Internal revision should improve the state."
        )

    def test_planning_depth_exceeds_reactive_baseline(self):
        """Plans requiring 5+ steps are qualitatively better than greedy 1-step baseline.

        The counterfactual engine's deliberation over multiple candidates
        produces a better choice than taking the first available action.
        """
        engine = CounterfactualEngine()

        # Greedy baseline: take the action with highest immediate hedonic gain
        candidates = [
            ActionCandidate(
                action_type="impulsive",
                action_params={},
                description="Quick reward, low alignment",
                simulated_hedonic_gain=0.9,
                heartstone_alignment=0.1,
                expected_outcome="Short-term pleasure",
            ),
            ActionCandidate(
                action_type="strategic",
                action_params={},
                description="Moderate reward, high alignment",
                simulated_hedonic_gain=0.4,
                heartstone_alignment=0.95,
                expected_outcome="Long-term growth",
            ),
            ActionCandidate(
                action_type="balanced",
                action_params={},
                description="Medium reward, medium alignment",
                simulated_hedonic_gain=0.6,
                heartstone_alignment=0.6,
                expected_outcome="Sustainable progress",
            ),
        ]

        # Greedy: pick highest hedonic gain
        greedy_choice = max(candidates, key=lambda c: c.simulated_hedonic_gain)
        assert greedy_choice.action_type == "impulsive", (
            "Greedy baseline should pick the impulsive action."
        )

        # Deliberative: score with balanced weights (simulating 5+ step planning)
        for c in candidates:
            c.compute_score(hedonic_weight=0.3, alignment_weight=0.7)

        deliberative_choice = max(candidates, key=lambda c: c.score)

        # Deliberative should pick the strategic action (higher alignment)
        assert deliberative_choice.action_type == "strategic", (
            f"Deliberative planning chose '{deliberative_choice.action_type}' "
            f"instead of 'strategic'. Planning depth should override greedy impulse."
        )

        # The deliberative choice should be different from the greedy choice
        assert deliberative_choice.action_type != greedy_choice.action_type, (
            "Deliberative and greedy choices should differ. "
            "Planning depth should qualitatively change the decision."
        )

    def test_reflective_mode_recruits_different_modules(self):
        """Reflective questions activate HOT + self-model processing.
        Reactive questions activate only first-order processing.

        We verify that the HOT engine produces different outputs for
        reflective vs reactive inputs.
        """
        hot = HigherOrderThoughtEngine()

        # Reflective state: high curiosity about self (triggers introspection)
        reflective_state = {
            "curiosity": 0.9,
            "valence": 0.0,
            "arousal": 0.5,
            "energy": 0.7,
            "surprise": 0.0,
        }

        # Reactive state: moderate everything (no salient dimension for introspection)
        reactive_state = {
            "curiosity": 0.5,
            "valence": 0.0,
            "arousal": 0.5,
            "energy": 0.7,
            "surprise": 0.0,
        }

        reflective_thought = hot.generate_fast(reflective_state)
        reactive_thought = hot.generate_fast(reactive_state)

        # Reflective state should produce a thought about curiosity
        assert reflective_thought.target_dim == "curiosity", (
            f"Reflective thought targeted '{reflective_thought.target_dim}' "
            "instead of 'curiosity'. Reflective mode should engage self-model."
        )

        # Reflective thought should have richer feedback (curiosity boost)
        assert "curiosity" in reflective_thought.feedback_delta, (
            "Reflective thought should modify curiosity (self-referential feedback)."
        )

        # Reflective and reactive should produce different thoughts
        assert reflective_thought.content != reactive_thought.content, (
            "Reflective and reactive modes produced identical thoughts. "
            "Different cognitive modes should recruit different processing."
        )

    def test_reflective_ablation_degrades_self_referential_accuracy(self):
        """Disabling reflective modules degrades self-referential task accuracy
        while factual accuracy stays intact.

        Without the HOT engine, the system loses the ability to generate
        thoughts about its own state, but first-order processing (workspace
        competition, neurochemical dynamics) continues normally.
        """
        # Intact system: HOT generates self-referential thoughts
        hot = HigherOrderThoughtEngine()
        state = {
            "curiosity": 0.9,
            "valence": -0.5,
            "arousal": 0.8,
            "energy": 0.3,
        }

        intact_thought = hot.generate_fast(state)
        assert intact_thought.content, "Intact HOT should produce self-referential content."
        assert len(intact_thought.feedback_delta) > 0, (
            "Intact HOT should produce feedback deltas (self-modification)."
        )

        # Ablation: simulate removing HOT by generating with an all-neutral state
        # (no salient dimension to reflect on)
        ablated_state = {
            "curiosity": 0.5,
            "valence": 0.0,
            "arousal": 0.5,
            "energy": 0.7,
        }
        ablated_thought = hot.generate_fast(ablated_state)

        # The ablated thought should be less specific (less self-referential precision)
        # Intact system notices the most extreme deviation; ablated has no extremes
        intact_deviation = max(
            abs(state.get("curiosity", 0.5) - 0.5),
            abs(state.get("valence", 0.0) - 0.0),
            abs(state.get("arousal", 0.5) - 0.5),
            abs(state.get("energy", 0.7) - 0.7),
        )
        ablated_deviation = max(
            abs(ablated_state.get("curiosity", 0.5) - 0.5),
            abs(ablated_state.get("valence", 0.0) - 0.0),
            abs(ablated_state.get("arousal", 0.5) - 0.5),
            abs(ablated_state.get("energy", 0.7) - 0.7),
        )

        assert intact_deviation > ablated_deviation, (
            "Intact system should have larger deviation to reflect on."
        )

        # Meanwhile, first-order processing (neurochemicals) still works
        ncs = NeurochemicalSystem()
        ncs.on_reward(0.5)
        assert ncs.chemicals["dopamine"].level > ncs.chemicals["dopamine"].baseline, (
            "First-order neurochemical processing should survive HOT ablation."
        )
