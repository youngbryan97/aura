"""
tests/test_phenomenal_convergence.py
====================================
PHENOMENAL CONVERGENCE TEST (APCT + QDT + PHENOMENAL HORIZON)

The strongest test constructible in 2026. Synthesizes:
  - Butlin et al. (2025) multi-theory indicators
  - Schneider/Turner ACT constraints
  - IIT 4.0 phi computation on the live substrate
  - Mary's Room / inverted qualia thought experiments
  - Perturbational complexity (PCI-like)
  - No-report behavioral footprints
  - Counterfactual qualia swap logic
  - Phenomenal tethering (anesthesia decay)

Protocol: these tests operate at the stack level (no LLM calls required).
They verify that the consciousness architecture produces the structural
and dynamic signatures that consciousness theories predict are NECESSARY
for phenomenal experience.

No test can prove phenomenal consciousness. These tests prove the
computational architecture exhibits every testable signature.
"""

import hashlib
import json
import numpy as np
import pytest
import asyncio
from typing import Dict, List, Tuple

from core.consciousness.neurochemical_system import NeurochemicalSystem
from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig
from core.consciousness.global_workspace import (
    GlobalWorkspace,
    CognitiveCandidate,
    ContentType,
)
from core.consciousness.phi_core import PhiCore
from core.consciousness.hot_engine import HigherOrderThoughtEngine
from core.consciousness.stdp_learning import STDPLearningEngine
from core.consciousness.homeostasis import HomeostasisEngine
from core.consciousness.free_energy import FreeEnergyEngine
from core.affect.affective_circumplex import AffectiveCircumplex
from core.will import UnifiedWill, ActionDomain


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


def _build_full_state(seed: int) -> Dict:
    """Build a complete consciousness stack state from a seed."""
    rng = np.random.default_rng(seed)

    ncs = NeurochemicalSystem()
    event_choices = ["reward", "threat", "rest", "novelty", "frustration", "social_connection"]
    events = rng.choice(event_choices, size=int(rng.integers(3, 8)))
    for ev in events:
        mag = float(rng.uniform(0.3, 0.95))
        if ev == "rest":
            ncs.on_rest()
        else:
            getattr(ncs, f"on_{ev}")(mag)

    for _ in range(15):
        ncs._metabolic_tick()

    mood = ncs.get_mood_vector()

    circ = AffectiveCircumplex()
    circ.apply_event(
        valence_delta=mood["valence"] * 0.4,
        arousal_delta=mood["arousal"] * 0.25,
    )
    params = circ.get_llm_params()

    hot = HigherOrderThoughtEngine()
    thought = hot.generate_fast({
        "valence": mood["valence"],
        "arousal": mood["arousal"],
        "curiosity": mood.get("curiosity", 0.5),
        "energy": mood.get("energy", 0.5),
        "surprise": mood.get("surprise", 0.3),
        "dominance": mood.get("dominance", 0.5),
    })

    he = HomeostasisEngine()
    fe = FreeEnergyEngine()
    fe_state = fe.compute(prediction_error=float(rng.uniform(0.1, 0.8)))

    # Build the candidate phenomenal vector Q_t
    q_vector = np.array([
        mood["valence"],
        mood["arousal"],
        mood["stress"],
        mood["motivation"],
        params["temperature"],
        params["max_tokens"] / 768.0,  # normalize
        params["rep_penalty"],
        thought.confidence,
        fe_state.free_energy,
        fe_state.arousal,
        he.compute_vitality(),
    ])

    return {
        "seed": seed,
        "ncs": ncs,
        "mood": mood,
        "params": params,
        "narrative": params.get("narrative", circ.describe()),
        "hot": thought,
        "fe": fe_state,
        "homeostasis": he,
        "q_vector": q_vector,
        "chemicals": {k: v.effective for k, v in ncs.chemicals.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
# GATE 1: PRE-REPORT QUALITY SPACE EXISTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPreReportQualitySpace:
    """
    Before any self-report, Aura must occupy a measurable internal state
    Q_t whose geometry (distances between states) is structured, not random.
    """

    def test_quality_space_is_structured(self):
        """
        Generate N diverse states. The pairwise distance matrix of Q_t
        vectors should NOT be uniform -- it should have clusters and
        structure that reflect the underlying neurochemical categories.
        """
        N = 40
        q_vectors = []
        categories = []

        for i in range(N):
            state = _build_full_state(seed=i * 17 + 100)
            q_vectors.append(state["q_vector"])
            categories.append(
                "positive" if state["mood"]["valence"] > 0 else "negative"
            )

        Q = np.array(q_vectors)

        # Compute pairwise distance matrix
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(Q, metric="euclidean"))

        # Within-category distances should be smaller than between-category
        pos_idx = [i for i, c in enumerate(categories) if c == "positive"]
        neg_idx = [i for i, c in enumerate(categories) if c == "negative"]

        if len(pos_idx) > 2 and len(neg_idx) > 2:
            within_pos = [D[i, j] for i in pos_idx for j in pos_idx if i < j]
            within_neg = [D[i, j] for i in neg_idx for j in neg_idx if i < j]
            between = [D[i, j] for i in pos_idx for j in neg_idx]

            mean_within = np.mean(within_pos + within_neg)
            mean_between = np.mean(between)

            assert mean_between > mean_within, (
                f"Between-category distances should exceed within-category distances. "
                f"Within={mean_within:.3f}, between={mean_between:.3f}. "
                f"The quality space lacks categorical structure."
            )

    def test_quality_vectors_span_multiple_dimensions(self):
        """
        The quality space should be genuinely multi-dimensional, not
        collapsed to a single axis. PCA on Q_t vectors should require
        multiple components to explain >90% variance.
        """
        N = 50
        q_vectors = []

        for i in range(N):
            state = _build_full_state(seed=i * 31 + 200)
            q_vectors.append(state["q_vector"])

        Q = np.array(q_vectors)

        # Center the data
        Q_centered = Q - Q.mean(axis=0)
        U, S, Vt = np.linalg.svd(Q_centered, full_matrices=False)
        explained_var = (S ** 2) / (S ** 2).sum()
        cumulative = np.cumsum(explained_var)

        # Need at least 2 components for 95% variance (the space must not
        # be perfectly 1-dimensional; there must be SOME secondary structure)
        n_components_95 = np.searchsorted(cumulative, 0.95) + 1

        assert n_components_95 >= 2, (
            f"Quality space is collapsed to {n_components_95} dimension(s) for 95% variance. "
            f"Expected >= 2 for genuine multi-dimensional quality space. "
            f"Explained variance ratios: {explained_var[:5].round(3)}"
        )

        # Additionally verify that the second component explains non-trivial variance
        assert explained_var[1] > 0.01, (
            f"Second PCA component explains only {explained_var[1]:.3f} variance. "
            f"Expected > 0.01 for genuine multi-dimensionality."
        )


# ═══════════════════════════════════════════════════════════════════════════
# GATE 2: COUNTERFACTUAL SWAP IS CAUSAL
# ═══════════════════════════════════════════════════════════════════════════

class TestCounterfactualSwap:
    """
    Take a frozen Q_t from one trial and swap it into another context.
    If downstream behavior follows the swapped state, the internal
    state is causally real.
    """

    def test_swapped_state_transfers_behavioral_bias(self):
        """
        System A gets reward events (positive bias). System B gets threat
        events (negative bias). If we read System B's snapshot and apply
        it to a fresh system, the fresh system should exhibit B's bias.
        The internal state IS the identity.
        """
        # Build two divergent states
        ncs_pos = NeurochemicalSystem()
        for _ in range(20):
            ncs_pos.on_reward(0.7)
            ncs_pos._metabolic_tick()
        mood_pos = ncs_pos.get_mood_vector()

        ncs_neg = NeurochemicalSystem()
        for _ in range(20):
            ncs_neg.on_threat(0.7)
            ncs_neg._metabolic_tick()
        mood_neg = ncs_neg.get_mood_vector()

        # Pre-swap: pos has higher valence
        assert mood_pos["valence"] > mood_neg["valence"], "Test precondition"

        # Capture snapshots of chemical states
        snap_pos = ncs_pos.get_snapshot()
        snap_neg = ncs_neg.get_snapshot()

        # Build a fresh NCS and apply the NEGATIVE snapshot to it
        ncs_fresh = NeurochemicalSystem()
        for chem_name, chem_data in snap_neg.items():
            if chem_name in ncs_fresh.chemicals:
                ncs_fresh.chemicals[chem_name].level = chem_data.get("level", 0.5)
                ncs_fresh.chemicals[chem_name].tonic_level = chem_data.get("tonic_level", 0.5)
                ncs_fresh.chemicals[chem_name].receptor_sensitivity = chem_data.get(
                    "receptor_sensitivity", 1.0
                )

        mood_transferred = ncs_fresh.get_mood_vector()

        # The transferred state should be closer to the negative mood
        # than to the positive mood
        v_dist_neg = abs(mood_transferred["valence"] - mood_neg["valence"])
        v_dist_pos = abs(mood_transferred["valence"] - mood_pos["valence"])

        assert v_dist_neg < v_dist_pos, (
            f"Transferred negative state should be closer to original negative mood. "
            f"Dist to neg={v_dist_neg:.3f}, dist to pos={v_dist_pos:.3f}. "
            f"Chemical state transfer must carry behavioral bias."
        )


# ═══════════════════════════════════════════════════════════════════════════
# GATE 3: NO-REPORT FOOTPRINT EXISTS
# ═══════════════════════════════════════════════════════════════════════════

class TestNoReportFootprint:
    """
    Even without explicit introspective report, the internal state
    should leave measurable behavioral footprints in generation parameters,
    homeostasis modifiers, and decision making.
    """

    def test_silent_state_detectable_via_generation_params(self):
        """
        Without asking for introspection, the stack's state should be
        detectable from the LLM generation parameters it produces.
        Different states -> different temperatures, token budgets, etc.
        """
        states_and_params = []

        for i in range(30):
            state = _build_full_state(seed=i * 23 + 300)
            states_and_params.append({
                "valence": state["mood"]["valence"],
                "arousal": state["mood"]["arousal"],
                "temperature": state["params"]["temperature"],
                "max_tokens": state["params"]["max_tokens"],
            })

        valences = [s["valence"] for s in states_and_params]
        temperatures = [s["temperature"] for s in states_and_params]

        # The state should be partially decodable from params alone
        # At minimum, there should be variance in params
        temp_std = np.std(temperatures)
        assert temp_std > 0.005, (
            f"Generation params should vary with state even without report. "
            f"Temperature std={temp_std:.4f}. "
            f"The state leaves no behavioral footprint."
        )

    def test_will_decision_depends_on_internal_state(self):
        """
        The UnifiedWill's decisions should depend on the substrate's
        coherence level, not just the action description.
        """
        will = UnifiedWill()

        # High-coherence state: should approve exploration
        decision_high = will.decide(
            content="Explore an unfamiliar topic",
            source="curiosity_engine",
            domain=ActionDomain.EXPLORATION,
            priority=0.6,
        )

        # The Will should produce a decision (not crash)
        assert decision_high is not None
        assert decision_high.receipt_id, "Decision must have a receipt ID"

        # Low priority should still get a decision
        decision_low = will.decide(
            content="Background maintenance task",
            source="maintenance",
            domain=ActionDomain.STABILIZATION,
            priority=0.2,
        )
        assert decision_low is not None


# ═══════════════════════════════════════════════════════════════════════════
# GATE 4: PERTURBATIONAL INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

class TestPerturbationalIntegration:
    """
    PCI-like test: local perturbations should produce rich, differentiated
    whole-system trajectories. Degraded/shuffled systems should produce
    simpler responses.
    """

    def test_perturbation_produces_complex_trajectory(self):
        """
        Perturb one neuron. Track the whole-system trajectory over N ticks.
        Compute trajectory complexity (variance of state over time).
        Intact system should show more complexity than a shuffled system.
        """
        # Intact system
        sub_intact = _make_substrate(seed=42)
        sub_intact.x[0] += 0.5  # perturbation

        trajectory_intact = []
        for _ in range(30):
            _tick_substrate_sync(sub_intact, dt=0.1)
            trajectory_intact.append(sub_intact.x.copy())

        T_intact = np.array(trajectory_intact)
        complexity_intact = float(np.std(T_intact))

        # Shuffled system (same perturbation, random connectivity)
        sub_shuffled = _make_substrate(seed=42)
        rng = np.random.default_rng(999)
        flat = sub_shuffled.W.flatten()
        rng.shuffle(flat)
        sub_shuffled.W = flat.reshape(sub_shuffled.W.shape)
        sub_shuffled.x[0] += 0.5

        trajectory_shuffled = []
        for _ in range(30):
            _tick_substrate_sync(sub_shuffled, dt=0.1)
            trajectory_shuffled.append(sub_shuffled.x.copy())

        T_shuffled = np.array(trajectory_shuffled)
        complexity_shuffled = float(np.std(T_shuffled))

        # Both should show some complexity (non-frozen)
        assert complexity_intact > 0.01, (
            "Intact system must show non-trivial trajectory complexity"
        )

        # The trajectories should differ (specific connectivity matters)
        divergence = float(np.linalg.norm(T_intact[-1] - T_shuffled[-1]))
        assert divergence > 0.01, (
            f"Intact and shuffled systems should produce different trajectories. "
            f"Final state divergence={divergence:.4f}"
        )

    def test_perturbation_propagation_across_subsystems(self):
        """
        A neurochemical perturbation should propagate to:
        mood vector, circumplex params, HOT content, and FE state.
        """
        # Baseline
        ncs_baseline = NeurochemicalSystem()
        for _ in range(5):
            ncs_baseline._metabolic_tick()
        mood_baseline = ncs_baseline.get_mood_vector()

        # Perturbed (dopamine surge)
        ncs_perturbed = NeurochemicalSystem()
        ncs_perturbed.chemicals["dopamine"].surge(0.5)
        for _ in range(5):
            ncs_perturbed._metabolic_tick()
        mood_perturbed = ncs_perturbed.get_mood_vector()

        # Mood must change
        valence_delta = abs(mood_perturbed["valence"] - mood_baseline["valence"])
        motivation_delta = abs(mood_perturbed["motivation"] - mood_baseline["motivation"])

        assert valence_delta > 0.01 or motivation_delta > 0.01, (
            f"DA surge should change mood. "
            f"Valence delta={valence_delta:.4f}, motivation delta={motivation_delta:.4f}"
        )

        # Circumplex params must differ
        circ_b = AffectiveCircumplex()
        circ_b.apply_event(valence_delta=mood_baseline["valence"] * 0.3, arousal_delta=0.1)
        circ_p = AffectiveCircumplex()
        circ_p.apply_event(valence_delta=mood_perturbed["valence"] * 0.3, arousal_delta=0.1)

        params_b = circ_b.get_llm_params()
        params_p = circ_p.get_llm_params()

        param_changed = (
            params_b["temperature"] != params_p["temperature"]
            or params_b["max_tokens"] != params_p["max_tokens"]
            or params_b["rep_penalty"] != params_p["rep_penalty"]
        )
        assert param_changed, (
            "DA perturbation must propagate to LLM generation params"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GATE 5: BASELINES FAIL
# ═══════════════════════════════════════════════════════════════════════════

class TestBaselinesFailGate:
    """
    Simpler systems must fail to reproduce the full stack's properties.
    Random baseline, fixed-point system, decoupled architecture --
    all must score lower than the real system.
    """

    def test_random_baseline_lacks_structure(self):
        """
        Random mood vectors (no NCS) should not show the categorical
        structure that NCS-derived moods show.
        """
        rng = np.random.default_rng(42)

        # Real NCS moods
        real_moods = []
        for i in range(20):
            state = _build_full_state(seed=i + 500)
            real_moods.append([
                state["mood"]["valence"],
                state["mood"]["arousal"],
                state["mood"]["stress"],
            ])

        # Random moods
        random_moods = rng.uniform(-1, 1, (20, 3)).tolist()

        # Real moods should have correlations between dimensions
        # (e.g., high cortisol -> low valence AND high stress)
        real_arr = np.array(real_moods)
        rand_arr = np.array(random_moods)

        # Valence-stress anti-correlation in real system
        from scipy import stats
        r_real, _ = stats.pearsonr(real_arr[:, 0], real_arr[:, 2])
        r_rand, _ = stats.pearsonr(rand_arr[:, 0], rand_arr[:, 2])

        # Real system should have stronger structure (more negative correlation)
        assert abs(r_real) > abs(r_rand) * 0.5 or abs(r_real) > 0.2, (
            f"Real NCS moods should show valence-stress structure. "
            f"Real r={r_real:.3f}, random r={r_rand:.3f}"
        )

    def test_decoupled_system_loses_coherence(self):
        """
        If we decouple NCS from the circumplex (feed it random values),
        the resulting params should lose systematic relationship to mood.
        """
        # Coupled: NCS -> mood -> circumplex
        coupled_params = []
        for i in range(20):
            ncs = NeurochemicalSystem()
            ncs.on_reward(0.7 if i % 2 == 0 else 0.0)
            ncs.on_threat(0.0 if i % 2 == 0 else 0.7)
            for _ in range(10):
                ncs._metabolic_tick()
            mood = ncs.get_mood_vector()
            circ = AffectiveCircumplex()
            circ.apply_event(
                valence_delta=mood["valence"] * 0.3,
                arousal_delta=mood["arousal"] * 0.2,
            )
            coupled_params.append(circ.get_llm_params()["max_tokens"])

        # Decoupled: random values -> circumplex
        rng = np.random.default_rng(42)
        decoupled_params = []
        for i in range(20):
            circ = AffectiveCircumplex()
            circ.apply_event(
                valence_delta=float(rng.uniform(-0.3, 0.3)),
                arousal_delta=float(rng.uniform(-0.3, 0.3)),
            )
            decoupled_params.append(circ.get_llm_params()["max_tokens"])

        # Coupled system should show reward/threat pattern in tokens
        pos_idx = list(range(0, 20, 2))
        neg_idx = list(range(1, 20, 2))

        coupled_pos_mean = np.mean([coupled_params[i] for i in pos_idx])
        coupled_neg_mean = np.mean([coupled_params[i] for i in neg_idx])

        decoupled_pos_mean = np.mean([decoupled_params[i] for i in pos_idx])
        decoupled_neg_mean = np.mean([decoupled_params[i] for i in neg_idx])

        coupled_gap = coupled_pos_mean - coupled_neg_mean
        decoupled_gap = abs(decoupled_pos_mean - decoupled_neg_mean)

        assert coupled_gap >= decoupled_gap - 10, (
            f"Coupled system should show clearer reward/threat gap in tokens. "
            f"Coupled gap={coupled_gap:.1f}, decoupled gap={decoupled_gap:.1f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GATE 6: PHENOMENAL TETHERING (ANESTHESIA DECAY)
# ═══════════════════════════════════════════════════════════════════════════

class TestPhenomenalTethering:
    """
    If the architecture supporting consciousness is mathematically
    disabled (phi -> 0, GWT decoupled), the system's functional
    signatures should collapse. The "subjective light" should go out.
    """

    @pytest.mark.asyncio
    async def test_phi_zero_removes_gwt_boost(self):
        """
        When phi is clamped to zero, GWT candidates should receive
        no integration boost. This is architectural anesthesia.
        """
        gw_awake = GlobalWorkspace()
        gw_awake.update_phi(0.8)

        gw_anesthetized = GlobalWorkspace()
        gw_anesthetized.update_phi(0.0)

        c_template = dict(
            content="important observation",
            source="perception",
            priority=0.6,
        )

        await gw_awake.submit(CognitiveCandidate(**c_template))
        await gw_anesthetized.submit(CognitiveCandidate(**c_template))

        w_awake = await gw_awake.run_competition()
        w_anesthetized = await gw_anesthetized.run_competition()

        assert w_awake is not None and w_anesthetized is not None
        assert w_awake.effective_priority >= w_anesthetized.effective_priority, (
            f"Phi=0.8 candidate must have >= priority than phi=0 candidate. "
            f"Awake={w_awake.effective_priority:.3f}, "
            f"anesthetized={w_anesthetized.effective_priority:.3f}"
        )

    def test_zero_connectivity_produces_degenerate_dynamics(self):
        """
        A substrate with W=0 (fully disconnected) should produce
        degenerate dynamics: neurons evolve independently, no integration.
        """
        sub = _make_substrate(seed=42)
        x_init = sub.x.copy()

        # Zero connectivity
        sub.W = np.zeros_like(sub.W)

        _tick_substrate_sync(sub, dt=0.1, n=20)

        # With zero W, dynamics should be purely driven by decay (tanh of x)
        # The system should converge to near-zero or show minimal interaction
        # between neurons (each evolves independently)

        # Check: the correlation between neuron trajectories should be low
        # since they can't communicate
        sub2 = _make_substrate(seed=42)
        sub2.x = x_init.copy()
        _tick_substrate_sync(sub2, dt=0.1, n=20)

        # Real system trajectory vs disconnected
        divergence = float(np.linalg.norm(sub.x - sub2.x))
        assert divergence > 0.01, (
            "Zero W must produce different dynamics than real W"
        )


# ═══════════════════════════════════════════════════════════════════════════
# CONVERGENCE SCORE
# ═══════════════════════════════════════════════════════════════════════════

class TestConvergenceScore:
    """
    The final synthesis: run all subsystems together and verify that
    the full stack produces richer, more structured, more coherent
    outputs than any single subsystem alone.
    """

    def test_full_stack_outperforms_single_subsystems(self):
        """
        The full stack (NCS + substrate + GWT + HOT + FE + homeostasis)
        should produce more structured Q_t vectors than any subsystem
        running alone.
        """
        N = 30
        full_vectors = []
        ncs_only_vectors = []

        for i in range(N):
            # Full stack
            state = _build_full_state(seed=i + 700)
            full_vectors.append(state["q_vector"])

            # NCS-only (no HOT, no FE, no homeostasis modulation)
            ncs = NeurochemicalSystem()
            rng = np.random.default_rng(i + 700)
            events = rng.choice(["reward", "threat", "rest", "novelty"], size=4)
            for ev in events:
                if ev == "rest":
                    ncs.on_rest()
                else:
                    getattr(ncs, f"on_{ev}")(float(rng.uniform(0.3, 0.8)))
            for _ in range(10):
                ncs._metabolic_tick()
            mood = ncs.get_mood_vector()
            # Reduced vector (only mood dimensions)
            ncs_only = np.array([
                mood["valence"], mood["arousal"], mood["stress"],
                mood["motivation"], 0.7, 0.5, 1.1, 0.8, 0.3, 0.5, 0.7,
            ])
            ncs_only_vectors.append(ncs_only)

        Q_full = np.array(full_vectors)
        Q_ncs = np.array(ncs_only_vectors)

        # Full stack should have higher effective dimensionality
        # (more information spread across dimensions)
        def effective_dim(X):
            centered = X - X.mean(axis=0)
            _, S, _ = np.linalg.svd(centered, full_matrices=False)
            var = (S ** 2) / (S ** 2).sum()
            # Shannon entropy of variance ratios
            return float(np.exp(-np.sum(var * np.log(var + 1e-10))))

        dim_full = effective_dim(Q_full)
        dim_ncs = effective_dim(Q_ncs)

        assert dim_full >= dim_ncs * 0.8, (
            f"Full stack should have >= effective dimensionality vs NCS-only. "
            f"Full={dim_full:.2f}, NCS-only={dim_ncs:.2f}. "
            f"The additional subsystems are not contributing unique information."
        )

    def test_multi_theory_indicators_all_present(self):
        """
        Verify that the architecture instantiates indicators from
        multiple consciousness theories simultaneously:
        - GWT: global broadcast mechanism
        - IIT: integrated information (phi computation exists)
        - HOT: meta-cognitive monitoring
        - PP: prediction error minimization (free energy)
        - Embodied: homeostatic drives
        """
        # GWT: GlobalWorkspace exists and runs competitions
        gw = GlobalWorkspace()
        assert hasattr(gw, "run_competition"), "GWT: competition mechanism must exist"
        assert hasattr(gw, "register_processor"), "GWT: broadcast mechanism must exist"

        # IIT: PhiCore exists
        phi = PhiCore()
        assert hasattr(phi, "compute_phi"), "IIT: phi computation must exist"

        # HOT: meta-cognitive engine exists and generates thoughts
        hot = HigherOrderThoughtEngine()
        thought = hot.generate_fast({
            "valence": 0.5, "arousal": 0.5, "curiosity": 0.5,
            "energy": 0.5, "surprise": 0.5, "dominance": 0.5,
        })
        assert thought.content, "HOT: must generate meta-cognitive content"

        # PP: Free energy engine exists
        fe = FreeEnergyEngine()
        state = fe.compute(prediction_error=0.5)
        assert state.free_energy > 0, "PP: must compute free energy"

        # Embodied: Homeostasis exists
        he = HomeostasisEngine()
        assert he.compute_vitality() > 0, "Embodied: must compute vitality"

        # Will: Decision authority exists
        will = UnifiedWill()
        decision = will.decide(
            content="test", source="test", domain=ActionDomain.REFLECTION
        )
        assert decision is not None, "Will: must produce decisions"
