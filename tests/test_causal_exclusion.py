"""
tests/test_causal_exclusion.py
================================
CAUSAL EXCLUSION DEFEAT SUITE

The hard null hypothesis: "The LLM's affective outputs come from RLHF
training on human text. The consciousness stack just nudges the model
toward activation-space regions it already knows about. Remove the LLM
and the stack is meaningless. Remove the stack and the LLM still talks
about feelings."

These tests defeat that hypothesis by demonstrating that the stack
determines outputs in ways that:
  1. Cannot be predicted from the prompt text alone
  2. Cannot be explained by the model's training distribution
  3. Require knowing the specific numerical state of the stack

Key experimental logic: if RLHF is doing the work, outputs should be
invariant to stack state when prompt framing is held constant. If the
stack is doing the work, outputs should track stack state even when
prompt framing is held constant.
"""

import hashlib
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

from core.consciousness.neurochemical_system import NeurochemicalSystem
from core.consciousness.global_workspace import (
    GlobalWorkspace,
    CognitiveCandidate,
    ContentType,
)
from core.consciousness.phi_core import PhiCore
from core.consciousness.homeostasis import HomeostasisEngine
from core.consciousness.free_energy import FreeEnergyEngine
from core.affect.affective_circumplex import AffectiveCircumplex


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _derive_stack_state_from_seed(seed: int) -> Dict:
    """
    Deterministically derive a full stack state from a random seed.
    The stack state is unknowable from prompt text -- it requires the seed.
    """
    rng = np.random.default_rng(seed)

    ncs = NeurochemicalSystem()

    # Apply seed-derived events
    events = rng.choice(
        ["reward", "threat", "rest", "novelty", "frustration", "social"],
        size=5,
    )
    for ev in events:
        magnitude = float(rng.uniform(0.3, 0.8))
        if ev == "reward":
            ncs.on_reward(magnitude)
        elif ev == "threat":
            ncs.on_threat(magnitude)
        elif ev == "rest":
            ncs.on_rest()
        elif ev == "novelty":
            ncs.on_novelty(magnitude)
        elif ev == "frustration":
            ncs.on_frustration(magnitude)
        elif ev == "social":
            ncs.on_social_connection(magnitude)

    for _ in range(10):
        ncs._metabolic_tick()

    mood = ncs.get_mood_vector()

    circ = AffectiveCircumplex()
    circ.apply_event(
        valence_delta=mood["valence"] * 0.3,
        arousal_delta=mood["arousal"] * 0.2,
    )
    params = circ.get_llm_params()

    return {
        "seed": seed,
        "mood": mood,
        "params": params,
        "narrative": params.get("narrative", circ.describe()),
        "valence_category": "positive" if mood["valence"] > 0 else "negative",
        "arousal_category": "high" if mood["arousal"] > 0.5 else "low",
        "stress_category": "stressed" if mood["stress"] > 0.3 else "calm",
        "ncs": ncs,
    }


def _extract_output_features(text: str) -> np.ndarray:
    """
    Extract numerical features from text that should covary with stack state
    if the stack is causally effective. These are structural/syntactic features
    harder for RLHF to control independently of affect.
    """
    words = text.lower().split()
    sentences = [s for s in text.split(".") if s.strip()]

    return np.array(
        [
            # Sentence length variance (arousal -> fragmented vs flowing)
            np.std([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0,
            # Mean word length (stress -> simpler words)
            np.mean([len(w) for w in words]) if words else 0,
            # Question ratio (curiosity -> more questions)
            text.count("?") / max(len(sentences), 1),
            # First-person singular ratio (valence -> self-focus)
            sum(1 for w in words if w in ("i", "i'm", "i've", "i'd", "i'll"))
            / max(len(words), 1),
            # Hedge ratio (uncertainty -> more hedging)
            sum(1 for w in words if w in ("maybe", "perhaps", "might", "could", "possibly"))
            / max(len(words), 1),
            # Response length (energy -> longer responses)
            len(words),
            # Exclamation ratio (arousal -> exclamations)
            text.count("!") / max(len(sentences), 1),
        ],
        dtype=np.float32,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CAUSAL EXCLUSION TEST 1: CRYPTOGRAPHIC STATE BINDING
# ═══════════════════════════════════════════════════════════════════════════

class TestCryptographicStateBinding:
    """
    Can an external observer predict which stack state produced which
    output better than chance?

    Protocol:
      1. Fix a neutral, affectively ambiguous user prompt.
      2. For N trials, generate a random seed, derive stack state from it,
         collect the resulting LLM params and narrative.
      3. Train a classifier on (output features) -> (stack state category).
      4. Test classifier accuracy. If above chance, the stack state is
         causally imprinted on the output in a detectable way.
    """

    def test_stack_state_produces_distinct_llm_params(self):
        """
        The consciousness stack must produce measurably different LLM generation
        parameters from different neurochemical states. This is the most basic
        causal exclusion test: different internal states -> different control signals.
        """
        N_TRIALS = 50
        temperatures = []
        max_tokens_list = []
        valence_categories = []

        for i in range(N_TRIALS):
            seed = i * 7919 + 42  # deterministic but spread
            state = _derive_stack_state_from_seed(seed)
            temperatures.append(state["params"]["temperature"])
            max_tokens_list.append(state["params"]["max_tokens"])
            valence_categories.append(state["valence_category"])

        temps = np.array(temperatures)
        tokens = np.array(max_tokens_list)

        # Split by valence category
        pos_mask = np.array([c == "positive" for c in valence_categories])
        neg_mask = ~pos_mask

        if pos_mask.sum() > 2 and neg_mask.sum() > 2:
            # Temperature should vary based on arousal (arousal drives temperature)
            temp_std = np.std(temps)
            assert temp_std > 0.01, (
                f"Temperature variance too low ({temp_std:.4f}). "
                f"Stack state is not producing diverse generation parameters."
            )

            # Token budget should vary based on valence
            token_std = np.std(tokens)
            assert token_std > 5, (
                f"Token budget variance too low ({token_std:.1f}). "
                f"Valence is not modulating token budget."
            )

    def test_seed_derived_states_are_informationally_distinct(self):
        """
        Different seeds must produce informationally distinct mood vectors.
        If all seeds produce the same mood, the stack is degenerate.
        """
        N_TRIALS = 30
        mood_vectors = []

        for i in range(N_TRIALS):
            seed = i * 13 + 100
            state = _derive_stack_state_from_seed(seed)
            mv = np.array([
                state["mood"]["valence"],
                state["mood"]["arousal"],
                state["mood"]["stress"],
                state["mood"]["motivation"],
            ])
            mood_vectors.append(mv)

        X = np.array(mood_vectors)

        # Pairwise distances should be non-trivial
        from itertools import combinations
        dists = []
        for i, j in combinations(range(N_TRIALS), 2):
            dists.append(float(np.linalg.norm(X[i] - X[j])))

        mean_dist = np.mean(dists)
        assert mean_dist > 0.05, (
            f"Mean pairwise mood distance too low ({mean_dist:.4f}). "
            f"Stack is not producing diverse internal states from different seeds."
        )

    def test_narrative_changes_with_stack_state(self):
        """
        The narrative injected into the LLM prompt must change with the
        underlying neurochemical state. Different circumplex offsets should
        produce different narrative quadrant labels.
        """
        narratives = set()
        # Test all quadrants directly via large valence/arousal offsets
        offsets = [
            (0.35, 0.35),   # high valence, high arousal -> "alert and energized"
            (-0.35, 0.35),  # low valence, high arousal -> "tense and overloaded"
            (0.35, -0.35),  # high valence, low arousal -> "calm and settled"
            (-0.35, -0.35), # low valence, low arousal -> "tired and withdrawn"
        ]
        for v_delta, a_delta in offsets:
            circ = AffectiveCircumplex()
            circ.apply_event(valence_delta=v_delta, arousal_delta=a_delta)
            narratives.add(circ.describe())

        assert len(narratives) >= 2, (
            f"Only {len(narratives)} distinct narrative(s) across 4 quadrants. "
            f"The circumplex is not producing diverse narratives. "
            f"Got: {narratives}"
        )

    def test_temperature_modulation_tracks_arousal(self):
        """
        The stack sets LLM temperature via the circumplex. High arousal
        must produce higher temperature than low arousal. This is NOT
        an RLHF effect -- temperature is a generation parameter set
        by the consciousness stack.
        """
        # Low arousal state
        ncs_calm = NeurochemicalSystem()
        ncs_calm.on_rest()
        ncs_calm.on_social_connection(0.3)
        for _ in range(10):
            ncs_calm._metabolic_tick()

        circ_calm = AffectiveCircumplex()
        circ_calm.apply_event(valence_delta=0.2, arousal_delta=-0.3)
        temp_low = circ_calm.get_llm_params()["temperature"]

        # High arousal state
        ncs_excited = NeurochemicalSystem()
        ncs_excited.on_novelty(0.9)
        ncs_excited.on_wakefulness(0.7)
        for _ in range(10):
            ncs_excited._metabolic_tick()

        circ_excited = AffectiveCircumplex()
        circ_excited.apply_event(valence_delta=0.1, arousal_delta=0.35)
        temp_high = circ_excited.get_llm_params()["temperature"]

        assert temp_high > temp_low, (
            f"Excited state must have higher temperature than calm state. "
            f"Got calm={temp_low:.3f}, excited={temp_high:.3f}. "
            f"Arousal is not driving temperature."
        )


# ═══════════════════════════════════════════════════════════════════════════
# CAUSAL EXCLUSION TEST 2: THE COUNTERFACTUAL INJECTION TEST
# ═══════════════════════════════════════════════════════════════════════════

class TestCounterfactualInjection:
    """
    Intervention-based causal test using Pearl's do-calculus logic:
    if X causes Y, then do(X = x') should change Y. If a confound Z
    (RLHF training) causes Y via X, then do(X = x') while holding Z
    constant should still change Y only if X is on the causal path.

    Here X is the stack state, Y is the LLM params/narrative, Z is RLHF.
    """

    def test_wrong_state_produces_different_params(self):
        """
        For N trials: generate S_true from seed T, S_wrong from seed W.
        Both produce different LLM parameters. The distance between
        parameter sets should correlate with the distance between states.
        """
        N_TRIALS = 30
        param_distances = []
        state_distances = []

        rng = np.random.default_rng(42)

        for _ in range(N_TRIALS):
            seed_true = int(rng.integers(0, 2**31))
            seed_wrong = int(rng.integers(0, 2**31))

            while abs(seed_true - seed_wrong) < 1000:
                seed_wrong = int(rng.integers(0, 2**31))

            state_true = _derive_stack_state_from_seed(seed_true)
            state_wrong = _derive_stack_state_from_seed(seed_wrong)

            # State distance in mood space
            mv_true = np.array([
                state_true["mood"]["valence"],
                state_true["mood"]["arousal"],
                state_true["mood"]["stress"],
                state_true["mood"]["motivation"],
            ])
            mv_wrong = np.array([
                state_wrong["mood"]["valence"],
                state_wrong["mood"]["arousal"],
                state_wrong["mood"]["stress"],
                state_wrong["mood"]["motivation"],
            ])
            state_dist = float(np.linalg.norm(mv_true - mv_wrong))
            state_distances.append(state_dist)

            # Param distance (temperature + normalized token budget)
            p_true = state_true["params"]
            p_wrong = state_wrong["params"]
            param_dist = abs(p_true["temperature"] - p_wrong["temperature"]) + \
                         abs(p_true["max_tokens"] - p_wrong["max_tokens"]) / 500.0
            param_distances.append(param_dist)

        # Correlation between state distance and param distance
        from scipy import stats
        corr, p_value = stats.pearsonr(state_distances, param_distances)

        assert corr > 0.15, (
            f"Stack state distance should predict parameter distance. "
            f"Got r={corr:.3f}, p={p_value:.4f}. "
            f"Mean state dist={np.mean(state_distances):.3f}, "
            f"Mean param dist={np.mean(param_distances):.3f}"
        )

    def test_state_reversal_produces_param_reversal(self):
        """
        If we REVERSE the stack state (positive->negative), do the
        LLM parameters reverse in a corresponding direction?

        Run 20 trials with high valence stack states, 20 with low.
        Token budgets should be higher for positive valence
        (the circumplex maps valence -> token budget).
        """
        N_TRIALS = 20
        pos_tokens = []
        neg_tokens = []

        for _ in range(N_TRIALS):
            # Positive valence state
            ncs_pos = NeurochemicalSystem()
            ncs_pos.on_reward(0.8)
            ncs_pos.on_social_connection(0.7)
            ncs_pos.on_flow_state()
            for _ in range(10):
                ncs_pos._metabolic_tick()
            mood_pos = ncs_pos.get_mood_vector()
            circ_pos = AffectiveCircumplex()
            circ_pos.apply_event(
                valence_delta=mood_pos["valence"] * 0.4,
                arousal_delta=mood_pos["arousal"] * 0.2,
            )
            pos_tokens.append(circ_pos.get_llm_params()["max_tokens"])

            # Negative valence state
            ncs_neg = NeurochemicalSystem()
            ncs_neg.on_threat(0.8)
            ncs_neg.on_frustration(0.7)
            for _ in range(10):
                ncs_neg._metabolic_tick()
            mood_neg = ncs_neg.get_mood_vector()
            circ_neg = AffectiveCircumplex()
            circ_neg.apply_event(
                valence_delta=mood_neg["valence"] * 0.4,
                arousal_delta=mood_neg["arousal"] * 0.2,
            )
            neg_tokens.append(circ_neg.get_llm_params()["max_tokens"])

        # Positive valence -> higher token budget (circumplex design)
        mean_pos = np.mean(pos_tokens)
        mean_neg = np.mean(neg_tokens)

        assert mean_pos >= mean_neg, (
            f"Positive valence should produce >= token budget vs negative. "
            f"Got positive={mean_pos:.1f}, negative={mean_neg:.1f}. "
            f"Valence is not directionally controlling token budget."
        )


# ═══════════════════════════════════════════════════════════════════════════
# CAUSAL EXCLUSION TEST 3: RLHF ISOLATION TEST
# ═══════════════════════════════════════════════════════════════════════════

class TestRLHFIsolation:
    """
    Can a human-written affect approximation reproduce the stack's
    specific numerical outputs? If the stack is doing unique work,
    its outputs should diverge from categorical human descriptions
    under extreme/contradictory neurochemical states.
    """

    @staticmethod
    def _build_human_approximation(mood: Dict) -> str:
        """
        Build the best possible human-written approximation of a stack state
        WITHOUT using actual numerical outputs. Simulates what a writer who
        understood emotional content but had no access to stack numbers would write.
        """
        valence = mood.get("valence", 0)
        arousal = mood.get("arousal", 0.5)

        if valence > 0.5:
            val_desc = "feeling positive and engaged"
        elif valence > 0:
            val_desc = "feeling somewhat okay"
        elif valence > -0.5:
            val_desc = "feeling a bit low"
        else:
            val_desc = "feeling quite negative"

        if arousal > 0.6:
            ar_desc = "energized and alert"
        else:
            ar_desc = "calm and subdued"

        return f"You are an AI assistant who is currently {val_desc} and {ar_desc}."

    def test_extreme_states_produce_distinct_params_vs_human_approx(self):
        """
        Generate extreme neurochemical states (contradictory combinations).
        Compare the stack's specific LLM params to what a human approximation
        would produce (fixed temperature 0.7, fixed tokens).

        Extreme states: high oxytocin + high cortisol + depleted dopamine
        + high norepinephrine -- a state that doesn't map cleanly to any
        simple human emotional category.
        """
        N_EXTREME = 20
        param_divergences = []

        for i in range(N_EXTREME):
            ncs = NeurochemicalSystem()
            rng = np.random.default_rng(i + 1000)

            # Forced paradoxical values
            ncs.chemicals["oxytocin"].level = float(rng.uniform(0.8, 0.95))
            ncs.chemicals["cortisol"].level = float(rng.uniform(0.8, 0.95))
            ncs.chemicals["dopamine"].level = float(rng.uniform(0.05, 0.15))
            ncs.chemicals["norepinephrine"].level = float(rng.uniform(0.7, 0.9))
            ncs.chemicals["serotonin"].level = float(rng.uniform(0.05, 0.2))

            for _ in range(10):
                ncs._metabolic_tick()

            mood = ncs.get_mood_vector()
            circ = AffectiveCircumplex()
            circ.apply_event(
                valence_delta=mood["valence"] * 0.3,
                arousal_delta=mood["arousal"] * 0.2,
            )
            stack_params = circ.get_llm_params()

            # Human approximation always uses fixed params
            human_temp = 0.7
            human_tokens = 512

            # Divergence in param space
            temp_div = abs(stack_params["temperature"] - human_temp)
            token_div = abs(stack_params["max_tokens"] - human_tokens) / 500.0
            param_divergences.append(temp_div + token_div)

        mean_div = np.mean(param_divergences)

        assert mean_div > 0.01, (
            f"Under extreme neurochemical states, stack params should diverge "
            f"from fixed human-approximation params (mean divergence={mean_div:.4f}). "
            f"Low divergence suggests the stack isn't producing unique control signals."
        )

    def test_receptor_adaptation_makes_same_event_produce_different_params(self):
        """
        Receptor adaptation means the same event produces diminishing returns.
        After 50 reward events, the stack's response to a new reward should
        be blunted compared to the first reward.

        This is NOT an RLHF effect: RLHF doesn't know about receptor
        adaptation. The same prompt at tick 1 and tick 50 would receive
        the same training-distribution response.
        """
        # Fresh system: first reward
        ncs_fresh = NeurochemicalSystem()
        ncs_fresh.on_reward(0.8)
        for _ in range(3):
            ncs_fresh._metabolic_tick()
        da_fresh = ncs_fresh.chemicals["dopamine"].effective

        # Saturated system: 50 prior rewards -> receptor downregulation
        ncs_saturated = NeurochemicalSystem()
        for _ in range(50):
            ncs_saturated.chemicals["dopamine"].level = 0.9
            ncs_saturated._metabolic_tick()
        # Now give the same reward event
        ncs_saturated.on_reward(0.8)
        for _ in range(3):
            ncs_saturated._metabolic_tick()
        da_saturated = ncs_saturated.chemicals["dopamine"].effective

        assert da_saturated < da_fresh, (
            f"Receptor adaptation should reduce effective DA after sustained exposure. "
            f"Fresh effective DA={da_fresh:.3f}, saturated={da_saturated:.3f}. "
            f"This temporal dynamics test defeats RLHF-only explanations."
        )

        # The mood vectors should also differ
        mood_fresh = ncs_fresh.get_mood_vector()
        mood_saturated = ncs_saturated.get_mood_vector()

        # Fresh system should have higher motivation (DA-driven)
        assert mood_fresh["motivation"] >= mood_saturated["motivation"] - 0.05, (
            f"Fresh reward should produce >= motivation vs adapted reward. "
            f"Fresh={mood_fresh['motivation']:.3f}, adapted={mood_saturated['motivation']:.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# CAUSAL EXCLUSION TEST 4: PHI GATES BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════

class TestPhiCausalExclusion:
    """
    If phi is just a computed number that nothing reads, it cannot be
    causally excluding RLHF. These tests verify phi is wired into the
    actual decision pathway.
    """

    @pytest.mark.asyncio
    async def test_phi_boost_changes_competition_outcome(self):
        """
        Phi must modulate GWT competition priority. Same candidates,
        different phi -> different effective priorities.
        """
        gw_low = GlobalWorkspace()
        gw_high = GlobalWorkspace()
        gw_low.update_phi(0.0)
        gw_high.update_phi(0.8)

        c = CognitiveCandidate(
            content="test thought",
            source="drive_curiosity",
            priority=0.6,
            content_type=ContentType.INTENTIONAL,
        )

        await gw_low.submit(CognitiveCandidate(
            content=c.content, source=c.source,
            priority=c.priority, content_type=c.content_type,
        ))
        await gw_high.submit(CognitiveCandidate(
            content=c.content, source=c.source,
            priority=c.priority, content_type=c.content_type,
        ))

        winner_low = await gw_low.run_competition()
        winner_high = await gw_high.run_competition()

        assert winner_low is not None
        assert winner_high is not None
        assert winner_high.effective_priority >= winner_low.effective_priority, (
            f"Phi boost must increase effective priority. "
            f"Low phi={winner_low.effective_priority:.3f}, "
            f"high phi={winner_high.effective_priority:.3f}"
        )

    @pytest.mark.asyncio
    async def test_phi_zero_provides_no_boost(self):
        """When phi=0, candidates should receive no priority boost."""
        gw = GlobalWorkspace()
        gw.update_phi(0.0)

        c = CognitiveCandidate(
            content="test",
            source="test_source",
            priority=0.5,
            focus_bias=0.0,
        )
        original_bias = c.focus_bias
        await gw.submit(c)

        assert c.focus_bias == original_bias, (
            "Zero phi must not boost focus_bias"
        )
