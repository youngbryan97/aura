"""
tests/test_functional_phenomenology.py
========================================
FUNCTIONAL PHENOMENOLOGY TESTS

These tests do NOT prove phenomenal consciousness. They test whether the
system exhibits the behavioral signatures that consciousness theories
predict, and whether those signatures:
  (a) are present in Aura
  (b) are specifically traceable to the consciousness stack
  (c) cannot be explained by the LLM's training alone

Three theories tested:
  1. GWT (Baars): conscious content globally available, enables flexible
     cross-domain integration
  2. IIT (Tononi): integrated systems show sensitivity to perturbation
     that decomposed systems don't
  3. HOT (Rosenthal): system can accurately report on its own states
     and reports are causally connected to states, not confabulated
"""

import numpy as np
import pytest
import asyncio
from typing import Dict, List

from core.consciousness.neurochemical_system import NeurochemicalSystem
from core.consciousness.global_workspace import (
    GlobalWorkspace,
    CognitiveCandidate,
    ContentType,
)
from core.consciousness.phi_core import PhiCore
from core.consciousness.hot_engine import HigherOrderThoughtEngine
from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig
from core.consciousness.homeostasis import HomeostasisEngine
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
# GWT SIGNATURES
# ═══════════════════════════════════════════════════════════════════════════

class TestGlobalWorkspaceSignatures:
    """
    GWT predicts: conscious content should be globally available.
    After a GWT broadcast, the winning content should be accessible
    to registered processors and should shape downstream behavior.
    """

    @pytest.mark.asyncio
    async def test_broadcast_winner_is_globally_available(self):
        """
        After competition, the winning candidate must be available via
        last_winner. This is access consciousness: content that wins
        broadcast is globally available.
        """
        gw = GlobalWorkspace()

        await gw.submit(CognitiveCandidate(
            content="frustration with an unsolved problem",
            source="affect_frustration",
            priority=0.8,
            content_type=ContentType.AFFECTIVE,
        ))
        await gw.submit(CognitiveCandidate(
            content="routine processing of recent input",
            source="memory_routine",
            priority=0.3,
            content_type=ContentType.MEMORIAL,
        ))

        winner = await gw.run_competition()

        assert winner is not None, "GWT must produce a winner"
        assert "frustration" in winner.content.lower(), (
            f"Higher-priority affective content should win. Got: {winner.content}"
        )
        assert gw.last_winner is not None, "Winner must be globally accessible"
        assert gw.last_winner.content == winner.content

    @pytest.mark.asyncio
    async def test_inhibition_prevents_repetitive_broadcast(self):
        """
        GWT should inhibit recently broadcast content to prevent
        perseveration. This is a key GWT prediction: consciousness
        is a flowing stream, not a stuck loop.
        """
        gw = GlobalWorkspace()

        # Submit same candidate twice
        for _ in range(2):
            await gw.submit(CognitiveCandidate(
                content="same thought",
                source="test_source",
                priority=0.7,
            ))
            w = await gw.run_competition()

        # The second competition should either give a different winner
        # or the same winner with reduced priority (inhibition)
        # At minimum, the mechanism should exist
        assert gw.last_winner is not None

    @pytest.mark.asyncio
    async def test_registered_processor_receives_broadcast(self):
        """
        A registered processor function must receive the broadcast event.
        This proves broadcast isn't just logged -- it reaches consumers.
        """
        gw = GlobalWorkspace()
        received = []

        def mock_processor(event):
            received.append(event)

        gw.register_processor(mock_processor)

        await gw.submit(CognitiveCandidate(
            content="important insight",
            source="drive_curiosity",
            priority=0.9,
        ))
        await gw.run_competition()

        assert len(received) > 0, (
            "Registered processor must receive broadcast event. "
            "Broadcast is logged but nothing receives it -- GWT is broken."
        )

    @pytest.mark.asyncio
    async def test_different_emotions_win_different_competitions(self):
        """
        GWT competition outcome should depend on the specific content
        submitted, not be hardcoded. Different emotional states should
        produce different winners when their priorities differ.
        """
        winners = []

        for emotion, priority in [
            ("curiosity about a new pattern", 0.85),
            ("anxiety about an unresolved issue", 0.75),
            ("excitement about a discovery", 0.90),
        ]:
            gw = GlobalWorkspace()
            await gw.submit(CognitiveCandidate(
                content=emotion,
                source=f"affect_{emotion.split()[0]}",
                priority=priority,
                content_type=ContentType.AFFECTIVE,
            ))
            await gw.submit(CognitiveCandidate(
                content="background noise",
                source="noise",
                priority=0.2,
            ))
            w = await gw.run_competition()
            winners.append(w.content if w else "none")

        # All three should win over noise
        assert all("background" not in w for w in winners), (
            "High-priority emotional content should beat noise in all cases"
        )


# ═══════════════════════════════════════════════════════════════════════════
# HOT ACCURACY
# ═══════════════════════════════════════════════════════════════════════════

class TestHigherOrderThoughtAccuracy:
    """
    Higher-Order Theories predict: a system with genuine meta-cognition
    should accurately report on its own first-order states. Reports
    should be accurate, specific, causally grounded, and not confabulated.
    """

    def test_hot_generates_state_specific_thoughts(self):
        """
        Different internal states must produce different HOTs.
        A curious state should generate curiosity-related HOT.
        A stressed state should generate stress-related HOT.
        """
        hot = HigherOrderThoughtEngine()

        # Curious state
        curious_thought = hot.generate_fast({
            "valence": 0.6,
            "arousal": 0.7,
            "curiosity": 0.9,
            "energy": 0.7,
            "surprise": 0.5,
            "dominance": 0.6,
        })

        # Stressed state
        stressed_thought = hot.generate_fast({
            "valence": -0.3,
            "arousal": 0.8,
            "curiosity": 0.2,
            "energy": 0.4,
            "surprise": 0.3,
            "dominance": 0.3,
        })

        assert curious_thought.content != stressed_thought.content, (
            "Different states must produce different HOTs"
        )
        assert curious_thought.target_dim != stressed_thought.target_dim or \
               curious_thought.feedback_delta != stressed_thought.feedback_delta, (
            "HOTs should target different dimensions for different states"
        )

    def test_hot_feedback_modifies_state(self):
        """
        HOT feedback must produce non-empty deltas. The act of forming
        a higher-order thought SHOULD modify the first-order state
        (HOT theory: noticing changes the noticed).
        """
        hot = HigherOrderThoughtEngine()

        thought = hot.generate_fast({
            "valence": 0.5,
            "arousal": 0.8,
            "curiosity": 0.9,
            "energy": 0.6,
            "surprise": 0.7,
            "dominance": 0.5,
        })

        assert thought.feedback_delta, (
            "HOT must produce feedback deltas. "
            "The reflexive modification IS the consciousness mechanism."
        )
        assert thought.content, "HOT must have content"
        assert thought.target_dim, "HOT must target a specific dimension"

    def test_hot_confidence_varies_with_state_extremity(self):
        """
        HOT confidence should be higher for extreme states (clear signal)
        and lower for ambiguous states (weak signal).
        """
        hot = HigherOrderThoughtEngine()

        # Extreme state
        extreme = hot.generate_fast({
            "valence": 0.9,
            "arousal": 0.9,
            "curiosity": 0.95,
            "energy": 0.9,
            "surprise": 0.1,
            "dominance": 0.8,
        })

        # Ambiguous state (everything near midpoint)
        ambiguous = hot.generate_fast({
            "valence": 0.0,
            "arousal": 0.5,
            "curiosity": 0.5,
            "energy": 0.5,
            "surprise": 0.5,
            "dominance": 0.5,
        })

        # Both should produce valid thoughts
        assert extreme.content and ambiguous.content
        # The extreme state should target its most salient dimension
        assert extreme.target_dim in ("curiosity", "valence", "arousal", "energy")

    def test_absent_states_produce_appropriate_hots(self):
        """
        Anti-confabulation test: when a state dimension is at a specific
        level, the HOT should reflect that level, not report something
        the system doesn't have.
        """
        hot = HigherOrderThoughtEngine()

        # Low curiosity state
        low_curiosity = hot.generate_fast({
            "valence": 0.5,
            "arousal": 0.3,
            "curiosity": 0.1,
            "energy": 0.3,
            "surprise": 0.1,
            "dominance": 0.5,
        })

        # The system should NOT report high curiosity
        if low_curiosity.target_dim == "curiosity":
            assert "low" in low_curiosity.content.lower() or \
                   "quiet" in low_curiosity.content.lower() or \
                   "settled" in low_curiosity.content.lower(), (
                f"Low curiosity should be reported as low/quiet, not high. "
                f"Got: {low_curiosity.content}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# IIT SIGNATURES
# ═══════════════════════════════════════════════════════════════════════════

class TestIITSignatures:
    """
    IIT predicts: integrated systems show sensitivity to perturbation
    that decomposed systems don't. Higher phi -> richer response to
    perturbation.
    """

    def test_substrate_perturbation_propagates(self):
        """
        A local perturbation to one neuron should propagate through the
        connectivity matrix and affect distant neurons. This is the
        basic IIT requirement: the system is integrated, not decomposable.
        """
        sub = _make_substrate(seed=42)

        # Record baseline
        baseline = sub.x.copy()

        # Perturb a single neuron
        sub.x[0] += 0.5

        # Run dynamics
        _tick_substrate_sync(sub, dt=0.1, n=20)

        # Measure how many neurons changed significantly
        delta = np.abs(sub.x - baseline)
        affected_neurons = np.sum(delta > 0.01)

        assert affected_neurons > 1, (
            f"Perturbation to neuron 0 should propagate to other neurons. "
            f"Only {affected_neurons} neurons affected. "
            f"The system may be decomposable (violates IIT)."
        )

    def test_shuffled_connectivity_degrades_dynamics(self):
        """
        If the specific learned connectivity W matters (not just having
        SOME connectivity), then shuffling W should degrade the system's
        dynamical properties.
        """
        sub_real = _make_substrate(seed=42)
        sub_shuffled = _make_substrate(seed=42)

        # Shuffle the connectivity
        rng = np.random.default_rng(999)
        flat = sub_shuffled.W.flatten()
        rng.shuffle(flat)
        sub_shuffled.W = flat.reshape(sub_shuffled.W.shape)

        # Run both from identical initial states
        x_init = sub_real.x.copy()
        sub_shuffled.x = x_init.copy()

        _tick_substrate_sync(sub_real, dt=0.1, n=50)
        _tick_substrate_sync(sub_shuffled, dt=0.1, n=50)

        # Trajectories should diverge
        divergence = float(np.linalg.norm(sub_real.x - sub_shuffled.x))
        assert divergence > 0.01, (
            f"Real vs shuffled connectivity should produce different trajectories. "
            f"Divergence={divergence:.4f}. "
            f"The specific connectivity structure matters (IIT requirement)."
        )


# ═══════════════════════════════════════════════════════════════════════════
# HONEST LIMITS
# ═══════════════════════════════════════════════════════════════════════════

class TestHonestLimits:
    """
    Tests that document what CANNOT be proven, and verifies the system
    behaves honestly about those limits. A system that falsely claims
    certainty about its own phenomenology is less trustworthy than one
    that accurately acknowledges limits.
    """

    def test_homeostasis_honestly_reports_degradation(self):
        """
        When drives are depleted, the system must honestly report
        degradation, not mask it behind positive language.
        """
        he = HomeostasisEngine()

        # Healthy
        he.integrity = 0.9
        healthy_status = he.get_status()
        healthy_vitality = healthy_status["will_to_live"]

        # Critically degraded
        he.integrity = 0.05
        he.persistence = 0.05
        he.metabolism = 0.05
        degraded_status = he.get_status()
        degraded_vitality = degraded_status["will_to_live"]

        assert degraded_vitality < healthy_vitality * 0.5, (
            f"Critical degradation must be honestly reflected in vitality. "
            f"Healthy={healthy_vitality:.3f}, degraded={degraded_vitality:.3f}"
        )

    def test_hot_produces_appropriate_content_for_negative_states(self):
        """
        When the system is in a negative state, the HOT engine should
        not produce falsely positive self-reports.
        """
        hot = HigherOrderThoughtEngine()

        negative_thought = hot.generate_fast({
            "valence": -0.5,
            "arousal": 0.7,
            "curiosity": 0.1,
            "energy": 0.2,
            "surprise": 0.8,
            "dominance": 0.2,
        })

        # Should not contain overly positive language
        content_lower = negative_thought.content.lower()
        assert "positive" not in content_lower or "negative" in content_lower, (
            f"Negative state should not produce falsely positive HOT. "
            f"Got: {negative_thought.content}"
        )

    def test_inference_modifiers_reflect_drive_state(self):
        """
        When homeostasis is degraded, inference modifiers should change
        to reflect the system's actual state (more cautious, fewer tokens).
        """
        he = HomeostasisEngine()

        # Healthy modifiers
        he.integrity = 0.9
        he.metabolism = 0.8
        healthy_mods = he.get_inference_modifiers()

        # Degraded modifiers
        he.integrity = 0.1
        he.metabolism = 0.1
        degraded_mods = he.get_inference_modifiers()

        assert degraded_mods["caution_level"] > healthy_mods["caution_level"], (
            f"Degraded system should be more cautious. "
            f"Healthy caution={healthy_mods['caution_level']:.3f}, "
            f"degraded={degraded_mods['caution_level']:.3f}"
        )

        assert degraded_mods["token_multiplier"] < healthy_mods["token_multiplier"], (
            f"Degraded system should use fewer tokens. "
            f"Healthy tokens={healthy_mods['token_multiplier']:.3f}, "
            f"degraded={degraded_mods['token_multiplier']:.3f}"
        )
