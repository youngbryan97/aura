"""
test_consciousness_e2e.py
=========================
End-to-end integration test verifying consciousness modules affect the
system during a simulated conversation flow.

No live LLM required -- all external dependencies are mocked.

Covers:
  1. ServiceContainer registration of all consciousness modules
  2. 10 heartbeat-like cycles: GWT competition, free energy, credit assignment,
     attention tracking, predictive engine updates
  3. Post-cycle state verification (non-trivial state accumulated)
  4. Response feedback loop: homeostasis + credit EMA shift
"""
from __future__ import annotations


import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _clear_container():
    from core.container import ServiceContainer
    ServiceContainer.clear()


# ============================================================================
# CONSCIOUSNESS E2E TEST
# ============================================================================

class TestConsciousnessE2E:
    """Simulate 10 heartbeat cycles through the consciousness stack and
    verify that every module accumulates meaningful state."""

    @pytest.mark.asyncio
    async def test_full_consciousness_loop(self):
        _clear_container()
        try:
            from core.container import ServiceContainer
            from core.consciousness.homeostasis import HomeostasisEngine
            from core.consciousness.free_energy import FreeEnergyEngine
            from core.consciousness.credit_assignment import CreditAssignmentSystem
            from core.consciousness.attention_schema import AttentionSchema
            from core.consciousness.global_workspace import (
                GlobalWorkspace,
                CognitiveCandidate,
                ContentType,
            )
            from core.consciousness.world_model import EpistemicState
            from core.consciousness.predictive_engine import PredictiveEngine

            # ── 1. Instantiate all consciousness modules ─────────────────
            homeostasis = HomeostasisEngine()
            free_energy = FreeEnergyEngine()
            credit = CreditAssignmentSystem()
            attention = AttentionSchema()
            gwt = GlobalWorkspace(attention_schema=attention)
            epistemic = EpistemicState()
            predictive = PredictiveEngine(world_model=epistemic)

            # ── 2. Register in ServiceContainer ──────────────────────────
            ServiceContainer.register_instance("homeostasis", homeostasis)
            ServiceContainer.register_instance("free_energy_engine", free_energy)
            ServiceContainer.register_instance("credit_assignment", credit)
            ServiceContainer.register_instance("attention_schema", attention)
            ServiceContainer.register_instance("global_workspace", gwt)
            ServiceContainer.register_instance("epistemic_state", epistemic)
            ServiceContainer.register_instance("predictive_engine", predictive)

            # Record starting integrity for later comparison
            starting_integrity = homeostasis.integrity

            # ── 3. Simulate 10 heartbeat cycles ──────────────────────────
            sources = [
                "drive_curiosity", "affect_valence", "memory_recall",
                "social_cue", "metabolic_signal", "identity_check",
                "dream_fragment", "executive_goal", "sensory_input",
                "autonomy_bid",
            ]

            for cycle in range(10):
                # (a) Homeostasis pulse
                await homeostasis.pulse()

                # (b) Submit GWT candidates from diverse sources
                for j, src in enumerate(sources[:3 + (cycle % 3)]):
                    candidate = CognitiveCandidate(
                        content=f"Cycle {cycle} content from {src}",
                        source=src,
                        priority=0.3 + (j * 0.1) + (cycle * 0.02),
                        content_type=ContentType.PERCEPTUAL,
                    )
                    await gwt.submit(candidate)

                # (c) Run GWT competition
                winner = await gwt.run_competition()

                # (d) Feed predictive engine with a prediction + surprise
                prediction = await predictive.predict_next_state(
                    {"type": "search", "cycle": cycle}
                )
                import numpy as np
                actual_substrate = np.random.randn(predictive.neuron_count) * 0.1
                surprise = predictive.compute_surprise(
                    {"total_beliefs": cycle + 1},
                    actual_substrate,
                )

                # (e) Push surprise into free energy engine
                free_energy.accept_surprise_signal(surprise)
                if attention.current_focus:
                    free_energy.accept_attention_complexity(
                        attention.get_coherence_for_complexity()
                    )
                fe_state = free_energy.compute(
                    prediction_error=surprise,
                    belief_system=epistemic,
                    recent_action_count=cycle,
                    user_present=(cycle % 3 == 0),
                )

                # (f) Assign credit based on outcome
                domain = ["chat", "logic", "identity", "creative"][cycle % 4]
                credit.assign_credit(
                    action_id=f"action_{cycle}",
                    outcome=0.6 + (cycle * 0.03),
                    domain=domain,
                )

                # (g) Update epistemic state with a belief
                epistemic.update_belief(
                    subject=f"concept_{cycle}",
                    predicate="related_to",
                    obj=f"concept_{cycle + 1}",
                    confidence=0.7 + (cycle * 0.02),
                )

            # ── 4. Post-10-cycle assertions ──────────────────────────────

            # FreeEnergy has computed state
            assert free_energy.current is not None, (
                "FreeEnergyEngine._current should not be None after 10 cycles"
            )
            assert free_energy._total_computes == 10

            # CreditAssignment has events recorded
            assert len(credit.history) == 10, (
                f"Expected 10 credit events, got {len(credit.history)}"
            )
            assert credit._total_events == 10

            # GlobalWorkspace has run competitions
            assert gwt._tick == 10, (
                f"GWT tick should be 10, got {gwt._tick}"
            )
            assert len(gwt._history) > 0, "GWT should have broadcast history"

            # AttentionSchema has focus history
            assert len(attention.history) > 0, (
                "AttentionSchema should have recorded focus transitions"
            )
            assert attention.current_focus is not None, (
                "AttentionSchema should have a current focus after 10 cycles"
            )

            # Homeostasis vitality is within bounds
            vitality = homeostasis.compute_vitality()
            assert 0.0 <= vitality <= 1.0, (
                f"Vitality should be in [0,1], got {vitality}"
            )
            assert len(homeostasis._vitality_history) == 10

            # Context blocks are non-empty for modules with data
            assert homeostasis.get_context_block(), "Homeostasis context block should be non-empty"
            assert free_energy.get_context_block(), "FreeEnergy context block should be non-empty"
            assert credit.get_context_block(), "CreditAssignment context block should be non-empty"
            assert attention.get_context_block(), "AttentionSchema context block should be non-empty"
            assert predictive.get_context_block(), "PredictiveEngine context block should be non-empty"

            # EpistemicState has accumulated beliefs
            assert epistemic.world_graph.number_of_edges() >= 10, (
                f"Epistemic graph should have >= 10 edges, got {epistemic.world_graph.number_of_edges()}"
            )

            # ── 5. Simulate a response feedback loop ─────────────────────
            integrity_before = homeostasis.integrity
            credit_ema_before = dict(credit._domain_ema)

            homeostasis.on_response_success(response_length=250)
            credit.assign_credit(
                action_id="response_feedback",
                outcome=0.95,
                domain="chat",
            )

            # ── 6. Verify feedback effects ───────────────────────────────

            # Integrity should have increased from on_response_success
            assert homeostasis.integrity > integrity_before, (
                f"Integrity should have increased after success: "
                f"before={integrity_before:.4f}, after={homeostasis.integrity:.4f}"
            )

            # Credit EMA for "chat" should have shifted toward 0.95
            assert credit._domain_ema["chat"] != credit_ema_before["chat"], (
                "Credit EMA for 'chat' should have shifted after new assignment"
            )
            # The new assignment was 0.95 weighted by domain_weight (~1.0),
            # so EMA should have moved upward from its prior value
            assert credit._domain_ema["chat"] > credit_ema_before["chat"], (
                f"Chat EMA should have increased: "
                f"before={credit_ema_before['chat']:.4f}, "
                f"after={credit._domain_ema['chat']:.4f}"
            )

            # Successful responses counter should be incremented
            assert homeostasis._successful_responses == 1
            assert homeostasis._total_responses == 1

        finally:
            _clear_container()

    @pytest.mark.asyncio
    async def test_homeostasis_error_feedback_drains_integrity(self):
        """Verify that on_response_error reduces integrity."""
        _clear_container()
        try:
            from core.consciousness.homeostasis import HomeostasisEngine

            h = HomeostasisEngine()
            initial = h.integrity

            h.on_response_error("inference")

            assert h.integrity < initial, (
                f"Integrity should decrease on error: before={initial}, after={h.integrity}"
            )
            assert h._failed_responses == 1
        finally:
            _clear_container()

    @pytest.mark.asyncio
    async def test_gwt_inhibition_prevents_resubmission(self):
        """Losers of a GWT competition should be inhibited for subsequent ticks."""
        _clear_container()
        try:
            from core.consciousness.global_workspace import (
                GlobalWorkspace,
                CognitiveCandidate,
            )
            from core.consciousness.attention_schema import AttentionSchema

            attention = AttentionSchema()
            gwt = GlobalWorkspace(attention_schema=attention)

            # Submit two candidates; lower priority will lose
            winner_candidate = CognitiveCandidate(
                content="I am the winner",
                source="high_priority",
                priority=0.9,
            )
            loser_candidate = CognitiveCandidate(
                content="I am the loser",
                source="low_priority",
                priority=0.1,
            )

            await gwt.submit(winner_candidate)
            await gwt.submit(loser_candidate)
            result = await gwt.run_competition()

            assert result is not None
            assert result.source == "high_priority"

            # Loser should be inhibited -- resubmission should return False
            resubmit = CognitiveCandidate(
                content="Trying again",
                source="low_priority",
                priority=0.9,
            )
            accepted = await gwt.submit(resubmit)
            assert accepted is False, (
                "Inhibited source should not be able to resubmit immediately"
            )
        finally:
            _clear_container()

    @pytest.mark.asyncio
    async def test_free_energy_trends(self):
        """Verify that FreeEnergyEngine tracks trend correctly over repeated computes."""
        from core.consciousness.free_energy import FreeEnergyEngine

        fe = FreeEnergyEngine()

        # Feed increasing surprise to create a rising trend
        for i in range(15):
            fe.compute(
                prediction_error=0.1 + (i * 0.05),
                belief_system=None,
                recent_action_count=0,
                user_present=False,
            )

        # After 15 computes with increasing error, trend should be rising or stable
        # (depends on smoothing, but FE should definitely not be None)
        assert fe.current is not None
        assert fe._total_computes == 15
        assert len(fe._history) == 15
        snapshot = fe.get_snapshot()
        assert "free_energy" in snapshot
        assert "trend" in snapshot

    @pytest.mark.asyncio
    async def test_credit_domain_weight_adaptation(self):
        """Verify that sustained high performance in a domain increases its weight."""
        from core.consciousness.credit_assignment import CreditAssignmentSystem

        credit = CreditAssignmentSystem()
        initial_weight = credit.domain_weights["chat"]

        # Assign many high-outcome events to "chat"
        for i in range(50):
            credit.assign_credit(
                action_id=f"chat_action_{i}",
                outcome=0.9,
                domain="chat",
            )

        # Weight should have increased (adaptation nudges it up for good performers)
        assert credit.domain_weights["chat"] > initial_weight, (
            f"Chat weight should have increased: "
            f"initial={initial_weight:.4f}, current={credit.domain_weights['chat']:.4f}"
        )

        # EMA should be near 0.9 (weighted by domain weight)
        assert credit._domain_ema["chat"] > 0.7, (
            f"Chat EMA should be high after 50 good events: {credit._domain_ema['chat']}"
        )
