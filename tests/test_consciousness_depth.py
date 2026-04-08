"""Consciousness Depth Integration Tests
═══════════════════════════════════════
Validates that deepened consciousness modules actually compute,
provide context blocks, close feedback loops, and degrade gracefully.

These tests run WITHOUT a live LLM — they exercise the consciousness
infrastructure directly with mocked/default state.
"""
import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ═════════════════════════════════════════════════════════════════════
# HOMEOSTASIS: Adaptive Setpoints + Drive Regulation
# ═════════════════════════════════════════════════════════════════════

class TestHomeostasisDepth:
    """Verify homeostasis uses proportional control toward adaptive setpoints."""

    def _make_engine(self):
        from core.consciousness.homeostasis import HomeostasisEngine
        return HomeostasisEngine()

    def test_initial_vitality_high(self):
        engine = self._make_engine()
        assert engine.compute_vitality() > 0.7

    @pytest.mark.asyncio
    async def test_proportional_control_approaches_setpoint(self):
        """When integrity is low, proportional control should bring it back up."""
        engine = self._make_engine()
        engine.integrity = 0.3  # Well below setpoint (0.90)
        initial_integrity = engine.integrity
        for _ in range(50):
            await engine.pulse()
        assert engine.integrity > initial_integrity, "Integrity should approach setpoint"
        assert engine.integrity > 0.5, "After 50 pulses, integrity should recover significantly"

    @pytest.mark.asyncio
    async def test_adaptive_setpoints_drift(self):
        """If a drive is chronically low, the setpoint should drift down."""
        engine = self._make_engine()
        engine.integrity = 0.3
        initial_setpoint = engine._setpoints["integrity"]
        # Hold integrity low for many pulses
        for _ in range(200):
            engine.integrity = 0.3  # Force low
            await engine.pulse()
            engine.integrity = 0.3  # Override the proportional control
        # Setpoint should have drifted down
        assert engine._setpoints["integrity"] < initial_setpoint, \
            "Setpoint should drift toward chronically achievable level"

    def test_dominant_deficiency(self):
        engine = self._make_engine()
        engine.curiosity = 0.1  # Very low
        drive, deficit = engine.get_dominant_deficiency()
        assert drive == "curiosity"
        assert deficit > 0.3

    def test_context_block_format(self):
        engine = self._make_engine()
        block = engine.get_context_block()
        assert "## HOMEOSTASIS" in block
        assert "Vitality:" in block
        assert len(block) < 300

    def test_inference_modifiers(self):
        engine = self._make_engine()
        mods = engine.get_inference_modifiers()
        assert "temperature_mod" in mods
        assert "token_multiplier" in mods
        assert "caution_level" in mods
        assert 0.5 <= mods["token_multiplier"] <= 1.5

    def test_response_success_feedback(self):
        engine = self._make_engine()
        engine.integrity = 0.5
        engine.on_response_success(response_length=300)
        assert engine.integrity > 0.5
        assert engine._successful_responses == 1

    def test_response_error_feedback(self):
        engine = self._make_engine()
        engine.integrity = 0.8
        engine.on_response_error("inference")
        assert engine.integrity < 0.8
        assert engine._failed_responses == 1

    def test_vitality_trend(self):
        engine = self._make_engine()
        # Empty history should be stable
        assert engine.get_vitality_trend() == "stable"


# ═════════════════════════════════════════════════════════════════════
# FREE ENERGY: Active Inference + Action Determination
# ═════════════════════════════════════════════════════════════════════

class TestFreeEnergyDepth:
    """Verify FreeEnergy computes, drives actions, and integrates signals."""

    def _make_engine(self):
        from core.consciousness.free_energy import FreeEnergyEngine
        return FreeEnergyEngine()

    def test_compute_produces_state(self):
        engine = self._make_engine()
        state = engine.compute(prediction_error=0.5)
        assert state is not None
        assert 0.0 <= state.free_energy <= 1.0
        assert state.dominant_action in ("engage", "update_beliefs", "act_on_world", "explore", "rest", "reflect")

    def test_high_surprise_drives_belief_update(self):
        engine = self._make_engine()
        # Need multiple computes with extreme prediction error to overcome
        # EMA smoothing and push normalized surprise past the 0.7 threshold.
        # pe=0.99 → raw_surprise=0.99 → nats ≈ 2.35 → normalized ≈ 0.78 > 0.7
        for _ in range(10):
            state = engine.compute(prediction_error=0.99)
        assert state.dominant_action == "update_beliefs"

    def test_low_fe_drives_rest(self):
        engine = self._make_engine()
        # Push FE very low over multiple computes
        for _ in range(20):
            state = engine.compute(prediction_error=0.0)
        assert state.free_energy < 0.3
        assert state.dominant_action == "rest"

    def test_accept_surprise_signal(self):
        engine = self._make_engine()
        engine.accept_surprise_signal(0.9)
        assert engine._last_surprise_signal == 0.9
        # Now compute should blend this in: raw_surprise = 0.6*0.3 + 0.4*0.9 = 0.54
        # Normalized surprise ≈ 0.22 — verify it's meaningfully above baseline (pe=0.3 alone → ~0.12)
        state = engine.compute(prediction_error=0.3)
        baseline_engine = self._make_engine()
        baseline_state = baseline_engine.compute(prediction_error=0.3)
        assert state.surprise > baseline_state.surprise, "Surprise signal should pull surprise upward"

    def test_accept_attention_complexity(self):
        engine = self._make_engine()
        engine.accept_attention_complexity(0.8)
        assert engine._last_attention_complexity == 0.8

    def test_action_hysteresis(self):
        """Action shouldn't flip-flop every tick."""
        engine = self._make_engine()
        engine._ACTION_HOLD_MIN = 3
        # Compute with moderate FE
        engine.compute(prediction_error=0.4)
        action1 = engine._current_action
        # Slightly different input shouldn't change action immediately
        engine.compute(prediction_error=0.42)
        action2 = engine._current_action
        assert action1 == action2, "Hysteresis should prevent rapid action switching"

    def test_context_block(self):
        engine = self._make_engine()
        engine.compute(prediction_error=0.5)
        block = engine.get_context_block()
        assert "## FREE ENERGY" in block
        assert "Drive:" in block
        assert len(block) < 300

    def test_action_urgency(self):
        engine = self._make_engine()
        # Many computes with extreme prediction error to overcome EMA smoothing
        for _ in range(20):
            engine.compute(prediction_error=0.95)
        urgency = engine.get_action_urgency()
        assert urgency > 0.3  # Sustained high surprise should create urgency

    def test_trend_detection(self):
        engine = self._make_engine()
        # Rising FE
        for i in range(15):
            engine.compute(prediction_error=0.1 + i * 0.05)
        assert engine.get_trend() in ("rising", "stable")

    def test_world_model_integration(self):
        """FreeEnergy should be able to compute with EpistemicState."""
        from core.consciousness.world_model import EpistemicState
        ws = EpistemicState()
        ws.update_belief("sun", "is", "hot", 0.99)
        ws.update_belief("water", "is", "wet", 0.95)
        engine = self._make_engine()
        state = engine.compute(prediction_error=0.3, belief_system=ws)
        assert state is not None


# ═════════════════════════════════════════════════════════════════════
# CREDIT ASSIGNMENT: Domain Tracking + Self-Modifying Weights
# ═════════════════════════════════════════════════════════════════════

class TestCreditAssignmentDepth:

    def _make_system(self):
        from core.consciousness.credit_assignment import CreditAssignmentSystem
        return CreditAssignmentSystem()

    def test_assign_and_retrieve(self):
        cas = self._make_system()
        cas.assign_credit("test_action", 0.8, "chat")
        perf = cas.get_domain_performance("chat")
        assert perf != 0.5  # Should have shifted from default

    def test_all_domain_performance(self):
        cas = self._make_system()
        cas.assign_credit("a1", 0.9, "chat")
        cas.assign_credit("a2", 0.3, "logic")
        perfs = cas.get_all_domain_performance()
        assert "chat" in perfs
        assert "logic" in perfs

    def test_self_modifying_weights(self):
        cas = self._make_system()
        initial_weight = cas.domain_weights["chat"]
        # Consistently high performance should increase weight
        for i in range(50):
            cas.assign_credit(f"action_{i}", 0.9, "chat")
        assert cas.domain_weights["chat"] > initial_weight

    def test_performance_trend(self):
        cas = self._make_system()
        # Rising performance
        for i in range(20):
            cas.assign_credit(f"a_{i}", 0.3 + i * 0.03, "chat")
        trend = cas.get_performance_trend("chat")
        assert trend in ("rising", "stable")

    def test_influence_scores(self):
        cas = self._make_system()
        cas.assign_credit("a1", 0.9, "chat")
        cas.assign_credit("a2", 0.1, "logic")
        scores = cas.get_influence_scores()
        assert scores["chat"] > scores["logic"]

    def test_context_block(self):
        cas = self._make_system()
        cas.assign_credit("a1", 0.8, "chat")
        block = cas.get_context_block()
        assert "## COGNITIVE CREDIT" in block
        assert len(block) < 300

    def test_new_domain_auto_created(self):
        cas = self._make_system()
        cas.assign_credit("a1", 0.7, "new_domain_xyz")
        assert "new_domain_xyz" in cas.domain_weights


# ═════════════════════════════════════════════════════════════════════
# PREDICTIVE ENGINE: Surprise + Trend + Feedback
# ═════════════════════════════════════════════════════════════════════

class TestPredictiveEngineDepth:

    def _make_engine(self):
        from core.consciousness.predictive_engine import PredictiveEngine
        return PredictiveEngine()

    @pytest.mark.asyncio
    async def test_predict_and_compute_surprise(self):
        engine = self._make_engine()
        await engine.predict_next_state({"type": "search"})
        surprise = engine.compute_surprise({"total_beliefs": 5})
        assert isinstance(surprise, float)
        assert 0.0 <= surprise <= 1.0

    def test_surprise_signal(self):
        engine = self._make_engine()
        signal = engine.get_surprise_signal()
        assert isinstance(signal, float)

    def test_context_block(self):
        engine = self._make_engine()
        # Need at least one surprise computation
        engine.compute_surprise({}, actual_substrate_x=np.random.randn(64))
        block = engine.get_context_block()
        assert "PRED" in block or "surprise" in block

    def test_surprise_trend(self):
        engine = self._make_engine()
        # Generate rising surprise
        for i in range(25):
            engine.compute_surprise({}, actual_substrate_x=np.random.randn(64) * (1 + i * 0.1))
        trend = engine.get_surprise_trend()
        assert trend in ("rising", "falling", "stable")

    def test_prediction_confidence(self):
        engine = self._make_engine()
        conf = engine.get_prediction_confidence()
        assert 0.0 <= conf <= 1.0


# ═════════════════════════════════════════════════════════════════════
# ATTENTION SCHEMA: Focus Bias + Coherence + Flow
# ═════════════════════════════════════════════════════════════════════

class TestAttentionSchemaDepth:

    def _make_schema(self):
        from core.consciousness.attention_schema import AttentionSchema
        return AttentionSchema()

    @pytest.mark.asyncio
    async def test_set_focus_and_context(self):
        schema = self._make_schema()
        await schema.set_focus("testing consciousness", "test_source", 0.8)
        block = schema.get_context_block()
        assert "ATT" in block or "ATTENTION" in block
        assert "testing consciousness" in block

    @pytest.mark.asyncio
    async def test_focus_bias(self):
        schema = self._make_schema()
        await schema.set_focus("topic A", "source_a", 0.8)
        bias = schema.get_focus_bias_for_source("source_a")
        assert bias > 0  # Should get a boost

    @pytest.mark.asyncio
    async def test_coherence_for_complexity(self):
        schema = self._make_schema()
        await schema.set_focus("topic A", "source_a", 0.8)
        complexity = schema.get_coherence_for_complexity()
        assert 0.0 <= complexity <= 1.0

    @pytest.mark.asyncio
    async def test_flow_state(self):
        schema = self._make_schema()
        # Same topic sustained > 5 ticks
        for _ in range(8):
            await schema.set_focus("same topic repeated here", "source_a", 0.8)
        assert schema.is_in_flow()

    @pytest.mark.asyncio
    async def test_coherence_drops_on_topic_change(self):
        schema = self._make_schema()
        await schema.set_focus("topic about cats", "source_a", 0.8)
        initial_coherence = schema.coherence
        await schema.set_focus("completely different quantum physics", "source_b", 0.7)
        assert schema.coherence < initial_coherence


# ═════════════════════════════════════════════════════════════════════
# WORLD MODEL: Belief Graph + Topic Retrieval
# ═════════════════════════════════════════════════════════════════════

class TestWorldModelDepth:

    def _make_model(self):
        from core.consciousness.world_model import EpistemicState
        return EpistemicState()

    def test_belief_update_and_retrieval(self):
        ws = self._make_model()
        ws.update_belief("sun", "is_a", "star", 0.99)
        beliefs = ws.get_beliefs("sun")
        assert len(beliefs) > 0

    def test_context_block_with_topic(self):
        ws = self._make_model()
        ws.update_belief("python", "is_a", "language", 0.95)
        ws.update_belief("rust", "is_a", "language", 0.90)
        block = ws.get_context_block(topic_hint="python")
        assert "python" in block.lower() or "WORLD" in block or "Belief" in block

    def test_summary(self):
        ws = self._make_model()
        ws.update_belief("a", "rel", "b", 0.8)
        ws.update_belief("c", "rel", "d", 0.6)
        summary = ws.get_summary()
        assert summary["total_beliefs"] >= 2
        assert "avg_confidence" in summary

    def test_relevant_beliefs(self):
        ws = self._make_model()
        ws.update_belief("python", "is_a", "language", 0.95)
        ws.update_belief("java", "is_a", "language", 0.85)
        ws.update_belief("sun", "is_a", "star", 0.99)
        relevant = ws.get_relevant_beliefs("python")
        assert len(relevant) >= 1
        assert any("python" in str(b).lower() for b in relevant)

    def test_extract_beliefs_from_response(self):
        ws = self._make_model()
        ws.extract_beliefs_from_response(
            "Python is a programming language. Rust is fast. Water is essential."
        )
        beliefs = ws.get_beliefs()
        assert len(beliefs) > 0  # Should have extracted at least one

    def test_contradiction_tracking(self):
        ws = self._make_model()
        ws.update_belief("sky", "color_is", "blue", 0.9)
        # Same edge (sky -> blue) but different predicate triggers dissonance
        ws.update_belief("sky", "actually_is", "blue", 0.95)
        # The contradiction count may or may not fire depending on graph structure
        # The important thing is the method works without crashing
        assert hasattr(ws, '_contradiction_count')
        # Test with actual contradiction: same subject->object, different predicate
        ws.world_graph.add_edge("cat", "animal", predicate="is_a", confidence=0.9)
        ws.update_belief("cat", "is_not", "animal", 0.95)  # Contradicts existing
        assert ws._contradiction_count >= 1


# ═════════════════════════════════════════════════════════════════════
# COUNTERFACTUAL ENGINE: Deliberation + Regret/Relief
# ═════════════════════════════════════════════════════════════════════

class TestCounterfactualEngineDepth:

    def _make_engine(self):
        from core.consciousness.counterfactual_engine import CounterfactualEngine
        return CounterfactualEngine()

    @pytest.mark.asyncio
    async def test_deliberate_and_select(self):
        engine = self._make_engine()
        actions = [
            {"type": "learn", "description": "Learn about topic"},
            {"type": "explore", "description": "Explore new area"},
            {"type": "rest", "description": "Consolidate knowledge"},
        ]
        candidates = await engine.deliberate(actions, {"hedonic_score": 0.5, "curiosity": 0.7})
        assert len(candidates) == 3
        best = engine.select(candidates)
        assert best is not None
        assert best.selected is True

    def test_record_outcome_and_learning(self):
        from core.consciousness.counterfactual_engine import ActionCandidate
        engine = self._make_engine()
        selected = ActionCandidate(
            action_type="learn", action_params={}, description="Learn",
            simulated_hedonic_gain=0.1, heartstone_alignment=0.8,
            expected_outcome="knowledge gained", score=0.5, selected=True,
        )
        engine.record_outcome(selected, actual_hedonic_change=0.2)
        assert len(engine._records) == 1
        assert engine._cumulative_relief > 0  # Actual > expected = relief

    def test_decision_quality(self):
        engine = self._make_engine()
        # Initially no data
        quality = engine.get_decision_quality()
        assert quality == 0.5  # Default

    def test_context_block(self):
        engine = self._make_engine()
        block = engine.get_context_block()
        assert "CFE" in block or "COUNTERFACTUAL" in block or "decisions" in block

    @pytest.mark.asyncio
    async def test_evaluate_autonomous_action(self):
        engine = self._make_engine()
        action = {"type": "learn", "description": "Study new framework"}
        context = {"hedonic_score": 0.5, "curiosity": 0.8}
        result = await engine.evaluate_autonomous_action(action, context)
        assert result is not None
        assert result.selected is True


# ═════════════════════════════════════════════════════════════════════
# GRACEFUL DEGRADATION
# ═════════════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    """Verify everything works when modules are unavailable."""

    def test_homeostasis_context_without_services(self):
        from core.consciousness.homeostasis import HomeostasisEngine
        engine = HomeostasisEngine()
        block = engine.get_context_block()
        assert block  # Should still produce output

    def test_free_energy_without_belief_system(self):
        from core.consciousness.free_energy import FreeEnergyEngine
        engine = FreeEnergyEngine()
        state = engine.compute(prediction_error=0.5, belief_system=None)
        assert state is not None
        assert state.free_energy > 0

    def test_credit_empty_history(self):
        from core.consciousness.credit_assignment import CreditAssignmentSystem
        cas = CreditAssignmentSystem()
        block = cas.get_context_block()
        assert block == ""  # No events = no block

    def test_attention_no_focus(self):
        from core.consciousness.attention_schema import AttentionSchema
        schema = AttentionSchema()
        block = schema.get_context_block()
        # Even without focus, may produce a "no focus" message — that's fine
        assert isinstance(block, str)
        bias = schema.get_focus_bias_for_source("anything")
        assert bias == 0.0

    def test_world_model_empty(self):
        from core.consciousness.world_model import EpistemicState
        ws = EpistemicState()
        block = ws.get_context_block()
        # Should handle empty graph gracefully
        assert isinstance(block, str)


# ═════════════════════════════════════════════════════════════════════
# FEEDBACK LOOP CONVERGENCE
# ═════════════════════════════════════════════════════════════════════

class TestFeedbackConvergence:
    """Verify that feedback loops converge and don't spiral."""

    @pytest.mark.asyncio
    async def test_homeostasis_converges(self):
        """Homeostasis should stabilize, not oscillate wildly."""
        from core.consciousness.homeostasis import HomeostasisEngine
        engine = HomeostasisEngine()
        engine.integrity = 0.2  # Start low
        vitalities = []
        for _ in range(100):
            await engine.pulse()
            vitalities.append(engine.compute_vitality())
        # Check convergence: later values should have less variance
        first_half_var = np.var(vitalities[:50])
        second_half_var = np.var(vitalities[50:])
        assert second_half_var <= first_half_var + 0.01, \
            "Homeostasis should converge (second half variance should be <= first half)"

    def test_credit_weights_bounded(self):
        """Credit weights should stay bounded even with extreme inputs."""
        from core.consciousness.credit_assignment import CreditAssignmentSystem
        cas = CreditAssignmentSystem()
        for i in range(500):
            cas.assign_credit(f"extreme_{i}", 1.0, "chat")
        assert cas.domain_weights["chat"] <= 2.0
        for i in range(500):
            cas.assign_credit(f"terrible_{i}", -1.0, "logic")
        assert cas.domain_weights["logic"] >= 0.3

    def test_free_energy_ema_bounded(self):
        """Free energy EMA should stay in [0, 1]."""
        from core.consciousness.free_energy import FreeEnergyEngine
        engine = FreeEnergyEngine()
        for _ in range(200):
            state = engine.compute(prediction_error=1.0)
        assert 0.0 <= state.free_energy <= 1.0
        for _ in range(200):
            state = engine.compute(prediction_error=0.0)
        assert 0.0 <= state.free_energy <= 1.0
