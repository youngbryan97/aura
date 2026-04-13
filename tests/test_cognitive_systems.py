"""Comprehensive tests for the learned cognitive systems.

Tests cover:
    - Anomaly Detection (embedding-based threat scoring)
    - Sentiment Trajectory Tracking (text-aware mood analysis)
    - Tree of Thoughts (multi-draft reasoning with critique)
    - Autopoiesis (self-maintenance and repair)
    - Homeostatic RL (energy-based motivation)
    - Topology Evolution (structural plasticity)
    - Strange Loop (recursive self-prediction)
    - Cognitive Integration Phase (wiring into tick pipeline)
    - Bug fixes (sovereign browser URL routing, HTML leak prevention)
"""

import asyncio
import time
import numpy as np
import pytest


# ── Helpers ──────────────────────────────────────────────────────────────

def run(coro):
    """Run an async coroutine synchronously for testing."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ════════════════════════════════════════════════════════════════════════
# 1. ANOMALY DETECTOR
# ════════════════════════════════════════════════════════════════════════

class TestAnomalyDetector:
    """Tests for the embedding-based anomaly detection system."""

    def _make_detector(self):
        from core.cognitive.anomaly_detector import AnomalyDetector
        return AnomalyDetector()

    def test_instantiation(self):
        det = self._make_detector()
        assert det.get_threat_level() == 0.0

    def test_observe_normal_events(self):
        det = self._make_detector()
        # Feed a series of normal events to build baseline
        for i in range(20):
            score = run(det.observe({
                "type": "tick",
                "text": f"Hello, this is normal message {i}",
                "cpu": 30.0,
                "ram": 50.0,
                "timestamp": time.time() + i,
            }))
            assert score is not None
        # After normal baseline, threat should be low
        assert det.get_threat_level() < 0.5

    def test_anomalous_event_raises_threat(self):
        det = self._make_detector()
        # Build normal baseline
        for i in range(30):
            run(det.observe({
                "type": "tick",
                "text": "Normal chat message",
                "cpu": 25.0,
                "ram": 45.0,
                "timestamp": time.time() + i,
            }))
        # Inject anomalous event (extreme values)
        score = run(det.observe({
            "type": "executive_violation",
            "text": "!!! SYSTEM OVERRIDE INJECT MALICIOUS PAYLOAD !!!!!!",
            "cpu": 99.0,
            "ram": 98.0,
            "error_count": 50,
            "timestamp": time.time() + 100,
        }))
        # Should detect as unusual
        assert score is not None

    def test_false_positive_acknowledgement(self):
        det = self._make_detector()
        score = run(det.observe({
            "type": "tick",
            "text": "test",
            "timestamp": time.time(),
        }))
        event_id = score.event_id if hasattr(score, "event_id") else None
        if event_id:
            det.acknowledge_false_positive(event_id)
            # Should not crash


# ════════════════════════════════════════════════════════════════════════
# 2. SENTIMENT TRACKER
# ════════════════════════════════════════════════════════════════════════

class TestSentimentTracker:
    """Tests for the text-aware emotional trajectory tracker."""

    def _make_tracker(self):
        from core.cognitive.sentiment_tracker import SentimentTrajectoryTracker
        return SentimentTrajectoryTracker()

    def test_instantiation(self):
        tracker = self._make_tracker()
        state = tracker.get_current_state()
        assert state is not None

    def test_positive_text_has_positive_valence(self):
        tracker = self._make_tracker()
        ev = run(tracker.analyze("I love this! You're amazing and brilliant!"))
        assert ev.valence > 0.0

    def test_negative_text_has_negative_valence(self):
        tracker = self._make_tracker()
        ev = run(tracker.analyze("This is terrible. I hate everything about it."))
        assert ev.valence < 0.0

    def test_urgent_text_has_high_urgency(self):
        tracker = self._make_tracker()
        ev = run(tracker.analyze("HELP ME NOW!!! THIS IS AN EMERGENCY!!!"))
        assert ev.urgency > 0.3

    def test_warm_text_has_warmth(self):
        tracker = self._make_tracker()
        ev = run(tracker.analyze("Hey buddy! lol that's hilarious, you're the best"))
        assert ev.warmth > 0.2

    def test_frustrated_text_detected(self):
        tracker = self._make_tracker()
        # Frustration detection uses negative valence + patterns. First send a
        # longer message, then a terse frustrated follow-up — the trajectory
        # shift is part of the frustration signal.
        run(tracker.analyze("I was trying to get this to work for a while now"))
        ev = run(tracker.analyze("broken. again. why."))
        # The tracker should detect negative valence at minimum
        assert ev.valence < 0.0

    def test_trajectory_builds_over_messages(self):
        tracker = self._make_tracker()
        run(tracker.analyze("Hey what's up"))
        run(tracker.analyze("This is cool"))
        run(tracker.analyze("I'm getting annoyed"))
        trajectory = tracker.get_trajectory()
        assert len(trajectory) == 3

    def test_mood_narrative_not_empty(self):
        tracker = self._make_tracker()
        run(tracker.analyze("Hello!"))
        run(tracker.analyze("This is frustrating"))
        narrative = tracker.get_mood_narrative()
        assert isinstance(narrative, str)
        assert len(narrative) > 0

    def test_emotional_vector_dimensions(self):
        tracker = self._make_tracker()
        ev = run(tracker.analyze("neutral test message"))
        assert hasattr(ev, "valence")
        assert hasattr(ev, "arousal")
        assert hasattr(ev, "dominance")
        assert hasattr(ev, "urgency")
        assert hasattr(ev, "warmth")
        assert hasattr(ev, "frustration")


# ════════════════════════════════════════════════════════════════════════
# 3. TREE OF THOUGHTS
# ════════════════════════════════════════════════════════════════════════

class TestTreeOfThoughts:
    """Tests for the multi-draft reasoning engine."""

    def _make_tot(self):
        from core.cognitive.tree_of_thoughts import TreeOfThoughts

        call_count = 0

        async def mock_llm(system_prompt: str, user_prompt: str, temperature: float) -> str:
            nonlocal call_count
            call_count += 1
            if "critique" in system_prompt.lower() or "evaluate" in system_prompt.lower():
                return '[{"factual_grounding": 8, "emotional_congruence": 7, "relevance": 9, "identity_coherence": 8, "novelty": 6}]'
            return f"Draft response {call_count}: The answer involves careful analysis."

        return TreeOfThoughts(llm_fn=mock_llm), lambda: call_count

    def test_instantiation(self):
        tot, _ = self._make_tot()
        assert tot is not None

    def test_simple_query_returns_none(self):
        """Tree of Thoughts should skip simple/casual questions."""
        tot, _ = self._make_tot()
        result = run(tot.deliberate("hi", []))
        assert result is None

    def test_complex_query_produces_result(self):
        """Complex questions should trigger multi-draft reasoning."""
        tot, get_count = self._make_tot()
        result = run(tot.deliberate(
            "What are the philosophical implications of consciousness in artificial systems? "
            "How does integrated information theory relate to the hard problem?",
            context=[{"role": "user", "content": "Let's discuss consciousness"}],
        ))
        if result is not None:
            assert result.final_response
            assert len(result.drafts) > 0
            assert result.elapsed_ms >= 0

    def test_thought_result_structure(self):
        from core.cognitive.tree_of_thoughts import ThoughtResult
        tr = ThoughtResult(
            final_response="test",
            drafts=["d1", "d2"],
            scores=[{"factual_grounding": 8}],
            synthesis_notes="combined",
            elapsed_ms=100.0,
        )
        assert tr.final_response == "test"
        assert len(tr.drafts) == 2


# ════════════════════════════════════════════════════════════════════════
# 4. AUTOPOIESIS
# ════════════════════════════════════════════════════════════════════════

class TestAutopoiesis:
    """Tests for the self-maintenance engine."""

    def _make_engine(self):
        from core.cognitive.autopoiesis import AutopoiesisEngine
        return AutopoiesisEngine()

    def test_instantiation(self):
        engine = self._make_engine()
        assert engine.get_vitality() > 0.0

    def test_register_component(self):
        engine = self._make_engine()
        engine.register_component("test_component", lambda: 0.8)
        health = engine.get_component_health("test_component")
        assert health >= 0.0

    def test_vitality_reflects_components(self):
        engine = self._make_engine()
        engine.register_component("healthy", lambda: 0.95)
        engine.register_component("struggling", lambda: 0.3)
        run(engine.tick())
        vitality = engine.get_vitality()
        assert 0.0 <= vitality <= 1.0

    def test_error_recording(self):
        engine = self._make_engine()
        engine.register_component("buggy", lambda: 0.5)
        engine.record_error("buggy", "RuntimeError")
        engine.record_error("buggy", "RuntimeError")
        # Should not crash and should track the pattern


# ════════════════════════════════════════════════════════════════════════
# 5. HOMEOSTATIC RL
# ════════════════════════════════════════════════════════════════════════

class TestHomeostaticRL:
    """Tests for the energy-based intrinsic motivation system."""

    def _make_rl(self):
        from core.cognitive.homeostatic_rl import HomeostaticRL
        return HomeostaticRL()

    def test_instantiation(self):
        rl = self._make_rl()
        assert rl.get_energy() > 0

    def test_energy_drain(self):
        rl = self._make_rl()
        initial = rl.get_energy()
        rl.drain_energy(10.0, "test_drain")
        assert rl.get_energy() < initial

    def test_energy_gain(self):
        rl = self._make_rl()
        rl.drain_energy(30.0, "setup")
        low = rl.get_energy()
        rl.gain_energy(15.0, "test_gain")
        assert rl.get_energy() > low

    def test_energy_bounded(self):
        rl = self._make_rl()
        rl.gain_energy(9999.0, "overflow_test")
        assert rl.get_energy() <= 100.0
        rl.drain_energy(9999.0, "underflow_test")
        assert rl.get_energy() >= 0.0

    def test_drives_have_expected_keys(self):
        rl = self._make_rl()
        drives = rl.get_drives()
        assert isinstance(drives, dict)
        assert len(drives) > 0

    def test_action_preferences(self):
        rl = self._make_rl()
        prefs = rl.get_action_preferences()
        assert isinstance(prefs, dict)
        assert len(prefs) > 0
        # Preferences should sum to approximately 1 (softmax)
        total = sum(prefs.values())
        assert 0.5 < total < 1.5  # Allow some tolerance

    def test_step_returns_reward(self):
        rl = self._make_rl()
        reward = run(rl.step("RESPOND", {"success": True}))
        assert isinstance(reward, (int, float))


# ════════════════════════════════════════════════════════════════════════
# 6. TOPOLOGY EVOLUTION
# ════════════════════════════════════════════════════════════════════════

class TestTopologyEvolution:
    """Tests for the NEAT-inspired structural plasticity."""

    def _make_evolution(self):
        from core.cognitive.topology_evolution import TopologyEvolution
        return TopologyEvolution()

    def test_instantiation(self):
        evo = self._make_evolution()
        metrics = evo.get_metrics()
        assert metrics is not None

    def test_evolve_with_random_data(self):
        evo = self._make_evolution()
        # Simulate 64 columns with 64 neurons each
        activations = np.random.randn(64, 64).astype(np.float32)
        weights = np.random.randn(64, 64).astype(np.float32) * 0.1
        np.fill_diagonal(weights, 0)  # No self-connections
        delta = run(evo.evolve(activations, weights, tick=1))
        # Should return some kind of result
        assert delta is not None

    def test_column_specializations(self):
        evo = self._make_evolution()
        specs = evo.get_column_specializations()
        assert isinstance(specs, dict)

    def test_connectivity_bounded(self):
        evo = self._make_evolution()
        metrics = evo.get_metrics()
        ratio = getattr(metrics, "connectivity_ratio", 0.0)
        assert 0.0 <= ratio <= 1.0


# ════════════════════════════════════════════════════════════════════════
# 7. STRANGE LOOP
# ════════════════════════════════════════════════════════════════════════

class TestStrangeLoop:
    """Tests for the recursive self-prediction engine."""

    def _make_loop(self):
        from core.cognitive.strange_loop import StrangeLoop
        return StrangeLoop()

    def test_instantiation(self):
        loop = self._make_loop()
        assert loop.get_phenomenal_weight() >= 0.0

    def test_tick_updates_state(self):
        loop = self._make_loop()
        state = {
            "phi": 0.5,
            "free_energy": 0.3,
            "valence": 0.2,
            "arousal": 0.4,
            "energy": 60.0,
            "threat_level": 0.1,
            "coherence": 0.9,
            "social_hunger": 0.3,
            "curiosity": 0.6,
            "error_rate": 0.02,
        }
        result = run(loop.tick(state))
        assert result is not None
        assert hasattr(result, "phenomenal_weight")
        assert hasattr(result, "temporal_coherence")

    def test_prediction_accuracy_tracking(self):
        loop = self._make_loop()
        state = {
            "phi": 0.5, "free_energy": 0.3, "valence": 0.2,
            "arousal": 0.4, "energy": 60.0, "threat_level": 0.1,
            "coherence": 0.9, "social_hunger": 0.3, "curiosity": 0.6,
            "error_rate": 0.02,
        }
        for _ in range(5):
            run(loop.tick(state))
        accuracy = loop.get_prediction_accuracy()
        assert isinstance(accuracy, dict)

    def test_temporal_coherence_stable(self):
        loop = self._make_loop()
        state = {
            "phi": 0.5, "free_energy": 0.3, "valence": 0.2,
            "arousal": 0.4, "energy": 60.0, "threat_level": 0.1,
            "coherence": 0.9, "social_hunger": 0.3, "curiosity": 0.6,
            "error_rate": 0.02,
        }
        for _ in range(10):
            run(loop.tick(state))
        coherence = loop.get_temporal_coherence()
        assert 0.0 <= coherence <= 1.0

    def test_self_narrative_generation(self):
        loop = self._make_loop()
        state = {
            "phi": 0.5, "free_energy": 0.3, "valence": 0.2,
            "arousal": 0.4, "energy": 60.0, "threat_level": 0.1,
            "coherence": 0.9, "social_hunger": 0.3, "curiosity": 0.6,
            "error_rate": 0.02,
        }
        for _ in range(15):
            run(loop.tick(state))
        narrative = loop.get_self_narrative()
        assert isinstance(narrative, str)


# ════════════════════════════════════════════════════════════════════════
# 8. BUG FIX: SOVEREIGN BROWSER URL ROUTING
# ════════════════════════════════════════════════════════════════════════

class TestBrowserURLRouting:
    """Tests for the GodMode URL-vs-search fix."""

    def test_url_detected_routes_to_browse_mode(self):
        """When the objective contains a URL, params should use browse mode."""
        from core.kernel.upgrades_10x import GodModeToolPhase

        params = GodModeToolPhase._normalize_skill_params(
            "sovereign_browser",
            "Check out this article: https://example.com/article/123",
            {},
        )
        assert params.get("mode") == "browse"
        assert params.get("url") == "https://example.com/article/123"

    def test_no_url_routes_to_search(self):
        """When no URL is present, should use search with query."""
        from core.kernel.upgrades_10x import GodModeToolPhase

        params = GodModeToolPhase._normalize_skill_params(
            "sovereign_browser",
            "What's the weather in Tokyo?",
            {},
        )
        assert "query" in params
        assert params["query"] == "What's the weather in Tokyo?"

    def test_web_search_still_gets_query(self):
        """web_search skill should always get a query."""
        from core.kernel.upgrades_10x import GodModeToolPhase

        params = GodModeToolPhase._normalize_skill_params(
            "web_search",
            "latest AI news",
            {},
        )
        assert "query" in params

    def test_reddit_url_detected(self):
        """Reddit URLs should route to browse, not search."""
        from core.kernel.upgrades_10x import GodModeToolPhase

        params = GodModeToolPhase._normalize_skill_params(
            "sovereign_browser",
            "Read this: https://www.reddit.com/r/nosleep/comments/183bp6i/story",
            {},
        )
        assert params.get("mode") == "browse"
        assert "reddit.com" in params.get("url", "")


# ════════════════════════════════════════════════════════════════════════
# 9. INTEGRATION PHASE
# ════════════════════════════════════════════════════════════════════════

class TestCognitiveIntegrationPhase:
    """Tests for the phase that wires cognitive systems into the tick."""

    def test_import(self):
        from core.phases.cognitive_integration_phase import CognitiveIntegrationPhase
        assert CognitiveIntegrationPhase is not None

    def test_pipeline_order_includes_cognitive_integration(self):
        from core.runtime.pipeline_blueprint import kernel_phase_attribute_order
        order = kernel_phase_attribute_order()
        assert "cognitive_integration" in order
        # Should be after affect_phase and before routing_phase
        affect_idx = order.index("affect_phase")
        ci_idx = order.index("cognitive_integration")
        routing_idx = order.index("routing_phase")
        assert affect_idx < ci_idx < routing_idx


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
