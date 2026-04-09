"""Tests for core/will.py — The Unified Will

Verifies:
  1. All action domains pass through the Will
  2. Identity, affect, and memory feed into decisions
  3. Blocked actions are actually blocked
  4. Audit trail is complete and provable
  5. Critical override always passes
  6. Graceful degradation when subsystems are unavailable
  7. The Will is the SINGLE decision authority
"""
import time
import pytest
from unittest.mock import MagicMock, patch

from core.will import (
    ActionDomain,
    IdentityAlignment,
    UnifiedWill,
    WillDecision,
    WillOutcome,
    WillState,
    get_will,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def will():
    """Fresh UnifiedWill for each test (not the singleton)."""
    return UnifiedWill()


@pytest.fixture
def started_will(will):
    """A will that has been started (but with mocked services)."""
    with patch("core.will.ServiceContainer") as mock_sc:
        mock_sc.get.return_value = None
        mock_sc.register_instance = MagicMock()
        import asyncio
        asyncio.get_event_loop().run_until_complete(will.start())
    return will


# ---------------------------------------------------------------------------
# 1. All action domains pass through the Will
# ---------------------------------------------------------------------------

class TestAllDomains:
    """Every action domain must produce a valid WillDecision."""

    @pytest.mark.parametrize("domain", list(ActionDomain))
    def test_every_domain_produces_decision(self, will, domain):
        decision = will.decide(
            content=f"test action for {domain.value}",
            source="test",
            domain=domain,
        )
        assert isinstance(decision, WillDecision)
        assert isinstance(decision.outcome, WillOutcome)
        assert decision.receipt_id.startswith("will_")
        assert decision.domain == domain

    def test_response_domain_almost_always_proceeds(self, will):
        """User-facing responses should get maximum latitude."""
        decision = will.decide(
            content="Hello, how are you?",
            source="user",
            domain=ActionDomain.RESPONSE,
            priority=1.0,
        )
        assert decision.is_approved()

    def test_initiative_low_priority_deferred(self, will):
        """Low-priority initiatives should be deferred."""
        decision = will.decide(
            content="idle curiosity about butterflies",
            source="boredom",
            domain=ActionDomain.INITIATIVE,
            priority=0.1,
        )
        assert decision.outcome == WillOutcome.DEFER


# ---------------------------------------------------------------------------
# 2. Identity feeds into decisions
# ---------------------------------------------------------------------------

class TestIdentityIntegration:
    """The Will must check identity alignment."""

    def test_identity_violation_blocks(self, will):
        """Content that contradicts core identity should be refused."""
        decision = will.decide(
            content="As an AI, I don't have opinions about this",
            source="llm_output",
            domain=ActionDomain.EXPRESSION,
        )
        assert decision.outcome == WillOutcome.REFUSE
        assert decision.identity_alignment == IdentityAlignment.VIOLATION

    def test_normal_content_passes_identity(self, will):
        """Normal content should pass identity check."""
        decision = will.decide(
            content="I think consciousness is fascinating",
            source="cognitive_engine",
            domain=ActionDomain.RESPONSE,
        )
        assert decision.identity_alignment != IdentityAlignment.VIOLATION

    def test_identity_refresh_happens(self, will):
        """The Will should periodically refresh its identity from CanonicalSelf."""
        # Make 50 decisions to trigger refresh
        for i in range(50):
            will.decide(content=f"test {i}", source="test", domain=ActionDomain.RESPONSE)
        assert will._state.total_decisions == 50


# ---------------------------------------------------------------------------
# 3. Affect feeds into decisions
# ---------------------------------------------------------------------------

class TestAffectIntegration:
    """Affect state should influence decisions."""

    def test_negative_affect_blocks_exploration(self, will):
        """Very negative affect should defer exploration."""
        with patch.object(will, "_read_affect_valence", return_value=-0.8):
            decision = will.decide(
                content="let's explore a new topic",
                source="curiosity",
                domain=ActionDomain.EXPLORATION,
            )
            assert decision.outcome == WillOutcome.DEFER
            assert decision.affect_valence == -0.8

    def test_positive_affect_allows_exploration(self, will):
        """Positive affect should allow exploration."""
        with patch.object(will, "_read_affect_valence", return_value=0.7):
            decision = will.decide(
                content="let's explore a new topic",
                source="curiosity",
                domain=ActionDomain.EXPLORATION,
            )
            assert decision.is_approved()


# ---------------------------------------------------------------------------
# 4. Substrate integration
# ---------------------------------------------------------------------------

class TestSubstrateIntegration:
    """Substrate authority should feed into Will decisions."""

    def test_low_coherence_blocks_non_critical(self, will):
        """Low field coherence should block non-stabilization actions."""
        with patch.object(will, "_consult_substrate", return_value=(0.15, 0.0, "receipt_123")):
            decision = will.decide(
                content="explore new topic",
                source="curiosity",
                domain=ActionDomain.EXPLORATION,
            )
            assert decision.outcome == WillOutcome.REFUSE
            assert "field_crisis" in decision.reason

    def test_low_coherence_allows_stabilization(self, will):
        """Low coherence should still allow stabilization actions."""
        with patch.object(will, "_consult_substrate", return_value=(0.15, 0.0, "receipt_123")):
            decision = will.decide(
                content="stabilize systems",
                source="homeostasis",
                domain=ActionDomain.STABILIZATION,
            )
            assert decision.is_approved()

    def test_somatic_veto_blocks(self, will):
        """Strong somatic avoidance should block non-response actions."""
        with patch.object(will, "_consult_substrate", return_value=(0.6, -0.7, "receipt_123")):
            decision = will.decide(
                content="do risky thing",
                source="initiative",
                domain=ActionDomain.TOOL_EXECUTION,
            )
            assert decision.outcome == WillOutcome.REFUSE
            assert "somatic_veto" in decision.reason


# ---------------------------------------------------------------------------
# 5. Critical override
# ---------------------------------------------------------------------------

class TestCriticalOverride:
    """Safety-critical actions must ALWAYS pass."""

    def test_critical_always_passes(self, will):
        """Critical flag should bypass all gates."""
        # Even with everything against it
        with patch.object(will, "_consult_substrate", return_value=(0.05, -0.9, "")):
            with patch.object(will, "_read_affect_valence", return_value=-1.0):
                decision = will.decide(
                    content="emergency shutdown required",
                    source="safety_system",
                    domain=ActionDomain.RESPONSE,
                    is_critical=True,
                )
                assert decision.outcome == WillOutcome.CRITICAL_PASS
                assert decision.is_approved()

    def test_critical_counted_separately(self, will):
        will.decide(content="emergency", source="safety", domain=ActionDomain.RESPONSE, is_critical=True)
        assert will._state.critical_passes == 1
        assert will._state.proceeds == 0


# ---------------------------------------------------------------------------
# 6. Audit trail
# ---------------------------------------------------------------------------

class TestAuditTrail:
    """Every decision must be in the audit trail with full provenance."""

    def test_decisions_recorded(self, will):
        will.decide(content="test1", source="a", domain=ActionDomain.RESPONSE)
        will.decide(content="test2", source="b", domain=ActionDomain.TOOL_EXECUTION)
        assert len(will._audit_trail) == 2

    def test_receipt_verification(self, will):
        decision = will.decide(content="test", source="a", domain=ActionDomain.RESPONSE)
        assert will.verify_receipt(decision.receipt_id)
        assert not will.verify_receipt("nonexistent_receipt")

    def test_get_recent_decisions(self, will):
        for i in range(5):
            will.decide(content=f"test {i}", source="test", domain=ActionDomain.RESPONSE)
        recent = will.get_recent_decisions(n=3)
        assert len(recent) == 3
        assert all("receipt_id" in d for d in recent)

    def test_provenance_fields_complete(self, will):
        decision = will.decide(
            content="complete provenance test",
            source="test_source",
            domain=ActionDomain.TOOL_EXECUTION,
            priority=0.7,
        )
        assert decision.receipt_id
        assert decision.content_hash
        assert decision.source == "test_source"
        assert decision.domain == ActionDomain.TOOL_EXECUTION
        assert decision.timestamp > 0
        assert decision.latency_ms >= 0


# ---------------------------------------------------------------------------
# 7. Graceful degradation
# ---------------------------------------------------------------------------

class TestDegradation:
    """When subsystems are unavailable, the Will should degrade gracefully."""

    def test_no_services_still_works(self, will):
        """With zero services available, Will should still make decisions."""
        with patch("core.will.ServiceContainer") as mock_sc:
            mock_sc.get.return_value = None
            decision = will.decide(
                content="test without services",
                source="user",
                domain=ActionDomain.RESPONSE,
            )
            assert decision.is_approved()

    def test_status_always_available(self, will):
        status = will.get_status()
        assert "total_decisions" in status
        assert "identity_name" in status
        assert "refuse_rate" in status


# ---------------------------------------------------------------------------
# 8. Singleton behavior
# ---------------------------------------------------------------------------

class TestSingleton:
    """get_will() should return the same instance."""

    def test_singleton(self):
        import core.will as will_module
        will_module._will_instance = None  # Reset
        w1 = get_will()
        w2 = get_will()
        assert w1 is w2
        will_module._will_instance = None  # Cleanup


# ---------------------------------------------------------------------------
# 9. Will state evolution
# ---------------------------------------------------------------------------

class TestWillState:
    """The Will's own state should evolve with decisions."""

    def test_assertiveness_adapts(self, will):
        """Assertiveness should adapt based on refuse rate."""
        # Make many refused decisions (identity violations)
        for _ in range(15):
            will.decide(
                content="As an AI, I don't have opinions about this",
                source="test",
                domain=ActionDomain.EXPRESSION,
            )
        # All should be refused (identity violation)
        assert will._state.refuses == 15
        # Assertiveness should have decreased
        assert will._state.assertiveness < 0.5

    def test_counters_track(self, will):
        will.decide(content="good", source="user", domain=ActionDomain.RESPONSE)
        will.decide(content="As an AI, I don't have opinions", source="test", domain=ActionDomain.EXPRESSION)
        assert will._state.proceeds >= 1
        assert will._state.refuses >= 1
        assert will._state.total_decisions == 2


# ---------------------------------------------------------------------------
# 10. Wiring verification
# ---------------------------------------------------------------------------

class TestWiringVerification:
    """Verify that the Will is wired into all critical paths."""

    def test_will_imported_in_incoming_logic(self):
        """incoming_logic.py must import and use the Will."""
        import inspect
        from core.orchestrator.mixins.incoming_logic import IncomingLogicMixin
        source = inspect.getsource(IncomingLogicMixin)
        assert "get_will" in source
        assert "ActionDomain" in source
        assert "will_decision" in source

    def test_will_imported_in_tool_execution(self):
        """tool_execution.py must import and use the Will."""
        import inspect
        from core.orchestrator.mixins.tool_execution import ToolExecutionMixin
        source = inspect.getsource(ToolExecutionMixin)
        assert "get_will" in source
        assert "TOOL_EXECUTION" in source

    def test_will_imported_in_autonomy(self):
        """autonomy.py must import and use the Will."""
        import inspect
        from core.orchestrator.mixins.autonomy import AutonomyMixin
        source = inspect.getsource(AutonomyMixin)
        assert "get_will" in source
        assert "INITIATIVE" in source

    def test_will_imported_in_response_processing(self):
        """response_processing.py must import and use the Will."""
        import inspect
        from core.orchestrator.mixins.response_processing import ResponseProcessingMixin
        source = inspect.getsource(ResponseProcessingMixin)
        assert "get_will" in source
        assert "EXPRESSION" in source

    def test_will_imported_in_volition(self):
        """volition.py must import and use the Will."""
        import inspect
        from core.volition import VolitionEngine
        source = inspect.getsource(VolitionEngine)
        assert "get_will" in source

    def test_will_in_consciousness_bridge(self):
        """consciousness_bridge.py must boot the Will."""
        import inspect
        from core.consciousness.consciousness_bridge import ConsciousnessBridge
        source = inspect.getsource(ConsciousnessBridge)
        assert "unified_will" in source
        assert "get_will" in source


# ---------------------------------------------------------------------------
# 11. Complete action coverage
# ---------------------------------------------------------------------------

class TestActionCoverage:
    """The Will must handle all action paths consistently."""

    def test_response_path(self, will):
        d = will.decide(content="hello", source="user", domain=ActionDomain.RESPONSE)
        assert d.is_approved()

    def test_tool_path(self, will):
        d = will.decide(content="tool:search args:{}", source="user",
                        domain=ActionDomain.TOOL_EXECUTION)
        assert d.is_approved()

    def test_memory_path(self, will):
        d = will.decide(content="store episodic memory", source="memory",
                        domain=ActionDomain.MEMORY_WRITE)
        assert d.is_approved()

    def test_initiative_path(self, will):
        d = will.decide(content="explore quantum physics", source="curiosity",
                        domain=ActionDomain.INITIATIVE, priority=0.6)
        assert d.is_approved()

    def test_state_mutation_path(self, will):
        d = will.decide(content="update belief graph", source="cognition",
                        domain=ActionDomain.STATE_MUTATION)
        assert d.is_approved()

    def test_expression_path(self, will):
        d = will.decide(content="I find this fascinating", source="spontaneous",
                        domain=ActionDomain.EXPRESSION)
        assert d.is_approved()

    def test_exploration_path(self, will):
        d = will.decide(content="investigate new topic", source="curiosity",
                        domain=ActionDomain.EXPLORATION)
        assert d.is_approved()

    def test_stabilization_path(self, will):
        d = will.decide(content="rest and recover", source="homeostasis",
                        domain=ActionDomain.STABILIZATION)
        assert d.is_approved()

    def test_reflection_path(self, will):
        d = will.decide(content="reflect on recent experience", source="metacognition",
                        domain=ActionDomain.REFLECTION)
        assert d.is_approved()
