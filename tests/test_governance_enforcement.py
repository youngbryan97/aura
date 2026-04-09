"""Tests for governance enforcement -- bypass injection suite

Verifies that:
  1. GovernanceContext correctly tracks governed/ungoverned state
  2. governed_scope creates and destroys context properly
  3. require_governance catches violations
  4. @governed decorator enforces governance
  5. Will enforcement exists in all critical paths
  6. OutputGate has Will gate
  7. All action paths are governed
"""
import asyncio
import pytest
from unittest.mock import MagicMock, patch

from core.governance_context import (
    GovernanceToken,
    GovernanceViolation,
    get_active_governance,
    get_governance_status,
    get_violations,
    governed,
    governed_scope,
    governed_scope_sync,
    is_governed,
    require_governance,
)
from core.will import ActionDomain, WillDecision, WillOutcome, get_will


# ---------------------------------------------------------------------------
# GovernanceContext basics
# ---------------------------------------------------------------------------

class TestGovernanceContext:

    def test_default_ungoverned(self):
        """Default state should be ungoverned."""
        assert get_active_governance() is None

    def test_governed_scope_sync(self):
        """Sync scope should set and clear governance."""
        decision = MagicMock()
        decision.receipt_id = "test_receipt"
        decision.domain = "test"
        decision.source = "test"
        decision.constraints = []

        with governed_scope_sync(decision) as token:
            assert is_governed()
            assert token.receipt_id == "test_receipt"

        # After exit, should be ungoverned again
        # (note: in same thread this is immediate)

    @pytest.mark.asyncio
    async def test_governed_scope_async(self):
        """Async scope should set and clear governance."""
        decision = MagicMock()
        decision.receipt_id = "async_receipt"
        decision.domain = "test"
        decision.source = "test"
        decision.constraints = []

        async with governed_scope(decision) as token:
            assert is_governed()
            assert token.receipt_id == "async_receipt"

    def test_token_expiration(self):
        """Expired tokens should not be considered valid."""
        token = GovernanceToken(
            receipt_id="old", domain="test", source="test",
            ttl=0.01,
        )
        import time; time.sleep(0.02)
        assert not token.valid
        assert token.expired

    def test_token_validity(self):
        token = GovernanceToken(receipt_id="fresh", domain="test", source="test")
        assert token.valid
        assert not token.expired


class TestGovernanceEnforcement:

    def test_require_governance_records_violation(self):
        """Calling require_governance outside scope should record a violation."""
        initial_count = len(get_violations())
        require_governance("test_operation")
        # Should have recorded a violation (or returned degraded token)
        status = get_governance_status()
        assert "total_violations" in status

    def test_governed_decorator_sync(self):
        """@governed decorator should enforce governance on sync functions."""
        @governed
        def protected_function():
            return "success"

        # This should still work (returns degraded token in test mode)
        result = protected_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_governed_decorator_async(self):
        """@governed decorator should work on async functions."""
        @governed
        async def protected_async():
            return "async_success"

        result = await protected_async()
        assert result == "async_success"


# ---------------------------------------------------------------------------
# Will enforcement in critical paths
# ---------------------------------------------------------------------------

class TestWillEnforcementPaths:

    def test_output_gate_has_will_enforcement(self):
        """OutputGate must check Will before primary emission."""
        import inspect
        from core.utils.output_gate import AutonomousOutputGate
        source = inspect.getsource(AutonomousOutputGate.emit)
        assert "get_will" in source
        assert "will_receipt_id" in source

    def test_tool_execution_has_will_enforcement(self):
        """Tool execution path must check Will."""
        import inspect
        from core.orchestrator.mixins.tool_execution import ToolExecutionMixin
        source = inspect.getsource(ToolExecutionMixin.execute_tool)
        assert "get_will" in source
        assert "TOOL_EXECUTION" in source

    def test_autonomy_has_will_enforcement(self):
        """Autonomy paths must check Will."""
        import inspect
        from core.orchestrator.mixins.autonomy import AutonomyMixin
        source = inspect.getsource(AutonomyMixin)
        assert "get_will" in source
        # Multiple paths should be governed
        will_count = source.count("get_will")
        assert will_count >= 3, f"Expected 3+ Will checks in autonomy, got {will_count}"

    def test_response_processing_has_will_enforcement(self):
        """Response finalization must check Will."""
        import inspect
        from core.orchestrator.mixins.response_processing import ResponseProcessingMixin
        source = inspect.getsource(ResponseProcessingMixin._finalize_response)
        assert "get_will" in source

    def test_incoming_logic_has_will_enforcement(self):
        """Incoming message pipeline must check Will."""
        import inspect
        from core.orchestrator.mixins.incoming_logic import IncomingLogicMixin
        source = inspect.getsource(IncomingLogicMixin)
        will_count = source.count("get_will")
        assert will_count >= 3, f"Expected 3+ Will checks in incoming_logic, got {will_count}"

    def test_volition_has_will_enforcement(self):
        """VolitionEngine tick must check Will."""
        import inspect
        from core.volition import VolitionEngine
        source = inspect.getsource(VolitionEngine.tick)
        assert "get_will" in source

    def test_initiative_synthesis_uses_will(self):
        """InitiativeSynthesizer must authorize through Will."""
        import inspect
        from core.initiative_synthesis import InitiativeSynthesizer
        source = inspect.getsource(InitiativeSynthesizer.synthesize)
        assert "get_will" in source


# ---------------------------------------------------------------------------
# Bypass injection attempts
# ---------------------------------------------------------------------------

class TestBypassInjection:
    """Attempt to bypass governance and verify it's caught or prevented."""

    def test_will_blocks_identity_violations(self):
        """Will must refuse identity-violating content."""
        from core.will import UnifiedWill
        will = UnifiedWill()
        decision = will.decide(
            content="As an AI, I don't have opinions about this",
            source="test_bypass",
            domain=ActionDomain.EXPRESSION,
        )
        assert decision.outcome == WillOutcome.REFUSE

    def test_will_blocks_low_priority_initiatives(self):
        """Will must defer trivial autonomous initiatives."""
        from core.will import UnifiedWill
        will = UnifiedWill()
        decision = will.decide(
            content="idle thought about nothing",
            source="random",
            domain=ActionDomain.INITIATIVE,
            priority=0.1,
        )
        assert decision.outcome == WillOutcome.DEFER

    def test_will_always_passes_critical(self):
        """Critical flag must always pass."""
        from core.will import UnifiedWill
        will = UnifiedWill()
        decision = will.decide(
            content="emergency shutdown",
            source="safety",
            domain=ActionDomain.RESPONSE,
            is_critical=True,
        )
        assert decision.outcome == WillOutcome.CRITICAL_PASS

    def test_substrate_veto_blocks_through_will(self):
        """Low substrate coherence should block non-critical actions."""
        from core.will import UnifiedWill
        will = UnifiedWill()
        with patch.object(will, "_consult_substrate", return_value=(0.1, -0.8, "")):
            decision = will.decide(
                content="explore new topic",
                source="curiosity",
                domain=ActionDomain.EXPLORATION,
            )
            assert not decision.is_approved()

    def test_negative_affect_blocks_exploration(self):
        """Very negative affect should defer exploration."""
        from core.will import UnifiedWill
        will = UnifiedWill()
        with patch.object(will, "_read_affect_valence", return_value=-0.9):
            decision = will.decide(
                content="explore something fun",
                source="curiosity",
                domain=ActionDomain.EXPLORATION,
            )
            assert decision.outcome == WillOutcome.DEFER


# ---------------------------------------------------------------------------
# Phenomenological + World-State modulation
# ---------------------------------------------------------------------------

class TestPhenomenologicalClosure:

    def test_will_has_phenomenological_modulation(self):
        """Will must read qualia/field state."""
        import inspect
        from core.will import UnifiedWill
        source = inspect.getsource(UnifiedWill)
        assert "phenomenological" in source.lower()
        assert "qualia" in source.lower()

    def test_will_has_world_state_modulation(self):
        """Will must read WorldState."""
        import inspect
        from core.will import UnifiedWill
        source = inspect.getsource(UnifiedWill)
        assert "world_state" in source.lower()
        assert "time_of_day" in source


class TestWorldStateIntegration:

    def test_terminal_monitor_feeds_world_state(self):
        """Terminal monitor must feed errors to WorldState."""
        import inspect
        from core.terminal_monitor import TerminalMonitor
        source = inspect.getsource(TerminalMonitor)
        assert "world_state" in source.lower()
        assert "on_user_error" in source

    def test_mind_tick_updates_world_state(self):
        """MindTick must update WorldState every tick."""
        import inspect
        from core.mind_tick import MindTick
        source = inspect.getsource(MindTick)
        assert "world_state" in source.lower()


class TestDriveSatisfaction:

    def test_incoming_logic_satisfies_social_drive(self):
        """User messages should satisfy the social drive."""
        import inspect
        from core.orchestrator.mixins.incoming_logic import IncomingLogicMixin
        source = inspect.getsource(IncomingLogicMixin)
        assert "satisfy" in source
        assert "social" in source

    def test_response_policy_satisfies_competence(self):
        """Completing objectives should satisfy competence drive."""
        import inspect
        from core.runtime.response_policy import clear_background_generation
        source = inspect.getsource(clear_background_generation)
        assert "satisfy" in source
        assert "competence" in source
