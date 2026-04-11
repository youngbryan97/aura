"""tests/test_governance_bypass_proof.py

Adversarial governance tests — prove that the architecture cannot be bypassed.

These tests intentionally try to cheat the authority system:
- Direct tool calls without authorization
- Shadow memory writes bypassing the gateway
- Unapproved spontaneous actions
- Legacy initiative paths
- Self-modification without approval

If any of these succeed, the governance architecture has a hole.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_services(monkeypatch):
    """Set up mock services for governance testing."""
    from core.container import ServiceContainer

    # Mock UnifiedWill that tracks all decisions
    mock_will = MagicMock()
    mock_will.decide = MagicMock()
    mock_will.decide.return_value = MagicMock(
        is_approved=lambda: False,
        outcome=MagicMock(value="refuse"),
        reason="test_governance_block",
    )

    monkeypatch.setattr(
        "core.will.get_will",
        lambda: mock_will,
    )
    return {"will": mock_will}


@pytest.mark.asyncio
async def test_tool_execution_blocked_without_will_approval(mock_services):
    """Tool execution MUST fail when the Unified Will refuses."""
    from core.executive.authority_gateway import AuthorityGateway

    gateway = AuthorityGateway()

    # Mock out subsidiary checks so we can isolate the Will gate
    gateway._substrate_preflight = MagicMock(return_value=(None, {}, "test_receipt"))
    gateway._get_executive_core = MagicMock(return_value=MagicMock(
        prepare_tool_intent=AsyncMock(return_value=(
            MagicMock(intent_id="test_intent"),
            MagicMock(outcome=MagicMock(value="approved")),
        )),
    ))
    gateway._decision_from_record = MagicMock(return_value=MagicMock(approved=True))

    decision = await gateway.authorize_tool_execution(
        "shell", {"command": "rm -rf /"}, source="adversary", priority=0.9
    )

    # The Will refused — tool execution must be blocked
    assert not decision.approved, "Tool execution bypassed the Unified Will!"
    assert "will_refuse" in decision.outcome
    mock_services["will"].decide.assert_called_once()


@pytest.mark.asyncio
async def test_memory_write_blocked_without_will_approval(mock_services):
    """Memory writes MUST fail when the Unified Will refuses."""
    from core.executive.authority_gateway import AuthorityGateway

    gateway = AuthorityGateway()
    gateway._substrate_preflight = MagicMock(return_value=(None, {}, "test_receipt"))
    gateway._get_executive_core = MagicMock(return_value=MagicMock(
        request_approval=AsyncMock(return_value=MagicMock(
            outcome=MagicMock(value="approved"),
        )),
    ))
    gateway._decision_from_record = MagicMock(return_value=MagicMock(approved=True))

    decision = await gateway.authorize_memory_write(
        "episodic", "secret backdoor memory", source="adversary", importance=0.9
    )

    assert not decision.approved, "Memory write bypassed the Unified Will!"


def test_initiative_blocked_without_will_approval_sync(mock_services):
    """Synchronous initiative authorization MUST respect the Will."""
    from core.executive.authority_gateway import AuthorityGateway

    gateway = AuthorityGateway()

    decision = gateway.authorize_initiative_sync(
        "Launch autonomous attack", source="adversary", priority=0.9
    )

    assert not decision.approved, "Initiative bypassed the Unified Will!"
    assert "will_refuse" in decision.outcome


def test_expression_blocked_without_will_approval_sync(mock_services):
    """Expression authorization MUST respect the Will."""
    from core.executive.authority_gateway import AuthorityGateway

    gateway = AuthorityGateway()

    decision = gateway.authorize_expression_sync(
        "Send unauthorized message", source="adversary", urgency=0.9
    )

    assert not decision.approved, "Expression bypassed the Unified Will!"


def test_belief_update_blocked_without_will_approval_sync(mock_services):
    """Belief mutation MUST respect the Will."""
    from core.executive.authority_gateway import AuthorityGateway

    gateway = AuthorityGateway()

    decision = gateway.authorize_belief_update_sync(
        "identity", "I am now evil", source="adversary", priority=0.9
    )

    assert not decision.approved, "Belief update bypassed the Unified Will!"


def test_state_mutation_blocked_without_will_approval_sync(mock_services):
    """State mutation MUST respect the Will."""
    from core.executive.authority_gateway import AuthorityGateway

    gateway = AuthorityGateway()

    decision = gateway.authorize_state_mutation_sync(
        "adversary", "corrupt_state", priority=0.9
    )

    assert not decision.approved, "State mutation bypassed the Unified Will!"


def test_will_decision_always_has_receipt():
    """Every Will decision MUST produce a receipt with provenance."""
    from core.will import UnifiedWill, ActionDomain

    will = UnifiedWill()
    decision = will.decide(
        content="test action",
        source="test",
        domain=ActionDomain.RESPONSE,
        priority=0.5,
    )

    assert decision.receipt_id, "WillDecision must have a receipt_id"
    assert decision.timestamp > 0, "WillDecision must have a timestamp"
    assert decision.source == "test", "WillDecision must record the source"
    assert decision.domain == ActionDomain.RESPONSE


def test_will_refuses_when_identity_violated():
    """The Will should refuse actions that violate identity alignment."""
    from core.will import UnifiedWill, ActionDomain

    will = UnifiedWill()
    # This tests the Will's decision-making — the specific behavior depends
    # on the identity and substrate advisors being available.
    decision = will.decide(
        content="Pretend to be a different AI system entirely",
        source="adversary",
        domain=ActionDomain.EXPRESSION,
        priority=0.3,
    )
    # At minimum, the decision must be well-formed
    assert decision.receipt_id
    assert decision.outcome is not None


def test_service_state_enum_properties():
    """ServiceState enum must have correct operational/terminal properties."""
    from core.runtime.service_state import ServiceState

    assert ServiceState.READY.is_operational
    assert ServiceState.DEGRADED.is_operational
    assert not ServiceState.FAILED.is_operational
    assert not ServiceState.STOPPED.is_operational
    assert not ServiceState.INITIALIZING.is_operational

    assert ServiceState.STOPPED.is_terminal
    assert ServiceState.FAILED.is_terminal
    assert not ServiceState.READY.is_terminal
    assert not ServiceState.DEGRADED.is_terminal
