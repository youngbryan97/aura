import pytest
from unittest.mock import Mock, AsyncMock
from core.environment.governance_bridge import EnvironmentGovernanceBridge
from core.environment.command import ActionIntent
from core.environment.environment_kernel import EnvironmentKernel
from core.environment.parsed_state import ParsedState
from tests.environment.final_blockers.conftest import ScriptedTerminalAdapter
from core.environment.action_gateway import GatewayDecision

@pytest.mark.asyncio
async def test_safe_observe_gets_environment_receipt_without_full_authority():
    bridge = EnvironmentGovernanceBridge()
    intent = ActionIntent(name="observe", requires_authority=False)
    decision = await bridge.decide_action(intent)
    
    assert decision.approved
    assert decision.will_receipt_id == "not_required"
    assert decision.authority_receipt_id == "not_required"

@pytest.mark.asyncio
async def test_risky_action_calls_unified_will_before_gateway_approval():
    will_mock = AsyncMock()
    will_mock.decide.return_value = Mock(status="PROCEED", receipt_id="will-123")
    bridge = EnvironmentGovernanceBridge(will_gateway=will_mock)
    
    intent = ActionIntent(name="quaff", requires_authority=True)
    decision = await bridge.decide_action(intent)
    
    assert will_mock.decide.called
    assert decision.will_receipt_id == "will-123"

@pytest.mark.asyncio
async def test_authority_gateway_required_for_effectful_irreversible_action():
    auth_mock = AsyncMock()
    auth_mock.authorize.return_value = "auth-456"
    bridge = EnvironmentGovernanceBridge(authority_gateway=auth_mock)
    
    intent = ActionIntent(name="quaff", requires_authority=True)
    decision = await bridge.decide_action(intent)
    
    assert auth_mock.authorize.called
    assert decision.authority_receipt_id == "auth-456"
    
@pytest.mark.asyncio
async def test_local_gateway_can_veto_even_with_will_authority_receipts():
    # Setup kernel and mock gateway
    adapter = ScriptedTerminalAdapter(["screen1"])
    kernel = EnvironmentKernel(adapter=adapter)
    kernel.command_compiler = Mock()
    
    # Give intent requiring authority
    intent = ActionIntent(name="dangerous_action", requires_authority=True)
    
    # Mock gateway to veto
    kernel.gateway.approve = Mock(return_value=GatewayDecision(action_intent=intent, approved=False, decision_id="veto_123", reason="critical_risk_blocks_risky_action"))
    
    await kernel.start(run_id="test_run")
    frame = await kernel.step(intent=intent)
    
    # It should not have compiled a command
    assert kernel.command_compiler.compile.called == False
    assert frame.receipt.status == "blocked"
