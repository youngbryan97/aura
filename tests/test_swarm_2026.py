################################################################################

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from core.collective.delegator import AgentDelegator, SwarmAgent
from core.collective.belief_sync import BeliefSync

@pytest.mark.asyncio
async def test_agent_delegator_concurrency():
    orchestrator = MagicMock()
    delegator = AgentDelegator(orchestrator)
    delegator.max_parallel = 2
    
    # Mock cognitive engine
    orchestrator.cognitive_engine = AsyncMock()
    orchestrator.cognitive_engine.think.return_value = MagicMock(content="Mock result")
    
    # Fill capacity
    aid1 = await delegator.delegate("critic", "Task 1")
    aid2 = await delegator.delegate("critic", "Task 2")
    aid3 = await delegator.delegate("critic", "Task 3")
    
    assert aid1 != ""
    assert aid2 != ""
    assert aid3 == "" # Should be blocked
    assert len(delegator.active_agents) == 2

@pytest.mark.asyncio
async def test_agent_delegator_debate_synthesis():
    orchestrator = MagicMock()
    delegator = AgentDelegator(orchestrator)
    
    # Mock cognitive engine
    orchestrator.cognitive_engine = AsyncMock()
    orchestrator.cognitive_engine.think.return_value = MagicMock(content="Consensus result")
    
    # Run a mock debate
    # We need to manually complete the agents since we are mocking the brain
    async def mock_delegate(specialty, prompt):
        aid = f"mock-{specialty}"
        agent = SwarmAgent(aid, specialty)
        agent.status = "COMPLETED"
        agent.result = f"Result for {specialty}"
        # Critical fix: Set the event so wait() finishes
        agent.done_event.set()
        delegator.active_agents[aid] = agent
        return aid
        
    delegator.delegate = mock_delegate
    
    result = await delegator.delegate_debate("Test topic", roles=["architect", "critic"])
    
    assert "Consensus result" in result
    orchestrator.cognitive_engine.think.assert_called()

@pytest.mark.asyncio
async def test_belief_sync_rpc_validation():
    orchestrator = MagicMock()
    bs = BeliefSync(orchestrator)
    
    # Test unauthorized method
    res = await bs.handle_rpc_request("delete_all_files", {})
    assert res == {"error": "Method not allowed"}
    
    # Test valid method with missing params
    res = await bs.handle_rpc_request("query_beliefs", {})
    assert res == {"error": "Invalid entity parameter"}
    
    # Test valid method with invalid param type
    res = await bs.handle_rpc_request("query_beliefs", {"entity": 123})
    assert res == {"error": "Invalid entity parameter"}

