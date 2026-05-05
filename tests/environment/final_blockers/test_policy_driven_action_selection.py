import pytest
from unittest.mock import Mock
from core.environment.command import ActionIntent
from core.environment.policy.policy_orchestrator import PolicyOrchestrator
from core.environment.parsed_state import ParsedState
from core.environment.homeostasis import Homeostasis, Resource
from core.environment.command import ActionIntent

@pytest.mark.asyncio
async def test_policy_low_hp_selects_survival_action_not_explore():
    # Set up
    orchestrator = PolicyOrchestrator()
    from core.environment.ontology import ResourceState
    parsed_state = ParsedState(
        environment_id="test", 
        context_id="test", 
        sequence_id=1, 
        self_state={"hp": 2, "max_hp": 25},
        resources={"health": ResourceState(name="health", value=2.0, max_value=25.0)}
    )
    homeostasis = Homeostasis()
    # mock extract to return low hp
    homeostasis.extract = Mock(return_value=[Resource(name="health", kind="health", value=2.0, max_value=25.0)])
    from core.environment.belief_graph import EnvironmentBeliefGraph
    belief = EnvironmentBeliefGraph()
    
    intent = orchestrator.select_action(
        parsed_state=parsed_state,
        belief=belief,
        homeostasis=homeostasis,
        episode=Mock(),
        recent_frames=[]
    )
    
    # Assert
    assert intent.name in ["wait", "retreat_to_safety", "stabilize_resource", "observe", "search", "inventory"]
