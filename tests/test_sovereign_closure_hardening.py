import time
import pytest
from unittest.mock import MagicMock, patch
from core.agency_core import AgencyCore
from core.volition import VolitionEngine
from core.container import ServiceContainer

@pytest.fixture(autouse=True)
def clean_container():
    ServiceContainer.reset()
    yield
    ServiceContainer.reset()

@pytest.mark.asyncio
async def test_goal_completion_lifecycle_by_match():
    # Instantiate AgencyCore with no orchestrator (or dummy)
    agency = AgencyCore(orchestrator=None)
    agency.state.pending_goals = []
    
    # Add a mock persistent goal
    goal = {
        "id": "goal_123",
        "text": "Ensure Persistence (Uplink)",
        "priority": 0.8,
    }
    
    # Pre-approve the state mutation gate
    with patch.object(agency, "_constitutional_runtime_live", return_value=False):
        added = agency.add_goal(goal)
        assert added is True
        assert len(agency.state.pending_goals) == 1
        assert agency.state.pending_goals[0]["status"] == "pending"

        # Test matching and completion via ID
        success = agency.complete_goal_by_match({"id": "goal_123"})
        assert success is True
        assert agency.state.pending_goals[0]["status"] == "completed"

        # Reset status to pending
        agency.state.pending_goals[0]["status"] = "pending"

        # Test matching and completion via text/description/objective matching
        success_text = agency.complete_goal_by_match({"text": "Ensure Persistence (Uplink)"})
        assert success_text is True
        assert agency.state.pending_goals[0]["status"] == "completed"

        # Reset status and test pursuit side effect commit
        agency.state.pending_goals[0]["status"] = "pending"
        action = {
            "type": "pursue_goal",
            "goal": {"id": "goal_123"},
        }
        
        # Trigger the commit action side effect
        await agency._commit_action_side_effects(action, time.time())
        assert agency.state.pending_goals[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_volition_engine_cooldown_registry():
    orchestrator = MagicMock()
    orchestrator.status.running = True
    orchestrator.cognitive_engine = MagicMock()
    
    # Pre-mock config to avoid FileNotFoundError
    with patch("core.volition.config") as mock_config:
        mock_config.paths = MagicMock()
        mock_config.paths.brain_dir = MagicMock()
        mock_config.paths.data_dir = MagicMock()
        
        engine = VolitionEngine(orchestrator)
        
        # Verify cooldown registry initialization
        assert hasattr(engine, "_goal_cooldowns")
        assert engine._goal_cooldowns == {}

        # Set up a potential list of goals
        potential_goals = [
            {"objective": "Explore new pathways", "origin": "intrinsic_curiosity"},
            {"objective": "Verify system health", "origin": "intrinsic_duty"},
        ]

        # Select a goal
        selected = engine._select_and_parse_goal(potential_goals)
        assert selected is not None
        objective = selected["objective"]
        assert objective in engine._goal_cooldowns
        
        # If we tick again with the same potential goals, the selected goal must be skipped since it's on cooldown
        selected_again = engine._select_and_parse_goal(potential_goals)
        assert selected_again is not None
        assert selected_again["objective"] != objective
        assert selected_again["objective"] in engine._goal_cooldowns

        # If both goals are on cooldown, _select_and_parse_goal should return None
        selected_third = engine._select_and_parse_goal(potential_goals)
        assert selected_third is None
