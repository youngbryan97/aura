import pytest
from unittest.mock import Mock
from core.environment.environment_kernel import EnvironmentKernel
from core.environment.command import ActionIntent
from core.environment.parsed_state import ParsedState


@pytest.mark.asyncio
async def test_kernel_observes_parses_before_and_after_execute(fake_adapter):
    fake_adapter.screens = ["before", "after"]
    kernel = EnvironmentKernel(adapter=fake_adapter)
    
    # Mock compiler to compile something
    kernel.command_compiler = Mock()
    kernel.command_compiler.compile.return_value = Mock(command_id="cmd_1")
    
    # Mock state compiler to return distinct ParsedStates
    kernel.state_compiler = Mock()
    parsed_before = ParsedState(environment_id="test", sequence_id=1, self_state={"local_coordinates": (10, 10)})
    parsed_after = ParsedState(environment_id="test", sequence_id=2, self_state={"local_coordinates": (11, 10)})
    kernel.state_compiler.compile.side_effect = [parsed_before, parsed_after]
    
    await kernel.start(run_id="test_run")
    
    intent = ActionIntent(name="move_east", expected_effect="position_changed")
    frame = await kernel.step(intent=intent)
    
    # Assert
    # It should have called state_compiler.compile twice (before and after)
    assert kernel.state_compiler.compile.call_count == 2
    
    # Check that outcome assessment has 'position_changed' 
    assert frame.outcome_assessment is not None
    assert "position_changed" in frame.outcome_assessment.observed_events
