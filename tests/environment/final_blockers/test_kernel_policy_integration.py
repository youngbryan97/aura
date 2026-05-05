import pytest
from unittest.mock import Mock
from core.environment.command import ActionIntent
from core.environment.environment_kernel import EnvironmentKernel
from core.environment.policy.policy_orchestrator import PolicyOrchestrator

class SpyPolicy(PolicyOrchestrator):
    def __init__(self):
        self.calls = []

    def select_action(self, *, parsed_state, belief, homeostasis, episode, recent_frames, **kwargs):
        self.calls.append({
            "parsed_state": parsed_state,
            "belief": belief,
            "homeostasis": homeostasis,
            "episode": episode,
            "recent_frames": recent_frames,
        })
        return ActionIntent(name="explore_frontier", expected_effect="frontier_progress", risk="safe")

@pytest.mark.asyncio
async def test_kernel_calls_policy_when_no_intent_provided(fake_adapter):
    adapter = fake_adapter
    kernel = EnvironmentKernel(adapter=adapter)
    kernel.command_compiler = Mock()
    kernel.command_compiler.compile.return_value = Mock(command_id="test_cmd")
    spy_policy = SpyPolicy()
    kernel.policy = spy_policy
    
    await kernel.start(run_id="test_run")
    await kernel.step(intent=None)
    
    assert len(spy_policy.calls) == 1
    call = spy_policy.calls[0]
    assert call["parsed_state"] is not None
    assert call["belief"] is not None
    assert call["homeostasis"] is not None
    
    # Assert black-box trace records the policy intent
    assert len(kernel.frames) > 0
    frame = kernel.frames[-1]
    assert frame.action_intent.name == "explore_frontier"
    assert frame.selected_option == "explore_frontier"

@pytest.mark.asyncio
async def test_policy_not_called_when_explicit_intent_supplied_unless_validation_requires(fake_adapter):
    adapter = fake_adapter
    kernel = EnvironmentKernel(adapter=adapter)
    kernel.command_compiler = Mock()
    kernel.command_compiler.compile.return_value = Mock(command_id="test_cmd")
    spy_policy = SpyPolicy()
    kernel.policy = spy_policy
    
    await kernel.start(run_id="test_run")
    explicit_intent = ActionIntent(name="inventory", expected_effect="show inventory", risk="safe")
    
    await kernel.step(intent=explicit_intent)
    
    assert len(spy_policy.calls) == 0
    assert kernel.frames[-1].action_intent.name == "inventory"
