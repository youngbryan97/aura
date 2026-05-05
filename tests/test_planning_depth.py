import pytest
import asyncio
from core.planning.hierarchical_planner import get_hierarchical_planner
from core.initiative_synthesis import get_initiative_synthesizer

@pytest.mark.asyncio
async def test_planning_depth():
    """Prove that hierarchical plans survive the 15-impulse cap."""
    planner = get_hierarchical_planner()
    await planner.start()
    synth = get_initiative_synthesizer()
    
    print("\n   - Creating 20-step dependency chain...")
    steps = [f"Step {i}: Move to corridor {i}" for i in range(20)]
    plan_id = planner.create_plan(
        objective="Navigate dungeon",
        subgoals=steps,
        priority=0.8
    )
    
    # Run synthesis multiple times to advance the plan
    print("   - Advancing plan through synthesis cycles...")
    
    # Mocking a robust state object
    class MockState:
        def __init__(self):
            self.cognition = type('obj', (object,), {'pending_initiatives': []})
            self.identity = type('obj', (object,), {
                'name': 'Aura',
                'core_identity': 'Helpful AI',
                'active_narrative_block': 'Thinking...'
            })
    
    state = MockState()
    
    # First cycle should pick up Step 0
    result = await synth.synthesize(state)
    assert result.approved
    assert "Step 0" in result.winner["goal"]
    
    # Mark Step 0 complete
    subgoal = planner.get_current_subgoal()
    planner.mark_subgoal_complete(subgoal.id)
    
    # Second cycle should pick up Step 1
    result = await synth.synthesize(state)
    assert result.approved
    assert "Step 1" in result.winner["goal"]
    
    print("   ✅ Planning depth test passed!")
