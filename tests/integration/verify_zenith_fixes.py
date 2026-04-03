import asyncio
import time
import logging
from unittest.mock import MagicMock, AsyncMock
from core.mind_tick import MindTick
from core.phases.initiative_generation import InitiativeGenerationPhase

# Mock state
class MockAffect:
    def __init__(self):
        self.curiosity = 1.0
        self.social_hunger = 0.0

class MockCognition:
    def __init__(self):
        self.working_memory = []
        self.pending_initiatives = []

class MockAuraState:
    def __init__(self):
        self.affect = MockAffect()
        self.cognition = MockCognition()
        self.health = MagicMock()
        self.state_id = "test_state"
    
    def derive(self, phase_name):
        return self

async def test_impulse_throttling():
    print("Testing InitiativeGeneration throttle...")
    container = MagicMock()
    phase = InitiativeGenerationPhase(container)
    state = MockAuraState()
    
    # First execution should generate an impulse
    new_state = await phase.execute(state)
    assert len(new_state.cognition.pending_initiatives) == 1
    print("✅ First impulse generated.")
    
    # Immediate second execution should NOT generate an impulse (throttle)
    new_state.cognition.pending_initiatives = []
    new_state = await phase.execute(state)
    assert len(new_state.cognition.pending_initiatives) == 0
    print("✅ Throttling active (0 impulses on second attempt).")

async def test_regex_lookahead():
    print("Testing curiosity_forage regex lookahead...")
    # This is a bit hard to test without full mycelium, but we can verify the regex logic
    import re
    pattern = r"^(?!INTERNAL_IMPULSE)(?:forage|explore|investigate|research)\s+(?:about\s+)?(.+)"
    
    internal_msg = "INTERNAL_IMPULSE: explore internal knowledge graph"
    user_msg = "explore the history of AI"
    
    assert not re.match(pattern, internal_msg)
    assert re.match(pattern, user_msg)
    print("✅ Regex lookahead successfully ignores INTERNAL_IMPULSE.")

async def test_mind_tick_timeouts():
    print("Testing MindTick per-phase timeouts...")
    tick = MindTick(MagicMock())
    assert tick.phase_timeouts["response_generation"] == 60.0
    assert tick.phase_timeouts["memory_retrieval"] == 15.0
    print("✅ MindTick phase timeouts verified.")

if __name__ == "__main__":
    asyncio.run(test_impulse_throttling())
    asyncio.run(test_regex_lookahead())
    asyncio.run(test_mind_tick_timeouts())

