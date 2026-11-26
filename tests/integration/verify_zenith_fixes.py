import asyncio
from types import SimpleNamespace

from core.container import ServiceContainer
from core.mind_tick import MindTick
from core.phases import initiative_generation as initiative_module
from core.phases.initiative_generation import InitiativeGenerationPhase

# Lightweight local state
class LocalAffect:
    def __init__(self):
        self.curiosity = 1.0
        self.social_hunger = 0.0
        self.arousal = 0.5

class LocalCognition:
    def __init__(self):
        self.working_memory = []
        self.pending_initiatives = []

class LocalAuraState:
    def __init__(self):
        self.affect = LocalAffect()
        self.cognition = LocalCognition()
        self.health = {}
        self.response_modifiers = {}
        self.state_id = "test_state"
    
    def derive(self, phase_name):
        return self

async def test_governance_absence_blocks_impulse():
    print("Testing InitiativeGeneration fail-closed governance...")
    ServiceContainer.clear()
    phase = InitiativeGenerationPhase(SimpleNamespace())
    state = LocalAuraState()

    new_state = await phase.execute(state)
    assert len(new_state.cognition.pending_initiatives) == 0
    print("Missing governance blocks autonomous initiative.")

async def test_impulse_throttling():
    print("Testing InitiativeGeneration throttle...")
    container = SimpleNamespace()
    phase = InitiativeGenerationPhase(container)
    state = LocalAuraState()

    async def approved_proposal_sink(current_state, goal, **kwargs):
        current_state.cognition.pending_initiatives.append(
            {
                "goal": goal,
                "source": kwargs.get("source", "initiative_generation"),
                "triggered_by": kwargs.get("triggered_by", "curiosity"),
            }
        )
        return current_state, {"action": "queued", "reason": "approved_for_throttle_verification"}

    original_proposal = initiative_module.propose_governed_initiative_to_state
    initiative_module.propose_governed_initiative_to_state = approved_proposal_sink
    try:
        new_state = await phase.execute(state)
        assert len(new_state.cognition.pending_initiatives) == 1
        print("First impulse generated.")

        new_state.cognition.pending_initiatives = []
        new_state = await phase.execute(state)
        assert len(new_state.cognition.pending_initiatives) == 0
        print("Throttling active (0 impulses on second attempt).")
    finally:
        initiative_module.propose_governed_initiative_to_state = original_proposal

async def test_regex_lookahead():
    print("Testing curiosity_forage regex lookahead...")
    # This is a bit hard to test without full mycelium, but we can verify the regex logic
    import re
    pattern = r"^(?!INTERNAL_IMPULSE)(?:forage|explore|investigate|research)\s+(?:about\s+)?(.+)"
    
    internal_msg = "INTERNAL_IMPULSE: explore internal knowledge graph"
    user_msg = "explore the history of AI"
    
    assert not re.match(pattern, internal_msg)
    assert re.match(pattern, user_msg)
    print("Regex lookahead successfully ignores INTERNAL_IMPULSE.")

async def test_mind_tick_timeouts():
    print("Testing MindTick per-phase timeouts...")
    tick = MindTick(SimpleNamespace())
    assert tick.phase_timeouts["response_generation"] == 120.0
    assert tick.phase_timeouts["memory_retrieval"] == 30.0
    assert tick.phase_timeouts["cognitive_routing"] == 120.0
    assert tick.phase_timeouts["memory_consolidation"] == 20.0
    print("MindTick phase timeouts verified.")

if __name__ == "__main__":
    asyncio.run(test_governance_absence_blocks_impulse())
    asyncio.run(test_impulse_throttling())
    asyncio.run(test_regex_lookahead())
    asyncio.run(test_mind_tick_timeouts())
