################################################################################

import asyncio
import sys
import time
import logging

logging.basicConfig(level=logging.INFO)

from core.evolution.persona_evolver import PersonaEvolver
from core.brain.personality_engine import get_personality_engine
from core.brain.cognitive_engine import CognitiveEngine

class MockOrchestrator:
    def __init__(self):
        self.cognitive_engine = CognitiveEngine()

async def main():
    try:
        orc = MockOrchestrator()
        
        # Mock some memories
        personality = get_personality_engine()
        personality.interaction_memories = [
            {"message": "You're really smart, I agree with you.", "sentiment": "positive", "timestamp": time.time()},
            {"message": "That's a great point. You are so helpful.", "sentiment": "positive", "timestamp": time.time()},
            {"message": "I love talking to you. Thanks for being here.", "sentiment": "positive", "timestamp": time.time()},
            {"message": "You're getting so much better at this. Brilliant!", "sentiment": "positive", "timestamp": time.time()}
        ] * 3 # 12 memories
        
        evolver = PersonaEvolver(orc)
        await evolver.run_evolution_cycle(force=True)
        
        print("Final Traits:", personality.traits)
        print("Final Emotions:", {k: v.base_level for k, v in personality.emotions.items()})
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())


##
