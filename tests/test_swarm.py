################################################################################

import asyncio
import logging

logging.basicConfig(level=logging.INFO)

from core.brain.cognitive_engine import CognitiveEngine

class MockOrchestrator:
    def __init__(self):
        self.cognitive_engine = CognitiveEngine()
        self.conversation_history = []

async def main():
    orc = MockOrchestrator()
    try:
        
        from core.collective.delegator import AgentDelegator
        delegator = AgentDelegator(orc)
        
        # Test 1: Volition Adding Interests
        print("\n--- Test 1: Volition Dynamic Interests ---")
        from core.volition import VolitionEngine
        vol = VolitionEngine(orc)
        vol.add_interest("swarm robotics phase transitions", "technical")
        print("Updated Technical Interests:", vol.technical_interests[-1])
        
        # Test 2: Swarm Consensus
        topic = "Should we migrate from JSON-based vector memory to a persistent SQLite-backed vector memory system?"
        print(f"\n--- Test 2: Initiating Swarm Debate on: {topic} ---")
        
        consensus = await delegator.delegate_debate(
            topic=topic,
            roles=["critic", "architect", "optimizer"],
            timeout=120.0
        )
        
        print("\n=== FINAL CONSENSUS ===")
        print(consensus)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())


##
