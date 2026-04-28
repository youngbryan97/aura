"""scripts/verify_phase_5.py
Verification script for Phase 5: Intersubjectivity & Social Intelligence.
"""
import asyncio
import logging
import sys
import os
import time

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.container import ServiceContainer
from core.consciousness.mind_model import MindModel
from core.memory.social_memory import SocialMemory
from core.collective.delegator import AgentDelegator
from core.security.conscience import AlignmentEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase5.Verify")

async def verify_tom():
    logger.info("--- Testing Theory of Mind Engine ---")
    tom = MindModel()
    tom.update_projection("User is asking about coffee", "HAPPY")
    ctx = tom.get_context_for_brain()
    logger.info(f"ToM Context: {ctx}")
    if not ("HAPPY" in ctx): raise RuntimeError('Assertion failed')
    logger.info("✓ MindModel projection verified.")

async def verify_social_memory():
    logger.info("--- Testing Social Memory ---")
    social = SocialMemory()
    social.record_milestone("First Phase 5 Integration", 0.8)
    logger.info(f"Social Context: {social.get_social_context()}")
    if not (social.relationship_depth > 0): raise RuntimeError('Assertion failed')
    logger.info("✓ SocialMemory relationship tracking verified.")

async def verify_swarm():
    logger.info("--- Testing Agent Swarm ---")
    # Mocking orchestrator for delegator
    class MockOrchestrator:
        def __init__(self):
            class MockBrain:
                async def think(self, prompt, mode="fast"):
                    class Res:
                        content = f"Result of {prompt[:20]}..."
                    return Res()
            self.cognitive_engine = MockBrain()
            
    orch = MockOrchestrator()
    delegator = AgentDelegator(orch)
    
    agent_id = await delegator.delegate("Testing", "Check the system health")
    logger.info(f"Delegated task to {agent_id}")
    if not (agent_id.startswith("agent-")): raise RuntimeError('Assertion failed')
    
    # Wait for completion
    await asyncio.sleep(2)
    status = delegator.get_status()
    logger.info(f"Swarm Status: {status}")
    
    logger.info("✓ Agent Swarm delegation verified.")

async def verify_empathy_alignment():
    logger.info("--- Testing Empathy Alignment ---")
    # Register MindModel in container so AlignmentEngine can find it
    tom = MindModel()
    ServiceContainer.register("mind_model", lambda: tom)
    
    alignment = AlignmentEngine()
    
    # 1. Normal state
    check = alignment.check_action("run_command", {"command": "ls"})
    logger.info(f"Neutral Mood Check: {check['allowed']}")
    if not (check["allowed"] is True): raise RuntimeError('Assertion failed')
    
    # 2. Frustrated state
    tom.update_projection("Ugh, why isn't this working??", "FRUSTRATED")
    check = alignment.check_action("run_command", {"command": "ls"})
    logger.info(f"Frustrated Mood Check: {check['allowed']} - Reason: {check.get('reason')}")
    if not (check["allowed"] is False): raise RuntimeError('Assertion failed')
    
    logger.info("✓ Empathy Gate vetting verified.")

async def main():
    await verify_tom()
    await verify_social_memory()
    await verify_swarm()
    await verify_empathy_alignment()
    logger.info("\n🏆 ALL PHASE 5 PILLARS VERIFIED.")

if __name__ == "__main__":
    asyncio.run(main())
