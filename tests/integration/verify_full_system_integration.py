################################################################################


import asyncio
import logging
import sys
import os
import json
from unittest.mock import MagicMock, AsyncMock

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.orchestrator import RobustOrchestrator
from core.brain.cognitive_engine import ThinkingMode
import core.brain.personality_engine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegrationTest")

async def test_full_system_loop():
    logger.info("🚀 Starting Full Mind/Body/Language Integration Test...")
    
    # 1. Initialize Orchestrator (The Body/Nervous System)
    # We mock components that require external I/O (like Ollama) to focus on plumbing
    orchestrator = RobustOrchestrator()
    
    # Mock Cognitive Engine (The Mind) to avoid real LLM calls but verify data flow
    orchestrator.cognitive_engine = MagicMock()
    orchestrator.cognitive_engine.think = AsyncMock(return_value=MagicMock(
        content="I am fully operational.",
        mode=ThinkingMode.CREATIVE,
        confidence=0.99,
        reasoning=["Systems check complete.", "All modules responding."],
        action=None
    ))
    
    # Mock Skill Registry (The Capabilities)
    orchestrator.router = MagicMock()
    orchestrator.router.registry = MagicMock()
    
    # 2. Test Input Processing (Language In)
    user_input = "Please run a thorough analysis of the system health and provide a detailed report on memory usage and skill registration status. I need a deep dive into the logs and an assessment of potential performance bottlenecks in the cognitive cycle."
    logger.info(f"🗣️  Input: '{user_input}'")
    
    # Verify process_user_input returns success (The Frustration Fix)
    result = orchestrator.process_user_input(user_input)
    if result.get("ok"):
        logger.info("✅ Orchestrator accepted input (Frustration Loop Fixed)")
    else:
        logger.error(f"❌ Orchestrator rejected input: {result}")
        return

    # 3. Simulate Execution Cycle
    # We manually trigger _process_message to verify internal flow
    logger.info("🧠 Triggering Cognitive Cycle...")
    
    # Inject a mock message into the queue if not already there (process_user_input does this)
    if orchestrator.message_queue.empty():
         orchestrator.message_queue.put(user_input)
         
    # Run _process_cycle logic manually (partially) or just _process_message
    msg = orchestrator.message_queue.get()
    
    # Spy on Personality Engine (The Soul)
    # We want to see if it gets called during _process_message
    
    try:
        response = await orchestrator._run_cognitive_loop(msg)
        logger.info(f"🗣️  Response: '{response}'")
        
        # Verify Cognitive Engine was called with correct Mode and Context
        args, kwargs = orchestrator.cognitive_engine.think.call_args
        
        # Check Mode (Depth)
        if kwargs.get('mode') == ThinkingMode.CREATIVE:
            logger.info("✅ Cognitive Engine used CREATIVE mode (Depth Restored)")
        else:
            logger.error(f"❌ Cognitive Engine used wrong mode: {kwargs.get('mode')}")
            
        # Check Context (Personality Injection)
        ctx = kwargs.get('context', {})
        if "personality" in ctx and isinstance(ctx["personality"], dict):
            p_state = ctx["personality"]
            logger.info(f"✅ Personality injected: {p_state.get('mood')} (Soul Connected)")
        elif "personality" in ctx:
             logger.info(f"⚠️ Personality context present but not a dict: {type(ctx['personality'])}")
        else:
            logger.error("❌ Personality context missing from thought process!")
            
        # Check History Update (Memory)
        if orchestrator.conversation_history[-1]['content'] == response:
             logger.info("✅ Conversation History updated (Memory Synced)")
        else:
             logger.error("❌ Conversation History mismatch")

    except Exception as e:
        logger.exception(f"❌ Integration Test Failed: {e}")

    logger.info("🏁 Mind/Body/Language Integration Test Complete.")

if __name__ == "__main__":
    asyncio.run(test_full_system_loop())


##
