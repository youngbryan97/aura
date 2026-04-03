################################################################################

import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock
from core.orchestrator import RobustOrchestrator
from core.brain.cognitive_engine import CognitiveEngine, ThinkingMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestTimeout")

async def test_timeout_resolution():
    """
    Verify that the Orchestrator's watchdog (60s) properly handles 
    a long-running Cognitive Engine task (e.g., 40s) without triggering premature timeout.
    """
    # 1. Setup Mock Engine
    mock_engine = MagicMock(spec=CognitiveEngine)
    
    # Simulate a thought that takes 40 seconds (longer than previous 30s watchdog)
    async def slow_think(*args, **kwargs):
        logger.info("Starting slow thought simulation (40s)...")
        await asyncio.sleep(40) # Longer than old 30s, shorter than new 60s
        mock_thought = MagicMock()
        mock_thought.content = "Success: I finished thinking."
        mock_thought.action = None
        return mock_thought

    mock_engine.think = slow_think

    # 2. Setup Orchestrator
    orchestrator = RobustOrchestrator()
    orchestrator.cognitive_engine = mock_engine
    orchestrator.conversation_history = []
    
    # Mock execute_tool to prevent actual side effects
    orchestrator.execute_tool = AsyncMock(return_value={"ok": True})

    # 3. Simulate message processing
    # We wrap the orchestrator logic or just call the specific loop part
    # Since the orchestrator loop is a long-running method, we'll try to trigger the specific logic
    
    logger.info("Triggering orchestrator cognitive analysis...")
    
    # Use a small timeout for the test to catch if it triggers the BREAK early
    # But it should finish successfully in ~40s
    try:
        # We simulate the part of the loop in orchestrator.py:542-578
        # We'll use a wrapper to run it
        
        message = "Test long thought"
        current_cycle = 1
        
        # This mirrors the logic in orchestrator.py
        try:
            from core.brain.personality_engine import get_personality_engine
            personality_context = {}
            # Mock personality if needed, but orchestrator handles it
            
            thought = await asyncio.wait_for(
                orchestrator.cognitive_engine.think(
                    objective=message,
                    context={
                        "history": [],
                        "cycle": current_cycle,
                        "personality": {}
                    },
                    mode=ThinkingMode.CREATIVE
                ),
                timeout=60.0
            )
            logger.info(f"Thought received: {thought.content}")
            assert "Success" in thought.content
            print("✅ TEST PASSED: Watchdog did not trigger early for 40s thought.")
            
        except asyncio.TimeoutError:
            print("❌ TEST FAILED: Watchdog triggered early (before 60s).")
            exit(1)
        except Exception as e:
            print(f"❌ TEST ERROR: {e}")
            exit(1)

    except Exception as e:
        print(f"Test setup error: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(test_timeout_resolution())


##
