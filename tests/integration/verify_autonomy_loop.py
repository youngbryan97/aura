################################################################################

import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock
from core.orchestrator import RobustOrchestrator
from core.brain.cognitive_engine import CognitiveEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestAutonomy")

async def test_autonomy_loop():
    """
    Verify that the Orchestrator's autonomy loop triggers when bored.
    """
    # 1. Setup Orchestrator with Mock Engine
    orchestrator = RobustOrchestrator()
    orchestrator.cognitive_engine = MagicMock(spec=CognitiveEngine)
    # Configure think to return a complete dict
    mock_brain = AsyncMock()
    mock_brain.think.return_value = {
        "content": "I am bored.", 
        "tool_calls": [{"name": "web_search", "args": {"query": "news"}}]
    }
    orchestrator.cognitive_engine.autonomous_brain = mock_brain
    
    # Mock system status
    orchestrator.status = MagicMock()
    orchestrator.status.cycle_count = 0
    
    # Mock emitter
    orchestrator._emit_thought_stream = MagicMock()
    
    # Mock execute_tool
    orchestrator.execute_tool = AsyncMock()

    # 2. Simulate Boredom Accumulation
    logger.info("Simulating 5 idle cycles...")
    for i in range(5):
        await orchestrator._perform_autonomous_thought()
        # On 5th cycle, it should trigger
        
    # 3. Verify Trigger
    # The thought process is async, so we await the called method
    if orchestrator.cognitive_engine.autonomous_brain.think.called:
        print("✅ TEST PASSED: Autonomous Brain triggered after boredom threshold.")
        call_args = orchestrator.cognitive_engine.autonomous_brain.think.call_args
        print(f"   Context boredom level: {call_args[1].get('context', {}).get('boredom_level')}")
        assert call_args[1].get('context', {}).get('boredom_level') >= 5
    else:
        print("❌ TEST FAILED: Autonomous Brain did NOT trigger.")
        exit(1)

    # 4. Verify Emitter
    if orchestrator._emit_thought_stream.called:
         print("✅ TEST PASSED: Thought stream emitted during reflection.")
    else:
         print("❌ TEST FAILED: No thoughts emitted.")
         exit(1)

if __name__ == "__main__":
    asyncio.run(test_autonomy_loop())


##
