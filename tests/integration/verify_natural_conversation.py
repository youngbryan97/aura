################################################################################

import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.brain.personality_engine import get_personality_engine
from core.orchestrator import RobustOrchestrator

async def test_time_awareness():
    print("--- Testing Time Awareness ---")
    personality = get_personality_engine()
    ctx = personality.get_time_context()
    print(f"Current Time Context: {ctx}")
    
    assert "period" in ctx
    assert "formatted" in ctx
    assert "energy_level" in ctx
    print("✓ Time context structure valid")

async def test_spontaneous_speech_handling():
    print("\n--- Testing Spontaneous Speech Handling ---")
    orch = RobustOrchestrator()
    orch.reply_queue = MagicMock()
    orch._emit_thought_stream = MagicMock()
    orch.execute_tool = AsyncMock()
    
    # Mock cognitive engine and brain
    orch.cognitive_engine = MagicMock()
    orch.cognitive_engine.autonomous_brain = MagicMock()
    
    # Mock a "check" where the brain decides to speak
    check_result = {
        "content": "I should say goodnight.",
        "tool_calls": [
            {"name": "speak", "args": {"message": "It's getting late, you should sleep."}}
        ]
    }
    
    orch.cognitive_engine.autonomous_brain.think = AsyncMock(return_value=check_result)
    
    # Force boredom to trigger
    orch.boredom = 10
    
    await orch._perform_autonomous_thought()
    
    # Verify speech was queued
    if orch.reply_queue.put.called:
        args = orch.reply_queue.put.call_args[0][0]
        print(f"✓ Speech queued: {args}")
        assert "sleep" in args
    else:
        print("❌ Speech NOT queued")

    # Verify thought emitted
    if orch._emit_thought_stream.called:
        print("✓ Thought emitted")
    else:
        print("❌ Thought NOT emitted")

if __name__ == "__main__":
    asyncio.run(test_time_awareness())
    asyncio.run(test_spontaneous_speech_handling())


##
