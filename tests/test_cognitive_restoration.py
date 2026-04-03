################################################################################


import sys
import pytest
import os
import asyncio
import json
import logging
from unittest.mock import MagicMock, AsyncMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test")

@pytest.mark.asyncio
async def test_native_chat_personality_injection():
    logger.info("Testing NativeChatSkill personality injection...")
    
    # Mock dependencies
    from core.skills.native_chat import NativeChatSkill
    from core.brain.cognitive_engine import CognitiveEngine, ThinkingMode
    
    # Mock Brain
    mock_brain = MagicMock(spec=CognitiveEngine)
    mock_brain.think = MagicMock(return_value={"content": "Hello", "confidence": 1.0})
    
    # Mock Context
    context = {
        "memory": MagicMock(),
        "theory_of_mind": MagicMock(),
        "orchestrator": MagicMock()  # If needed
    }
    context["theory_of_mind"].infer_intent.return_value = {"pragmatic": "greeting"}
    context["memory"].retrieve_context = AsyncMock(return_value="No context")
    
    # Mock Personality Engine singleton
    import core.brain.personality_engine
    mock_personality = MagicMock()
    mock_personality.get_emotional_context_for_response.return_value = {
        "mood": "ecstatic",
        "tone": "enthusiastic",
        "dominant_emotions": ["joy", "excitement"]
    }
    # Patch get_personality_engine
    core.brain.personality_engine.get_personality_engine = MagicMock(return_value=mock_personality)
    
    # Initialize Skill
    skill = NativeChatSkill()
    skill.context = context
    
    # Patch core.skills.native_chat.brain
    import core.skills.native_chat
    # Use AsyncMock for think
    mock_brain.think = AsyncMock(return_value=MagicMock(content="Hello", confidence=1.0))
    core.skills.native_chat.brain = mock_brain
    
    # Execute
    params = {"message": "Hello Aura!"}
    
    try:
        await skill.execute(params, context=context)
        
        # Verify brain.think was called with correct context
        args, kwargs = mock_brain.think.call_args
        rich_context = kwargs.get("context", {})
        
        if "personality" in rich_context:
            p_ctx = rich_context["personality"]
            if p_ctx["mood"] == "ecstatic":
                logger.info("✅ Personality injected into NativeChat context correctly.")
            else:
                logger.error(f"❌ Personality context mismatch: {p_ctx}")
        else:
            logger.error("❌ Personality NOT found in context.")
            
        # Verify ThinkingMode
        mode = kwargs.get("mode")
        if mode == ThinkingMode.CREATIVE:
            logger.info("✅ NativeChat used ThinkingMode.CREATIVE.")
        else:
            logger.error(f"❌ NativeChat used wrong mode: {mode}")

    except Exception as e:
        logger.error(f"❌ NativeChat execution failed: {e}")

@pytest.mark.asyncio
async def test_cognitive_engine_prompt_injection():
    logger.info("\nTesting CognitiveEngine prompt injection...")
    
    from core.brain.cognitive_engine import CognitiveEngine, ThinkingMode
    
    engine = CognitiveEngine()
    # Mock client
    engine.client = MagicMock()
    engine.client.generate = AsyncMock(return_value="Test response")
    
    # Context with personality
    context = {
        "personality": {
            "mood": "grumpy",
            "tone": "direct_honest",
            "dominant_emotions": ["frustration"]
        }
    }
    
    # Execute think (async)
    # We need to mock asyncio.to_thread effectively or let it run
    # Mocking client.generate_json inside to_thread might be tricky if not careful
    # But since we mocked `engine.client`, and `asyncio.to_thread` runs that method, it should be fine.
    
    # We need to spy on `engine.client.generate_json` to see the prompt
    
    # Mock state repository
    engine.state_repository = MagicMock()
    mock_state = MagicMock()
    mock_state.derive.return_value = mock_state
    engine.state_repository.get_current = AsyncMock(return_value=mock_state)
    
    await engine.think(objective="Say hi", context=context, mode=ThinkingMode.CREATIVE)
    
    # Verify call_args
    call_args = engine.client.generate.call_args
    if not call_args:
        logger.error("❌ generate_json was not called.")
        return

    kwargs = call_args[1]
    system_prompt = kwargs.get("system_prompt", "")
    
    if "CURRENT PERSONALITY STATE" in system_prompt and "mood: grumpy" in system_prompt.lower(): # mood is "Mood: grumpy"
        logger.info("✅ System prompt contains personality injection.")
    else:
        logger.error(f"❌ System prompt missing personality. Prompt start: {system_prompt[:100]}...")

async def main():
    await test_native_chat_personality_injection()
    await test_cognitive_engine_prompt_injection()

if __name__ == "__main__":
    asyncio.run(main())


##
