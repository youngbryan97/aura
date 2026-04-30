import asyncio
import logging
import sys

import pytest

from core.container import ServiceContainer
from core.brain.llm.llm_router import IntelligentLLMRouter
from core.brain.inference_gate import InferenceGate

pytestmark = pytest.mark.skip(reason="manual LLM smoke test requires a live inference stack")

async def test_generation():
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components if not already
    gate = ServiceContainer.get("inference_gate")
    if not gate._initialized:
        await gate.initialize()
    
    router = ServiceContainer.get("llm_router")
    
    print("--- Testing Foreground Generation (32B) ---")
    try:
        res = await router.generate("Hello, who are you?", origin="user")
        print(f"Response: {res}")
    except Exception as e:
        print(f"Foreground Error: {e}")

    print("\n--- Testing Background Generation (7B) ---")
    try:
        res = await router.generate("Summarize the current state.", is_background=True)
        print(f"Response: {res}")
    except Exception as e:
        print(f"Background Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_generation())
