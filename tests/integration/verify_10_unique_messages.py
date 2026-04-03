import asyncio
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.orchestrator import RobustOrchestrator
from core.container import ServiceContainer
from unittest.mock import AsyncMock, MagicMock

async def verify_10_diverse_messages():
    print("🚀 Starting 10 Unique 'Real' Messages Verification (v3)...")
    print("Using 8-bit Qwen model on M5... this will take a few seconds to prefill.")
    
    # 1. Initialize Mental Stack Services
    from core.brain.llm_health_router import HealthAwareLLMRouter
    from core.brain.llm.autonomous_brain_integration import AutonomousCognitiveEngine
    from core.capability_engine import CapabilityEngine
    from core.event_bus import get_event_bus
    from core.api_adapter import get_api_adapter
    from core.cognitive_integration_layer import CognitiveIntegrationLayer

    # Setup Router & ACE (Discovery Layer)
    router = HealthAwareLLMRouter()
    ServiceContainer.register_instance("llm_router", router)
    
    # Populates router with MLX tiers from model_registry
    ace = AutonomousCognitiveEngine(
        registry=CapabilityEngine(),
        llm_router=router,
        event_bus=get_event_bus()
    )
    
    # Setup CIL & Adapter
    cil = CognitiveIntegrationLayer()
    await cil.initialize()
    ServiceContainer.register_instance("cognitive_integration", cil)
    ServiceContainer.register_instance("api_adapter", get_api_adapter())

    # 2. Setup Orchestrator
    orchestrator = RobustOrchestrator()
    orchestrator.cognitive_integration = cil

    # Start orchestrator
    loop_task = asyncio.create_task(orchestrator.run())
    await asyncio.sleep(1.0) # Wait for event loop to settle
    
    messages = [
        "Hello Aura, are you online?",
        "What is your current cognitive state?",
        "How do you feel about working on an M5 Mac?",
        "Tell me a unique fact about quantum computing.",
        "Who is your favorite philosopher?",
        "What are your primary goals for this session?",
        "Explain the concept of 'emergent intelligence'.",
        "How do you handle unified memory on Apple Silicon?",
        "What have you learned about Bryan so far?",
        "Goodbye Aura, stand by for optimization."
    ]

    success_count = 0
    results = []
    
    for i, msg in enumerate(messages):
        print(f"[{i+1}/10] Human: {msg}")
        start_time = time.time()
        
        # Use _process_message which calls the full cognitive pipeline
        result = await orchestrator._process_message(msg)
        
        duration = time.time() - start_time
        if result.get("ok"):
            response = result.get("response")
            print(f"      🤖 Aura: {response} ({duration:.2f}s)")
            results.append((msg, response))
            success_count += 1
        else:
            print(f"      ❌ Failed: {result.get('error')}")

    print("\n" + "="*60)
    print(f"FINAL VERIFICATION REPORT: {success_count}/10 successful turns")
    print("="*60)
    for i, (m, r) in enumerate(results):
        print(f"{i+1}. Q: {m}")
        print(f"   A: {r}\n")
    
    await orchestrator.stop()
    await loop_task
    
    if success_count == 10:
        print("🏆 DIVERSE VERIFICATION SUCCESSFUL")
        return True
    return False

if __name__ == "__main__":
    success = asyncio.run(verify_10_diverse_messages())
    sys.exit(0 if success else 1)
