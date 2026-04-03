################################################################################

import asyncio
import time
import logging
import sys
import os

# Set up path for core imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import SovereignOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Benchmark")

async def benchmark():
    print("\n🚀 STARTING LATENCY BENCHMARK...")
    # 1. Initialize Orchestrator
    orchestrator = SovereignOrchestrator()
    await orchestrator.start()
    
    # Wait for initialization
    await asyncio.sleep(2)
    
    tests = [
        ("Hi Aura", "Greeting (Reflex)"),
        ("How are you?", "State Check (Reflex)"),
        ("What is 123 * 456?", "Cognition (Thinking)")
    ]
    
    for msg, category in tests:
        start_time = time.time()
        print(f"\n[Test] Category: {category} | Message: '{msg}'")
        
        # We simulate the message handling
        # Since orchestrator.process_user_input is async but prints, 
        # we'll measure the time until the response is generated.
        await orchestrator._handle_incoming_message(msg)
        
        duration = time.time() - start_time
        print(f"⏱️  Latency: {duration:.2f}s")
        
        if duration < 3.0 and "Reflex" in category:
            print("✅ PASS: Low latency")
        elif "Reflex" in category:
            print("❌ FAIL: Latency too high for reflex")

if __name__ == "__main__":
    asyncio.run(benchmark())


##
