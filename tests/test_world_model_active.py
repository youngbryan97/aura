import pytest
################################################################################


import asyncio
import logging
import sys
import os

# Set up path for core imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import SovereignOrchestrator
from core.world_model.belief_graph import belief_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test.WorldModel")

@pytest.mark.asyncio
async def test_active_surprise():
    print("\n🧪 TESTING ACTIVE WORLD MODELING (SURPRISE-DRIVEN RE-THINKING)...")
    
    from core.service_registration import register_all_services
    register_all_services()
    
    orchestrator = SovereignOrchestrator()
    await orchestrator.start()
    
    # 1. Clear Beliefs for clean test
    belief_graph.edges = {}
    
    # 2. Mock a situation: Ask Aura to check a file that "should" exist but won't
    # We simulate this by checking a non-existent file path
    test_msg = "Check if /tmp/ghost_file.txt exists."
    
    print(f"\n[Test] Message: '{test_msg}'")
    print("Expected Behavior: Aura expects file to exist (potentially), finds it missing (Surprise!), and then reflects.")
    
    # We run the cognitive loop
    # Note: In a real test we'd mock the LLM response to ensure it has an 'expectation'
    # For now, we rely on her actual thinking (Autonomous Brain)
    
    await orchestrator._handle_incoming_message(test_msg)
    
    # 3. Verify Belief Graph Update
    # After 'ls' or 'cat' fails, BeliefGraph should have a belief about ghost_file.txt state
    beliefs = belief_graph.get_beliefs_about("/tmp/ghost_file.txt")
    print(f"\n📊 Extracted Beliefs about ghost_file: {beliefs}")
    
    if beliefs:
        print("✅ PASS: Beliefs updated autonomously.")
    else:
        print("❌ FAIL: No beliefs formed.")

    # 4. Check Trace logs
    trace_dir = os.path.expanduser("~/.aura/traces")
    traces = os.listdir(trace_dir)
    print(f"\n📂 Cognitive Traces generated: {len(traces)}")
    if traces:
        print(f"✅ PASS: Cognitive Trace saved.")
    else:
        print("❌ FAIL: No trace saved.")

if __name__ == "__main__":
    asyncio.run(test_active_surprise())


##
