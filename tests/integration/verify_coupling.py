################################################################################

"""Verify Substrate-Behavior Coupling."""
import asyncio
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.brain.consciousness.conscious_core import ConsciousnessCore
from core.brain.consciousness.liquid_substrate import SubstrateConfig

class MockOrchestrator:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.impulses = []
    
    async def handle_impulse(self, impulse):
        print(f"🚀 Mock Orchestrator received impulse: {impulse}")
        self.impulses.append(impulse)

async def test_coupling():
    print("Testing Substrate-Behavior Coupling...")
    core = ConsciousnessCore()
    orchestrator = MockOrchestrator()
    core.orchestrator_ref = orchestrator
    
    # Ensure a clean slate for telemetry
    from core.config import config
    telemetry_path = config.paths.data_dir / "telemetry" / "causal_behavior.jsonl"
    if telemetry_path.exists():
        get_task_tracker().create_task(get_storage_gateway().delete(telemetry_path, cause='test_coupling'))

    core.start()
    
    try:
        # Force a state into the "Boredom" Basin
        # v < -0.1, a < -0.2
        print("💉 Injecting 'Boredom' stimulus...")
        # valence is idx 0, arousal is idx 1
        boredom_vec = np.zeros(64)
        boredom_vec[0] = -0.5 # Negative valence
        boredom_vec[1] = -0.6 # Low arousal
        
        await core.substrate.inject_stimulus(boredom_vec, weight=2.0)
        
        # Wait for the volition loop to trigger (it runs at 1Hz)
        print("⏳ Waiting for volition to emerge from substrate...")
        # Check for 5 seconds
        for _ in range(10):
            await asyncio.sleep(0.5)
            state = await core.substrate.get_state_summary()
            print(f"Current State: V={state['valence']:.2f}, A={state['arousal']:.2f}, D={state['dominance']:.2f}")
            if orchestrator.impulses:
                break
        
    finally:
        core.stop()
        
    if orchestrator.impulses:
        print(f"✅ Causal Impulse Triggered: {orchestrator.impulses[0]}")
        # Run analysis (AFTER core is stopped and files are closed)
        from scripts.prove_coupling import analyze_coupling
        analyze_coupling()
    else:
        print("❌ No impulse triggered. Check refractory periods or basin definitions.")

if __name__ == "__main__":
    asyncio.run(test_coupling())


##
