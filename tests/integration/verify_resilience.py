################################################################################


import asyncio
import sys
import os
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.resilience.state_manager import StateManager
from core.orchestrator import RobustOrchestrator

async def test_resilience():
    print("--- Testing Resilience (State Snapshot) ---")
    
    # 1. Setup Mock System
    print("Initializing Orchestrator...")
    orchestrator = RobustOrchestrator()
    
    # Simulate some activity
    orchestrator.status.cycle_count = 42
    orchestrator.conversation_history = [
        {"role": "user", "content": "Hello Aura"},
        {"role": "assistant", "content": "Hello User"}
    ]
    orchestrator.boredom = 5
    
    # 2. Save Snapshot
    print("Saving Snapshot...")
    orchestrator._save_state("test_verification")
    
    # Verify file exists
    snapshot_path = Path(orchestrator.state_manager.snapshot_dir) / "latest_snapshot.json"
    if snapshot_path.exists():
        print(f"✓ Snapshot file created at {snapshot_path}")
    else:
        print("❌ Snapshot file missing!")
        return

    # 3. Simulate Crash (New Instance)
    print("Simulating Crash & Restart (New Instance)...")
    new_orchestrator = RobustOrchestrator()
    
    # 4. Verify Restoration
    # Note: _load_state is called in __init__
    
    if new_orchestrator.status.cycle_count == 42:
        print("✓ Cycle Count Restored (42)")
    else:
        print(f"❌ Cycle Count Mismatch: {new_orchestrator.status.cycle_count}")
        
    if len(new_orchestrator.conversation_history) == 2:
        print("✓ Conversation History Restored")
    else:
        print(f"❌ History Mismatch: len={len(new_orchestrator.conversation_history)}")

    if new_orchestrator.boredom == 5:
        print("✓ Boredom Level Restored")
    else:
        print(f"❌ Boredom Mismatch: {new_orchestrator.boredom}")
        
    print("--- Resilience Test Complete ---")

if __name__ == "__main__":
    asyncio.run(test_resilience())


##
