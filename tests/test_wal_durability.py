import pytest
import os
import json
import time
import multiprocessing
from pathlib import Path
from core.resilience.cognitive_wal import CognitiveWAL

def simulate_crash_worker(wal_path):
    """Worker that writes to WAL and then 'crashes' (exits)."""
    wal = CognitiveWAL(filepath=str(wal_path))
    wal.log_intent(
        turn_id="crash_turn_1",
        action="move_north",
        target="dungeon",
        context={"hp": 20, "pos": [5, 5]}
    )
    # Simulate work... and then kill self
    os._exit(1)

def test_wal_durability():
    wal_path = Path("/tmp/test_wal.jsonl")
    if wal_path.exists():
        wal_path.unlink()
        
    print("\n   - Simulating crash mid-intent...")
    p = multiprocessing.Process(target=simulate_crash_worker, args=(wal_path,))
    p.start()
    p.join()
    
    # Now try to recover
    print("   - Restarting and recovering state...")
    wal = CognitiveWAL(filepath=str(wal_path))
    recovered = wal.recover_state()
    
    # Assert: the intent should be there because it was logged BEFORE execution
    assert len(recovered) == 1, "Failed to recover pending intent after crash"
    assert recovered[0]["id"] == "crash_turn_1"
    assert recovered[0]["context"]["hp"] == 20
    
    print("   ✅ WAL durability test passed!")

if __name__ == "__main__":
    test_wal_durability()
