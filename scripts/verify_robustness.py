"""
verify_robustness.py
───────────────────
Stress-tests Aura's new architectural sensors by intentionally
introducing code smells and verifying detection.
"""
from core.runtime.atomic_writer import atomic_write_text
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.self_modification.ast_analyzer import ASTAnalyzer

def test_robustness():
    analyzer = ASTAnalyzer(PROJECT_ROOT)
    
    # 1. Test Deadlock Detection
    deadlock_code = """
class DeadlockDemo:
    def __init__(self):
        import threading
        self._lock = threading.Lock()
    
    def dangerous_method(self):
        with self._lock:
            print("Acquired once")
            with self._lock:  # NESTED LOCK ON SAME OBJECT
                print("Acquired twice - DEADLOCK")
"""
    temp_file = PROJECT_ROOT / "temp_deadlock_test.py"
    atomic_write_text(temp_file, deadlock_code)
    
    print("🔍 Testing Deadlock Detection...")
    results = analyzer.analyze_file(temp_file)
    deadlock_smells = [s for s in results.get("smells", []) if s["type"] == "deadlock_risk"]
    
    if deadlock_smells:
        print(f"✅ SUCCESS: Detected {len(deadlock_smells)} deadlock risks.")
        for s in deadlock_smells: print(f"   -> {s['message']} (Line {s['lineno']})")
    else:
        print("❌ FAILURE: Deadlock not detected.")
    
    # 2. Test Async Stall Detection
    stall_code = """
import asyncio
import time

async def stalling_loop():
    while True:
        # Synchrous blocker in async loop!
        time.sleep(10)
        print("Still stalling...")
"""
    temp_file_2 = PROJECT_ROOT / "temp_stall_test.py"
    atomic_write_text(temp_file_2, stall_code)
    
    print("\n🔍 Testing Async Stall Detection...")
    results_2 = analyzer.analyze_file(temp_file_2)
    stall_smells = [s for s in results_2.get("smells", []) if s["type"] == "blocking_call"]
    
    if stall_smells:
        print(f"✅ SUCCESS: Detected {len(stall_smells)} blocking calls in async context.")
        for s in stall_smells: print(f"   -> {s['message']} (Line {s['lineno']})")
    else:
        print("❌ FAILURE: Stall not detected.")

    # 3. Test Resource Leak Detection
    leak_code = """
def leak_resources():
    f = open("log.txt", "w")
    f.write("leaking")
    # No close, no with!
"""
    temp_file_3 = PROJECT_ROOT / "temp_leak_test.py"
    atomic_write_text(temp_file_3, leak_code)
    
    print("\n🔍 Testing Resource Leak Detection...")
    results_3 = analyzer.analyze_file(temp_file_3)
    leak_smells = [s for s in results_3.get("smells", []) if s["type"] == "resource_leak"]
    
    if leak_smells:
        print(f"✅ SUCCESS: Detected {len(leak_smells)} resource leaks.")
        for s in leak_smells: print(f"   -> {s['message']} (Line {s['lineno']})")
    else:
        print("❌ FAILURE: Resource leak not detected.")

    # Cleanup
    for f in [temp_file, temp_file_2, temp_file_3]:
        if f.exists(): f.unlink()

if __name__ == "__main__":
    test_robustness()
