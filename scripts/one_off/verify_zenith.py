from core.runtime.atomic_writer import atomic_write_text
import asyncio
import sys
import os
from pathlib import Path

# Fix paths
sys.path.insert(0, str(Path.cwd()))

async def test_zenith_fixes():
    print("🧪 Testing Zenith Audit Fixes...")

    # 1. Test Mycelial Network (Cycle Detection)
    print("\n1. Testing Mycelial Network...")
    try:
        from core.mycelial.graph import get_mycelial
        mycelium = get_mycelial()
        # Add a safe edge
        await mycelium.add_edge("memory_A", "skill_B")
        # Try to create a cycle
        success = await mycelium.add_edge("skill_B", "memory_A")
        if not success:
            print("✅ Cycle detection blocked correctly.")
        else:
            print("❌ Cycle detection failed!")
    except Exception as e:
        print(f"❌ Mycelial test error: {e}")

    # 2. Test Safety Registry
    print("\n2. Testing Safety Registry...")
    try:
        from core.agency.safety_registry import get_safety_registry
        safety = get_safety_registry()
        await safety.disable_skill("dangerous_skill")
        is_allowed = await safety.is_allowed("dangerous_skill")
        if not is_allowed:
            print("✅ Skill revocation working.")
        else:
            print("❌ Skill revocation failed!")
    except Exception as e:
        print(f"❌ Safety registry test error: {e}")

    # 3. Test Hybrid Store (Unicode & Pruning)
    print("\n3. Testing Hybrid Store...")
    try:
        from core.memory.hybrid_store import get_hybrid_store
        store = get_hybrid_store()
        await store.store("Testing Zenith memory fix with emoji 🛡️", {"confidence": 0.9})
        results = await store.retrieve("Zenith")
        if results and "🛡️" in results[0]['content']:
            print("✅ Hybrid store unicode-safe and retrieving.")
        else:
            print("❌ Hybrid store retrieval failed.")
    except Exception as e:
        print(f"❌ Hybrid store test error: {e}")

    # 4. Test Safe Optimizer (Safety checks)
    print("\n4. Testing Safe Optimizer...")
    try:
        from core.adaptation.safe_optimizer import get_safe_optimizer
        opt = get_safe_optimizer()
        # Mock a small file
        test_data = Path("/tmp/lora_test.txt")
        atomic_write_text(test_data, "dummy dataset content")
        await opt.optimize_lora(str(test_data), "base_model")
        print("✅ Safe optimizer executed without crash.")
    except Exception as e:
        print(f"❌ Safe optimizer test error: {e}")

    print("\n✨ Zenith Audit Fixes Verification Complete.")

if __name__ == "__main__":
    asyncio.run(test_zenith_fixes())
