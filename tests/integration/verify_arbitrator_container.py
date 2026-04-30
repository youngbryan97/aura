from core.utils.task_tracker import get_task_tracker
import sys
import unittest.mock as mock

# Mock pydantic to avoid version mismatch errors
mock_pydantic = mock.MagicMock()
sys.modules['pydantic'] = mock_pydantic

import asyncio
import time
import types
from core.resilience.resource_arbitrator import get_resource_arbitrator
from core.container import ServiceContainer

async def test_arbitrator():
    arbitrator = get_resource_arbitrator()
    
    print("Test 1: Inference blocks Evolution")
    async with arbitrator.inference_context():
        print("Main: Inference lock acquired.")
        
        evo_blocked = True
        
        async def background_evo():
            nonlocal evo_blocked
            async with arbitrator.evolution_context():
                evo_blocked = False
                print("Background: Evolution lock acquired!")
        
        evo_task = get_task_tracker().create_task(background_evo())
        await asyncio.sleep(1) # Wait for task to try lock
        
        if evo_blocked:
            print("✅ Evolution correctly blocked by Inference.")
        else:
            print("❌ Evolution NOT blocked by Inference!")
            
    await asyncio.sleep(0.5)
    if not evo_blocked:
        print("✅ Evolution proceeded after Inference release.")
    else:
        print("❌ Evolution still blocked after Inference release!")

async def test_container():
    print("\nTest 2: ServiceContainer Hardening")
    # Reset registry for test
    ServiceContainer._registration_locked = False
    ServiceContainer._services = {} # Reset to plain dict
    
    ServiceContainer.register("test_service", lambda: {"id": 1})
    ServiceContainer.lock_registration()
    print("Container locked.")
    
    # 1. Test register() early return
    ServiceContainer.register("rogue_service", lambda: {"id": 666})
    if "rogue_service" not in ServiceContainer._services:
        print("✅ Successfully prevented rogue registration (Early return).")
    else:
        print("❌ Rogue service found in registry!")

    # 2. Test direct mutation of MappingProxy
    try:
        ServiceContainer._services["rogue_direct"] = mock.MagicMock()
        print("❌ Managed to directly mutate frozen registry!")
    except TypeError:
        print("✅ Correctly rejected direct mutation of frozen registry (MappingProxy verified).")

async def main():
    await test_arbitrator()
    await test_container()

if __name__ == "__main__":
    asyncio.run(main())

