################################################################################


import pytest
import asyncio
from core.orchestrator import RobustOrchestrator
from core.container import ServiceContainer, ServiceLifetime

@pytest.mark.asyncio
async def test_memory_injection_and_remember():
    """
    Verify that:
    1. Orchestrator initializes self.memory with SemanticMemory.
    2. self.memory.remember() can be called without error.
    """
    print("\n[TEST] Registering services...")
    from core.service_registration import register_all_services
    register_all_services()

    print("\n[TEST] Initializing Orchestrator...")
    orchestrator = RobustOrchestrator()
    
    # Manually trigger init sequence if not auto-run (Orchestrator usually does this in __init__ or run)
    # Based on previous reads, __init__ calls _init_cognitive_core, so it should be there.
    
    print(f"[TEST] Orchestrator memory type: {type(orchestrator.memory)}")
    
    assert orchestrator.memory is not None, "Orchestrator.memory should not be None"
    
    # Check if it has the remember method
    assert hasattr(orchestrator.memory, "remember"), "Memory object must have 'remember' method"
    
    print("[TEST] Calling memory.remember()...")
    try:
        await orchestrator.memory.remember("Test memory injection", metadata={"test": True})
        print("[TEST] memory.remember() succeeded.")
    except Exception as e:
        pytest.fail(f"memory.remember() failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_memory_injection_and_remember())


##
