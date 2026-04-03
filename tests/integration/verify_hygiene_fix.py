import asyncio
import logging
import sys
import os

# Ensure we can import from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.phases.memory_consolidation import MemoryConsolidationPhase
from core.container import ServiceContainer

# Configure logging to see results
logging.basicConfig(level=logging.INFO)

async def test_hygiene_fix():
    print("\n" + "="*50)
    print("Testing MemoryConsolidationPhase Hygiene Fix...")
    print("="*50)
    
    # 1. Setup mock services
    mock_memory = type('MockMemory', (), {'consolidate': lambda self, x: print(f"Consolidated: {len(x)} items")})()
    ServiceContainer.register_instance("memory_synthesizer", mock_memory)
    
    # 2. Setup phase
    phase = MemoryConsolidationPhase(container=ServiceContainer)
    
    # 3. Create a "dirty" working memory with a non-dict object (The Orchestrator Leak)
    class FakeOrchestrator:
        def __repr__(self): return "<RobustOrchestrator object at 0x...>"
        
    dirty_memory = [
        {"role": "user", "content": "hello"},
        FakeOrchestrator(), # This should be filtered out
        {"role": "aura", "content": "hi"}
    ]
    
    # 4. Structure state correctly
    # state.cognition.working_memory
    class MockCognition:
        def __init__(self, wm):
            self.working_memory = wm
            
    class MockState:
        def __init__(self, wm):
            self.cognition = MockCognition(wm)
    
    state = MockState(dirty_memory)
    
    print(f"Memory before execution (length {len(dirty_memory)}): {dirty_memory}")
    
    # 5. Execute phase
    try:
        await phase.execute(state)
        print(f"Memory after execution (length {len(state.cognition.working_memory)}): {state.cognition.working_memory}")
        
        # Verify filtering
        if any(not isinstance(m, dict) for m in state.cognition.working_memory):
            print("✗ FAILURE: Non-dict items still present in memory.")
        elif len(state.cognition.working_memory) == 2:
            print("✓ SUCCESS: Non-dict item (Orchestrator) filtered out safely.")
        else:
            print(f"？ UNEXPECTED: Memory length is {len(state.cognition.working_memory)}")
            
    except AttributeError as e:
        print(f"✗ FAILURE: AttributeError still present: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"✗ FAILURE: Unexpected error type: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_hygiene_fix())

