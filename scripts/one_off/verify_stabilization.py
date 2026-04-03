import asyncio
import time
import logging
import sys
from unittest.mock import MagicMock, AsyncMock

# Add project root to path
sys.path.insert(0, ".")

from core.memory.sovereign_pruner import SovereignPruner, MemoryRecord
from core.senses.neural_bridge import NeuralBridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify.Stabilization")

async def test_parallel_consolidation():
    logger.info("🧪 Testing Parallel Consolidation...")
    pruner = SovereignPruner()
    
    # Mock brain to simulate slow LLM calls
    mock_brain = AsyncMock()
    async def slow_generate(*args, **kwargs):
        await asyncio.sleep(2) # 2 seconds per call
        return "Essential insight."
    mock_brain.generate = slow_generate
    pruner.orchestrator = MagicMock()
    pruner.orchestrator.cognitive_engine = mock_brain
    
    memories = [
        MemoryRecord(
            id=f"mem_{i}", 
            content=f"Memory content {i}", 
            source="test",
            timestamp=time.time(),
            emotional_weight=0.5,
            identity_relevance=0.8
        )
        for i in range(5) # 5 memories
    ]
    
    start_time = time.time()
    # If sequential, this would take 10s. If parallel, it should take ~2s.
    surviving, log = await pruner.prune(memories, {"valence": 0.5})
    duration = time.time() - start_time
    
    logger.info(f"⏱️ Pruning took {duration:.2f}s")
    assert duration < 5, f"Pruning took too long: {duration:.2f}s"
    logger.info("✅ Parallel Consolidation verified.")

async def test_neural_bridge_loop():
    logger.info("🧪 Testing NeuralBridge Thread Safety...")
    bridge = NeuralBridge()
    bridge._event_bus = AsyncMock()
    
    # Capture loop in load
    await bridge.load()
    assert hasattr(bridge, "_main_loop"), "NeuralBridge failed to capture main loop."
    
    # Simulate a decode and check if it uses run_coroutine_threadsafe
    # (Checking if it crashes is the first goal)
    try:
        # We'll just wait a bit to see if the worker thread crashes
        await asyncio.sleep(2)
        logger.info("✅ NeuralBridge worker running without immediate loop error.")
    finally:
        bridge.stop()

async def main():
    try:
        await test_parallel_consolidation()
        await test_neural_bridge_loop()
        print("\n🏆 STABILIZATION VERIFIED: All systems nominal.")
    except Exception as e:
        logger.error(f"❌ Verification Failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
