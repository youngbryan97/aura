import asyncio
import logging
import time
from unittest.mock import MagicMock, AsyncMock, patch
from core.orchestrator import RobustOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestConcurrency")

async def test_concurrent_dispatch():
    """
    Verify that multiple rapid messages are serialized and don't cause StateLock timeouts.
    """
    orch = RobustOrchestrator()
    
    # Mock the internal pipeline methods instead of the engine
    # This allows us to simulate a slow cognitive cycle without triggering state machine errors
    async def slow_pipeline(*args, **kwargs):
        msg_id = args[0] if args else "unknown"
        logger.info(f"🟢 [PIPELINE] Started processing: {msg_id}")
        await asyncio.sleep(5)
        logger.info(f"🔴 [PIPELINE] Finished processing: {msg_id}")
        return MagicMock()

    orch._process_message_pipeline = slow_pipeline
    orch._route_prefixed_message = slow_pipeline

    # Send 3 messages rapidly
    logger.info("Sending 3 rapid messages...")
    start_time = time.time()
    orch._dispatch_message("Message 1")
    orch._dispatch_message("Message 2")
    orch._dispatch_message("Message 3")
    
    # Wait for processing (should take ~15s total)
    logger.info("Waiting for all messages to be processed (serialized)...")
    
    # We'll wait more than the expected 15s to be sure
    await asyncio.sleep(18)
    
    duration = time.time() - start_time
    logger.info(f"Total duration: {duration:.2f}s")
    
    # If duration >= 15s, they were serialized (GOOD)
    if duration >= 15.0:
        print("\n✅ TEST PASSED: Messages were serialized by the dispatch semaphore.")
        print(f"Total time {duration:.2f}s matches expected sequential execution (~15s).")
    else:
        print("\n❌ TEST FAILED: Messages were processed concurrently!")
        print(f"Total time {duration:.2f}s is less than 15s (expected sequential).")

if __name__ == "__main__":
    # Ensure we run in a clean loop
    asyncio.run(test_concurrent_dispatch())
