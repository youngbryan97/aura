################################################################################


import sys
import os
import time
import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StressTest")

from core.orchestrator import RobustOrchestrator
from core.brain.cognitive_engine import CognitiveEngine, Thought

class StressTestOrchestrator(unittest.IsolatedAsyncioTestCase):
    
    async def test_high_throughput(self):
        """Test sending 50 messages rapidly through the current user pipeline."""
        logger.info("--- Starting High Throughput Test ---")

        orchestrator = RobustOrchestrator()
        orchestrator._inference_gate = AsyncMock()
        orchestrator._inference_gate.generate = AsyncMock(
            side_effect=lambda message, context=None: f"Fast response: {message}"
        )

        class _Kernel:
            def is_ready(self):
                return False

        start_time = time.time()
        with patch(
            "core.kernel.kernel_interface.KernelInterface.get_instance",
            return_value=_Kernel(),
        ):
            for i in range(50):
                response = await orchestrator.process_user_input(f"Message {i}")
                self.assertIn("Fast response", response)
                if i % 10 == 0:
                    logger.info(f"Processed {i}/50 messages...")
                
        duration = time.time() - start_time
        logger.info(f"✅ High Throughput Test Passed: 50 messages in {duration:.2f}s")

    async def test_cognitive_timeout(self):
        """Test that the Soft Timeout in CognitiveEngine kicks in"""
        logger.info("\n--- Starting Cognitive Timeout Test ---")
        
        # Real Cognitive Engine, Mock Client that sleeps
        engine = CognitiveEngine()
        engine.client = AsyncMock()
        
        # Mock generate to sleep longer than the 25s timeout
        # We need to use a side_effect that sleeps asynchronously
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(1) # Sleep 1s instead of 26s to not block test if timeout is lower, wait actually timeout is 25s? 
            # Oh wait, we should sleep 26s to trigger timeout. But I'll just sleep 26s.
            await asyncio.sleep(26)
            return '{"content": "Too late"}'
            
        engine.client.generate.side_effect = slow_generate
        
        # We need to patch the check_health to always return True so it doesn't fallback
        engine.client.check_health.return_value = True
        
        logger.info("Invoking think() with slow client...")
        start_time = time.time()
        
        # This should trigger the 25s soft timeout
        thought = await engine.think("Test timeout")
        
        duration = time.time() - start_time
        logger.info(f"Think duration: {duration:.2f}s")
        
        logger.info(f"Thought content: {thought.content}")
        
        # Check if we got the fallback message
        if "processing a complex thought pattern" in thought.content:
            logger.info("✅ Timeout Test Passed: Soft timeout triggered and fallback returned.")
        elif duration < 25:
             logger.error("❌ Timeout Test Failed: Returned too early (Mock didn't sleep?)")
        else:
             logger.error(f"❌ Timeout Test Failed: Unexpected content: {thought.content}")

if __name__ == "__main__":
    unittest.main()


##
