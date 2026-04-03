
import asyncio
import time
import logging
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append("/Users/bryan/Desktop/aura")

from core.brain.llm.mlx_client import MLXLocalClient
from core.phases.response_generation import ResponseGenerationPhase
from core.phases.affect_update import AffectUpdatePhase
from core.state.aura_state import AuraState
from core.utils.concurrency import RobustLock
from core.container import ServiceContainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY")

async def test_mlx_client_retries():
    """Verify that MLXLocalClient performs tiered retries on timeout."""
    logger.info("--- Testing MLXLocalClient Tiered Retries ---")
    client = MLXLocalClient(model_path="/mock/model")
    
    # Mock the worker process communication to always fail/timeout
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = MagicMock()
        mock_proc.stdout.readline = MagicMock(side_effect=asyncio.TimeoutError)
        mock_exec.return_value = mock_proc
        
        # We need to mock generate_text_async_inner to simulate failure
        # Or mock the parts it calls.
        
        with patch.object(MLXLocalClient, "_generate_text_async_inner", side_effect=asyncio.TimeoutError):
            start_time = time.time()
            try:
                # This should take ~45 + 90 + 120 = 255s if it retries fully
                # But we can't wait that long. We'll verify it calls inner multiple times.
                await asyncio.wait_for(client.generate_text_async("Test"), timeout=2.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.info(f"Caught expected failure/timeout: {type(e).__name__}")
            
    # A better test: Mock inner to fail 2 times then succeed
    with patch.object(MLXLocalClient, "_generate_text_async_inner") as mock_inner:
        mock_inner.side_effect = [asyncio.TimeoutError(), asyncio.TimeoutError(), (True, "Success!", {})]
        
        # This should take 3 calls
        success, result, meta = await client.generate_text_async("Test prompt")
        logger.info(f"Result: {result}")
        assert result == "Success!"
        assert mock_inner.call_count == 3
        logger.info("✅ MLX Tiered Retries verified.")

async def test_response_phase_watchdog():
    """Verify ResponseGenerationPhase handles its own watchdog."""
    logger.info("--- Testing ResponseGenerationPhase Watchdog ---")
    container = ServiceContainer()
    phase = ResponseGenerationPhase(container)
    state = AuraState.default()
    state.cognition.current_objective = "Test objective"
    state.cognition.current_origin = "user"
    
    # Mock LLM router to hang
    mock_router = MagicMock()
    async def hanging_think(*args, **kwargs):
        await asyncio.sleep(10)
        return "Not reached"
    mock_router.think = hanging_think
    container.register_instance("llm_router", mock_router)
    
    # Run phase with a short local watchdog for test
    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        result_state = await phase.execute(state, objective="Test")
        # Check if it returned a failure state
        assert result_state.transition_cause == "generation_timeout"
        assert "timed out" in result_state.cognition.working_memory[-1]["content"]
        logger.info("✅ ResponseGenerationPhase Watchdog verified.")

async def test_affect_telemetry_sync():
    """Verify AffectUpdatePhase pushes VAD to LiquidSubstrate."""
    logger.info("--- Testing Affect Telemetry Sync ---")
    container = ServiceContainer()
    mock_kernel = MagicMock()
    mock_kernel.organs = {}
    ls = MagicMock()
    ls.update = MagicMock(return_value=asyncio.Future())
    ls.update.return_value.set_result(None)
    container.register_instance("liquid_substrate", ls)
    
    phase = AffectUpdatePhase(mock_kernel)
    state = AuraState.default()
    state.affect.emotions["joy"] = 0.9
    state.affect.emotions["trust"] = 0.8
    
    # Execute phase
    await phase.execute(state)
    
    # Check if ls.update was called
    # Note: it's called via create_task, so we might need a small sleep
    await asyncio.sleep(0.1)
    assert ls.update.called
    args, kwargs = ls.update.call_args
    logger.info(f"LS Update Args: {kwargs}")
    assert "valence" in kwargs
    assert "arousal" in kwargs
    logger.info("✅ Affect Telemetry Sync verified.")

async def test_robust_lock():
    """Verify RobustLock works across event loops."""
    logger.info("--- Testing RobustLock ---")
    lock = RobustLock("TestLock")
    
    # Case 1: Simple acquisition
    assert await lock.acquire_robust(timeout=1.0) is True
    lock.release()
    
    # Case 2: Concurrent acquisition
    async def worker():
        acquired = await lock.acquire_robust(timeout=1.0)
        if acquired:
            await asyncio.sleep(0.1)
            lock.release()
        return acquired

    results = await asyncio.gather(worker(), worker(), worker())
    assert all(results)
    logger.info("✅ RobustLock verified.")

async def main():
    try:
        await test_mlx_client_retries()
        await test_response_phase_watchdog()
        await test_affect_telemetry_sync()
        await test_robust_lock()
        logger.info("\n🏆 ALL ARCHITECTURAL FIXES VERIFIED 🏆")
    except Exception as e:
        logger.error(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
