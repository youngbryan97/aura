import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock

# Configure logging
logging.basicConfig(level=logging.DEBUG)

async def test_mind_tick_resilience():
    print("🚀 Starting MindTick Resilience Test...")
    
    # Mock Orchestrator and Status
    mock_orch = MagicMock()
    mock_orch.status.cycle_count = 0
    mock_orch.state_repo.get_current = AsyncMock(return_value=MagicMock())
    
    # Mock EventBus to hang
    from core.event_bus import get_event_bus
    bus = get_event_bus()
    
    async def hung_publish(*args, **kwargs):
        print("⏳ Mock EventBus: Hanging publish for 10s...")
        await asyncio.sleep(10)
        print("✅ Mock EventBus: Publish finally finished (should have timed out).")

    bus.publish = hung_publish

    # Import MindTick and initialize (avoiding full bootstrap for speed)
    from core.mind_tick import MindTick, CognitiveMode
    
    # Patch _bootstrap_phases to avoid loading all phases
    MindTick._bootstrap_phases = lambda self: None
    
    mt = MindTick(mock_orch)
    mt.mode = CognitiveMode.CRITICAL # Fast ticks (0.1s)
    
    print("💓 Starting MindTick...")
    await mt.start()
    
    # Wait for a few ticks.
    # With a 5s timeout on publish, each tick should take ~5s if it hangs.
    # Uptime was 26s in user report, so we wait for ~12s to see at least 2 ticks.
    print("⏳ Waiting for cycles to increment despite hangs...")
    for i in range(15):
        await asyncio.sleep(1)
        count = mock_orch.status.cycle_count
        print(f"   [T+{i}s] Cycle Count: {count}")
        if count >= 2:
            print("✅ SUCCESS: Cycle count incremented past stall point!")
            break
    
    await mt.stop()
    
    if mock_orch.status.cycle_count < 2:
        print("❌ FAILURE: Cycle count stayed below 2.")
        exit(1)
    else:
        print(f"🎉 Final Cycle Count: {mock_orch.status.cycle_count}")

if __name__ == "__main__":
    asyncio.run(test_mind_tick_resilience())
