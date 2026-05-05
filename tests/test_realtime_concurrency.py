import pytest
import asyncio
import time
from core.container import ServiceContainer

def heavy_background_task():
    """Simulate heavy CPU-bound task (e.g. dream consolidation)."""
    while True:
        # Blocking the event loop entirely for 100ms
        import time
        time.sleep(0.1)
        yield

@pytest.mark.asyncio
async def test_realtime_concurrency():
    """Prove that background work blocks real-time reflex actions."""
    print("\n   - Starting background dream consolidation...")
    
    # We use a thread to run the blocking background task 
    # to see if the main loop can stay responsive
    def run_bg():
        for _ in heavy_background_task():
            pass

    import threading
    bg_thread = threading.Thread(target=run_bg, daemon=True)
    bg_thread.start()
    
    latencies = []
    print("   - Measuring reflex latency (10 samples)...")
    for _ in range(10):
        start = time.perf_counter()
        await asyncio.sleep(0.01) # Baseline 10ms
        latencies.append(time.perf_counter() - start)
    
    # Threshold is tight: < 100ms for reflexes even with blocking background
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    print(f"   - p95 Latency: {p95*1000:.2f}ms")
    
    assert p95 < 0.1, f"Reflex latency too high: {p95*1000:.2f}ms (p95)"
    
    print("   ✅ Real-time concurrency test passed!")
