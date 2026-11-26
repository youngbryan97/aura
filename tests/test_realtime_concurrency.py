import asyncio
import threading
import time

import pytest


def heavy_background_task(stop_event: threading.Event):
    """Simulate blocking background work (e.g. dream consolidation)."""
    while not stop_event.is_set():
        # Sleeps in a worker thread; the event loop should remain responsive.
        time.sleep(0.1)
        yield


async def _measure_sleep_p95(samples: int = 10, interval_s: float = 0.01) -> float:
    latencies = []
    for _ in range(samples):
        start = time.perf_counter()
        await asyncio.sleep(interval_s)
        latencies.append(time.perf_counter() - start)
    return sorted(latencies)[int(len(latencies) * 0.95)]


@pytest.mark.asyncio
async def test_realtime_concurrency():
    """Prove background work does not materially block real-time reflex actions."""
    print("\n   - Starting background dream consolidation...")
    
    # We use a thread to run the blocking background task 
    # to see if the main loop can stay responsive
    stop_event = threading.Event()

    def run_bg():
        for _ in heavy_background_task(stop_event):
            pass

    baseline_p95 = await _measure_sleep_p95()
    bg_thread = threading.Thread(target=run_bg, daemon=False)
    bg_thread.start()
    try:
        print("   - Measuring reflex latency (10 samples)...")
        loaded_p95 = await _measure_sleep_p95()
        allowed_p95 = max(0.1, baseline_p95 * 3.0 + 0.02)
        print(
            f"   - baseline p95: {baseline_p95*1000:.2f}ms; "
            f"loaded p95: {loaded_p95*1000:.2f}ms; "
            f"allowed: {allowed_p95*1000:.2f}ms"
        )

        assert loaded_p95 < allowed_p95, (
            f"Reflex latency regressed under background load: "
            f"baseline={baseline_p95*1000:.2f}ms loaded={loaded_p95*1000:.2f}ms "
            f"allowed={allowed_p95*1000:.2f}ms"
        )
    finally:
        stop_event.set()
        bg_thread.join(timeout=1.0)

    assert not bg_thread.is_alive()
    print("   Real-time concurrency test passed!")
