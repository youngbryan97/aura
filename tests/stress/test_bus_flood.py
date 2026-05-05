import asyncio
import logging
import pytest
import time
from core.bus.actor_bus import ActorBus

logger = logging.getLogger("Aura.TestBusFlood")

@pytest.mark.asyncio
async def test_actor_bus_flood_handling():
    """Stress test the ActorBus by flooding it with messages beyond its queue limits."""
    bus = ActorBus()
    bus.start()
    
    # We will spam it with 10000 messages.
    # The telemetry queue maxsize is 100, so it will drop messages using put_nowait/get_nowait.
    
    start = time.perf_counter()
    tasks = []
    for i in range(10000):
        tasks.append(bus.broadcast_telemetry("flood_test", {"count": i}))
        
    await asyncio.gather(*tasks, return_exceptions=True)
    end = time.perf_counter()
    
    await bus.stop()
    
    # Just need to make sure the bus didn't crash and we survived.
    assert end - start < 5.0, "Flood took too long, possible deadlock"
    logger.info(f"Bus flooded successfully in {end - start:.3f}s")
