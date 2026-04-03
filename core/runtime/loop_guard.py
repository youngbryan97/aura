import asyncio
import time
import logging

logger = logging.getLogger("Runtime.LoopGuard")

class LoopLagMonitor:
    def __init__(self, threshold_s: float = 0.5):
        self.threshold_s = threshold_s
        self.last = time.perf_counter()

    async def start(self):
        logger.info("⏱️ LoopLagMonitor active (Threshold: %.1fs)", self.threshold_s)
        while True:
            await asyncio.sleep(0.5)
            now = time.perf_counter()
            lag = now - self.last - 0.5
            self.last = now
            if lag > self.threshold_s:
                logger.warning("🐌 EVENT LOOP PRESSURE DETECTED: %.3fs", lag)
