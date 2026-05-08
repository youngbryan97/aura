import asyncio
import time
import logging
from contextlib import suppress
from typing import Optional

logger = logging.getLogger("Runtime.LoopGuard")


class LoopLagMonitor:
    """Cancellable event-loop lag monitor."""

    def __init__(self, threshold_s: float = 0.5, sample_interval_s: float = 0.5):
        if threshold_s <= 0:
            raise ValueError("threshold_s must be > 0")
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be > 0")
        self.threshold_s = float(threshold_s)
        self.sample_interval_s = float(sample_interval_s)
        self.last = time.perf_counter()
        self._stop_event: Optional[asyncio.Event] = None

    async def start(self, stop_event: Optional[asyncio.Event] = None):
        logger.info(
            "LoopLagMonitor active (threshold=%.3fs interval=%.3fs)",
            self.threshold_s,
            self.sample_interval_s,
        )
        self._stop_event = stop_event or asyncio.Event()
        self.last = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.sample_interval_s,
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                now = time.perf_counter()
                lag = now - self.last - self.sample_interval_s
                self.last = now
                if lag > self.threshold_s:
                    logger.warning("EVENT LOOP PRESSURE DETECTED: %.3fs", lag)
        except asyncio.CancelledError:
            logger.debug("LoopLagMonitor cancelled")
            raise

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

    async def run_for(self, duration_s: float) -> None:
        """Run the monitor for a bounded interval; useful for canaries/tests."""
        if duration_s < 0:
            raise ValueError("duration_s must be >= 0")
        stop_event = asyncio.Event()
        task = asyncio.create_task(self.start(stop_event), name="loop-lag-monitor")
        try:
            await asyncio.sleep(duration_s)
            stop_event.set()
            await task
        finally:
            stop_event.set()
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
