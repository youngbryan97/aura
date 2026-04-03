import asyncio
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("Aura.GPU.Sentinel")

from enum import IntEnum

class GPUPriority(IntEnum):
    REFLECTION = 0  # Background thoughts, deep reasoning
    REFLEX = 1      # Voice, STT, Immediate reaction

class GPUSentinel:
    """A cross-thread sentinel for Apple Silicon GPU (Metal) access.
    
    Supports priority levels and pre-emption (high priority tasks signal 
    lower priority tasks to yield).
    """
    _instance: Optional["GPUSentinel"] = None
    _lock = threading.RLock()
    _preempt_flag = threading.Event() # Set when a REFLEX task is waiting
    
    # Lock Holder Tracking
    _holder_thread: Optional[threading.Thread] = None
    _holder_task: Optional[asyncio.Task] = None
    _lock_time_mono: float = 0.0

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GPUSentinel, cls).__new__(cls)
        return cls._instance

    def acquire(self, priority: GPUPriority = GPUPriority.REFLECTION, timeout: float = 60.0) -> bool:
        """Synchronous lock acquisition with priority awareness."""
        acquired = False
        if priority == GPUPriority.REFLEX:
            if not self._preempt_flag.is_set():
                logger.debug("GPU Sentinel: REFLEX priority pre-emption requested.")
            self._preempt_flag.set() # Warn lower priority tasks to finish ASAP
            
        # Death detection for previous holder
        if self._holder_thread is not None:
            if not self._holder_thread.is_alive():
                logger.warning("GPU Sentinel: Owner thread died without release. Forcing cleanup.")
                self.release()
            
        # Stale lock watchdog (150s)
        now = time.monotonic()
        if self._lock_time_mono > 0 and (now - self._lock_time_mono > 150):
            logger.critical("🚨 GPU Sentinel: LOCK STALE (>150s). Forcible override initiated.")
            self.release()

        try:
            acquired = self._lock.acquire(timeout=timeout)
            if acquired:
                self._holder_thread = threading.current_thread()
                try:
                    self._holder_task = asyncio.current_task()
                except RuntimeError:
                    self._holder_task = None
                self._lock_time_mono = time.monotonic()
                
                if priority == GPUPriority.REFLEX:
                    self._preempt_flag.clear()
            return acquired
        finally:
            if not acquired and priority == GPUPriority.REFLEX:
                self._preempt_flag.clear()

    def should_yield(self) -> bool:
        """Called by REFLECTION tasks periodically to see if they should yield for a REFLEX task."""
        return self._preempt_flag.is_set()

    def release(self):
        """Synchronous lock release."""
        try:
            self._lock.release()
        except RuntimeError:
            # Already released or not held by us?
            # In health monitoring cases, we might force release.
            pass
        finally:
            self._holder_thread = None
            self._holder_task = None
            self._lock_time_mono = 0.0

    async def acquire_async(self, priority: GPUPriority = GPUPriority.REFLECTION, timeout: float = 60.0) -> bool:
        """Asynchronous lock acquisition via to_thread."""
        try:
            acquired = await asyncio.to_thread(self.acquire, priority, timeout)
            return acquired
        except Exception as e:
            logger.error("GPU Sentinel: Error while acquiring lock: %s", e)
            return False

    async def release_async(self):
        """Asynchronous lock release."""
        self.release()

def get_gpu_sentinel() -> GPUSentinel:
    return GPUSentinel()
