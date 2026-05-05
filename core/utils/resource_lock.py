"""core/utils/resource_lock.py — Global Resource Lock

Coordinates heavy resource usage (browser, GPU) with background tasks.
When Playwright is active, heavy metabolic tasks pause.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger("Aura.ResourceLock")


class ResourceLock:
    """Global resource coordination."""

    def __init__(self):
        self._loop = None
        self._browser_idle = None
        self._gpu_semaphore = None
        self._browser_sessions = 0
        self._total_browser_sessions = 0

    def _ensure_primitives(self):
        """Lazily create async primitives for the current event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        if self._loop is not loop:
            self._loop = loop
            self._browser_idle = asyncio.Event()
            self._browser_idle.set()
            self._gpu_semaphore = asyncio.Semaphore(1)

    @asynccontextmanager
    async def browser_session(self):
        """Context manager for browser operations.
        Clears the idle flag so heavy background tasks pause.
        """
        self._ensure_primitives()
        self._browser_sessions += 1
        self._total_browser_sessions += 1
        if self._browser_idle:
            self._browser_idle.clear()
        logger.debug("🌐 Browser session started (%d active)", self._browser_sessions)
        try:
            yield
        finally:
            self._browser_sessions -= 1
            if self._browser_sessions <= 0:
                self._browser_sessions = 0
                if self._browser_idle:
                    self._browser_idle.set()
                logger.debug("🌐 Browser session ended — background tasks resumed")

    async def wait_for_browser_idle(self, timeout: Optional[float] = 30.0):
        """Wait until no browser session is active.
        
        Call this before starting heavy metabolic tasks.
        Returns True if idle, False if timed out.
        """
        self._ensure_primitives()
        if not self._browser_idle or self._browser_idle.is_set():
            return True

        logger.debug("⏳ Waiting for browser to finish before heavy task...")
        try:
            await asyncio.wait_for(self._browser_idle.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning("Browser session still active after %ds — proceeding anyway", timeout)
            return False

    @asynccontextmanager
    async def gpu_session(self, owner: str = "unknown"):
        """Mutex for GPU-heavy operations (one at a time) with timeout."""
        self._ensure_primitives()
        if self._gpu_semaphore:
            try:
                # Use asyncio.timeout for Python 3.11+
                if hasattr(asyncio, "timeout"):
                    async with asyncio.timeout(30.0):
                        await self._gpu_semaphore.acquire()
                else:
                    await asyncio.wait_for(self._gpu_semaphore.acquire(), timeout=30.0)
            except (asyncio.TimeoutError, TimeoutError):
                logger.error("🚨 DEADLOCK DETECTED: Could not acquire GPU semaphore within 30s for %s", owner)
                raise RuntimeError(f"GPU semaphore deadlock for {owner}")
        
        logger.debug("gpu_session acquired by %s", owner)
        try:
            yield
        finally:
            if self._gpu_semaphore:
                self._gpu_semaphore.release()
                logger.debug("gpu_session released by %s", owner)

    @property
    def browser_active(self) -> bool:
        self._ensure_primitives()
        return not self._browser_idle.is_set() if self._browser_idle else False

    def get_stats(self) -> dict:
        self._ensure_primitives()
        return {
            "browser_active": self.browser_active,
            "active_browser_sessions": self._browser_sessions,
            "total_browser_sessions": self._total_browser_sessions,
        }


# Singleton
_lock: Optional[ResourceLock] = None

def get_resource_lock() -> ResourceLock:
    global _lock
    if _lock is None:
        _lock = ResourceLock()
    return _lock