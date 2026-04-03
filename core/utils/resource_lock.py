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
        # Browser lock: SET = idle (tasks can run), CLEARED = browser active (tasks wait)
        self._browser_idle = asyncio.Event()
        self._browser_idle.set()  # Start as idle

        # GPU lock: for heavy compute operations
        self._gpu_semaphore = asyncio.Semaphore(1)

        self._browser_sessions = 0
        self._total_browser_sessions = 0

    @asynccontextmanager
    async def browser_session(self):
        """Context manager for browser operations.
        Clears the idle flag so heavy background tasks pause.
        """
        self._browser_sessions += 1
        self._total_browser_sessions += 1
        self._browser_idle.clear()
        logger.debug("🌐 Browser session started (%d active)", self._browser_sessions)
        try:
            yield
        finally:
            self._browser_sessions -= 1
            if self._browser_sessions <= 0:
                self._browser_sessions = 0
                self._browser_idle.set()
                logger.debug("🌐 Browser session ended — background tasks resumed")

    async def wait_for_browser_idle(self, timeout: Optional[float] = 30.0):
        """Wait until no browser session is active.
        
        Call this before starting heavy metabolic tasks.
        Returns True if idle, False if timed out.
        """
        if self._browser_idle.is_set():
            return True

        logger.debug("⏳ Waiting for browser to finish before heavy task...")
        try:
            await asyncio.wait_for(self._browser_idle.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning("Browser session still active after %ds — proceeding anyway", timeout)
            return False

    @asynccontextmanager
    async def gpu_session(self):
        """Mutex for GPU-heavy operations (one at a time)."""
        await self._gpu_semaphore.acquire()
        try:
            yield
        finally:
            self._gpu_semaphore.release()

    @property
    def browser_active(self) -> bool:
        return not self._browser_idle.is_set()

    def get_stats(self) -> dict:
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