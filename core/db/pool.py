import aiosqlite
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict

logger = logging.getLogger("Aura.DBPool")

class ConnectionPool:
    """
    Per-caller aiosqlite connections.  SQLite with WAL supports many concurrent
    readers; we serialise writes through db_writer_queue.  Each acquire() opens
    a fresh connection that the caller owns for the duration of its operation,
    then closes it.  For high-throughput scenarios a bounded semaphore limits
    the number of simultaneous connections per path.
    """
    _MAX_CONNS_PER_PATH = 5

    def __init__(self):
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._sem_lock = asyncio.Lock()

    async def _get_semaphore(self, path: str) -> asyncio.Semaphore:
        async with self._sem_lock:
            if path not in self._semaphores:
                self._semaphores[path] = asyncio.Semaphore(self._MAX_CONNS_PER_PATH)
            return self._semaphores[path]

    @asynccontextmanager
    async def connection(self, path: str):
        """Async context manager: yields a fresh, configured aiosqlite connection."""
        sem = await self._get_semaphore(path)
        async with sem:
            async with aiosqlite.connect(path) as conn:
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA busy_timeout=5000")
                yield conn

    # Keep acquire/release for backward-compat callers that
    # don't yet use the context-manager API, but log a deprecation.
    async def acquire(self, path: str) -> aiosqlite.Connection:
        logger.warning(
            "ConnectionPool.acquire() is deprecated and not concurrency-safe. "
            "Use ConnectionPool.connection() context manager instead."
        )
        conn = await aiosqlite.connect(path)
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        return conn

    async def release(self, path: str):
        pass  # No-op: connections are now per-caller

pool = ConnectionPool()
