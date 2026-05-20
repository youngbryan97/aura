import asyncio
import logging
import random
import sqlite3
from contextlib import asynccontextmanager

import aiosqlite

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
        self._semaphores: dict[str, asyncio.Semaphore] = {}
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
            async with aiosqlite.connect(path, timeout=30.0) as conn:
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute("PRAGMA synchronous=NORMAL;")
                await conn.execute("PRAGMA cache_size=-64000;")
                await conn.execute("PRAGMA temp_store=MEMORY;")
                await conn.execute("PRAGMA mmap_size=30000000000;")
                yield conn

    # Keep acquire/release for backward-compat callers that
    # don't yet use the context-manager API, but log a deprecation.
    async def acquire(self, path: str) -> aiosqlite.Connection:
        logger.warning(
            "ConnectionPool.acquire() is deprecated and not concurrency-safe. "
            "Use ConnectionPool.connection() context manager instead."
        )
        conn = await aiosqlite.connect(path, timeout=30.0)
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA synchronous=NORMAL;")
        await conn.execute("PRAGMA cache_size=-64000;")
        await conn.execute("PRAGMA temp_store=MEMORY;")
        await conn.execute("PRAGMA mmap_size=30000000000;")
        return conn

    async def release(self, conn: object | None = None) -> None:
        """Close a legacy acquired connection when one is supplied."""
        close = getattr(conn, "close", None)
        if callable(close):
            await close()
        elif conn is not None:
            logger.debug("ConnectionPool.release() received non-connection legacy token: %r", conn)


async def execute_write_with_backoff(conn: aiosqlite.Connection | sqlite3.Connection, query: str, params: tuple = (), max_retries: int = 5):
    """Executes atomic state changes utilizing a randomized exponential backoff loop to resolve race contentions."""
    base_delay = 0.05
    for attempt in range(max_retries):
        try:
            if hasattr(conn, "commit") and not hasattr(conn, "execute_insert"):
                # Standard synchronous sqlite3.Connection
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    return cursor.lastrowid
            else:
                # aiosqlite Connection
                async with conn.execute(query, params) as cursor:
                    await conn.commit()
                    return cursor.lastrowid
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 0.05)
                logger.warning(f"⚠️ DB locked conflict. Attempt {attempt+1}/{max_retries}. Sleeping {sleep_time:.3f}s")
                await asyncio.sleep(sleep_time)
            else:
                raise e


pool = ConnectionPool()
