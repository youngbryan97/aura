from core.runtime.errors import record_degradation
import asyncio
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

logger = logging.getLogger("Aura.DatabaseCoordinator")

class DatabaseCoordinator:
    """Centralized Async Database Writer.
    
    Prevents SQLite concurrency locks by funneling all write operations through 
    a single managed queue.
    """
    
    def __init__(self):
        self._write_queue = asyncio.Queue()
        self._worker_task = None
        self._connections: Dict[str, sqlite3.Connection] = {}
        self._running = False
        logger.info("🗄️ DatabaseCoordinator initialized.")

    async def start(self):
        """Start the background write worker."""
        if self._running:
            return
        self._running = True
        try:
            from core.utils.task_tracker import get_task_tracker

            self._worker_task = get_task_tracker().create_task(
                self._process_writes(),
                name="aura.database_coordinator",
            )
        except Exception:
            self._worker_task = get_task_tracker().create_task(self._process_writes(), name="aura.database_coordinator")
        logger.info("🗄️ DatabaseCoordinator worker started.")

    async def stop(self):
        """Stop the background worker gracefully."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in database_coordinator.py: %s', _e)
        
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()
        logger.info("🗄️ DatabaseCoordinator stopped.")

    async def execute_write(self, db_path: str, query: str, params: Tuple = ()) -> asyncio.Future:
        """Enqueue a write operation and return a future for the result."""
        result_future = asyncio.get_running_loop().create_future()
        await self._write_queue.put((db_path, query, params, result_future))
        return result_future

    async def _process_writes(self):
        """Background loop processing write operations sequentially."""
        while self._running:
            try:
                db_path, query, params, future = await self._write_queue.get()
                
                try:
                    conn = self._get_connection(db_path)
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    conn.commit()
                    
                    if not future.done():
                        future.set_result(cursor.lastrowid or True)
                except Exception as e:
                    record_degradation('database_coordinator', e)
                    logger.error("Database write error on %s: %s", db_path, e)
                    if not future.done():
                        future.set_exception(e)
                finally:
                    self._write_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('database_coordinator', e)
                logger.error("DatabaseCoordinator loop error: %s", e)
                await asyncio.sleep(1)

    def _get_connection(self, db_path: str) -> sqlite3.Connection:
        """Get or create a cached database connection."""
        if db_path not in self._connections:
            # Ensure parent directories exist
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._connections[db_path] = sqlite3.connect(db_path, check_same_thread=False)
            # Optimize for high-speed writes
            self._connections[db_path].execute("PRAGMA journal_mode=WAL")
            self._connections[db_path].execute("PRAGMA synchronous=NORMAL")
            
        return self._connections[db_path]

    def checkpoint_wal(self):
        """Checkpoint all WAL files to prevent unbounded WAL growth."""
        logger.info("🗄️ [DB] Starting WAL checkpoint on all active databases...")
        for path, conn in self._connections.items():
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                logger.info("✓ [DB] WAL checkpoint: %s", path)
            except Exception as e:
                record_degradation('database_coordinator', e)
                logger.error("❌ [DB] WAL checkpoint failed for %s: %s", path, e)

    def vacuum_all_databases(self):
        """Perform VACUUM on all active connections to prevent file bloat."""
        logger.info("🧹 [DB] Starting VACUUM on all active databases...")
        for path, conn in self._connections.items():
            try:
                conn.execute("VACUUM")
                logger.info("✓ [DB] Vacuumed: %s", path)
            except Exception as e:
                record_degradation('database_coordinator', e)
                logger.error("❌ [DB] Vacuum failed for %s: %s", path, e)

# Global singleton
_coordinator = DatabaseCoordinator()

def get_db_coordinator():
    return _coordinator
