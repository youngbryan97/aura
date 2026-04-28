"""Aura Zenith Atomic State Manager.
Uses aiosqlite for thread-safe, resilient state persistence.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import aiosqlite

from core.config import config
from core.memory import db_config

logger = logging.getLogger("Aura.AtomicState")

class AtomicStateManager:
    """Manages system state with ACID guarantees using SQLite.
    """
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (Path(config.paths.data_dir) / "atomic_state.db")
        self._db: Optional[aiosqlite.Connection] = None

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await db_config.configure_connection_async(str(self.db_path))
        return self._db

    async def initialize(self):
        """Create tables if they don't exist."""
        get_task_tracker().create_task(get_storage_gateway().create_dir(self.db_path.parent, cause='AtomicStateManager.initialize'))
        db = await self._get_conn()
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
        logger.info(f"📁 Atomic State DB initialized at {self.db_path}")

    async def get(self, key: str, default: Any = None) -> Any:
        """Get state value by key."""
        db = await self._get_conn()
        async with db.execute("SELECT value FROM system_state WHERE key = ?", (key,)) as cursor:
            row = await cursor.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    return row[0]
        return default

    async def set(self, key: str, value: Any):
        """Set state value by key atomically."""
        val_str = json.dumps(value)
        db = await self._get_conn()
        await db.execute("""
            INSERT OR REPLACE INTO system_state (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, val_str))
        await db.commit()

    async def delete(self, key: str):
        """Delete key from state."""
        db = await self._get_conn()
        await db.execute("DELETE FROM system_state WHERE key = ?", (key,))
        await db.commit()

    async def get_all(self) -> Dict[str, Any]:
        """Retrieve the entire state dictionary."""
        state = {}
        db = await self._get_conn()
        async with db.execute("SELECT key, value FROM system_state") as cursor:
            async for key, value in cursor:
                try:
                    state[key] = json.loads(value)
                except json.JSONDecodeError:
                    state[key] = value
        return state
