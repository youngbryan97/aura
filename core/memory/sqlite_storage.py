"""aiosqlite-backed Atomic Memory System.

Replaces the synchronous sqlite3 SQLiteMemory for scalability and non-blocking I/O.
Maintains the same API via threadsafe synchronous wrappers, but deeply uses async natively.

C-10 FIX: Implemented persistent aiosqlite connection to remove file open/close overhead.
H-11 FIX: Consistent log formatting using %s placeholders.
H-04 FIX: Thread-safe synchronous wrappers using threading.Lock.
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import sys
import aiosqlite
from core.container import ServiceContainer
from core.health.degraded_events import record_degraded_event
from core.memory import db_config
from core.memory.base import MemoryEvent

logger = logging.getLogger("Kernel.Memory.SQLite")

class SQLiteMemory:
    """Scalable, non-blocking memory storage using aiosqlite.
    Drop-in replacement for AtomicStorage/sync SQLiteMemory.
    
    C-10 FIX: Uses a single persistent aiosqlite connection.
    H-04 FIX: Synchronous wrappers are protected by a threading.Lock.
    """
    
    def __init__(self, storage_file: str = "autonomy_engine/memory/atomic.db"):
        if sys.version_info < (3, 10):
            raise RuntimeError("Aura Memory Subsystem (SQLiteMemory) requires Python 3.10+")
        self.storage_file = Path(storage_file)
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._init_lock_obj: Optional[asyncio.Lock] = None
        self._conn: Optional[aiosqlite.Connection] = None
        self._sync_lock = threading.Lock()  # H-04 FIX: Thread-safe sync wrappers

    @property
    def _init_lock(self) -> asyncio.Lock:
        if self._init_lock_obj is None:
            self._init_lock_obj = asyncio.Lock()
        return self._init_lock_obj
        
    async def _get_conn(self) -> aiosqlite.Connection:
        """Returns the persistent connection, initializing it if necessary."""
        if self._conn is not None:
            return self._conn
            
        async with self._init_lock:
            if self._conn is not None:
                return self._conn
            
            # C-10 FIX: Open persistent connection
            self._conn = await db_config.configure_connection_async(str(self.storage_file))
            await self._conn.execute("PRAGMA journal_mode=WAL;")
            await self._conn.execute("PRAGMA synchronous=NORMAL;")
            await self._conn.execute("PRAGMA cache_size=-64000;")
            await self._ensure_schema()
            self._initialized = True
            logger.info("aiosqlite Memory initialized at %s", self.storage_file)
            return self._conn

    async def _ensure_schema(self):
        """Initialize the database schema."""
        if self._conn is None:
            return
            
        # episodic: Stores chronological events
        await self._conn.execute('''
            CREATE TABLE IF NOT EXISTS episodic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                event_type TEXT,
                goal TEXT,
                outcome TEXT,
                cost REAL,
                metadata TEXT
            )
        ''')
    
        # semantic: Key-value store for facts
        await self._conn.execute('''
            CREATE TABLE IF NOT EXISTS semantic (
                key TEXT PRIMARY KEY,
                value TEXT,
                last_modified REAL
            )
        ''')
        
        # goals: Active goals
        await self._conn.execute('''
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal TEXT,
                status TEXT,
                created_at REAL
            )
        ''')
        
        # Indices for speed
        await self._conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON episodic(timestamp)')
        await self._conn.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON episodic(event_type)')
        
        await self._conn.commit()

    async def on_stop_async(self):
        """Close the persistent connection."""
        async with self._init_lock:
            if self._conn:
                await self._conn.close()
                self._conn = None
                self._initialized = False
                logger.debug("SQLite connection closed")

    def _constitutional_runtime_live(self) -> bool:
        return (
            ServiceContainer.has("executive_core")
            or ServiceContainer.has("aura_kernel")
            or ServiceContainer.has("kernel_interface")
            or bool(getattr(ServiceContainer, "_registration_locked", False))
        )

    async def _approve_memory_write(
        self,
        memory_type: str,
        content: str,
        *,
        importance: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            from core.constitution import get_constitutional_core

            approved, reason = await get_constitutional_core().approve_memory_write(
                memory_type,
                str(content or "")[:240],
                source="sqlite_memory",
                importance=max(0.0, min(1.0, float(importance or 0.0))),
                metadata=dict(metadata or {}),
            )
            if not approved:
                record_degraded_event(
                    "sqlite_memory",
                    "memory_write_blocked",
                    detail=str(content or "")[:160],
                    severity="warning",
                    classification="background_degraded",
                    context={"reason": reason},
                )
            return approved
        except Exception as exc:
            if self._constitutional_runtime_live():
                record_degraded_event(
                    "sqlite_memory",
                    "memory_write_gate_failed",
                    detail=str(content or "")[:160],
                    severity="warning",
                    classification="background_degraded",
                    context={"error": type(exc).__name__},
                    exc=exc,
                )
                return False
            logger.debug("SQLiteMemory constitutional gate unavailable: %s", exc)
            return True

    def close(self):
        """Synchronous connection close."""
        with self._sync_lock:
            try:
                loop = asyncio.get_running_loop()
                asyncio.run_coroutine_threadsafe(self.on_stop_async(), loop).result()
            except RuntimeError:
                asyncio.run(self.on_stop_async())

    # --- Async Native Methods ---

    async def log_event_async(self, event: Union[Dict[str, Any], MemoryEvent]) -> bool:
        """Log an episodic event asynchronously."""
        try:
            conn = await self._get_conn()
            if isinstance(event, dict):
                # Ensure defaults
                if 'timestamp' not in event: event['timestamp'] = time.time()
                if 'metadata' not in event: event['metadata'] = {}
            else:
                event = event.to_dict()

            await conn.execute('''
                INSERT INTO episodic (timestamp, event_type, goal, outcome, cost, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                event.get('timestamp'),
                event.get('event_type'),
                event.get('goal'),
                json.dumps(event.get('outcome')),
                event.get('cost', 0.0),
                json.dumps(event.get('metadata', {}))
            ))
            await conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to log event asynchronously: %s", e)
            return False

    async def commit_interaction(self, context: str, action: str, outcome: str, success: bool, emotional_valence: float = 0.0, importance: float = 0.5):
        """Unified commit for an interaction (compatibility with MemoryFacade)."""
        return await self.record_episode_async(
            context=context,
            action=action,
            outcome=outcome,
            success=success,
            emotional_valence=emotional_valence,
            importance=importance
        )

    async def get_recent_events_async(self, count: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent events asynchronously."""
        try:
            conn = await self._get_conn()
            conn.row_factory = aiosqlite.Row
            async with conn.execute('''
                SELECT * FROM episodic 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (count,)) as cursor:
                rows = await cursor.fetchall()
                
            events = []
            for row in rows:
                evt = dict(row)
                # Parse JSON fields
                try:
                    evt['outcome'] = json.loads(evt['outcome']) if evt['outcome'] else None
                    evt['metadata'] = json.loads(evt['metadata']) if evt['metadata'] else {}
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug("Malformed JSON in event row %s: %s", evt.get('id', '?'), e)
                events.append(evt)
            
            return list(reversed(events)) # Return chronological order
        except Exception as e:
            logger.error("Failed to get events asynchronously: %s", e)
            return []

    async def update_semantic_async(self, key: str, value: Any) -> bool:
        """Update a semantic memory asynchronously."""
        try:
            conn = await self._get_conn()
            await conn.execute('''
                INSERT OR REPLACE INTO semantic (key, value, last_modified)
                VALUES (?, ?, ?)
            ''', (key, json.dumps(value), time.time()))
            await conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to update semantic asynchronously: %s", e)
            return False

    async def get_semantic_async(self, key: str, default: Any = None) -> Any:
        """Get a semantic memory asynchronously."""
        try:
            conn = await self._get_conn()
            async with conn.execute('SELECT value FROM semantic WHERE key = ?', (key,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return default
        except Exception as e:
            logger.error("Failed to get semantic asynchronously: %s", e)
            return default

    async def add(self, content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """Direct add for compatibility with Vector/Semantic APIs."""
        if not await self._approve_memory_write(
            "sqlite_observation",
            content,
            importance=float(kwargs.get("importance", 0.5) or 0.5),
            metadata=metadata,
        ):
            return False
        event = {
            "event_type": "observation",
            "outcome": content,
            "metadata": metadata or {}
        }
        return await self.log_event_async(event)

    async def record_episode_async(self, context: str, action: str, outcome: str, success: bool, emotional_valence: float, importance: float) -> int:
        """Record a structured episode and return its ID."""
        if not await self._approve_memory_write(
            "sqlite_episode",
            f"{context} | {action} | {outcome}",
            importance=importance,
            metadata={"success": success, "valence": emotional_valence},
        ):
            return 0
        try:
            conn = await self._get_conn()
            await conn.execute('''
                INSERT INTO episodic (timestamp, event_type, goal, outcome, cost, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                action,
                context,
                json.dumps(outcome),
                0.0,
                json.dumps({
                    "success": success,
                    "valence": emotional_valence,
                    "importance": importance
                })
            ))
            await conn.commit()
            async with conn.execute("SELECT last_insert_rowid()") as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0
        except Exception as e:
            logger.error("Failed to record episode: %s", e)
            return 0

    async def recall_recent_async(self, limit: int = 5) -> List[Any]:
        """Fetch recent events as objects for Facade compatibility."""
        rows = await self.get_recent_events_async(count=limit)
        # Use a simple dynamic object to satisfy Facade expectations
        class MemObj:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        
        events = []
        for r in rows:
            events.append(MemObj(
                timestamp=r.get('timestamp'),
                context=r.get('goal'),
                action=r.get('event_type'),
                outcome=r.get('outcome'),
                success=r.get('metadata', {}).get('success', True)
            ))
        return events

    async def get_hot_memory(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent episodic context."""
        return await self.get_recent_events_async(count=limit)

    async def prune_low_salience(self, threshold_days: int = 30, min_salience: float = -0.2) -> int:
        """Strategic Forgetting: Removes old, low-salience memories."""
        try:
            conn = await self._get_conn()
            now = time.time()
            threshold_s = threshold_days * 86400
            
            # Use json_extract to filter by valence stored in the metadata JSON blob
            # Note: This requires SQLite with JSON1 extension (standard in Python 3.9+)
            query = """
                DELETE FROM episodic 
                WHERE (timestamp < ?) 
                AND (CAST(json_extract(metadata, '$.valence') AS REAL) < ?)
            """
            cursor = await conn.execute(query, (now - threshold_s, min_salience))
            count = cursor.rowcount
            await conn.commit()
            logger.info("Pruned %d low-salience memories from episodic store.", count)
            return count
        except Exception as e:
            logger.error("Failed to prune low salience memory: %s", e)
            return 0

    async def clear_episodic_async(self) -> bool:
        """Clear episodic memory asynchronously."""
        conn = await self._get_conn()
        await conn.execute('DELETE FROM episodic')
        await conn.commit()
        return True

    def save(self) -> bool:
        """No-op for SQLite as it sends to disk immediately."""
        return True

    # --- Legacy Synchronous Wrappers ---
    def _run_sync(self, coro):
        with self._sync_lock: # H-04 FIX: Thread safety for sync wrappers
            try:
                try:
                    loop = self.loop or asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    import threading
                    # If we are in the same thread as the event loop, we CANNOT call .result()
                    # as it will deadlock immediately.
                    if threading.current_thread() is threading.main_thread() or (hasattr(loop, '_thread_id') and threading.get_ident() == loop._thread_id):
                        logger.warning("⚠️ RECURSIVE SYNC CALL: Memory operation attempted sync from within event loop. This WILL deadlock. Returning None.")
                        return None
                    
                    return asyncio.run_coroutine_threadsafe(coro, loop).result()
                else:
                    return asyncio.run(coro)
            except Exception as e:
                logger.error("Error in _run_sync: %s", e)
                return None

    def log_event(self, event: Union[Dict[str, Any], MemoryEvent]) -> bool:
        return self._run_sync(self.log_event_async(event))

    def commit_interaction_sync(self, context: str, action: str, outcome: str, success: bool, emotional_valence: float = 0.0, importance: float = 0.5):
        return self._run_sync(self.commit_interaction(context, action, outcome, success, emotional_valence, importance))

    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        return self._run_sync(self.get_recent_events_async(count))

    def update_semantic(self, key: str, value: Any) -> bool:
        return self._run_sync(self.update_semantic_async(key, value))

    def get_semantic(self, key: str, default: Any = None) -> Any:
        return self._run_sync(self.get_semantic_async(key, default))

    def clear_episodic(self) -> bool:
        return self._run_sync(self.clear_episodic_async())
