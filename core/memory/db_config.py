"""Database Configuration & Optimization
-------------------------------------
Centralizes SQLite configuration to ensure WAL (Write-Ahead Logging) mode is enabled.
WAL mode significantly improves concurrency, allowing readers to not block writers.
"""

from core.runtime.errors import record_degradation
import logging
import sqlite3
from pathlib import Path
from core.db.pool import pool as aio_pool

import threading

logger = logging.getLogger("Aura.DBConfig")

# ISSUE 26 fix: Thread-local connection cache
_thread_local = threading.local()

def configure_connection(db_path: str) -> sqlite3.Connection:
    """Creates a connection to the SQLite DB and enables WAL mode (synchronous)."""
    # ISSUE 26 fix: Cache check
    if not hasattr(_thread_local, "connections"):
        _thread_local.connections = {}
    
    if db_path in _thread_local.connections:
        return _thread_local.connections[db_path]
        
    path = Path(db_path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        
    conn = sqlite3.connect(str(path), check_same_thread=False)
    
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA cache_size=-8000;")
        conn.execute("PRAGMA wal_autocheckpoint=1000;")
        conn.commit()
        _thread_local.connections[db_path] = conn
    except Exception as e:
        record_degradation('db_config', e)
        logger.warning("Failed to set PRAGMA options on %s: %s", db_path, e)
        
    return conn

async def configure_connection_async(db_path: str):
    """Creates an aiosqlite connection and enables WAL mode.
    Now routes through the centralized core.db.pool to prevent connection churn.
    """
    path = Path(db_path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        
    conn = await aio_pool.acquire(str(path))
    
    try:
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA synchronous=NORMAL;")
        await conn.execute("PRAGMA busy_timeout=30000;")  # 30s — prevents lock errors under heavy metabolic load
        await conn.execute("PRAGMA cache_size=-8000;")     # 8MB cache — reduces I/O pressure
        await conn.execute("PRAGMA wal_autocheckpoint=100;")  # Limit WAL file growth under sustained writes
        await conn.commit()
    except Exception as e:
        record_degradation('db_config', e)
        logger.warning("Failed to set async PRAGMA options on %s: %s", db_path, e)
        
    return conn