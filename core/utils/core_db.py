import sqlite3
import json
import logging
from pathlib import Path
from core.memory import db_config

logger = logging.getLogger("Aura.CoreDB")

class AuraCoreDB:
    """Centralized SQLite management for core sovereign data (Values, Fallback Memories)."""
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AuraCoreDB, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        from core.config import config
        self.db_path = config.paths.data_dir / "sovereign_core.db"
        self._init_db()
        self._initialized = True

    def _init_db(self):
        """Ensure core tables exist."""
        conn = db_config.configure_connection(str(self.db_path))
        try:
            with conn:
                # Table for AlignmentEngine / Values
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS kv_store (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at REAL
                    )
                """)
                # Table for VectorMemory Fallback
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS vector_fallback (
                        id TEXT PRIMARY KEY,
                        collection TEXT,
                        content TEXT,
                        metadata TEXT,
                        timestamp REAL
                    )
                """)
                # Indexing for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_vfallback_coll ON vector_fallback(collection)")
            logger.info("✓ Aura Core DB Initialized at %s", self.db_path)
        except Exception as e:
            logger.error("Failed to initialize core DB: %s", e)
        finally:
            conn.close()

    def get_connection(self):
        """Returns a configured synchronous SQLite connection."""
        return db_config.configure_connection(str(self.db_path))

    async def get_connection_async(self):
        """Returns a configured asynchronous aiosqlite connection."""
        return await db_config.configure_connection_async(str(self.db_path))

# Singleton accessor
def get_core_db() -> AuraCoreDB:
    return AuraCoreDB()