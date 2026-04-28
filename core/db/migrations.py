"""
Lightweight schema migration system for Aura's SQLite databases.
No external dependency — pure Python + sqlite3.

Usage:
    migrator = Migrator("~/.aura/data/knowledge.db")
    migrator.register(1, "Add confidence column", sql_v1)
    migrator.register(2, "Add metadata index", sql_v2)
    migrator.run()   # Applies only unapplied migrations in order
"""
from core.runtime.errors import record_degradation
import hashlib
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Union

logger = logging.getLogger("Aura.Migrations")


class Migration(NamedTuple):
    version: int
    description: str
    up_sql: str                          # SQL to apply
    checksum: str = ""                   # Auto-computed from up_sql


class Migrator:

    _SCHEMA_TABLE = """
    CREATE TABLE IF NOT EXISTS _aura_migrations (
        version     INTEGER PRIMARY KEY,
        description TEXT NOT NULL,
        checksum    TEXT NOT NULL,
        applied_at  REAL NOT NULL
    );
    """

    def __init__(self, db_path: Union[str, Path]):
        path_str = str(db_path)
        if path_str.startswith("~/.aura/data/"):
            data_dir = os.environ.get("AURA_DATA_DIR")
            if data_dir:
                path_str = path_str.replace("~/.aura/data", data_dir, 1)
            else:
                try:
                    from core.config import config
                    suffix = path_str[len("~/.aura/data/"):]
                    path_str = str(config.paths.data_dir / suffix)
                except Exception as _exc:
                    record_degradation('migrations', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)
        elif path_str.startswith("~/.aura"):
            aura_root = os.environ.get("AURA_ROOT")
            if aura_root:
                path_str = path_str.replace("~/.aura", aura_root, 1)
            else:
                try:
                    from core.config import config
                    suffix = path_str[len("~/.aura/"):]
                    path_str = str(config.paths._effective_home_dir() / suffix)
                except Exception as _exc:
                    record_degradation('migrations', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)
        
        self.db_path = str(Path(path_str).expanduser())
        self._migrations: List[Migration] = []

    def register(self, version: int, description: str, up_sql: str):
        checksum = hashlib.sha256(up_sql.encode()).hexdigest()[:16]
        self._migrations.append(Migration(version, description, up_sql, checksum))
        self._migrations.sort(key=lambda m: m.version)

    def run_all(self) -> int:
        """Alias for run() to match standard maintenance interface."""
        self.reconcile_legacy_schema()
        return self.run()

    def run(self) -> int:
        """Apply all unapplied migrations. Returns count of migrations applied."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")  # WAL for concurrent reads
        con.execute("PRAGMA foreign_keys=ON")
        applied = 0

        try:
            con.execute(self._SCHEMA_TABLE)
            con.commit()

            applied_versions = {
                row[0] for row in con.execute("SELECT version FROM _aura_migrations")
            }

            for migration in self._migrations:
                if migration.version in applied_versions:
                    continue

                logger.info(
                    "Applying migration v%d: %s", migration.version, migration.description
                )
                try:
                    # [BOOT FIX] Use executescript to correctly handle triggers and multi-line SQL.
                    # This avoids the "incomplete input" error caused by naive semicolon splitting.
                    con.executescript(migration.up_sql)
                    
                    con.execute(
                        "INSERT INTO _aura_migrations (version, description, checksum, applied_at) "
                        "VALUES (?, ?, ?, ?)",
                        (migration.version, migration.description,
                         migration.checksum, time.time()),
                    )
                    con.commit()
                    applied += 1
                    logger.info("✅ Migration v%d applied.", migration.version)
                except Exception as e:
                    record_degradation('migrations', e)
                    con.rollback()
                    logger.error(
                        "❌ Migration v%d FAILED: %s — database unchanged.", migration.version, e
                    )
                    raise RuntimeError(
                        f"Migration v{migration.version} ('{migration.description}') failed: {e}"
                    ) from e

        finally:
            con.close()

        if applied == 0:
            logger.debug("All migrations already applied.")
        return applied

        if applied == 0:
            logger.debug("All migrations already applied.")
        return applied

    def reconcile_legacy_schema(self):
        """Detect and fix legacy schema issues (ISSUE 4)."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        try: 
            cursor = con.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge'")
            if cursor.fetchone():
                cursor.execute("PRAGMA table_info(knowledge)")
                columns = {row[1] for row in cursor.fetchall()}
                if "learned_at" in columns and "created_at" not in columns:
                    logger.warning("⚠️ Legacy 'learned_at' column detected. Reconciling...")
                    try:
                        con.execute("ALTER TABLE knowledge RENAME COLUMN learned_at TO created_at")
                        con.commit()  # Commit immediately after each DDL
                        logger.info("✅ Renamed 'learned_at' to 'created_at'.")
                    except sqlite3.OperationalError:
                        logger.warning("ALTER RENAME failed, trying manual migration...")
                        con.execute("BEGIN")
                        con.execute("ALTER TABLE knowledge RENAME TO knowledge_old")
                        con.execute("""
                            CREATE TABLE knowledge (
                                id TEXT PRIMARY KEY, content TEXT, type TEXT, source TEXT,
                                confidence REAL, created_at REAL, updated_at REAL, metadata TEXT
                            )
                        """)
                        con.execute("""
                            INSERT INTO knowledge
                            SELECT id, content, type, source, confidence,
                                   learned_at, learned_at, metadata
                            FROM knowledge_old
                        """)
                        con.execute("DROP TABLE knowledge_old")
                        con.commit()
                        logger.info("✅ Manual reconciliation complete.")
                # Re-fetch columns after potential rename
                cursor.execute("PRAGMA table_info(knowledge)")
                columns = {row[1] for row in cursor.fetchall()}
                if "updated_at" not in columns:
                    con.execute("ALTER TABLE knowledge ADD COLUMN updated_at REAL DEFAULT 0.0")
                    con.execute("UPDATE knowledge SET updated_at = created_at WHERE updated_at = 0.0")
                    con.commit()
                    logger.info("✅ Added 'updated_at' column.")
        except Exception as e:
            record_degradation('migrations', e)
            logger.error("Failed to reconcile legacy schema: %s", e)
            con.rollback()
        finally:
            con.close()


def _build_knowledge_migrator(db_path: Optional[Union[str, Path]] = None) -> Migrator:
    """Build and configure the knowledge graph migrator (ISSUE 5)."""
    m = Migrator(db_path or "~/.aura/data/knowledge.db")
    m.register(1, "Initial schema", _SQL_V1)
    m.register(2, "Add full-text search", _SQL_V2)
    m.register(3, "Add skills and goals tables", _SQL_V3)
    m.register(4, "Add execution audit log", _SQL_V4)
    return m

def get_migrator(db_path: Optional[Union[str, Path]] = None) -> Migrator:
    """Convenience factory for the knowledge migrator."""
    return _build_knowledge_migrator(db_path)

# ── Knowledge Graph Migrations ────────────────────────────────────────────────

_SQL_V1 = """
-- Support legacy schema reconciliation
-- Rename 'learned_at' to 'created_at' if it exists (SQLite 3.25.0+)
PRAGMA foreign_keys=OFF;

-- 1. Create the target table if it doesn't exist (clean slate)
CREATE TABLE IF NOT EXISTS knowledge (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    type        TEXT NOT NULL DEFAULT 'fact',
    source      TEXT,
    confidence  REAL DEFAULT 0.8,
    metadata    TEXT DEFAULT '{}',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

-- 2. If it's a legacy table, it might have learned_at.
-- We'll try to add the new columns and potentially copy data.
-- Since this script runs once for v1, we'll use a safer approach for legacy users.
-- If 'learned_at' column exists in an existing 'knowledge' table:
-- (This is handled gracefully by SQL logic or subsequent migrations if needed)

-- Ensure indexes
CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge(type);
CREATE INDEX IF NOT EXISTS idx_knowledge_created ON knowledge(created_at DESC);
"""

_SQL_V2 = """
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    id UNINDEXED,
    content,
    tokenize = 'porter ascii'
);
CREATE TRIGGER IF NOT EXISTS knowledge_fts_insert
AFTER INSERT ON knowledge BEGIN
    INSERT INTO knowledge_fts(id, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS knowledge_fts_update
AFTER UPDATE ON knowledge BEGIN
    UPDATE knowledge_fts SET content = new.content WHERE id = new.id;
END;
CREATE TRIGGER IF NOT EXISTS knowledge_fts_delete
AFTER DELETE ON knowledge BEGIN
    DELETE FROM knowledge_fts WHERE id = old.id;
END;
"""

_SQL_V3 = """
CREATE TABLE IF NOT EXISTS skills (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_used   REAL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS goals (
    id          TEXT PRIMARY KEY,
    objective   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    priority    INTEGER DEFAULT 5,
    created_at  REAL NOT NULL,
    completed_at REAL
);

CREATE TABLE IF NOT EXISTS people (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    notes       TEXT,
    last_seen   REAL,
    interaction_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_goals_priority ON goals(priority DESC);
"""

_SQL_V4 = """
CREATE TABLE IF NOT EXISTS execution_log (
    id          TEXT PRIMARY KEY,
    skill_name  TEXT NOT NULL,
    status      TEXT NOT NULL,
    duration_ms REAL,
    error       TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_execlog_skill ON execution_log(skill_name);
CREATE INDEX IF NOT EXISTS idx_execlog_created ON execution_log(created_at DESC);
-- Auto-prune: keep only last 10000 entries
CREATE TRIGGER IF NOT EXISTS prune_execution_log
AFTER INSERT ON execution_log
WHEN (SELECT COUNT(*) FROM execution_log) > 10000
BEGIN
    DELETE FROM execution_log WHERE id IN (
        SELECT id FROM execution_log ORDER BY created_at ASC LIMIT 100
    );
END;
"""
