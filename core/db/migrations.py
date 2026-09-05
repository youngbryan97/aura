"""Schema migrations for Aura's SQLite databases.

No external dependency — pure Python and sqlite3.

    migrator = Migrator("~/.aura/data/knowledge.db")
    migrator.register(1, "Add confidence column", sql_v1)
    migrator.register(2, "Add metadata index", sql_v2)
    migrator.run()   # Applies only unapplied migrations in order

Two things this recorded and did not check, until now.

**The checksum was write-only.** Every applied migration stored a hash of
its SQL, and ``run()`` then built a set of applied version numbers and
skipped those. So editing migration 3 after it had run anywhere left the
column holding one hash, the source holding another, and the migrator
skipping the version because the NUMBER matched. The checksum could not
detect the drift it exists to detect. :meth:`verify` now compares them and
:meth:`run` refuses to continue on a mismatch, because a database whose
schema no longer matches the code that believes it wrote it is the one
state a migrator must never proceed from.

**An interrupted migration left no trace.** ``executescript`` commits
before it runs, so the DDL landed in its own transaction and the ledger
row was inserted afterwards in another. A crash between the two left the
schema changed and unrecorded, and the next boot re-applied the same
migration — harmless while every statement happens to say ``IF NOT
EXISTS``, and silent corruption the first time one does not. The ledger
now records ``started`` before the DDL and ``applied`` after it, so an
interrupted migration is a state the next boot can see and act on rather
than a gap it cannot distinguish from never having run.
"""
import hashlib
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import NamedTuple

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Migrations")

#: Checksums written before this module used the full digest are the first
#: 16 hex characters of the same sha256. Comparing on the shorter length
#: keeps every database written by an earlier build verifiable instead of
#: declaring all of them drifted on the first boot after the upgrade.
_LEGACY_CHECKSUM_LENGTH = 16


def checksum_of(up_sql: str) -> str:
    return hashlib.sha256(up_sql.encode()).hexdigest()


def checksums_match(stored: str, computed: str) -> bool:
    if not stored:
        return False
    if len(stored) == _LEGACY_CHECKSUM_LENGTH:
        return computed[:_LEGACY_CHECKSUM_LENGTH] == stored
    return stored == computed


class LedgerRow(NamedTuple):
    version: int
    description: str
    checksum: str
    applied_at: float
    status: str


class Migration(NamedTuple):
    version: int
    description: str
    up_sql: str                          # SQL to apply
    checksum: str = ""                   # Auto-computed from up_sql


class MigrationDriftError(RuntimeError):
    """An applied migration's SQL is not the SQL that was applied.

    Raised instead of continuing. Everything after this point in a boot
    assumes the schema matches the code; when it does not, the safe move is
    to stop where the operator can see it rather than to run queries against
    a shape nobody believes in.
    """


class InterruptedMigrationError(RuntimeError):
    """A migration recorded that it started and never recorded finishing."""


#: One row per applied migration. ``status`` is written BEFORE the DDL and
#: updated after it, so the two-phase record survives a crash between them.
STATUS_STARTED = "started"
STATUS_APPLIED = "applied"


class Migrator:

    _SCHEMA_TABLE = """
    CREATE TABLE IF NOT EXISTS _aura_migrations (
        version     INTEGER PRIMARY KEY,
        description TEXT NOT NULL,
        checksum    TEXT NOT NULL,
        applied_at  REAL NOT NULL
    );
    """

    #: Databases written before the ledger had a status column still exist.
    #: Adding it with a default of ``applied`` is the truth for those rows:
    #: they were written by the single-phase path, which only ever inserted
    #: after the DDL succeeded.
    _LEDGER_UPGRADES = (
        ("status", f"ALTER TABLE _aura_migrations ADD COLUMN status TEXT NOT NULL "
                   f"DEFAULT '{STATUS_APPLIED}'"),
        ("started_at", "ALTER TABLE _aura_migrations ADD COLUMN started_at REAL"),
    )

    def __init__(self, db_path: str | Path) -> None:
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
                except (ImportError, AttributeError, RuntimeError) as _exc:
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
                except (ImportError, AttributeError, RuntimeError) as _exc:
                    record_degradation('migrations', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)
        
        self.db_path = str(Path(path_str).expanduser())
        self._migrations: list[Migration] = []

    def register(self, version: int, description: str, up_sql: str) -> None:
        if any(m.version == version for m in self._migrations):
            raise ValueError(
                f"migration v{version} is registered twice; version numbers are "
                "the identity the ledger stores"
            )
        self._migrations.append(
            Migration(version, description, up_sql, checksum_of(up_sql))
        )
        self._migrations.sort(key=lambda m: m.version)

    def _open(self) -> sqlite3.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _prepare_ledger(self, con: sqlite3.Connection) -> None:
        con.execute(self._SCHEMA_TABLE)
        existing = {row[1] for row in con.execute("PRAGMA table_info(_aura_migrations)")}
        for column, statement in self._LEDGER_UPGRADES:
            if column not in existing:
                con.execute(statement)
        con.commit()

    def ledger(self) -> "list[LedgerRow]":
        """What the database says has been applied to it."""
        con = self._open()
        try:
            self._prepare_ledger(con)
            return [
                LedgerRow(int(v), str(d), str(c), float(a or 0.0), str(s or STATUS_APPLIED))
                for v, d, c, a, s in con.execute(
                    "SELECT version, description, checksum, applied_at, status "
                    "FROM _aura_migrations ORDER BY version"
                )
            ]
        finally:
            con.close()

    def verify(self) -> "list[str]":
        """Every difference between what was applied and what the code says.

        Three kinds, all of which the version-number check could not see:

        * an applied migration whose SQL has changed since — the drift the
          checksum column was added for and never used to detect;
        * an applied version this code no longer registers, which means the
          database is ahead of the binary, so a downgrade;
        * a migration that recorded starting and never recorded finishing.
        """
        registered = {m.version: m for m in self._migrations}
        problems: list[str] = []
        for row in self.ledger():
            if row.status == STATUS_STARTED:
                problems.append(
                    f"v{row.version} ('{row.description}') started and never "
                    "finished; the schema may be half-applied"
                )
                continue
            migration = registered.get(row.version)
            if migration is None:
                problems.append(
                    f"v{row.version} ('{row.description}') is applied to this "
                    "database and is not registered in this build: the database "
                    "is ahead of the code"
                )
                continue
            if not checksums_match(row.checksum, migration.checksum):
                problems.append(
                    f"v{row.version} ('{row.description}') was applied with "
                    f"checksum {row.checksum} and this build computes "
                    f"{migration.checksum}: the migration was edited after it ran"
                )
        return problems

    def run_all(self) -> int:
        """Alias for run() to match standard maintenance interface."""
        self.reconcile_legacy_schema()
        return self.run()

    def run(self, *, recover_interrupted: bool = True) -> int:
        """Apply what has not been applied, after checking what has.

        ``verify`` runs first and a drift raises. A half-applied migration is
        re-run once when ``recover_interrupted`` — every registered statement
        in this module is idempotent by construction, so replaying the script
        reaches the same schema — and raises if the replay fails.
        """
        con = self._open()
        applied = 0

        try:
            self._prepare_ledger(con)

            interrupted = [
                row for row in self._ledger_rows(con) if row.status == STATUS_STARTED
            ]
            if interrupted and not recover_interrupted:
                raise InterruptedMigrationError(
                    "half-applied migrations: "
                    + ", ".join(f"v{row.version}" for row in interrupted)
                )

            drift = [
                problem
                for problem in self.verify()
                if "started and never finished" not in problem
            ]
            if drift:
                raise MigrationDriftError(
                    f"{self.db_path} does not match this build:\n  - "
                    + "\n  - ".join(drift)
                )

            registered = {m.version: m for m in self._migrations}
            for row in interrupted:
                migration = registered.get(row.version)
                if migration is None:
                    raise InterruptedMigrationError(
                        f"v{row.version} was interrupted and this build no longer "
                        "registers it, so it cannot be finished or rolled back"
                    )
                logger.warning(
                    "Migration v%d was interrupted; replaying it.", migration.version
                )
                self._apply(con, migration, replay=True)
                applied += 1

            ledger_versions = {row.version for row in self._ledger_rows(con)}
            for migration in self._migrations:
                if migration.version in ledger_versions:
                    continue
                logger.info(
                    "Applying migration v%d: %s", migration.version, migration.description
                )
                self._apply(con, migration)
                applied += 1
        finally:
            con.close()

        if applied == 0:
            logger.debug("All migrations already applied.")
        return applied

    def _ledger_rows(self, con: sqlite3.Connection) -> "list[LedgerRow]":
        return [
            LedgerRow(int(v), str(d), str(c), float(a or 0.0), str(st or STATUS_APPLIED))
            for v, d, c, a, st in con.execute(
                "SELECT version, description, checksum, applied_at, status "
                "FROM _aura_migrations ORDER BY version"
            )
        ]

    def _apply(
        self, con: sqlite3.Connection, migration: Migration, *, replay: bool = False
    ) -> None:
        """Record that it started, run it, record that it finished.

        ``executescript`` issues a COMMIT before it runs, so the intent row
        cannot share a transaction with the DDL. Writing the intent first is
        what makes the gap between them visible: a crash leaves ``started``
        in the ledger instead of leaving nothing.
        """
        now = time.time()
        if not replay:
            con.execute(
                "INSERT INTO _aura_migrations "
                "(version, description, checksum, applied_at, status, started_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    migration.version,
                    migration.description,
                    migration.checksum,
                    0.0,
                    STATUS_STARTED,
                    now,
                ),
            )
            con.commit()

        try:
            con.executescript(migration.up_sql)
            con.execute(
                "UPDATE _aura_migrations SET status = ?, applied_at = ?, "
                "checksum = ?, description = ? WHERE version = ?",
                (
                    STATUS_APPLIED,
                    time.time(),
                    migration.checksum,
                    migration.description,
                    migration.version,
                ),
            )
            con.commit()
            logger.info("✅ Migration v%d applied.", migration.version)
        except (sqlite3.Error, OSError) as e:
            record_degradation("migrations", e)
            con.rollback()
            logger.error(
                "❌ Migration v%d FAILED: %s — the ledger records it as unfinished.",
                migration.version,
                e,
            )
            raise RuntimeError(
                f"Migration v{migration.version} ('{migration.description}') failed: {e}"
            ) from e

    def reconcile_legacy_schema(self) -> None:
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
        except (sqlite3.Error, OSError) as e:
            record_degradation('migrations', e)
            logger.error("Failed to reconcile legacy schema: %s", e)
            con.rollback()
        finally:
            con.close()


def _build_knowledge_migrator(db_path: str | Path | None = None) -> Migrator:
    """Build and configure the knowledge graph migrator (ISSUE 5)."""
    m = Migrator(db_path or "~/.aura/data/knowledge.db")
    m.register(1, "Initial schema", _SQL_V1)
    m.register(2, "Add full-text search", _SQL_V2)
    m.register(3, "Add skills and goals tables", _SQL_V3)
    m.register(4, "Add execution audit log", _SQL_V4)
    return m

def get_migrator(db_path: str | Path | None = None) -> Migrator:
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
