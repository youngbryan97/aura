"""core/persistence/db_maintenance.py
=====================================
Production database maintenance for Aura's SQLite stores.

Handles:
  - WAL checkpoint scheduling (prevents unbounded WAL growth)
  - Auto-vacuum with incremental mode (reclaims space without full locks)
  - Tiered retention (receipts: 30d, life_trace: 90d, memories: permanent)
  - Periodic integrity checks (PRAGMA integrity_check on slow schedule)
  - Database size monitoring with alerting

Designed to run as a background periodic task from MetabolicCoordinator
or as a standalone maintenance pass via `make seal`.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Persistence.Maintenance")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _record_db_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("db_maintenance", exc, severity=severity, action=action)


def _quote_identifier(identifier: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"unsafe SQLite identifier: {identifier!r}")
    return f'"{identifier}"'

# ── Retention Policies ────────────────────────────────────────────────────

@dataclass
class RetentionPolicy:
    """Defines how long data in a specific table should be kept."""
    table_name: str
    timestamp_column: str
    max_age_days: int
    description: str = ""
    # If True, archive to a separate table before deleting
    archive_before_delete: bool = False
    # Maximum rows to delete per maintenance pass (prevents long locks)
    batch_size: int = 500


# Default retention policies — tuned for production
DEFAULT_RETENTION_POLICIES = [
    RetentionPolicy(
        table_name="receipts",
        timestamp_column="created_at",
        max_age_days=30,
        description="Execution receipts older than 30 days",
        batch_size=1000,
    ),
    RetentionPolicy(
        table_name="life_trace",
        timestamp_column="timestamp",
        max_age_days=90,
        description="LifeTrace events older than 90 days",
        archive_before_delete=True,
        batch_size=500,
    ),
    RetentionPolicy(
        table_name="degraded_events",
        timestamp_column="timestamp",
        max_age_days=14,
        description="Degradation events older than 14 days",
        batch_size=2000,
    ),
    RetentionPolicy(
        table_name="thought_stream",
        timestamp_column="created_at",
        max_age_days=60,
        description="Thought stream entries older than 60 days",
        batch_size=500,
    ),
    RetentionPolicy(
        table_name="incident_log",
        timestamp_column="reported_at",
        max_age_days=180,
        description="Incident logs older than 180 days",
        batch_size=200,
    ),
]


@dataclass
class MaintenanceResult:
    """Result of a single maintenance pass."""
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    wal_checkpointed: bool = False
    wal_pages_moved: int = 0
    vacuum_run: bool = False
    integrity_ok: bool | None = None
    rows_deleted: dict[str, int] = field(default_factory=dict)
    rows_archived: dict[str, int] = field(default_factory=dict)
    db_size_bytes: int = 0
    wal_size_bytes: int = 0
    errors: list[str] = field(default_factory=list)
    skipped_policies: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        if self.completed_at <= 0:
            return time.time() - self.started_at
        return self.completed_at - self.started_at

    @property
    def total_rows_deleted(self) -> int:
        return sum(self.rows_deleted.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": round(self.duration_s, 2),
            "wal_checkpointed": self.wal_checkpointed,
            "wal_pages_moved": self.wal_pages_moved,
            "vacuum_run": self.vacuum_run,
            "integrity_ok": self.integrity_ok,
            "rows_deleted": self.rows_deleted,
            "rows_archived": self.rows_archived,
            "total_rows_deleted": self.total_rows_deleted,
            "db_size_bytes": self.db_size_bytes,
            "wal_size_bytes": self.wal_size_bytes,
            "errors": self.errors,
            "skipped_policies": self.skipped_policies,
        }


class DatabaseMaintenance:
    """Production database maintenance engine.

    Usage:
        maint = DatabaseMaintenance("/path/to/aura_state.db")
        result = maint.run_maintenance()
    """

    def __init__(
        self,
        db_path: str | None = None,
        retention_policies: list[RetentionPolicy] | None = None,
        *,
        vacuum_interval_hours: float = 24.0,
        checkpoint_interval_minutes: float = 15.0,
        integrity_check_interval_hours: float = 168.0,  # Weekly
        max_db_size_mb: float = 500.0,
    ):
        if db_path is None:
            runtime_dir = os.environ.get(
                "AURA_ENV_RUNTIME_DIR",
                str(Path.home() / ".aura" / "live-source" / "data"),
            )
            db_path = str(Path(runtime_dir) / "aura_state.db")

        self._db_path = db_path
        self._retention_policies = retention_policies or list(DEFAULT_RETENTION_POLICIES)
        self._vacuum_interval_s = vacuum_interval_hours * 3600
        self._checkpoint_interval_s = checkpoint_interval_minutes * 60
        self._integrity_interval_s = integrity_check_interval_hours * 3600
        self._max_db_size_bytes = int(max_db_size_mb * 1024 * 1024)

        self._last_vacuum_time: float = 0.0
        self._last_checkpoint_time: float = 0.0
        self._last_integrity_time: float = 0.0
        self._last_result: MaintenanceResult | None = None
        self._total_passes: int = 0

    def _get_connection(self) -> sqlite3.Connection | None:
        """Get a maintenance connection with appropriate settings."""
        try:
            path = Path(self._db_path)
            if not path.exists():
                logger.debug("Maintenance: DB does not exist yet: %s", self._db_path)
                return None
            conn = sqlite3.connect(self._db_path, timeout=10.0)
            conn.execute("PRAGMA busy_timeout=10000;")
            return conn
        except (sqlite3.Error, OSError) as exc:
            _record_db_degradation(
                exc,
                action="skipped database maintenance pass because SQLite connection could not be opened",
            )
            logger.warning("Maintenance: Cannot connect to DB: %s", exc)
            return None

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            _quote_identifier(table_name)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return cursor.fetchone() is not None
        except (sqlite3.Error, ValueError) as exc:
            _record_db_degradation(
                exc,
                severity="warning",
                action=f"treated table {table_name!r} as missing during maintenance",
            )
            return False

    def _column_exists(self, conn: sqlite3.Connection, table: str, column: str) -> bool:
        """Check if a column exists in a table."""
        try:
            quoted_table = _quote_identifier(table)
            _quote_identifier(column)
            cursor = conn.execute(f"PRAGMA table_info({quoted_table})")
            columns = [row[1] for row in cursor.fetchall()]
            return column in columns
        except (sqlite3.Error, ValueError) as exc:
            _record_db_degradation(
                exc,
                severity="warning",
                action=f"treated column {table}.{column} as missing during maintenance",
            )
            return False

    def run_checkpoint(self, conn: sqlite3.Connection, result: MaintenanceResult) -> None:
        """Run WAL checkpoint to prevent unbounded WAL growth."""
        now = time.time()
        if now - self._last_checkpoint_time < self._checkpoint_interval_s:
            return

        try:
            # PRAGMA wal_checkpoint(PASSIVE) doesn't block readers/writers
            cursor = conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
            row = cursor.fetchone()
            if row:
                # Returns (blocked, log_pages, checkpointed_pages)
                result.wal_pages_moved = row[2] if len(row) > 2 else 0
            result.wal_checkpointed = True
            self._last_checkpoint_time = now
            logger.debug(
                "Maintenance: WAL checkpoint complete (%d pages moved).",
                result.wal_pages_moved,
            )
        except sqlite3.Error as exc:
            _record_db_degradation(
                exc,
                action="left WAL checkpoint schedule unchanged after passive checkpoint failed",
            )
            result.errors.append(f"checkpoint: {exc}")

    def run_retention(self, conn: sqlite3.Connection, result: MaintenanceResult) -> None:
        """Apply tiered retention policies — delete old rows in batches."""
        for policy in self._retention_policies:
            try:
                table = _quote_identifier(policy.table_name)
                timestamp_column = _quote_identifier(policy.timestamp_column)
                archive_table = _quote_identifier(f"{policy.table_name}_archive")
            except ValueError as exc:
                _record_db_degradation(
                    exc,
                    action=f"skipped unsafe retention policy for {policy.table_name!r}",
                )
                result.errors.append(f"retention_policy_invalid_{policy.table_name}: {exc}")
                result.skipped_policies.append(policy.table_name)
                continue

            if not self._table_exists(conn, policy.table_name):
                result.skipped_policies.append(policy.table_name)
                continue
            if not self._column_exists(conn, policy.table_name, policy.timestamp_column):
                result.skipped_policies.append(policy.table_name)
                continue

            try:
                cutoff = time.time() - (policy.max_age_days * 86400)

                # Archive before delete if configured
                if policy.archive_before_delete:
                    archive_table_name = f"{policy.table_name}_archive"
                    if not self._table_exists(conn, archive_table_name):
                        try:
                            conn.execute(
                                f"CREATE TABLE IF NOT EXISTS {archive_table} "
                                f"AS SELECT * FROM {table} WHERE 0"
                            )
                        except sqlite3.Error as exc:
                            _record_db_degradation(
                                exc,
                                action=f"continued retention for {policy.table_name} without archive table creation",
                            )
                            result.errors.append(f"archive_create_{policy.table_name}: {exc}")

                    if self._table_exists(conn, archive_table_name):
                        try:
                            cursor = conn.execute(
                                f"INSERT INTO {archive_table} "
                                f"SELECT * FROM {table} "
                                f"WHERE {timestamp_column} < ? "
                                f"LIMIT ?",
                                (cutoff, policy.batch_size),
                            )
                            archived = cursor.rowcount
                            if archived > 0:
                                result.rows_archived[policy.table_name] = archived
                        except sqlite3.Error as exc:
                            _record_db_degradation(
                                exc,
                                action=f"continued retention delete for {policy.table_name} after archive insert failed",
                            )
                            result.errors.append(
                                f"archive_{policy.table_name}: {exc}"
                            )

                # Delete expired rows in batches
                cursor = conn.execute(
                    f"DELETE FROM {table} "
                    f"WHERE rowid IN ("
                    f"  SELECT rowid FROM {table} "
                    f"  WHERE {timestamp_column} < ? "
                    f"  LIMIT ?"
                    f")",
                    (cutoff, policy.batch_size),
                )
                deleted = cursor.rowcount
                if deleted > 0:
                    result.rows_deleted[policy.table_name] = deleted
                    logger.info(
                        "Maintenance: Deleted %d expired rows from %s (>%dd old).",
                        deleted,
                        policy.table_name,
                        policy.max_age_days,
                    )
            except (sqlite3.Error, ValueError) as exc:
                _record_db_degradation(
                    exc,
                    action=f"skipped retention delete for {policy.table_name} after policy operation failed",
                )
                result.errors.append(f"retention_{policy.table_name}: {exc}")

        # Commit all retention changes
        try:
            conn.commit()
        except sqlite3.Error as exc:
            _record_db_degradation(
                exc,
                action="rolled back retention transaction after commit failed",
                severity="degraded",
            )
            try:
                conn.rollback()
            except sqlite3.Error as rollback_exc:
                _record_db_degradation(
                    rollback_exc,
                    action="left retention transaction state to connection close after rollback failed",
                    severity="degraded",
                )
            result.errors.append(f"retention_commit: {exc}")

    def run_vacuum(self, conn: sqlite3.Connection, result: MaintenanceResult) -> None:
        """Run incremental auto-vacuum if due."""
        now = time.time()
        if now - self._last_vacuum_time < self._vacuum_interval_s:
            return

        try:
            # Check current auto_vacuum mode
            cursor = conn.execute("PRAGMA auto_vacuum;")
            mode = cursor.fetchone()
            current_mode = mode[0] if mode else 0

            if current_mode == 0:
                # Switch to incremental auto_vacuum (mode 2)
                # This requires a VACUUM to take effect
                conn.execute("PRAGMA auto_vacuum=INCREMENTAL;")
                conn.execute("VACUUM;")
                logger.info("Maintenance: Switched to INCREMENTAL auto_vacuum.")
            elif current_mode == 2:
                # Run incremental vacuum — reclaim up to 100 pages
                conn.execute("PRAGMA incremental_vacuum(100);")
                logger.debug("Maintenance: Incremental vacuum completed (100 pages).")

            result.vacuum_run = True
            self._last_vacuum_time = now
        except sqlite3.Error as exc:
            _record_db_degradation(
                exc,
                action="left vacuum schedule unchanged after incremental vacuum failed",
            )
            result.errors.append(f"vacuum: {exc}")

    def run_integrity_check(
        self, conn: sqlite3.Connection, result: MaintenanceResult
    ) -> None:
        """Run periodic integrity check (weekly by default)."""
        now = time.time()
        if now - self._last_integrity_time < self._integrity_interval_s:
            return

        try:
            cursor = conn.execute("PRAGMA integrity_check(1);")
            row = cursor.fetchone()
            ok = row is not None and row[0] == "ok"
            result.integrity_ok = ok
            self._last_integrity_time = now

            if ok:
                logger.info("Maintenance: Integrity check PASSED.")
            else:
                error_msg = row[0] if row else "no response"
                logger.error("Maintenance: Integrity check FAILED: %s", error_msg)
                result.errors.append(f"integrity: {error_msg}")

                # Report to incident manager
                try:
                    from core.resilience.incident_manager import get_incident_manager
                    get_incident_manager().report(
                        source="db_maintenance",
                        title="Database integrity check failed",
                        detail=error_msg[:500],
                        severity="critical",
                    )
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
                    _record_db_degradation(
                        _exc,
                        severity="degraded",
                        action="left integrity failure in maintenance result after incident reporting failed",
                    )
                    logger.debug("Suppressed %s in core.persistence.db_maintenance: %s", type(_exc).__name__, _exc)
        except sqlite3.Error as exc:
            _record_db_degradation(
                exc,
                action="left last successful integrity timestamp unchanged after integrity check failed",
                severity="degraded",
            )
            result.errors.append(f"integrity_check: {exc}")

    def check_size(self, result: MaintenanceResult) -> None:
        """Monitor database and WAL file sizes."""
        try:
            db_path = Path(self._db_path)
            if db_path.exists():
                result.db_size_bytes = db_path.stat().st_size

            wal_path = Path(f"{self._db_path}-wal")
            if wal_path.exists():
                result.wal_size_bytes = wal_path.stat().st_size

            # Alert if DB is too large
            if result.db_size_bytes > self._max_db_size_bytes:
                logger.warning(
                    "Maintenance: DB size %.1f MB exceeds limit %.1f MB",
                    result.db_size_bytes / (1024 * 1024),
                    self._max_db_size_bytes / (1024 * 1024),
                )
                try:
                    from core.resilience.incident_manager import get_incident_manager
                    get_incident_manager().report(
                        source="db_maintenance",
                        title="Database size exceeds limit",
                        detail=f"{result.db_size_bytes / (1024*1024):.1f} MB > {self._max_db_size_bytes / (1024*1024):.1f} MB",
                        severity="warning",
                    )
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
                    _record_db_degradation(
                        _exc,
                        severity="warning",
                        action="kept database-size warning in logs after incident reporting failed",
                    )
                    logger.debug("Suppressed %s in core.persistence.db_maintenance: %s", type(_exc).__name__, _exc)

            # Alert if WAL is growing too large (> 50MB)
            if result.wal_size_bytes > 50 * 1024 * 1024:
                logger.warning(
                    "Maintenance: WAL size %.1f MB — consider TRUNCATE checkpoint.",
                    result.wal_size_bytes / (1024 * 1024),
                )

            # Record to metrics
            try:
                from core.observability.metrics import get_metrics
                get_metrics().set_gauge("db_size_bytes", float(result.db_size_bytes))
                get_metrics().set_gauge("wal_size_bytes", float(result.wal_size_bytes))
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
                _record_db_degradation(
                    _exc,
                    severity="debug",
                    action="completed size check without metrics gauge update",
                )
                logger.debug("Suppressed %s in core.persistence.db_maintenance: %s", type(_exc).__name__, _exc)

        except (OSError, sqlite3.Error) as exc:
            _record_db_degradation(
                exc,
                action="completed maintenance pass without filesystem size telemetry",
            )
            result.errors.append(f"size_check: {exc}")

    def run_maintenance(self, *, force: bool = False) -> MaintenanceResult:
        """Execute a full maintenance pass.

        Args:
            force: If True, run all operations regardless of intervals.

        Returns:
            MaintenanceResult with details of what was done.
        """
        result = MaintenanceResult()

        if force:
            self._last_vacuum_time = 0.0
            self._last_checkpoint_time = 0.0
            self._last_integrity_time = 0.0

        conn = self._get_connection()
        if conn is None:
            result.errors.append("no_connection")
            result.completed_at = time.time()
            return result

        phases = (
            ("checkpoint", lambda: self.run_checkpoint(conn, result)),
            ("retention", lambda: self.run_retention(conn, result)),
            ("vacuum", lambda: self.run_vacuum(conn, result)),
            ("integrity", lambda: self.run_integrity_check(conn, result)),
            ("size", lambda: self.check_size(result)),
        )
        try:
            for phase_name, phase in phases:
                try:
                    phase()
                except (sqlite3.Error, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    _record_db_degradation(
                        exc,
                        action=f"continued database maintenance pass after {phase_name} phase failed",
                        severity="degraded",
                    )
                    result.errors.append(f"{phase_name}_phase: {exc}")
        finally:
            try:
                conn.close()
            except sqlite3.Error as exc:
                _record_db_degradation(
                    exc,
                    severity="warning",
                    action="completed database maintenance pass after connection close failed",
                )

        result.completed_at = time.time()
        self._last_result = result
        self._total_passes += 1

        if result.total_rows_deleted > 0 or result.vacuum_run:
            logger.info(
                "Maintenance pass #%d complete in %.2fs: "
                "deleted=%d, vacuum=%s, checkpoint=%s, integrity=%s",
                self._total_passes,
                result.duration_s,
                result.total_rows_deleted,
                result.vacuum_run,
                result.wal_checkpointed,
                result.integrity_ok,
            )
        else:
            logger.debug(
                "Maintenance pass #%d complete in %.2fs (no changes).",
                self._total_passes,
                result.duration_s,
            )

        return result

    def get_status(self) -> dict[str, Any]:
        """Return current maintenance status for observability."""
        return {
            "total_passes": self._total_passes,
            "last_result": self._last_result.to_dict() if self._last_result else None,
            "last_checkpoint_age_s": round(
                time.time() - self._last_checkpoint_time, 1
            )
            if self._last_checkpoint_time > 0
            else None,
            "last_vacuum_age_s": round(time.time() - self._last_vacuum_time, 1)
            if self._last_vacuum_time > 0
            else None,
            "last_integrity_age_s": round(
                time.time() - self._last_integrity_time, 1
            )
            if self._last_integrity_time > 0
            else None,
        }


# ── Singleton ─────────────────────────────────────────────────────────────

_instance: DatabaseMaintenance | None = None


def get_db_maintenance() -> DatabaseMaintenance:
    """Get the singleton DatabaseMaintenance instance."""
    global _instance
    if _instance is None:
        _instance = DatabaseMaintenance()
    return _instance
