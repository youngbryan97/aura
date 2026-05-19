import sqlite3
import time
from pathlib import Path


def test_db_maintenance_degradation_audit_is_clean():
    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/persistence/db_maintenance.py")) == []


def test_retention_policy_rejects_unsafe_identifiers_without_executing_sql(tmp_path):
    from core.persistence.db_maintenance import (
        DatabaseMaintenance,
        MaintenanceResult,
        RetentionPolicy,
    )

    db_path = tmp_path / "aura_state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE safe_table(id INTEGER PRIMARY KEY, created_at REAL)")
        conn.execute("INSERT INTO safe_table(created_at) VALUES (?)", (time.time(),))

    policy = RetentionPolicy(
        table_name='safe_table"; DROP TABLE safe_table; --',
        timestamp_column="created_at",
        max_age_days=1,
    )
    maint = DatabaseMaintenance(db_path=str(db_path), retention_policies=[policy])
    result = MaintenanceResult()

    with sqlite3.connect(db_path) as conn:
        maint.run_retention(conn, result)
        still_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='safe_table'"
        ).fetchone()
        row_count = conn.execute("SELECT COUNT(*) FROM safe_table").fetchone()[0]

    assert still_exists is not None
    assert row_count == 1
    assert result.skipped_policies == [policy.table_name]
    assert any("retention_policy_invalid" in error for error in result.errors)


def test_retention_deletes_only_expired_rows_in_batch(tmp_path):
    from core.persistence.db_maintenance import (
        DatabaseMaintenance,
        MaintenanceResult,
        RetentionPolicy,
    )

    db_path = tmp_path / "aura_state.db"
    now = time.time()
    old = now - (40 * 86400)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE receipts(id INTEGER PRIMARY KEY, created_at REAL)")
        conn.executemany(
            "INSERT INTO receipts(created_at) VALUES (?)",
            [(old,), (old,), (now,)],
        )

    maint = DatabaseMaintenance(
        db_path=str(db_path),
        retention_policies=[
            RetentionPolicy(
                table_name="receipts",
                timestamp_column="created_at",
                max_age_days=30,
                batch_size=1,
            )
        ],
    )
    result = MaintenanceResult()

    with sqlite3.connect(db_path) as conn:
        maint.run_retention(conn, result)
        remaining = conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]

    assert result.rows_deleted == {"receipts": 1}
    assert remaining == 2


def test_maintenance_pass_continues_after_phase_failure(monkeypatch, tmp_path):
    from core.persistence.db_maintenance import DatabaseMaintenance

    db_path = tmp_path / "aura_state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE receipts(id INTEGER PRIMARY KEY, created_at REAL)")

    maint = DatabaseMaintenance(db_path=str(db_path))
    phases_seen = []

    def _raise_checkpoint(conn, result):
        phases_seen.append("checkpoint")
        result.errors.append("checkpoint-started")
        error = sqlite3.Error("checkpoint broke")
        raise error

    def _retention(conn, result):
        phases_seen.append("retention")
        result.rows_deleted["receipts"] = 0

    def _vacuum(conn, result):
        phases_seen.append("vacuum")
        result.vacuum_run = True

    def _integrity(conn, result):
        phases_seen.append("integrity")
        result.integrity_ok = True

    def _size(result):
        phases_seen.append("size")
        result.db_size_bytes = db_path.stat().st_size

    monkeypatch.setattr(maint, "run_checkpoint", _raise_checkpoint)
    monkeypatch.setattr(maint, "run_retention", _retention)
    monkeypatch.setattr(maint, "run_vacuum", _vacuum)
    monkeypatch.setattr(maint, "run_integrity_check", _integrity)
    monkeypatch.setattr(maint, "check_size", _size)

    result = maint.run_maintenance(force=True)

    assert phases_seen == ["checkpoint", "retention", "vacuum", "integrity", "size"]
    assert any("checkpoint_phase" in error for error in result.errors)
    assert result.vacuum_run is True
    assert result.integrity_ok is True
