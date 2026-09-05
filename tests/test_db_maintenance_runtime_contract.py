import asyncio
import sqlite3
import time
from pathlib import Path

from core.runtime.sqlite_support import connecting


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
    with connecting(sqlite3.connect(db_path)) as conn:
        conn.execute("CREATE TABLE safe_table(id INTEGER PRIMARY KEY, created_at REAL)")
        conn.execute("INSERT INTO safe_table(created_at) VALUES (?)", (time.time(),))

    policy = RetentionPolicy(
        table_name='safe_table"; DROP TABLE safe_table; --',
        timestamp_column="created_at",
        max_age_days=1,
    )
    maint = DatabaseMaintenance(db_path=str(db_path), retention_policies=[policy])
    result = MaintenanceResult()

    with connecting(sqlite3.connect(db_path)) as conn:
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
    with connecting(sqlite3.connect(db_path)) as conn:
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

    with connecting(sqlite3.connect(db_path)) as conn:
        maint.run_retention(conn, result)
        remaining = conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]

    assert result.rows_deleted == {"receipts": 1}
    assert remaining == 2


def test_maintenance_pass_continues_after_phase_failure(monkeypatch, tmp_path):
    from core.persistence.db_maintenance import DatabaseMaintenance

    db_path = tmp_path / "aura_state.db"
    with connecting(sqlite3.connect(db_path)) as conn:
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

    def _vacuum(conn, result, *, allow_full_vacuum=False):
        phases_seen.append("vacuum")
        assert allow_full_vacuum is True
        result.vacuum_run = True

    def _integrity(conn, result, *, thorough=False):
        phases_seen.append("integrity")
        assert thorough is True
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


def test_routine_maintenance_never_promotes_itself_to_full_vacuum(tmp_path):
    from core.persistence.db_maintenance import DatabaseMaintenance

    db_path = tmp_path / "aura_state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE payload(value TEXT)")

    maint = DatabaseMaintenance(db_path=str(db_path), vacuum_interval_hours=0)
    result = maint.run_maintenance()

    assert result.vacuum_run is False
    assert "full_vacuum_requires_explicit_maintenance" in result.deferred_phases
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 0


def test_explicit_maintenance_may_install_incremental_vacuum(tmp_path):
    from core.persistence.db_maintenance import DatabaseMaintenance

    db_path = tmp_path / "aura_state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE payload(value TEXT)")

    maint = DatabaseMaintenance(db_path=str(db_path), vacuum_interval_hours=0)
    result = maint.run_maintenance(force=True)

    assert result.vacuum_run is True
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2


def test_async_routine_maintenance_defers_while_a_person_is_waiting(tmp_path):
    from core.persistence.db_maintenance import DatabaseMaintenance
    from core.runtime.foreground_guard import _reset_for_tests, begin_foreground_turn

    db_path = tmp_path / "aura_state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE payload(value TEXT)")
    maint = DatabaseMaintenance(db_path=str(db_path))

    _reset_for_tests()
    lease = begin_foreground_turn()
    try:
        result = asyncio.run(maint.run_maintenance_async())
    finally:
        lease.close()
        _reset_for_tests()

    assert result.deferred_phases == ["foreground_chat_active"]
    assert maint.get_status()["total_passes"] == 0


def test_async_routine_maintenance_leaves_event_loop_responsive(monkeypatch, tmp_path):
    import threading

    from core.persistence.db_maintenance import DatabaseMaintenance
    from core.runtime.foreground_guard import _reset_for_tests

    db_path = tmp_path / "aura_state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE payload(value TEXT)")
    maint = DatabaseMaintenance(db_path=str(db_path))
    entered = threading.Event()
    release = threading.Event()

    def _slow_run(*, force=False):
        assert force is False
        entered.set()
        release.wait(timeout=1)
        return type("Result", (), {"total_rows_deleted": 0})()

    async def _run():
        _reset_for_tests()
        monkeypatch.setattr(maint, "run_maintenance", _slow_run)
        task = asyncio.create_task(maint.run_maintenance_async())
        for _ in range(1000):
            if entered.is_set():
                break
            await asyncio.sleep(0.001)
        assert entered.is_set()
        heartbeat = asyncio.create_task(asyncio.sleep(0))
        await asyncio.wait_for(heartbeat, timeout=0.1)
        release.set()
        return await asyncio.wait_for(task, timeout=1)

    asyncio.run(_run())
