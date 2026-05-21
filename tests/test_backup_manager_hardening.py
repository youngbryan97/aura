from __future__ import annotations

import sqlite3

import pytest

from core.backup import BackupManager


@pytest.mark.asyncio
async def test_vacuum_reports_partial_database_failure(monkeypatch, tmp_path):
    good_db = tmp_path / "a.db"
    bad_db = tmp_path / "b.db"
    good_db.touch()
    bad_db.touch()

    manager = BackupManager()
    manager.data_dir = tmp_path
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr(
        "core.resilience.database_coordinator.get_db_coordinator",
        lambda: type("Coordinator", (), {"_connections": {}})(),
    )

    seen: list[str] = []

    def _vacuum(path):
        seen.append(path.name)
        if path == bad_db:
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(BackupManager, "_vacuum_database_sync", staticmethod(_vacuum))

    assert await manager.run_vacuum() is False
    assert seen == ["a.db", "b.db"]
    assert manager._last_vacuum_at == 0.0
    assert manager._last_vacuum_attempt_at > 0.0
    assert manager._last_vacuum_failures == [f"{bad_db}: OperationalError"]


@pytest.mark.asyncio
async def test_vacuum_defers_when_background_policy_fails_closed(monkeypatch):
    manager = BackupManager()

    def _policy_failure(*_args, **_kwargs):
        reason = "policy unavailable"
        raise RuntimeError(reason)

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        _policy_failure,
    )
    monkeypatch.setattr(
        BackupManager,
        "_vacuum_database_sync",
        staticmethod(lambda _path: pytest.fail("vacuum should be deferred")),
    )

    assert await manager.run_vacuum() is False
    assert manager._last_vacuum_attempt_at == 0.0


@pytest.mark.asyncio
async def test_get_health_reports_last_vacuum_failures(tmp_path):
    manager = BackupManager()
    manager.backup_dir = tmp_path
    manager._last_vacuum_failures = ["state.db: OperationalError"]

    health = await manager.get_health()

    assert health["status"] == "online"
    assert health["last_vacuum_failures"] == ["state.db: OperationalError"]
