"""A database large with the state log it was told to keep is not one that has outgrown itself.

Third day run, 6 November 2026: her state database was 1019 MB, nearly all of it 8090 state-log rows of about 125 KB
kept at the run's 100,000-row retention; "Maintenance: DB size 1019.2 MB exceeds limit 500.0 MB" was said 13 times and
opened an incident.
"""
from __future__ import annotations

import logging
import sqlite3

import pytest

from core.persistence.db_maintenance import DatabaseMaintenance, MaintenanceResult


def _database(path, *, state_rows: int, other_bytes: int = 0):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE state_log (state_id TEXT, state_json TEXT, timestamp REAL)")
    conn.executemany("INSERT INTO state_log VALUES (?, ?, 0)", [(str(i), "x" * 20_000) for i in range(state_rows)])
    if other_bytes:
        conn.execute("CREATE TABLE other (blob TEXT)")
        conn.executemany("INSERT INTO other VALUES (?)", [("y" * 20_000,) for _ in range(other_bytes // 20_000)])
    conn.commit()
    conn.close()


@pytest.fixture
def incidents(monkeypatch):
    seen = []
    import core.resilience.incident_manager as manager

    monkeypatch.setattr(manager, "get_incident_manager", lambda: type("M", (), {"report": lambda self, **k: seen.append(k)})())
    monkeypatch.setenv("AURA_STATE_LOG_MAX_ROWS", "1000")
    return seen


def test_over_the_limit_only_by_the_state_log_it_keeps(tmp_path, incidents, caplog):
    path = tmp_path / "aura_state.db"
    _database(path, state_rows=60)  # about 1.2 MB of state log
    maintenance = DatabaseMaintenance(db_path=str(path), max_db_size_mb=0.5)
    with caplog.at_level(logging.INFO, logger="Aura.Persistence.Maintenance"):
        maintenance.check_size(MaintenanceResult())
    assert incidents == []
    assert any("by the state log it keeps: 60 rows" in r.getMessage() for r in caplog.records)
    assert not any(r.levelno >= logging.WARNING and "exceeds limit" in r.getMessage() for r in caplog.records)


def test_over_the_limit_by_something_else_is_still_an_incident(tmp_path, incidents):
    path = tmp_path / "aura_state.db"
    _database(path, state_rows=5, other_bytes=1_200_000)
    DatabaseMaintenance(db_path=str(path), max_db_size_mb=0.5).check_size(MaintenanceResult())
    assert incidents and incidents[0]["title"] == "Database size exceeds limit"
