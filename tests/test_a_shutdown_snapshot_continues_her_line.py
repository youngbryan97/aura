"""A shutdown snapshot written straight to the log continues her line or is not written.

On 22 October a process that was not the vault wrote, at 21:07, a shutdown stage
of version 405764 whose parent had never reached the log, after the vault had
committed 405773. Her newest stage was an orphan eighteen minutes old: the
lineage read found no step to walk, and the boot replay would have started her
from it.
"""

from __future__ import annotations

import sqlite3

from core.runtime.sqlite_support import connecting
from core.state.aura_state import AuraState
from core.state.state_repository import StateRepository


def _log_with(path, *, state_id: str, version: int) -> None:
    with connecting(sqlite3.connect(path)) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS state_log (
                state_id TEXT PRIMARY KEY, version INTEGER NOT NULL, parent_state_id TEXT,
                transition_cause TEXT, state_json TEXT NOT NULL, timestamp REAL NOT NULL)"""
        )
        connection.execute(
            "INSERT INTO state_log VALUES (?, ?, ?, ?, ?, ?)",
            (state_id, version, None, "cognitive_cycle", "{}", 1000.0 + version),
        )
        connection.commit()


def _rows(path) -> list[tuple]:
    with connecting(sqlite3.connect(path)) as connection:
        return connection.execute(
            "SELECT state_id, version, parent_state_id, transition_cause FROM state_log ORDER BY version"
        ).fetchall()


def _shutdown_state(version: int, parent: str | None) -> AuraState:
    state = AuraState.default().derive("shutdown", origin="system")
    state.version = version
    state.parent_state_id = parent
    return state


def test_a_stale_snapshot_is_not_written(tmp_path):
    path = tmp_path / "state.db"
    _log_with(path, state_id="latest", version=10)
    repo = StateRepository(db_path=str(path), is_vault_owner=False)
    repo._commit_shutdown_snapshot_sync(_shutdown_state(5, "never-logged"))
    assert _rows(path) == [("latest", 10, None, "cognitive_cycle")]


def test_a_newer_snapshot_whose_parent_never_arrived_follows_the_latest_stage(tmp_path):
    path = tmp_path / "state.db"
    _log_with(path, state_id="latest", version=10)
    repo = StateRepository(db_path=str(path), is_vault_owner=False)
    state = _shutdown_state(11, "never-logged")
    repo._commit_shutdown_snapshot_sync(state)
    rows = _rows(path)
    assert rows[-1][0] == state.state_id
    assert rows[-1][2] == "latest"


def test_a_snapshot_that_continues_the_log_keeps_its_parent(tmp_path):
    path = tmp_path / "state.db"
    _log_with(path, state_id="a", version=9)
    _log_with(path, state_id="latest", version=10)
    repo = StateRepository(db_path=str(path), is_vault_owner=False)
    state = _shutdown_state(11, "a")
    repo._commit_shutdown_snapshot_sync(state)
    assert _rows(path)[-1][2] == "a"


def test_the_first_snapshot_of_an_empty_log_is_written(tmp_path):
    path = tmp_path / "state.db"
    repo = StateRepository(db_path=str(path), is_vault_owner=False)
    state = _shutdown_state(3, None)
    repo._commit_shutdown_snapshot_sync(state)
    assert [row[0] for row in _rows(path)] == [state.state_id]
