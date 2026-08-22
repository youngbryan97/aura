"""The migration ledger records four things and now checks all four.

Before this, ``run()`` read one column — the version number — and skipped
every version it found. The checksum next to it was written on every apply
and read by nothing, so a migration edited after it had run left the
database holding one hash and the source holding another, and the migrator
stepped over it because the number matched. The column existed; the check
did not.

The cases below are the ones a migration framework is judged on: a fresh
database reaches current, a database from an earlier release reaches
current, an edited migration is refused rather than skipped, an interrupted
apply is visible on the next boot, and a database ahead of the binary is
named as a downgrade instead of silently accepted.
"""
from __future__ import annotations

import sqlite3

import pytest

from core.db.migrations import (
    STATUS_APPLIED,
    STATUS_STARTED,
    InterruptedMigrationError,
    MigrationDriftError,
    Migrator,
    checksum_of,
    checksums_match,
)

V1 = "CREATE TABLE IF NOT EXISTS thing (id TEXT PRIMARY KEY);"
V2 = "CREATE INDEX IF NOT EXISTS idx_thing ON thing(id);"
V3 = "CREATE TABLE IF NOT EXISTS other (id TEXT PRIMARY KEY);"


def _migrator(path, *pairs):
    m = Migrator(path)
    for version, description, sql in pairs:
        m.register(version, description, sql)
    return m


def _current(path):
    return _migrator(
        path, (1, "thing", V1), (2, "index", V2), (3, "other", V3)
    )


def _tables(path):
    con = sqlite3.connect(path)
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


# ─────────────────────────────────────────────── the ordinary paths


def test_a_fresh_database_reaches_current(tmp_path):
    db = tmp_path / "fresh.db"
    assert _current(db).run() == 3
    assert {"thing", "other"} <= _tables(db)
    assert _current(db).verify() == []
    # A second run is a no-op, not a re-apply.
    assert _current(db).run() == 0


def test_a_database_from_an_earlier_release_reaches_current(tmp_path):
    db = tmp_path / "old.db"
    _migrator(db, (1, "thing", V1)).run()
    assert "other" not in _tables(db)

    assert _current(db).run() == 2
    assert {"thing", "other"} <= _tables(db)
    assert _current(db).verify() == []


# ─────────────────────────────────────────────── the checks that were absent


def test_an_edited_migration_is_refused_not_skipped(tmp_path):
    """The defect this file exists for."""
    db = tmp_path / "drift.db"
    _current(db).run()

    edited = _migrator(
        db,
        (1, "thing", V1 + "\nALTER TABLE thing ADD COLUMN sneaked TEXT;"),
        (2, "index", V2),
        (3, "other", V3),
    )
    problems = edited.verify()
    assert len(problems) == 1
    assert "edited after it ran" in problems[0]

    with pytest.raises(MigrationDriftError) as caught:
        edited.run()
    assert "v1" in str(caught.value)


def test_a_database_ahead_of_the_binary_is_named_as_such(tmp_path):
    db = tmp_path / "ahead.db"
    _current(db).run()

    older_build = _migrator(db, (1, "thing", V1), (2, "index", V2))
    problems = older_build.verify()
    assert any("ahead of the code" in p for p in problems)
    with pytest.raises(MigrationDriftError):
        older_build.run()


def test_an_interrupted_apply_is_visible_on_the_next_boot(tmp_path):
    db = tmp_path / "interrupted.db"
    _migrator(db, (1, "thing", V1)).run()

    # The exact state a crash between the DDL and the ledger update leaves.
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO _aura_migrations "
        "(version, description, checksum, applied_at, status, started_at) "
        "VALUES (?, ?, ?, 0.0, ?, 1.0)",
        (2, "index", checksum_of(V2), STATUS_STARTED),
    )
    con.commit()
    con.close()

    stalled = _migrator(db, (1, "thing", V1), (2, "index", V2))
    assert any("started and never finished" in p for p in stalled.verify())

    with pytest.raises(InterruptedMigrationError):
        stalled.run(recover_interrupted=False)

    # The default is to finish it, and the ledger says so afterwards.
    assert stalled.run() == 1
    rows = {row.version: row for row in stalled.ledger()}
    assert rows[2].status == STATUS_APPLIED
    assert rows[2].applied_at > 0


def test_the_ledger_records_intent_before_the_ddl(tmp_path):
    """A crash inside the DDL must leave `started`, not nothing."""
    db = tmp_path / "half.db"
    broken = _migrator(db, (1, "thing", V1), (2, "bad", "NOT VALID SQL AT ALL;"))
    with pytest.raises(RuntimeError):
        broken.run()

    rows = {row.version: row for row in Migrator(db).ledger()}
    assert rows[2].status == STATUS_STARTED, (
        "the failed migration left no trace, so the next boot cannot tell it "
        "apart from one that never ran"
    )


# ─────────────────────────────────────────────── compatibility and hygiene


def test_a_checksum_written_by_an_earlier_build_still_verifies(tmp_path):
    """Old rows hold the first 16 characters of the same digest."""
    db = tmp_path / "legacy.db"
    _migrator(db, (1, "thing", V1)).run()

    con = sqlite3.connect(db)
    con.execute(
        "UPDATE _aura_migrations SET checksum = ? WHERE version = 1",
        (checksum_of(V1)[:16],),
    )
    con.commit()
    con.close()

    assert _migrator(db, (1, "thing", V1)).verify() == []
    assert checksums_match(checksum_of(V1)[:16], checksum_of(V1))
    assert not checksums_match(checksum_of(V1)[:16], checksum_of(V2))


def test_a_ledger_without_a_status_column_is_upgraded(tmp_path):
    """Databases written before the two-phase record still open."""
    db = tmp_path / "preupgrade.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE _aura_migrations (version INTEGER PRIMARY KEY, "
        "description TEXT NOT NULL, checksum TEXT NOT NULL, applied_at REAL NOT NULL)"
    )
    con.execute(
        "INSERT INTO _aura_migrations VALUES (1, 'thing', ?, 1.0)",
        (checksum_of(V1)[:16],),
    )
    con.execute(V1)
    con.commit()
    con.close()

    migrator = _current(db)
    assert migrator.verify() == []
    assert migrator.run() == 2
    assert {row.status for row in migrator.ledger()} == {STATUS_APPLIED}


def test_a_version_registered_twice_is_refused(tmp_path):
    migrator = Migrator(tmp_path / "dup.db")
    migrator.register(1, "thing", V1)
    with pytest.raises(ValueError):
        migrator.register(1, "thing again", V2)


def test_the_shipped_knowledge_migrator_verifies_against_itself(tmp_path):
    """The registered set is internally consistent on a fresh database."""
    from core.db.migrations import get_migrator

    db = tmp_path / "knowledge.db"
    migrator = get_migrator(db)
    assert migrator.run() == len(migrator._migrations)
    assert get_migrator(db).verify() == []
