"""An arm left its memories on disk for the next arm to recall.

The fork carried every service's Python state and none of the files those
services write and read back. One arm of three turns under the memory workload
changed seventeen files in its own state root, including three new episodic
memories, the outcome ledger, the identity chronicle and tensions, and the
restore put back none of them. So every sham arm began with what the displaced
arm had just stored.

The fork now carries the run's own state files. A quick run's root was 24 MB
over 3,178 files, so a snapshot stats every file and reads only the ones that
moved; SQLite databases are imaged through the backup API, which is what lets a
restore land under a connection that is still open.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.subject import snapshot as fork

pytestmark = pytest.mark.unit


@pytest.fixture
def root(monkeypatch, tmp_path: Path) -> Path:
    state = tmp_path / "run_001" / "state"
    state.mkdir(parents=True)
    monkeypatch.setenv("AURA_STATE_ROOT", str(state))
    monkeypatch.setattr(fork, "_STORE_CACHE", {})
    return state


def _database(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS episodic (id INTEGER PRIMARY KEY, what TEXT)")
    connection.executemany("INSERT INTO episodic (what) VALUES (?)", [(f"row {i}",) for i in range(rows)])
    connection.commit()
    connection.close()


def test_plain_state_files_come_back_with_their_bytes_and_stamp(root: Path) -> None:
    (root / "data").mkdir()
    tensions = root / "data" / "tensions.json"
    tensions.write_text('{"open": 1}')
    ledger = root / "data" / "executive_ledger.jsonl"
    ledger.write_text('{"turn": 1}\n')
    before = tensions.stat().st_mtime_ns
    saved = fork._store_state()
    tensions.write_text('{"open": 4}')
    with ledger.open("a") as handle:
        handle.write('{"turn": 2}\n')
    fork._restore_stores(saved)
    assert tensions.read_text() == '{"open": 1}'
    assert ledger.read_text() == '{"turn": 1}\n'
    assert tensions.stat().st_mtime_ns == before


def test_a_memory_the_arm_created_is_gone_after_the_restore(root: Path) -> None:
    episodic = root / "memory" / "episodic"
    episodic.mkdir(parents=True)
    (episodic / "mem-before.json").write_text("{}")
    saved = fork._store_state()
    (episodic / "mem-during-the-arm.json").write_text('{"what": "the displaced arm"}')
    fork._restore_stores(saved)
    assert sorted(p.name for p in episodic.iterdir()) == ["mem-before.json"]


def test_database_rows_come_back_through_a_connection_held_open(root: Path) -> None:
    database = root / "data" / "outcome_ledger.db"
    _database(database, rows=3)
    held = sqlite3.connect(database)
    saved = fork._store_state()
    held.executemany("INSERT INTO episodic (what) VALUES (?)", [("arm",)] * 5)
    held.commit()
    assert held.execute("SELECT COUNT(*) FROM episodic").fetchone()[0] == 8
    fork._restore_stores(saved)
    assert held.execute("SELECT COUNT(*) FROM episodic").fetchone()[0] == 3
    held.close()


def test_a_file_that_did_not_move_is_not_read_again(root: Path, monkeypatch) -> None:
    (root / "substrate_state.npy").write_bytes(b"x" * 1024)
    fork._store_state()
    reads = []
    original = Path.read_bytes

    def counting(self):
        reads.append(self.name)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    fork._store_state()
    assert "substrate_state.npy" not in reads


def test_one_snapshot_restores_as_often_as_a_trial_needs(root: Path) -> None:
    """A trial restores one snapshot before each of its arms."""
    note = root / "self_model.json"
    note.write_text("before")
    saved = fork._store_state()
    for arm in ("plain", "again", "moved", "outside"):
        note.write_text(f"after the {arm} arm")
        fork._restore_stores(saved)
        assert note.read_text() == "before"


def test_a_process_that_was_not_isolated_forks_no_state_files(monkeypatch) -> None:
    monkeypatch.delenv("AURA_STATE_ROOT", raising=False)
    assert fork._store_state() is None
    fork._restore_stores(None)


def test_a_root_too_large_to_be_one_runs_own_refuses(root: Path, monkeypatch) -> None:
    (root / "big.bin").write_bytes(b"x" * 64)
    monkeypatch.setattr(fork, "STORE_BOUND_BYTES", 16)
    with pytest.raises(RuntimeError, match="not one run"):
        fork._store_state()


def test_a_restore_into_a_different_root_writes_nothing(root: Path, monkeypatch, tmp_path: Path) -> None:
    (root / "tensions.json").write_text("before")
    saved = fork._store_state()
    other = tmp_path / "run_002" / "state"
    other.mkdir(parents=True)
    monkeypatch.setenv("AURA_STATE_ROOT", str(other))
    fork._restore_stores(saved)
    assert list(other.iterdir()) == []


def test_a_write_ahead_store_keeps_its_journal_mode_after_a_restore(root: Path) -> None:
    """A write-ahead image cannot be deserialized as it is, so the fork keeps it
    with the version bytes rewritten, and has to put the mode back."""
    database = root / "data" / "goal_lifecycle.db"
    _database(database, rows=2)
    held = sqlite3.connect(database)
    assert held.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    saved = fork._store_state()
    held.execute("INSERT INTO episodic (what) VALUES ('arm')")
    held.commit()
    fork._restore_stores(saved)
    assert held.execute("SELECT COUNT(*) FROM episodic").fetchone()[0] == 2
    held.close()
    again = sqlite3.connect(database)
    assert again.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    again.close()


def test_a_restore_that_fails_says_so(root: Path, monkeypatch) -> None:
    """Skipping a failed restore is how every database stayed unrestored."""
    database = root / "data" / "outcome_ledger.db"
    _database(database, rows=1)
    saved = fork._store_state()
    writer = sqlite3.connect(database)
    writer.execute("INSERT INTO episodic (what) VALUES ('arm')")
    writer.commit()
    writer.close()

    def refuses(*_args, **_kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(fork, "_sqlite_restore", refuses)
    with pytest.raises(RuntimeError, match="could not restore"):
        fork._restore_stores(saved)


def test_a_file_named_like_a_database_that_is_not_one_is_copied_as_bytes(root: Path) -> None:
    fake = root / "notes.db"
    fake.write_bytes(b"not a database at all")
    saved = fork._store_state()
    fake.write_bytes(b"changed by the arm")
    fork._restore_stores(saved)
    assert fake.read_bytes() == b"not a database at all"
