"""A backup that has never been restored is a hope. A drill that cannot fail is a slogan.

`make restore-test` printed "Simulating state corruption..." and simulated
nothing: it restored an archive over a tree that was already correct and said
the drill passed. And `make backup` had raised ModuleNotFoundError on every
run since 2026-08-05, so the newest archive it could have restored was from
July. And that archive held the repository's `data/`, not the live runtime's
`~/.aura/data`, so the conversations, memory and state the desktop writes
were in no backup at all. Three ways to have no backup while a green target
said otherwise.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import state_backup  # noqa: E402


def _store(path: Path, rows: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("CREATE INDEX idx_body ON t(body)")
        conn.executemany("INSERT INTO t (body) VALUES (?)", [(f"row {i} " * 40,) for i in range(rows)])


@pytest.fixture
def a_small_world(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    live = tmp_path / "live"
    _store(repo / "data" / "campaign.db")
    (repo / "data" / "notes.json").write_text('{"kept": true}')
    _store(live / "aura_state.db")
    _store(live / "conversations.db")
    (live / ".continuity_hmac.key").write_bytes(b"k" * 64)
    _store(live / "memory" / "quarantine" / "ledger.db.corrupt.1", rows=10)
    with sqlite3.connect(live / "memory" / "quarantine" / "ledger.db.corrupt.1") as conn:
        pass
    # Damage the quarantined one for real, the way the runtime found it.
    q = live / "memory" / "quarantine" / "ledger.db.corrupt.1"
    with q.open("r+b") as fh:
        fh.seek(q.stat().st_size // 2)
        fh.write(b"\xff" * 512)
    monkeypatch.setattr(state_backup, "_port_serving", lambda port=8000: False)
    out = tmp_path / "backups"
    return repo, live, out


def test_the_archive_carries_the_live_data_directory(a_small_world) -> None:
    repo, live, out = a_small_world
    archive = state_backup.create_backup(repo, out, live_data=live)
    import tarfile

    with tarfile.open(archive) as tar:
        names = set(tar.getnames())
    assert "aura_data/aura_state.db" in names
    assert "aura_data/conversations.db" in names
    assert "data/campaign.db" in names


def test_a_file_over_the_bound_is_skipped_and_named(a_small_world, monkeypatch) -> None:
    repo, live, out = a_small_world
    monkeypatch.setattr(state_backup, "PER_FILE_BOUND_BYTES", 1_000_000)
    (live / "state").mkdir()
    (live / "state" / "huge.bin").write_bytes(b"\0" * 2_000_000)
    state_backup.create_backup(repo, out, live_data=live)
    import json

    entry = json.loads((out / "manifest.jsonl").read_text().splitlines()[-1])
    assert [s["path"] for s in entry["skipped_over_bound"]] == ["aura_data/state/huge.bin"]


def test_the_drill_damages_restores_and_proves_it(a_small_world, capsys) -> None:
    repo, live, out = a_small_world
    state_backup.create_backup(repo, out, live_data=live)
    assert state_backup.drill(out) == 0
    said = capsys.readouterr().out
    assert "failed quick_check while damaged" in said
    assert "restore drill passed" in said


def test_a_quarantined_store_is_carried_and_not_counted_against_the_archive(a_small_world) -> None:
    repo, live, out = a_small_world
    state_backup.create_backup(repo, out, live_data=live)
    assert state_backup.verify_backup(out) == 0


def test_restore_puts_live_data_back_where_the_runtime_reads_it(a_small_world, tmp_path) -> None:
    repo, live, out = a_small_world
    archive = state_backup.create_backup(repo, out, live_data=live)
    target_repo = tmp_path / "restored_repo"
    target_live = tmp_path / "restored_live"
    target_repo.mkdir()
    target_live.mkdir()
    assert state_backup.restore_archive(archive, target_repo, target_live) == 0
    assert (target_live / "aura_state.db").exists()
    assert (target_repo / "data" / "campaign.db").exists()
    assert not (target_repo / "aura_data").exists(), "live data must not land under the repository"
