"""A store moved out of the state root was moved out of the fork with it.

The fork held every file under the run's state root. Thirty-eight variables can
put a store somewhere else, and under the test suite five of them point at a
per-test directory beside the root, the memory store among them. A memory arm
run after a tool-use arm recalled up to 0.13 differently from the same arm run
after a conversation arm, because the memories the tool-use arm wrote sat in
that directory through every restore.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from core.config import config
from core.subject import snapshot as fork

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
PATH_VARIABLE = re.compile(r'"(AURA_[A-Z0-9_]*(?:ROOT|DIR|PATH|DB|HOME|FILE))"')


@pytest.fixture
def root(monkeypatch, tmp_path: Path) -> Path:
    state = tmp_path / "run_001" / "state"
    state.mkdir(parents=True)
    monkeypatch.setenv("AURA_STATE_ROOT", str(state))
    for name in (*fork.REDIRECTED_DIRECTORIES, *fork.REDIRECTED_FILES):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config.paths, "home_dir_override", state)
    monkeypatch.setattr(fork, "_STORE_CACHE", {})
    return state


def test_every_path_variable_the_tree_reads_is_followed_or_says_why_not() -> None:
    read: set[str] = set()
    for path in (REPO / "core").rglob("*.py"):
        read.update(PATH_VARIABLE.findall(path.read_text(encoding="utf-8", errors="ignore")))
    directories, files, excused = set(fork.REDIRECTED_DIRECTORIES), set(fork.REDIRECTED_FILES), set(fork.NOT_STORES)
    assert not (directories & files) and not ((directories | files) & excused)
    listed = directories | files | excused
    assert not read - listed, f"path variables the fork neither follows nor explains: {sorted(read - listed)}"
    assert not listed - read, f"listed variables nothing reads any more: {sorted(listed - read)}"


def test_a_memory_written_beside_the_root_is_gone_after_the_restore(root: Path, monkeypatch, tmp_path: Path) -> None:
    beside = tmp_path / "aura-runtime"
    monkeypatch.setenv("AURA_TEST_RUNTIME_ROOT", str(beside))
    episodic = beside / "memory" / "episodic"
    episodic.mkdir(parents=True)
    (episodic / "mem-before.json").write_text("{}")
    saved = fork._store_state()
    (episodic / "mem-during-the-arm.json").write_text('{"what": "the tool-use arm"}')
    fork._restore_stores(saved)
    assert sorted(p.name for p in episodic.iterdir()) == ["mem-before.json"]


def test_a_database_one_variable_names_comes_back(root: Path, monkeypatch, tmp_path: Path) -> None:
    database = tmp_path / "elsewhere" / "chat_delivery.sqlite3"
    database.parent.mkdir()
    monkeypatch.setenv("AURA_CHAT_DELIVERY_DB", str(database))
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE delivered (id INTEGER PRIMARY KEY)")
    connection.executemany("INSERT INTO delivered DEFAULT VALUES", [()] * 3)
    connection.commit()
    saved = fork._store_state()
    connection.executemany("INSERT INTO delivered DEFAULT VALUES", [()] * 5)
    connection.commit()
    fork._restore_stores(saved)
    assert connection.execute("SELECT COUNT(*) FROM delivered").fetchone()[0] == 3
    connection.close()


def test_a_store_the_arm_created_where_there_was_none_is_removed(root: Path, monkeypatch, tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    queue = tmp_path / "queues" / "persist-retry-queue.jsonl"
    queue.parent.mkdir()
    monkeypatch.setenv("AURA_RECEIPT_ROOT", str(receipts))
    monkeypatch.setenv("AURA_MEMORY_PERSIST_RETRY_QUEUE_PATH", str(queue))
    saved = fork._store_state()
    (receipts / "memory_write").mkdir(parents=True)
    (receipts / "memory_write" / "receipt.json").write_text("{}")
    queue.write_text('{"retry": 1}\n')
    fork._restore_stores(saved)
    assert not receipts.exists()
    assert not queue.exists()


def test_a_place_inside_another_is_held_once(root: Path, monkeypatch, tmp_path: Path) -> None:
    beside = tmp_path / "aura-runtime"
    monkeypatch.setenv("AURA_TEST_RUNTIME_ROOT", str(beside))
    monkeypatch.setenv("AURA_RECEIPT_ROOT", str(beside / "receipts"))
    monkeypatch.setenv("AURA_ALLOSTASIS_DIR", str(root / "data" / "allostasis"))
    assert sorted(fork._store_places(root)) == sorted([(root, True), (beside, True)])


def test_the_configured_home_is_forked_when_it_is_not_the_root(root: Path, monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    engrams = home / "engrams"
    engrams.mkdir(parents=True)
    monkeypatch.setattr(config.paths, "home_dir_override", home)
    saved = fork._store_state()
    (engrams / "written-by-the-arm.json").write_text("{}")
    fork._restore_stores(saved)
    assert list(engrams.iterdir()) == []
