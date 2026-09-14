"""The retention report must tell a store something reads from one nothing does.

Q02. The 46GB orphan at data/state/aura_state.db was written by the proxy's
standalone fallback for two months and read by nothing for five; the live
store beside it is 15MB and pruned. A report that could not tell them apart
would list both as "state" and leave the owner none the wiser.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import disk_retention  # noqa: E402


def _state_log(path: Path, *, rows: int, written_at: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE state_log (state_id TEXT PRIMARY KEY, version INTEGER, "
            "parent_state_id TEXT, transition_cause TEXT, state_json TEXT, timestamp REAL)"
        )
        for i in range(rows):
            conn.execute(
                "INSERT INTO state_log VALUES (?, ?, ?, ?, ?, ?)",
                (f"s{i}", i, None, "test", "{}", written_at),
            )


def test_the_vaults_store_is_live_and_the_orphan_is_not(tmp_path, monkeypatch) -> None:
    home = tmp_path / "aura"
    monkeypatch.setattr(disk_retention, "AURA_HOME", home)
    _state_log(home / "data" / "aura_state.db", rows=3, written_at=time.time())
    _state_log(home / "data" / "state" / "aura_state.db", rows=700, written_at=time.time() - 160 * 86400)

    rows = {Path(r["path"]).relative_to(home).as_posix(): r for r in disk_retention.state_stores()}
    assert rows["data/aura_state.db"]["status"] == "live"
    assert "prune" in rows["data/aura_state.db"]["bound"]
    orphan = rows["data/state/aura_state.db"]
    assert orphan["status"] == "unreferenced"
    assert orphan["rows"] == 700
    assert orphan["last_write_days_ago"] == pytest.approx(160, abs=1)


def test_the_report_removes_nothing(tmp_path, monkeypatch) -> None:
    home = tmp_path / "aura"
    monkeypatch.setattr(disk_retention, "AURA_HOME", home)
    orphan = home / "data" / "state" / "aura_state.db"
    _state_log(orphan, rows=1, written_at=time.time() - 200 * 86400)
    before = orphan.stat().st_size
    disk_retention.state_stores()
    assert orphan.exists() and orphan.stat().st_size == before


def test_a_fusion_a_manifest_names_is_live_and_one_nothing_names_is_not(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    fused = repo / "training" / "fused-model"
    for name in ("named-fusion", "orphan-fusion"):
        (fused / name).mkdir(parents=True)
        (fused / name / "weights.bin").write_bytes(b"\\0" * (disk_retention.GIB // 2 + 1))
    (fused / "active.json.rollback").write_text('{"active_model_path": "training/fused-model/named-fusion"}')
    (repo / "core").mkdir()
    monkeypatch.setattr(disk_retention, "REPO_ROOT", repo)

    rows = {Path(r["path"]).name: r for r in disk_retention.fusions_and_adapters()}
    assert rows["named-fusion"]["status"] == "live"
    assert rows["orphan-fusion"]["status"] == "unreferenced"
