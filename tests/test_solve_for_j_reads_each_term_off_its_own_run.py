"""The J* tool reads the three runs, opens the state log read-only, and writes the triple."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tool() -> Any:
    name = "solve_for_j_under_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "solve_for_j.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def _state_log(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute(
        "create table state_log (state_id text, version integer, parent_state_id text, "
        "transition_cause text, state_json text, timestamp real)"
    )
    identity = {"name": "Aura Luna", "core_values": ["honesty"], "personality_growth": {"openness": 0.1}}
    for index, parent in enumerate([None, "s0", "s1"]):
        connection.execute(
            "insert into state_log values (?, ?, ?, ?, ?, ?)",
            (f"s{index}", index, parent, "tick", json.dumps({"identity": identity}), float(index)),
        )
    connection.commit()
    connection.close()
    return path


def _reports(tmp: Path) -> tuple[Path, Path]:
    carrier = tmp / "carrier.json"
    carrier.write_text(
        json.dumps(
            {
                "authority": {"authoritative": False, "blockers": ["only a sample of the bipartitions was scored"]},
                "exclusion": {"status": "NOT_FOUND"},
                "placement": {"level": "L4"},
            }
        )
    )
    content = tmp / "content.json"
    content.write_text(
        json.dumps(
            {
                "classes": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
                "internal": {"0-1": 1.0, "0-2": 2.0, "1-2": 3.0},
                "internal_floor": {"0-0": 0.1, "1-1": 0.1, "2-2": 0.1},
                "verdict": "ONE_STRUCTURE",
                "authority": {"authoritative": True, "blockers": []},
            }
        )
    )
    return carrier, content


def test_the_tool_writes_the_triple_with_each_terms_own_reasons(tool: Any, tmp_path: Path) -> None:
    carrier, content = _reports(tmp_path)
    log = _state_log(tmp_path / "aura_state.db")
    out_root = tmp_path / "bridge"
    code = tool.main(
        ["--carrier", str(carrier), "--content", str(content), "--state-log", str(log), "--out-root", str(out_root)]
    )
    assert code == 0
    (written,) = out_root.glob("run_*/j_star_report.json")
    report = json.loads(written.read_text())
    assert report["status"] == "UNRESOLVED"
    assert report["unresolved"] == {"carrier": ["only a sample of the bipartitions was scored"]}
    assert report["structure"]["status"] == "IDENTIFIED" and report["structure"]["rigid"]
    assert report["lineage"]["status"] == "RECORDED"
    assert len(report["lineage"]["graph"]["stages"]) == 3
    assert report["bridge_status"]["phenomenal_bridge"] == "UNVALIDATED"


def test_a_missing_state_log_leaves_the_lineage_unmeasured_and_says_where_it_looked(tool: Any, tmp_path: Path) -> None:
    carrier, content = _reports(tmp_path)
    out_root = tmp_path / "bridge"
    tool.main(
        ["--carrier", str(carrier), "--content", str(content), "--state-log", str(tmp_path / "absent.db"), "--out-root", str(out_root)]
    )
    (written,) = out_root.glob("run_*/j_star_report.json")
    report = json.loads(written.read_text())
    assert report["lineage"]["status"] == "NOT_MEASURED"
    assert "absent.db" in report["sources"]["state_log_note"]


def test_each_solve_is_its_own_run(tool: Any, tmp_path: Path) -> None:
    carrier, content = _reports(tmp_path)
    out_root = tmp_path / "bridge"
    for _ in range(2):
        tool.main(["--carrier", str(carrier), "--content", str(content), "--state-log", str(tmp_path / "x.db"), "--out-root", str(out_root)])
    assert sorted(p.parent.name for p in out_root.glob("run_*/j_star_report.json")) == ["run_001", "run_002"]


def test_the_newest_of_several_state_logs_is_the_one_read_and_all_are_listed(tool: Any, tmp_path: Path) -> None:
    carrier, content = _reports(tmp_path)
    old = _state_log(tmp_path / "old.db")
    new_dir = tmp_path / "live"
    new_dir.mkdir()
    new = _state_log(new_dir / "aura_state.db")
    connection = sqlite3.connect(new)
    connection.execute("update state_log set timestamp = timestamp + 1000")
    connection.commit()
    connection.close()
    out_root = tmp_path / "bridge"
    tool.main(
        [
            "--carrier", str(carrier), "--content", str(content),
            "--state-log", str(old), "--state-log", str(new), "--state-log", str(tmp_path / "gone.db"),
            "--out-root", str(out_root),
        ]
    )
    (written,) = out_root.glob("run_*/j_star_report.json")
    sources = json.loads(written.read_text())["sources"]
    assert sources["state_log"] == str(new)
    assert [entry["path"] for entry in sources["state_logs_surveyed"]] == [str(old), str(new), str(tmp_path / "gone.db")]
    assert sources["state_logs_surveyed"][2]["exists"] is False
