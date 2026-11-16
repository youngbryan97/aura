"""An empty requested selection or missing requested register cannot pass."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_test_chunks as runner


@pytest.fixture
def run(monkeypatch, tmp_path):
    discovered, launched = [], []

    def discover(_):
        discovered.append(True)
        return [Path("test_example.py")]

    def chunk(index, *args, **kwargs):
        launched.append(index)
        return True, "chunk passed", []

    monkeypatch.setattr(runner, "discover_test_files", discover)
    monkeypatch.setattr(runner, "run_chunk", chunk)
    monkeypatch.setattr(runner, "working_tree_revision", lambda: "fixed-revision")

    def invoke(*args):
        return runner.main(["--progress-file", str(tmp_path / "progress.log"), *args])

    return invoke, discovered, launched


@pytest.mark.parametrize("selection", ["", ",", ",,", " , , "])
def test_an_explicit_empty_selection_starts_no_discovery_or_test_work(run, selection):
    invoke, discovered, launched = run
    with pytest.raises(SystemExit) as exc:
        invoke("--only-chunks", selection)
    assert exc.value.code == 2
    assert not discovered and not launched


def test_omitting_the_selection_keeps_the_normal_run(run):
    invoke, discovered, launched = run
    assert invoke() == 0
    assert discovered == [True] and launched == [1]


@pytest.mark.parametrize("blocked", ["parent", "destination"])
def test_a_requested_register_write_failure_fails_the_run(run, tmp_path, capsys, blocked):
    invoke, _, launched = run
    if blocked == "parent":
        parent = tmp_path / "regular-file"
        parent.write_text("keep this")
        target = parent / "register.json"
    else:
        target = tmp_path / "directory"
        target.mkdir()

    assert invoke("--defect-register", str(target)) == 1
    assert launched == [1]
    assert "defect register write failed" in capsys.readouterr().err
    if blocked == "parent":
        assert parent.read_text() == "keep this"
    else:
        assert target.is_dir()


def test_a_written_register_keeps_the_run_green(run, tmp_path, monkeypatch):
    invoke, _, launched = run
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    target = tmp_path / "new" / "register.json"
    assert invoke("--defect-register", str(target)) == 0
    assert launched == [1]
    record = json.loads(target.read_text())
    assert record["schema"] == "aura.test_defect_register.v1"
    assert record["source_revision"]["measured"]
    assert target.stat().st_mode & 0o777 == 0o600


def test_a_failed_register_replacement_keeps_the_previous_report(run, tmp_path, monkeypatch, capsys):
    import os

    invoke, _, launched = run
    target = tmp_path / "register.json"
    previous = b'{"schema":"aura.test_defect_register.v1","old_report":true}\n'
    target.write_bytes(previous)
    original = os.replace

    def replace(source, destination, **kwargs):
        if Path(destination) == target:
            assert Path(source) != target
            assert json.loads(Path(source).read_bytes())["source_revision"]["measured"]
            raise OSError("replacement refused")
        return original(source, destination, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    assert invoke("--defect-register", str(target)) == 1
    assert launched == [1]
    assert "replacement refused" in capsys.readouterr().err
    assert target.read_bytes() == previous


@pytest.mark.parametrize("continue_on_failure, attempted", [(False, 1), (True, 2)])
def test_a_missing_interpreter_leaves_a_failed_chunk_record(tmp_path, capsys, continue_on_failure, attempted):
    for name in ("a", "b"):
        (tmp_path / f"test_{name}.py").write_text("def test_example():\n    assert True\n")
    register = tmp_path / "defects.json"
    progress = tmp_path / "progress.log"
    args = [
        "--tests-dir", str(tmp_path), "--chunks", "2", "--python", str(tmp_path / "missing-python"),
        "--defect-register", str(register), "--progress-file", str(progress),
    ]
    if continue_on_failure:
        args.append("--continue-on-failure")

    assert runner.main(args) == 1
    record = json.loads(register.read_text())
    assert record["chunk_outcomes"] == {"passed": 0, "failed": attempted, "empty": 0}
    assert record["real_failures"] == record["order_dependent"] == []
    assert len(record["unisolated_chunks"]) == attempted
    assert all("could not start" in chunk and "FileNotFoundError" in chunk
               for chunk in record["unisolated_chunks"])
    assert progress.read_text().count("LAUNCH ERROR") == attempted
    assert f"{attempted} chunk(s) failed without isolatable test ids" in capsys.readouterr().out
