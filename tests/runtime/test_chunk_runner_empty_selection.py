"""Deselected chunks supply no passing test evidence."""
from __future__ import annotations

import json

import pytest

from tools import run_test_chunks as runner


def _suite(tmp_path, *, offline_second=False):
    for name, live in (("a", True), ("b", not offline_second)):
        marker = "@pytest.mark.live\n" if live else ""
        (tmp_path / f"test_{name}.py").write_text(
            "import pytest\n\n" + marker + "def test_example():\n    assert True\n"
        )


def _run(tmp_path, *args):
    return runner.main([
        "--tests-dir", str(tmp_path), "--chunks", "2",
        "--chunk-timeout", "30", "--retry-timeout", "30",
        "--defect-register", str(tmp_path / "defects.json"),
        "--progress-file", str(tmp_path / "progress.log"),
        *args, "--", "-p", "no:randomly",
    ])


@pytest.mark.parametrize("selection", [[], ["--only-chunks", "1"]])
def test_a_run_with_every_test_deselected_has_no_passing_verdict(tmp_path, capsys, selection):
    _suite(tmp_path)
    assert _run(tmp_path, *selection) == 1
    record = json.loads((tmp_path / "defects.json").read_text())
    assert record["chunk_outcomes"] == {
        "passed": 0, "failed": 0, "empty": 1 if selection else 2,
    }
    output = capsys.readouterr().out
    assert "no selected tests ran" in output
    assert "chunks passed" not in output


def test_an_empty_chunk_does_not_stop_a_later_nonempty_chunk(tmp_path, capsys):
    _suite(tmp_path, offline_second=True)
    assert _run(tmp_path) == 0
    record = json.loads((tmp_path / "defects.json").read_text())
    assert record["chunk_outcomes"] == {"passed": 1, "failed": 0, "empty": 1}
    output = capsys.readouterr().out
    assert "1 chunks passed" in output
    assert "2 chunks passed" not in output
