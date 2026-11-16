"""A retry classifies the original failure and cannot erase it."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_test_chunks as runner

NODE = "tests/test_example.py::test_example"


@pytest.fixture
def exercise(monkeypatch, tmp_path):
    def run(results, retry_code=0):
        files = [Path(f"test_{i}.py") for i in range(len(results))]
        monkeypatch.setattr(runner, "discover_test_files", lambda _: files)
        monkeypatch.setattr(runner, "split_chunks", lambda *a: [[file] for file in files])
        monkeypatch.setattr(runner, "working_tree_revision", lambda: "fixed-revision")
        monkeypatch.setattr(runner, "free_memory_gb", lambda: 8.0)
        chunks = iter(results)
        monkeypatch.setattr(runner, "run_chunk", lambda *a, **kw: next(chunks))
        monkeypatch.setattr(
            runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=retry_code),
        )
        register = tmp_path / "defects.json"
        code = runner.main([
            "--continue-on-failure", "--defect-register", str(register),
            "--progress-file", str(tmp_path / "progress.log"),
        ])
        return code, json.loads(register.read_text())

    return run


def test_an_order_dependent_failure_keeps_the_run_red(exercise, capsys):
    code, register = exercise([(False, "chunk failed", [NODE])])
    assert register["order_dependent"] == [NODE]
    assert register["real_failures"] == []
    assert code == 1
    assert "all 1 chunks passed" not in capsys.readouterr().out


def test_a_failed_chunk_with_no_ids_cannot_hide_beside_an_order_defect(exercise, capsys):
    code, register = exercise([
        (False, "chunk 1 KILLED", []),
        (False, "chunk 2 failed", [NODE]),
    ])
    assert register["order_dependent"] == [NODE]
    assert code == 1
    assert "1 chunk(s) failed without isolatable test ids" in capsys.readouterr().out


@pytest.mark.parametrize("retry_code", [2, 3, 4, 5, -9])
def test_a_retry_that_did_not_run_the_test_is_unmeasured(exercise, retry_code):
    code, register = exercise([(False, "chunk failed", [NODE])], retry_code)
    assert code == 1
    assert register["real_failures"] == register["order_dependent"] == []
    assert str(retry_code) in register["retry_errors"][NODE]


def test_a_failure_reproduced_alone_is_still_a_real_failure(exercise):
    code, register = exercise([(False, "chunk failed", [NODE])], 1)
    assert code == 1
    assert register["real_failures"] == [NODE]


def test_only_a_run_of_passing_chunks_is_green(exercise):
    code, register = exercise([(True, "chunk 1 passed", []), (True, "chunk 2 passed", [])])
    assert code == 0
    assert register["order_dependent"] == register["real_failures"] == []


def test_a_retry_launch_error_preserves_the_register(exercise, monkeypatch):
    def cannot_start(*args, **kwargs):
        raise OSError("process limit reached")

    original = runner.memory_refusal

    def admit(*args):
        monkeypatch.setattr(runner.subprocess, "run", cannot_start)
        return original(*args)

    monkeypatch.setattr(runner, "memory_refusal", admit)
    code, register = exercise([(False, "chunk failed", [NODE])])
    assert code == 1
    assert register["real_failures"] == register["order_dependent"] == []
    assert "process limit reached" in register["retry_errors"][NODE]
