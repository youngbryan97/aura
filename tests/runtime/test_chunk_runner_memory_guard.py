"""Every pytest child obeys the runner's requested memory floor."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_test_chunks as runner


def _vm_stat(free=10, inactive=20, page_size=4096):
    return (
        f"Mach Virtual Memory Statistics: (page size of {page_size} bytes)\n"
        f"Pages free: {free}.\nPages inactive: {inactive}.\n"
    )


@pytest.mark.parametrize("page_size", [4096, 16384])
def test_memory_reading_uses_free_and_inactive_pages(monkeypatch, page_size):
    monkeypatch.setattr(
        runner.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=_vm_stat(page_size=page_size)),
    )
    assert runner.free_memory_gb() == 30 * page_size / 1024**3


@pytest.mark.parametrize("output,returncode", [
    (_vm_stat(), 1),
    ("Pages free: 10.\nPages inactive: 20.\n", 0),
    ("Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 10.\n", 0),
    (_vm_stat(page_size=0), 0),
    (_vm_stat(free=-1), 0),
])
def test_incomplete_or_failed_memory_reading_is_unknown(monkeypatch, output, returncode):
    monkeypatch.setattr(
        runner.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(returncode=returncode, stdout=output),
    )
    assert runner.free_memory_gb() is None


def test_zero_available_pages_is_a_measured_zero(monkeypatch):
    monkeypatch.setattr(
        runner.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=_vm_stat(0, 0)),
    )
    assert runner.free_memory_gb() == 0.0


@pytest.mark.parametrize("available", [None, 0.0, 1.0, float("nan"), float("inf")])
def test_chunk_never_starts_without_its_memory_floor(monkeypatch, tmp_path, available):
    monkeypatch.setattr(runner, "free_memory_gb", lambda: available)
    launched = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: launched.append(a))
    progress = tmp_path / "progress.log"
    ok, message, ids = runner.run_chunk(
        1, 1, [Path("test_unused.py")], marker="", timeout_s=1,
        python="unused", extra_args=[], min_free_gb=4, progress_file=progress,
    )
    assert not ok and ids == []
    assert "REFUSED" in message and "--min-free-gb" in message
    assert not launched
    assert "REFUSED" in progress.read_text()


@pytest.mark.parametrize("available", [None, 0.0, 2.0])
def test_retry_is_unmeasured_when_its_memory_floor_is_unavailable(monkeypatch, tmp_path, available):
    node = "tests/test_example.py::test_example"
    monkeypatch.setattr(runner, "discover_test_files", lambda _: [Path(node.split("::")[0])])
    monkeypatch.setattr(runner, "working_tree_revision", lambda: "fixed-revision")
    monkeypatch.setattr(runner, "run_chunk", lambda *a, **kw: (False, "chunk failed", [node]))
    monkeypatch.setattr(runner, "free_memory_gb", lambda: available)
    launched = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: launched.append(a))
    register = tmp_path / "defects.json"
    progress = tmp_path / "progress.log"

    rc = runner.main([
        "--min-free-gb", "4", "--defect-register", str(register),
        "--progress-file", str(progress),
    ])

    assert rc == 1
    assert not launched
    record = json.loads(register.read_text())
    assert record["real_failures"] == record["order_dependent"] == []
    assert node in record["retry_refusals"]
    assert "REFUSED" in progress.read_text()


def test_each_retry_takes_a_fresh_memory_reading(monkeypatch, tmp_path):
    nodes = ["tests/test_example.py::test_a", "tests/test_example.py::test_b"]
    monkeypatch.setattr(runner, "discover_test_files", lambda _: [Path("tests/test_example.py")])
    monkeypatch.setattr(runner, "working_tree_revision", lambda: "fixed-revision")
    monkeypatch.setattr(runner, "run_chunk", lambda *a, **kw: (False, "chunk failed", nodes))
    readings = iter([8.0, 1.0])
    monkeypatch.setattr(runner, "free_memory_gb", lambda: next(readings))
    launched = []

    def retry(cmd, **kwargs):
        launched.append(cmd)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner.subprocess, "run", retry)
    register = tmp_path / "defects.json"
    assert runner.main([
        "--min-free-gb", "4", "--defect-register", str(register),
        "--progress-file", str(tmp_path / "progress.log"),
    ]) == 1
    assert len(launched) == 1 and nodes[0] in launched[0]
    record = json.loads(register.read_text())
    assert record["real_failures"] == [nodes[0]]
    assert list(record["retry_refusals"]) == [nodes[1]]


@pytest.mark.parametrize("floor", ["nan", "inf", "-1"])
def test_invalid_memory_floor_is_rejected_before_discovery(monkeypatch, floor):
    discovered = []
    monkeypatch.setattr(runner, "discover_test_files", lambda _: discovered.append(True))
    with pytest.raises(SystemExit) as exc:
        runner.main(["--min-free-gb", floor])
    assert exc.value.code == 2 and not discovered
