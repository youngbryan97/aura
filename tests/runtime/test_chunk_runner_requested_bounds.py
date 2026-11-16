"""Invalid resource bounds must fail before test discovery or launch."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools import run_test_chunks as runner


@pytest.fixture
def run(monkeypatch, tmp_path):
    discovered, launched = [], []

    def discover(_):
        discovered.append(True)
        return [Path("test_example.py")]

    def chunk(*args, **kwargs):
        launched.append(True)
        return True, "chunk passed", []

    monkeypatch.setattr(runner, "discover_test_files", discover)
    monkeypatch.setattr(runner, "run_chunk", chunk)
    monkeypatch.setattr(runner, "working_tree_revision", lambda: "fixed-revision")

    def invoke(*args):
        return runner.main(["--progress-file", str(tmp_path / "progress.log"), *args])

    return invoke, discovered, launched


@pytest.mark.parametrize("count", ["0", "-1", "-40"])
def test_a_nonpositive_chunk_count_cannot_collapse_the_suite_into_one_process(run, count):
    invoke, discovered, launched = run
    with pytest.raises(SystemExit) as exc:
        invoke("--chunks", count)
    assert exc.value.code == 2
    assert not discovered and not launched


@pytest.mark.parametrize("flag", ["--chunk-timeout", "--retry-timeout"])
@pytest.mark.parametrize("seconds", ["0", "-1", "nan", "inf", "-inf"])
def test_a_timeout_must_be_finite_and_positive(run, flag, seconds):
    invoke, discovered, launched = run
    with pytest.raises(SystemExit) as exc:
        invoke(f"{flag}={seconds}")
    assert exc.value.code == 2
    assert not discovered and not launched


def test_finite_positive_resource_bounds_still_allow_the_run(run):
    invoke, discovered, launched = run
    assert invoke("--chunks", "40", "--chunk-timeout", "0.5", "--retry-timeout", "2") == 0
    assert discovered == [True] and launched == [True]
