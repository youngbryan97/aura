"""A run that spans several trees is not one register.

The chunked suite takes hours — 30 chunks in 314 minutes on 2026-09-18,
10.5 minutes each — and this repository is a shared checkout that several
agents commit to. Twenty-six commits landed between the first chunk and
the thirty-third of that run. Chunks either side of a commit tested
different code, so the 169 collected failures were a mixture across 26
trees: a green chunk said nothing about the tree the red chunk found, and
a failure could have been fixed hours before the summary was read.

Nothing in the runner said so. It reported a clean-looking list.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_test_chunks.py"


def _runner_source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_the_runner_can_name_the_tree_it_is_reading():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import run_test_chunks
    finally:
        sys.path.pop(0)

    revision = run_test_chunks.working_tree_revision()
    assert isinstance(revision, str)
    # This repository is a git checkout, so it must resolve to something.
    assert revision, "the runner cannot tell which revision it is about to test"


def test_an_uncommitted_edit_counts_as_a_different_tree():
    """A dirty tree changes what runs as surely as a commit does."""

    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import run_test_chunks
    finally:
        sys.path.pop(0)

    source = inspect.getsource(run_test_chunks.working_tree_revision)
    assert "--porcelain" in source
    assert "-dirty" in source


def test_the_run_records_the_revision_it_began_on():
    source = _runner_source()
    assert "RUN begins revision=" in source


def test_a_moving_tree_is_reported_while_the_run_is_still_readable():
    source = _runner_source()
    # Said when it happens...
    assert "MOVED chunk" in source
    assert "tested different code" in source
    # ...and again in the summary, which is where a reader arrives hours
    # later and would otherwise see a list with no warning on it.
    assert "THE TREE MOVED" in source
    assert "worktree pinned to a revision" in source


def test_a_run_on_one_tree_says_that_too():
    """The absence of a warning must be a statement, not a silence."""

    source = _runner_source()
    assert "unchanged for the whole run" in source


def test_the_revision_probe_never_raises():
    """It runs inside a loop that must not die over bookkeeping."""

    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import run_test_chunks
    finally:
        sys.path.pop(0)

    source = inspect.getsource(run_test_chunks.working_tree_revision)
    assert "except (OSError, subprocess.SubprocessError, ValueError)" in source
    assert "timeout=" in source


def test_the_probe_reads_the_repository_not_the_cwd():
    """A chunk runs with cwd=ROOT, but the probe must not depend on that."""

    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import run_test_chunks
    finally:
        sys.path.pop(0)

    source = inspect.getsource(run_test_chunks.working_tree_revision)
    assert '"-C", str(ROOT)' in source


def test_the_runner_still_starts():
    done = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-1500:]
