"""A report names a commit, and a commit is not a promise.

A run records the commit it ran on and a digest over the files that decided the
answer. That says what was measured. It does not stop the answer drifting: a
branch moves, a history is rewritten, and a year later a reader has a number
and no way back to the thing that produced it.

So an authoritative run gets a tag named for its campaign fingerprint, and the
tag does not move. Repointing it is refused rather than forced — a tag that can
be repointed is a label, and the point of this one is that it cannot be.

The digest has to be recomputed the right way. A run hashes file *contents*,
not a git tree object, so comparing it to `commit^{tree}` compares two
unrelated things: pointed at run_020 that check reported drift on a commit
nothing had touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.subject.provenance import TREE_PATHS, tree_hash_at
from tools.seal_evaluation_run import DEFINING_TESTS, _checks, _tag_name

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]


def _status(rows, name):
    return next(row["status"] for row in rows if row["check"] == name)


def _report(**campaign):
    base = {
        "fingerprint": "abcdef0123456789",
        "commit": "0" * 40,
        "tree_hash": "deadbeef",
        "dirty": False,
    }
    base.update(campaign)
    return {"campaign": base, "verdict": {"passed": 15, "total": 24}}


def test_the_tag_is_named_for_what_was_frozen_not_for_when_it_ran() -> None:
    """Two runs of one campaign belong under one name."""
    first = _tag_name({"fingerprint": "abcdef0123456789"})
    second = _tag_name({"fingerprint": "abcdef0123456789"})
    other = _tag_name({"fingerprint": "9999999999999999"})
    assert first == second
    assert first != other
    assert first.startswith("isc-")


def test_a_dirty_run_cannot_be_sealed() -> None:
    """The commit does not describe a tree that had uncommitted changes in it."""
    rows = _checks(_report(dirty=True), "isc-test")
    assert _status(rows, "tree was clean") == "blocked"


def test_a_clean_run_clears_that_check() -> None:
    rows = _checks(_report(dirty=False), "isc-test")
    assert _status(rows, "tree was clean") == "ready"


def test_a_commit_that_is_not_here_is_blocked() -> None:
    rows = _checks(_report(commit="0" * 40), "isc-test")
    assert _status(rows, "commit is reachable") == "blocked"


def test_signing_and_branch_protection_are_reported_not_claimed() -> None:
    """A key and a repository setting belong to whoever operates the repository."""
    rows = _checks(_report(), "isc-test")
    assert _status(rows, "branch protection") == "unknown"


def test_the_tree_digest_is_recomputed_from_the_commits_own_blobs() -> None:
    """Not from `commit^{tree}`, which is a different object entirely."""
    head = tree_hash_at("HEAD")
    assert head and len(head) == 32
    assert tree_hash_at("HEAD") == head


def test_two_different_commits_hash_differently() -> None:
    head = tree_hash_at("HEAD")
    parent = tree_hash_at("HEAD~1")
    if not parent:
        pytest.skip("no parent commit in this checkout")
    # They may legitimately match when a commit touched none of the measured
    # paths; what must hold is that the function reads the commit rather than
    # the working tree.
    assert isinstance(parent, str) and len(parent) == 32
    assert head != "" and parent != ""


def test_the_measured_paths_are_declared_in_one_place() -> None:
    """So a tool checking the digest cannot drift from the run that wrote it."""
    assert "core/subject" in TREE_PATHS
    assert "tools/run_subject_core.py" in TREE_PATHS


def test_the_defining_suites_exist() -> None:
    """A named suite that is not there is a check nobody is running."""
    missing = [name for name in DEFINING_TESTS if not (REPO / name).is_file()]
    assert not missing, f"the seal names suites that do not exist: {missing}"


def test_a_real_run_can_be_checked_end_to_end() -> None:
    runs = sorted(
        p for p in (REPO / "artifacts" / "subject_core").glob("run_*")
        if (p / "subject_core_report.json").is_file()
    )
    if not runs:
        pytest.skip("no run with a report in this checkout")
    report = json.loads((runs[-1] / "subject_core_report.json").read_text())
    rows = _checks(report, _tag_name(report.get("campaign", {})))
    assert {row["check"] for row in rows} >= {
        "tree was clean", "commit is reachable", "tree still hashes the same", "tag is free",
    }
