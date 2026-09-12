"""A raise has to carry what landed, not a sentence standing in for it.

The complexity ratchet went red on 2026-09-01 and stayed red for eleven days.
By the time anyone looked, satisfying it meant writing one `--because` line for
38,926 lines across 319 files and five new organs — which reads as a reset
rather than a record, and is why nobody wrote one. The baseline now remembers
the commit it was written at, and a raise measures that window itself.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config/cognitive_complexity_baseline.json"


def _complexity():
    spec = importlib.util.spec_from_file_location(
        "cognitive_complexity_attribution", ROOT / "tools/cognitive_complexity.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_baseline_remembers_where_it_was_written():
    baseline = json.loads(BASELINE.read_text())
    commit = str(baseline.get("at_commit", ""))
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), (
        "without a commit there is no window to measure, and a raise is a sentence again"
    )


def test_the_commit_it_names_is_one_this_repository_has():
    commit = json.loads(BASELINE.read_text())["at_commit"]
    found = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-t", commit],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if found.returncode != 0:
        pytest.skip("this checkout does not hold the baseline commit")
    assert found.stdout.strip() == "commit"


def test_nothing_landed_between_a_commit_and_itself():
    module = _complexity()
    head = module.head_commit()
    if not head:
        pytest.skip("no git here")
    assert module.landed_since(head) == {}


def test_an_unreadable_commit_reports_nothing_rather_than_raising():
    module = _complexity()
    assert module.landed_since("") == {}
    assert module.landed_since("not-a-commit") == {}


def _a_commit_far_enough_back() -> str:
    """A commit with kernel changes after it, so there is a window to measure."""
    walked = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--format=%H", "-n", "400", "--", "core/"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    commits = walked.stdout.split()
    return commits[-1] if len(commits) > 20 else ""


def test_the_window_is_measured_per_package_and_per_file():
    """Against a real range, not against whatever the baseline happens to hold.

    Reading the baseline's own commit would make this skip the moment the
    ratchet is current — green without checking anything, which is the shape
    it exists to catch.
    """
    module = _complexity()
    commit = _a_commit_far_enough_back()
    if not commit:
        pytest.skip("this checkout has too little history to measure a window")
    landed = module.landed_since(commit)
    assert landed, f"no kernel change found between {commit[:9]} and HEAD"
    assert set(landed) >= {
        "since", "net_kernel_lines", "files_touched", "per_package",
        "largest_files", "commits_by_author",
    }
    assert landed["files_touched"] >= len(landed["largest_files"])
    assert sum(landed["per_package"].values()) == landed["net_kernel_lines"], (
        "the per-package split has to add up to the total it splits"
    )
    assert all(
        row["path"].startswith("core/") for row in landed["largest_files"]
    ), "the kernel measure must only count kernel files"
    assert landed["per_package"].keys() <= set(module.KERNEL), (
        "a package outside the kernel has no business in the kernel measure"
    )


def test_the_last_raise_carries_its_measurement():
    history = json.loads(BASELINE.read_text()).get("history", [])
    assert history, "a ratchet with no history of its raises is a number"
    latest = history[-1]
    assert latest["because"].strip(), "a raise without a reason is a reset"
    if "landed" not in latest:
        pytest.skip("this raise predates the measurement")
    landed = latest["landed"]
    assert landed["net_kernel_lines"] > 0
    assert landed["files_touched"] > 0
    assert sum(landed["per_package"].values()) == landed["net_kernel_lines"]
