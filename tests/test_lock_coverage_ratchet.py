"""Lockdep's coverage is the scope of its claim, so it may only grow.

``lockdep_report()["splats"] == 0`` backs a registered claim about lock
ordering. Coverage was never measured, so a zero meaning "clean across the
locks we watch" read as "clean across the runtime" — and capability_engine
was wrapped only *after* it deadlocked the boot path.

Converting every raw lock at once would be a large untested change to the
most deadlock-sensitive code in the system. This makes the direction
one-way instead.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools import lint_lock_coverage as coverage

BASELINE = Path(coverage.BASELINE)


def test_raw_lock_constructions_never_increase():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    current = coverage.measure()

    allowed = int(baseline["raw_lock_calls"])
    actual = int(current["raw_lock_calls"])

    if actual > allowed:
        previous = baseline.get("raw_by_file") or {}
        grew = [
            f"{name}: {int(previous.get(name, 0))} -> {count}"
            for name, count in sorted(current["raw_by_file"].items())
            if count > int(previous.get(name, 0))
        ]
        raise AssertionError(
            f"raw lock constructions rose {allowed} -> {actual}. Use checked_lock / "
            "checked_async_lock (core/runtime/lockdep.py), or instrument(name) to "
            "adopt an existing lock — lockdep only sees the locks it wraps.\n  "
            + "\n  ".join(grew)
        )


def test_the_baseline_is_refreshed_when_coverage_improves():
    """A stale baseline hides progress and lets a later regression hide in it."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    current = coverage.measure()
    assert int(current["raw_lock_calls"]) == int(baseline["raw_lock_calls"]), (
        "raw lock count moved; refresh with "
        "`python tools/lint_lock_coverage.py --write-baseline`"
    )


def test_coverage_is_reported_so_the_claim_can_state_its_scope():
    current = coverage.measure()
    checked = int(current["files_using_checked_locks"])
    raw = int(current["files_with_raw_locks"])
    assert checked > 0, "no file uses checked locks; lockdep would see nothing"
    assert raw + checked > 100, "the scanner stopped seeing the codebase"


def test_the_refresh_command_cannot_loosen_the_ratchet(tmp_path, monkeypatch):
    """A ratchet whose maintenance path can raise it is not a ratchet.

    --write-baseline wrote whatever it measured. A run after new raw locks
    landed would have recorded the debt as the new normal, and the gate would
    have passed on it forever — the failure this file exists to prevent,
    reached through its own refresh command.
    """
    import json

    import tools.lint_lock_coverage as gate

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"raw_lock_calls": 1, "raw_by_file": {}}) + "\n")
    monkeypatch.setattr(gate, "BASELINE", baseline)

    assert gate.main(["--write-baseline"]) == 1, "a rise was written"
    assert json.loads(baseline.read_text())["raw_lock_calls"] == 1


def test_the_refresh_command_still_tightens(tmp_path, monkeypatch):
    import json

    import tools.lint_lock_coverage as gate

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"raw_lock_calls": 10_000, "raw_by_file": {}}) + "\n"
    )
    monkeypatch.setattr(gate, "BASELINE", baseline)

    assert gate.main(["--write-baseline"]) == 0
    assert json.loads(baseline.read_text())["raw_lock_calls"] < 10_000
