"""A ground read NOT_MEASURED and the evidence was on disk the whole time.

`solve_for_j` reads five grounds from the runs it is given, and a ground nobody
gave it reads NOT_MEASURED. A ground it WAS given and could not read read the
same thing, in the same words, so the two were indistinguishable.

MEASURED on the decisive chain of 24 October. It resolved its paths with a glob
that ran before the campaign had written its report, so `--campaign` arrived as
an empty string, `Path("")` resolved to the working directory, and the markers
ground reported "no campaign was read". The campaign's report had been on disk
since 21:41 and J* ran at 23:05. Re-solved with the file passed by hand, the
markers ground holds: perturbational_complexity, reentry and global_access all
pass, and the bridge went from three grounds holding to four.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]


def _solve(tmp_path: Path, *args: str) -> tuple[str, dict]:
    out = tmp_path / "out"
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "solve_for_j.py"), "--out-root", str(out), *args],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    written = sorted(out.glob("run_*/j_star_report.json"))
    report = json.loads(written[-1].read_text()) if written else {}
    return done.stdout + done.stderr, report


def test_an_empty_path_is_named_rather_than_read_as_absent(tmp_path):
    said, report = _solve(tmp_path, "--campaign", "")
    assert "campaign" in report.get("unread_sources", {}), report.get("unread_sources")
    assert "!!" in said and "campaign" in said, said[-400:]


def test_a_path_with_nothing_at_it_is_named(tmp_path):
    missing = tmp_path / "not_here" / "subject_core_report.json"
    said, report = _solve(tmp_path, "--reports", str(missing))
    assert "reports" in report.get("unread_sources", {})
    assert str(missing) in said


def test_a_ground_nobody_gave_is_not_reported_as_unread(tmp_path):
    """Giving nothing is a different thing from giving something unreadable."""
    _said, report = _solve(tmp_path)
    assert report.get("unread_sources") == {}
