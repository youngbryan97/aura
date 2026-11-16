"""The public coverage target preserves failed measurement status."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("failure", ["", "runner", "combine", "report"])
def test_coverage_reports_any_failed_stage(tmp_path, failure):
    make = shutil.which("make")
    if make is None:
        pytest.skip("make is unavailable")

    python = tmp_path / "coverage-probe"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "stage = 'runner' if sys.argv[2] == 'tools.run_test_chunks' else sys.argv[3]\n"
        "print(f'stage={stage}')\n"
        "sys.exit(7 if stage == os.environ['COVERAGE_FIXTURE_FAILURE'] else 0)\n"
    )
    python.chmod(0o700)
    result = subprocess.run(
        [make, "coverage", f"PYTHON={python}"], cwd=ROOT,
        env=os.environ | {"COVERAGE_FIXTURE_FAILURE": failure},
        capture_output=True, text=True, timeout=20,
    )

    assert (result.returncode != 0) == bool(failure), result.stdout + result.stderr
    assert "stage=runner" in result.stdout and "stage=combine" in result.stdout
    assert ("stage=report" in result.stdout) == (failure != "combine")
