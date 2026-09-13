"""The completion tracker runs the way the integrity gate runs it.

`make subject-core-integrity` calls `python tools/isc_completion_status.py
--check` from the repository root with nothing on PYTHONPATH. Run that way,
Python puts tools/ first on the import path, and the tracker's first import,
from `tools.report_expression`, raised ModuleNotFoundError. The gate that
checks the completion tracker could not start.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_tracker_imports_without_the_repository_on_pythonpath() -> None:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    done = subprocess.run(
        [sys.executable, "tools/isc_completion_status.py", "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-2000:]
