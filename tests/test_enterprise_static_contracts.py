from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config" / "aura_enterprise_gate_baseline.json"
GATE = ROOT / "tools" / "aura_enterprise_gate.py"


def _run_static_gate(tmp_path: Path) -> dict:
    report_path = tmp_path / "enterprise_gate.json"
    env = os.environ.copy()
    env.setdefault("AURA_TEST_MODE", "1")
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    proc = subprocess.run(
        [
            sys.executable,
            str(GATE),
            "--root",
            str(ROOT),
            "--skip-compile",
            "--skip-pytest-collect",
            "--baseline",
            str(BASELINE),
            "--fail-on-regression",
            "--out",
            str(report_path),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout[-4000:]
    return json.loads(report_path.read_text(encoding="utf-8"))


def test_enterprise_gate_baseline_blocks_static_regressions(tmp_path: Path):
    report = _run_static_gate(tmp_path)

    assert report["python_files"] >= 2000
    assert report["counts"]["broad_exception_review"] <= 4730
    assert not [
        finding
        for finding in report["findings"]
        if finding["kind"] == "baseline_regression"
    ]


def test_enterprise_gate_has_zero_secret_literals_and_shell_true(tmp_path: Path):
    report = _run_static_gate(tmp_path)
    counts = report["counts"]

    assert counts.get("potential_secret", 0) == 0
    assert counts.get("subprocess_shell_true", 0) == 0
