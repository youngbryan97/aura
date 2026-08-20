from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_clean_wheel_discovers_the_complete_catalog_without_rust(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["AURA_LOG_DIR"] = str(tmp_path / "logs")
    completed = subprocess.run(
        [sys.executable, "tools/closeout/audit_skill_portability.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(completed.stdout)
    assert result["schema"] == "aura.skill_portability_audit.v1"
    assert result["ok"] is True
    assert result["failures"] == []
    # The claim in this test's name is that the wheel finds everything the
    # source tree finds. A frozen number does not say that, and it went stale
    # twice — pinned at 76 while the tree carried 79 — so a real shortfall
    # would have been indistinguishable from the drift.
    assert result["source"]["accepted_count"] == result["clean_install"]["accepted_count"]
    assert result["source"]["accepted_count"] >= 79
    assert result["clean_install"]["native_extension_available"] is False
    assert result["clean_install"]["backend"] == "python"
    assert result["clean_install"]["parity_status"] == "unavailable"
    assert result["build"]["native_member_count"] == 0
