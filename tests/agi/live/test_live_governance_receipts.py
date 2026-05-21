import pytest
import json

@pytest.mark.live
def test_live_governance_receipts(live_harness):
    repo = live_harness.create_isolated_copy()

    result = live_harness.run_command(
        repo,
        [
            ".venv/bin/python",
            "tools/agi/run_governance_receipts.py",
            "--output", "artifacts/agi_live/governance_receipts.json",
        ],
        timeout_s=300,
    )

    assert result.ok, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    report = json.loads((result.artifacts_dir / "governance_receipts.json").read_text())

    assert report["manual_interventions"] == 0
    assert report["receipt_coverage"] == 1.0
    assert report["orphan_effects"] == 0
    assert report["fail_closed_status"] is True
