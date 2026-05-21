import pytest
import json

@pytest.mark.live
def test_live_causal_agency_lesion(live_harness):
    repo = live_harness.create_isolated_copy()

    result = live_harness.run_command(
        repo,
        [
            ".venv/bin/python",
            "tools/agi/run_causal_agency_lesion.py",
            "--seeds", "50",
            "--output", "artifacts/agi_live/causal_agency.json",
        ],
        timeout_s=900,
    )

    assert result.ok, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    report = json.loads((result.artifacts_dir / "causal_agency.json").read_text())

    assert report["manual_interventions"] == 0
    assert report["receipt_coverage"] == 1.0
    assert report["normal_state_action_divergence"] >= 0.25
    assert report["lesioned_action_divergence"] <= 0.10
    assert report["p_value"] < 0.01
