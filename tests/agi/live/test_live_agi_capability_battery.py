import pytest
import json

@pytest.mark.live
def test_live_agi_capability_battery(live_harness):
    """
    Live environment execution test of the expanded External AGI Capability Battery.
    """
    repo = live_harness.create_isolated_copy()

    result = live_harness.run_command(
        repo,
        [
            ".venv/bin/python",
            "tools/agi/run_agi_capability_battery.py",
            "--output", "artifacts/agi_live/capability_battery.json",
            "--markdown", "artifacts/agi_live/CAPABILITY_BATTERY_RESULTS.md",
            "--seeds", "20",  # Use 20 seeds to keep execution times fast
        ],
        timeout_s=600,
    )

    assert result.ok, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Load and verify JSON capability report
    report = json.loads((result.artifacts_dir / "capability_battery.json").read_text())

    assert report["categories_evaluated"] == 17
    assert report["aura_scores"]["mean_score"] > 0.80
    
    # Assert statistical separation
    assert report["aura_scores"]["lower_ci"] > report["baselines_and_ablations"]["base_model_with_tools"]["upper_ci"], (
        "Statistical separation failed! Full Aura must beat base model with tools."
    )
    assert report["aura_scores"]["lower_ci"] > report["baselines_and_ablations"]["react_tool_agent"]["upper_ci"], (
        "Statistical separation failed! Full Aura must beat ReAct tool agent."
    )
