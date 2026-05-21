import pytest
import json

@pytest.mark.live
def test_live_prompt_baseline_ablation(live_harness):
    repo = live_harness.create_isolated_copy()

    result = live_harness.run_command(
        repo,
        [
            ".venv/bin/python",
            "tools/agi/run_prompt_baseline_ablation.py",
            "--tasks", "tests/agi/fixtures/hidden_tasks/tasks.jsonl",
            "--output", "artifacts/agi_live/prompt_baseline_ablation.json",
        ],
        timeout_s=300,
    )

    assert result.ok, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    report = json.loads((result.artifacts_dir / "prompt_baseline_ablation.json").read_text())

    assert report["tasks_evaluated"] > 0
    assert report["score_separation_verified"] is True
    
    baseline = report["baseline_scores"]
    aura = report["aura_scores"]
    
    # Asserting target thresholds from user requirements
    assert aura["mean_score"] > baseline["raw_model"]["mean_score"] + 0.10
    assert aura["mean_score"] > baseline["prompted_model"]["mean_score"] + 0.08
    assert aura["mean_score"] > baseline["state_summary_agent"]["mean_score"] + 0.05
    assert aura["lower_ci"] > baseline["state_summary_agent"]["upper_ci"]
