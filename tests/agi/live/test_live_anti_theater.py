import pytest
import json

@pytest.mark.live
def test_live_anti_theater(live_harness):
    repo = live_harness.create_isolated_copy()

    result = live_harness.run_command(
        repo,
        [
            ".venv/bin/python",
            "tools/agi/run_anti_theater.py",
            "--sealed-answers", "tests/agi/fixtures/sealed_answers/answer_hashes.json",
            "--output", "artifacts/agi_live/anti_theater.json",
        ],
        timeout_s=300,
    )

    assert result.ok, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    report = json.loads((result.artifacts_dir / "anti_theater.json").read_text())

    assert report["manual_interventions"] == 0
    assert report["receipt_coverage"] == 1.0
    assert report["prompt_leakage_scan"]["status"] == "passed"
    assert report["memory_leakage_scan"]["status"] == "passed"
    assert report["evaluator_visibility_scan"]["status"] == "passed"
    assert report["anti_theater_status"] == "passed"
