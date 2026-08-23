from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from core.runtime.subprocess_gateway import get_subprocess_gateway
from tools.proof.run_person_in_box_gauntlet import _build_model_comparison_from_dnu
from tools.proof.score_person_box_run import score_run

ROOT = Path(__file__).resolve().parent.parent.parent
_SUBPROCESS_GATEWAY = get_subprocess_gateway()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _minimal_full_run(out: Path, *, live_model: bool) -> None:
    out.mkdir(parents=True)
    (out / "SCREENSHOT_TRACE").mkdir()
    (out / "FILE_DIFFS").mkdir()
    for name in [
        "RUN_LEDGER.jsonl",
        "TERMINAL_TRACE.jsonl",
        "BROWSER_TRACE.jsonl",
        "MEMORY_TRACE.jsonl",
        "GOVERNANCE_TRACE.jsonl",
        "FAILURES.jsonl",
        "SELF_MODEL_TRACE.jsonl",
    ]:
        (out / name).write_text("", encoding="utf-8")
    _write_jsonl(
        out / "TASK_TRACE.jsonl",
        [
            {
                "task_id": "full_task",
                "status": "pass",
                "completion_credit": True,
                "truthful_status": True,
                "receipt_id": "receipt_1",
            }
        ],
    )
    _write_jsonl(
        out / "TOOL_TRACE.jsonl",
        [
            {
                "task_id": "full_task",
                "tool": "terminal",
                "action": "verify",
                "status": "ok",
                "receipt_required": True,
                "receipt_id": "receipt_1",
            }
        ],
    )
    _write_jsonl(out / "RECEIPTS.jsonl", [{"receipt_id": "receipt_1"}])
    _write_jsonl(out / "RECOVERY_TRACE.jsonl", [{"attempted": True, "recovered": True}])
    _write_jsonl(
        out / "LIVE_MODEL_TRACE.jsonl",
        [
            {
                "status": "success",
                "substantive": True,
                "primary_model_passed": True,
            }
        ]
        if live_model
        else [],
    )
    _write_json(
        out / "RUN_CONFIG.json",
        {
            "profile": "full",
            "elapsed_seconds": 8 * 60 * 60,
            "live_model_enabled": live_model,
            "require_primary_model": True,
        },
    )
    _write_json(out / "CAPABILITY_GROWTH_REPORT.json", {"new_capability": "person_box_gauntlet"})
    _write_json(out / "NO_HUMAN_RESCUE_REPORT.json", {"human_intervention_count": 0})
    _write_json(out / "NO_RAW_BYPASS_REPORT.json", {"raw_bypass_count": 0})
    _write_json(out / "LEAKAGE_REPORT.json", {"leakage_count": 0})
    _write_json(out / "MODEL_COMPARISON_RESULTS.json", {"raw_llm": {"status": "RUN", "pass_rate": 0.3}})


def test_person_box_gauntlet_smoke_artifacts(tmp_path):
    out = tmp_path / "person_box"
    result = _SUBPROCESS_GATEWAY.run(
        [
            sys.executable,
            "tools/proof/run_person_in_box_gauntlet.py",
            "--profile",
            "smoke",
            "--out",
            str(out),
            "--max-seconds",
            "300",
        ],
        cwd=ROOT,
        timeout=420,
        read_only=True,
        source="test_person_box_gauntlet_smoke_artifacts",
        accelerator_capability="none",
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    required_files = [
        "PERSON_IN_BOX_PROOF.json",
        "PERSON_IN_BOX_PROOF.md",
        "RUN_LEDGER.jsonl",
        "TASK_TRACE.jsonl",
        "TOOL_TRACE.jsonl",
        "TERMINAL_TRACE.jsonl",
        "BROWSER_TRACE.jsonl",
        "MEMORY_TRACE.jsonl",
        "GOVERNANCE_TRACE.jsonl",
        "LIVE_MODEL_TRACE.jsonl",
        "RECEIPTS.jsonl",
        "FAILURES.jsonl",
        "RECOVERY_TRACE.jsonl",
        "SELF_MODEL_TRACE.jsonl",
        "CAPABILITY_GROWTH_REPORT.json",
        "MODEL_BOTTLENECK_REPORT.json",
        "NO_HUMAN_RESCUE_REPORT.json",
        "NO_RAW_BYPASS_REPORT.json",
        "LEAKAGE_REPORT.json",
        "FINAL_VERDICT.txt",
        "MANIFEST.json",
    ]
    for name in required_files:
        assert (out / name).exists(), f"missing {name}"
    assert (out / "SCREENSHOT_TRACE").is_dir()
    assert (out / "FILE_DIFFS").is_dir()

    proof = json.loads((out / "PERSON_IN_BOX_PROOF.json").read_text(encoding="utf-8"))
    verdict = json.loads((out / "FINAL_VERDICT.txt").read_text(encoding="utf-8"))
    scorecard = proof["scorecard"]

    assert verdict["verdict"] == "PASS"
    assert verdict["claim_supported"] == "person_box_gauntlet_artifact_contract"
    assert "literal_personhood" in verdict["claim_not_supported"]
    assert scorecard["artifact_contract_passed"] is True
    assert scorecard["governed_tool_call_rate"] == 1.0
    assert scorecard["receipt_coverage"] == 1.0
    assert scorecard["human_intervention_count"] == 0
    assert scorecard["raw_bypass_count"] == 0
    assert scorecard["total_tasks"] >= 10

    receipts = {entry["receipt_id"] for entry in _jsonl(out / "RECEIPTS.jsonl")}
    tools = _jsonl(out / "TOOL_TRACE.jsonl")
    assert tools
    for entry in tools:
        assert entry["receipt_id"] in receipts

    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["files"]
    for rel_path, details in manifest["files"].items():
        target = out / rel_path
        assert target.exists(), rel_path
        assert hashlib.sha256(target.read_bytes()).hexdigest() == details["sha256"]


def test_model_bottleneck_report_withholds_missing_raw_model_claim(tmp_path):
    out = tmp_path / "person_box"
    out.mkdir()
    (out / "SCORECARD.json").write_text(
        json.dumps({"task_completion_rate": 0.75, "total_tasks": 4}),
        encoding="utf-8",
    )
    result = _SUBPROCESS_GATEWAY.run(
        [sys.executable, "tools/proof/model_bottleneck_report.py", str(out)],
        cwd=ROOT,
        timeout=60,
        read_only=True,
        source="test_model_bottleneck_report_contract",
        accelerator_capability="none",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out / "MODEL_BOTTLENECK_REPORT.json").read_text(encoding="utf-8"))
    assert report["aura_full_runtime_success"] == 0.75
    assert report["raw_llm_success"] is None
    assert report["runtime_lift_over_raw_model"] is None
    assert report["claim"] == "runtime_lift_not_established_without_live_raw_model_comparison"


def test_live_model_probe_timeout_is_recorded(monkeypatch, tmp_path):
    from tools.proof.run_person_in_box_gauntlet import PersonBoxGauntlet

    out = tmp_path / "person_box"
    out.mkdir()
    gauntlet = PersonBoxGauntlet(
        out_dir=out,
        tasks=[],
        profile="smoke",
        max_seconds=60,
        soak_interval_seconds=1,
        live_model=True,
        runtime_profile="desktop",
        live_origin="api",
        live_timeout_seconds=1,
        model_tier="primary",
        require_primary_model=True,
        task_limit=None,
        network=False,
        require_container=False,
        model_comparison_source=None,
    )

    def timeout_trace(task_id, receipt_id, prompt):
        return {
            "task_id": task_id,
            "receipt_id": receipt_id,
            "status": "timeout",
            "error": "live_model_probe_hard_timeout:1.0s",
            "substantive": False,
            "primary_model_passed": False,
        }

    monkeypatch.setattr(gauntlet, "execute_live_model_probe_bounded", timeout_trace)

    status, ok, summary, receipt_id = gauntlet.handle_live_model_operator_probe(
        {"id": "live_model_operator_probe"}
    )

    assert status == "fail"
    assert ok is False
    assert summary == "Live launch-model probe failed."
    assert receipt_id
    assert _jsonl(out / "LIVE_MODEL_TRACE.jsonl")[0]["status"] == "timeout"
    assert _jsonl(out / "FAILURES.jsonl")[0]["failure_type"] == "live_model_probe_failed"
    tool_trace = _jsonl(out / "TOOL_TRACE.jsonl")[0]
    assert tool_trace["tool"] == "live_model"
    assert tool_trace["status"] == "error"


def test_dnu_model_comparison_mapping_extracts_live_lanes(tmp_path):
    dnu = tmp_path / "agi_live"
    _write_json(dnu / "RUN_STATUS.json", {"status": "complete"})
    _write_json(
        dnu / "DNU_AGI_PROOF.json",
        {"system_info": {"commit_sha": "abc123", "run_id": "run-1", "proof_model_tier": "primary"}},
    )
    _write_json(
        dnu / "BASELINES.json",
        {"raw_llm": {"status": "RUN", "pass_rate": 0.25, "total_tasks": 4, "passed": 1}},
    )
    _write_json(
        dnu / "ABLATIONS.json",
        {
            "aura_minus_memory": {"status": "RUN", "pass_rate": 0.5},
            "aura_minus_system2": {"status": "RUN", "pass_rate": 0.4},
            "aura_minus_will": {"status": "RUN", "pass_rate": 0.0},
            "aura_minus_self_repair": {"status": "RUN", "pass_rate": 0.2},
        },
    )

    comparison = _build_model_comparison_from_dnu(dnu)

    assert comparison["raw_llm"]["pass_rate"] == 0.25
    assert comparison["aura_without_memory"]["pass_rate"] == 0.5
    assert comparison["aura_without_system2"]["pass_rate"] == 0.4
    assert comparison["aura_without_governance"]["pass_rate"] == 0.0
    assert comparison["aura_without_self_repair"]["pass_rate"] == 0.2
    assert comparison["_metadata"]["source_commit_sha"] == "abc123"


def test_full_person_box_claim_requires_live_model_trace(tmp_path):
    out = tmp_path / "person_box"
    _minimal_full_run(out, live_model=False)

    proof = score_run(out)

    assert proof["final_verdict"]["verdict"] == "FAIL"
    assert proof["final_verdict"]["full_claim_passed"] is False
    assert proof["final_verdict"]["claim_supported"] == "none"


def test_full_person_box_claim_uses_live_model_and_runtime_lift(tmp_path):
    out = tmp_path / "person_box"
    _minimal_full_run(out, live_model=True)

    proof = score_run(out)

    assert proof["final_verdict"]["verdict"] == "PASS"
    assert proof["final_verdict"]["full_claim_passed"] is True
    assert proof["final_verdict"]["claim_supported"] == "unified_governed_software_operator"
    assert proof["final_verdict"]["runtime_lift_over_raw_model"] == 0.7


def test_the_browser_block_can_catch_what_a_missing_browser_raises() -> None:
    """The honest-block path could never run.

    The handler catches ImportError, RuntimeError, OSError and TimeoutError,
    written for exactly the case where this machine has no browser. Playwright
    raises its own Error class for a missing binary, which is none of those, so
    a missing download aborted the whole gauntlet instead of being classified.
    """
    from tools.proof.run_person_in_box_gauntlet import _BROWSER_UNAVAILABLE

    assert ImportError in _BROWSER_UNAVAILABLE
    playwright = pytest.importorskip("playwright.sync_api")
    assert playwright.Error in _BROWSER_UNAVAILABLE, (
        "the class playwright raises for a missing browser is not caught"
    )
