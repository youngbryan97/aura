from __future__ import annotations

from pathlib import Path

from core.unity.unity_receipts import write_unity_results_artifact


_UNITY_RESULTS = {
    "tests_total": 0,
    "tests_passed": 0,
    "tests_failed": 0,
    "outcomes": {},
}


def pytest_configure(config):
    _UNITY_RESULTS["tests_total"] = 0
    _UNITY_RESULTS["tests_passed"] = 0
    _UNITY_RESULTS["tests_failed"] = 0
    _UNITY_RESULTS["outcomes"] = {}


def pytest_runtest_logreport(report):
    if "tests/unity/" not in report.nodeid or report.when != "call":
        return
    _UNITY_RESULTS["tests_total"] += 1
    if report.passed:
        _UNITY_RESULTS["tests_passed"] += 1
        _UNITY_RESULTS["outcomes"][report.nodeid] = "passed"
    elif report.failed:
        _UNITY_RESULTS["tests_failed"] += 1
        _UNITY_RESULTS["outcomes"][report.nodeid] = "failed"
    else:
        _UNITY_RESULTS["outcomes"][report.nodeid] = "other"


def pytest_sessionfinish(session, exitstatus):
    outcomes = dict(_UNITY_RESULTS.get("outcomes", {}) or {})
    if not outcomes:
        return

    def _passed(fragment: str) -> bool:
        return any(fragment in nodeid and outcome == "passed" for nodeid, outcome in outcomes.items())

    gating_cases = [
        "test_low_unity_blocks_external_tool_action",
        "test_low_unity_allows_stabilization",
        "test_low_unity_defers_memory_write_when_drafts_are_unstable",
    ]
    gating_passes = sum(1 for case in gating_cases if _passed(case))
    gating_rate = gating_passes / len(gating_cases)

    artifact_payload = {
        "timestamp": __import__("time").time(),
        "unity_layer_version": "1.0",
        "tests_passed": int(_UNITY_RESULTS.get("tests_passed", 0) or 0),
        "tests_total": int(_UNITY_RESULTS.get("tests_total", 0) or 0),
        "lesion_effects": {
            "temporal_binding": _passed("test_temporal_binding_lesion_changes_behavior"),
            "ownership_binding": _passed("test_self_world_lesion_reduces_authorship_confidence"),
            "draft_reconciliation": _passed("test_draft_lesion_removes_conflict_preservation"),
        },
        "fragmentation_cases": {
            "self_reported": _passed("test_fragmented_unity_reports_measurable_cause"),
            "repaired": _passed("test_projected_repair_improves_unity_without_faking_resolution"),
            "gated": _passed("test_end_to_end_unity_changes_tool_decision"),
        },
        "repair_success_rate": 1.0 if _passed("test_projected_repair_improves_unity_without_faking_resolution") else 0.0,
        "false_unity_claims": 0 if _passed("test_fragmented_unity_reports_measurable_cause") and _passed("test_integrated_frame_warns_against_false_clarity") else 1,
        "false_fragmentation_claims": 0 if _passed("test_nominal_state_does_not_force_fragmentation_language") else 1,
        "memory_conflict_preservation_rate": 1.0 if _passed("test_conflicting_drafts_become_conflicted_memory") and _passed("test_memory_metadata_carries_unity_fields") else 0.0,
        "action_gating_success_rate": round(gating_rate, 4),
    }

    target = Path(session.config.rootpath) / "artifacts" / "unity" / "latest" / "UNITY_RESULTS.json"
    write_unity_results_artifact(target, artifact_payload)
