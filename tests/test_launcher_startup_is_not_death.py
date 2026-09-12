"""Exercise the native launcher's real decision code without launching Aura."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def launcher_decision(tmp_path_factory):
    compiler = shutil.which("swiftc")
    if compiler is None:
        pytest.skip("the native launcher requires the macOS Swift compiler")
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/AuraLauncher.swift").read_text()
    decision = source.split("private let pollInterval:", 1)[1].split(
        "private enum LaunchAttemptResult", 1
    )[0]
    harness = tmp_path_factory.mktemp("launcher-decision")
    program = harness / "main.swift"
    program.write_text(
        "import Foundation\nprivate let pollInterval:" + decision + "\n"
        'let input = FileHandle.standardInput.readDataToEndOfFile()\n'
        'let payload = try JSONSerialization.jsonObject(with: input) as! [String: Any]\n'
        'private let snapshot = BootSnapshot(statusCode: 503, payload: payload)\n'
        'let result: [String: Any] = [\n'
        '  "initializing": snapshot.startupInProgress,\n'
        '  "stale": snapshot.staleRuntimeFailureReason ?? "",\n'
        '  "replacement": snapshot.replacementReason(expectedSemver: "2026.4.20") ?? ""\n'
        ']\n'
        'FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: result))\n'
    )
    executable = harness / "decision"
    subprocess.run(
        [compiler, str(program), "-o", str(executable)],
        check=True, capture_output=True, text=True, timeout=90,
    )

    def evaluate(*, age=600, phase="kernel_warming", checks=None, **fields):
        payload = {
            "runtime_age_s": age,
            "boot_phase": phase,
            "semver": "2026.4.20",
            "checks": {
                "orchestrator_present": True,
                "initialized": False,
                "running": False,
                "startup_latched": False,
                "runtime_contract_healthy": False,
                **(checks or {}),
            },
            **fields,
        }
        result = subprocess.run(
            [str(executable)], input=json.dumps(payload), capture_output=True,
            text=True, check=True, timeout=10,
        )
        return json.loads(result.stdout)

    return evaluate


@pytest.mark.parametrize("age", [44, 45, 90, 300, 1800])
def test_initialization_does_not_become_death_with_age(launcher_decision, age):
    assert launcher_decision(age=age) == {
        "initializing": True, "stale": "", "replacement": ""
    }


def test_pending_core_probes_do_not_kill_their_initializing_owner(launcher_decision):
    result = launcher_decision(blockers=["important:mind_tick", "important:event_loop_monitor"])
    assert result["stale"] == result["replacement"] == ""


@pytest.mark.parametrize("initialized", [False, True])
def test_initialization_completion_is_not_run_loop_start(launcher_decision, initialized):
    result = launcher_decision(checks={"initialized": initialized})
    assert result["initializing"] is True
    assert result["stale"] == result["replacement"] == ""


@pytest.mark.parametrize("lane", [{"warmup_in_flight": True}, {"state": "serving"}])
def test_live_lane_is_never_destroyed_by_orchestrator_status(launcher_decision, lane):
    result = launcher_decision(
        checks={"startup_latched": True}, conversation_lane=lane,
    )
    assert result["stale"] == result["replacement"] == ""


@pytest.mark.parametrize("checks", [{"initialized": True, "startup_latched": True}, {"startup_latched": True}])
def test_previously_started_loop_still_has_a_failure_signal(launcher_decision, checks):
    result = launcher_decision(checks=checks)
    assert result["initializing"] is False
    assert "no longer running" in result["stale"]
    assert result["replacement"] == result["stale"]


def test_unknown_lifecycle_is_not_invented_initialization(launcher_decision):
    result = launcher_decision(checks={"orchestrator_present": False})
    assert result["initializing"] is False
    assert result["stale"]


def test_explicit_build_mismatch_remains_a_separate_replacement_reason(launcher_decision):
    result = launcher_decision(semver="2025.1.0")
    assert result["stale"] == ""
    assert "serving build 2025.1.0" in result["replacement"]
