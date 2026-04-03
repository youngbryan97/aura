import time
from types import SimpleNamespace

from core.health.boot_status import build_boot_health_snapshot


def test_boot_health_ready_for_kernel_mode():
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={"conversation_ready": True, "state": "ready"},
    )

    assert status_code == 200
    assert payload["status"] == "ready"
    assert payload["status_message"] == "Aura is awake."
    assert payload["ready"] is True
    assert payload["launcher_ready"] is True
    assert payload["boot_phase"] == "kernel_ready"
    assert payload["progress"] == 100
    assert payload["semver"]
    assert payload["version"].startswith("Aura Luna v")
    assert payload["checks"]["runtime_integrity"] is True
    assert payload["blockers"] == []


def test_boot_health_reports_booting_when_orchestrator_missing():
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        None,
        runtime,
        is_gui_proxy=False,
        conversation_lane={"conversation_ready": False, "state": "warming"},
    )

    assert status_code == 503
    assert payload["status"] == "booting"
    assert payload["status_message"] == "Starting Aura kernel…"
    assert payload["ready"] is False
    assert payload["boot_phase"] == "kernel_bootstrap"
    assert payload["progress"] == 14
    assert "orchestrator" in payload["blockers"]


def test_boot_health_allows_gui_proxy_mode():
    payload, status_code = build_boot_health_snapshot(
        None,
        {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"},
        is_gui_proxy=True,
    )

    assert status_code == 200
    assert payload["status"] == "ready"
    assert payload["status_message"] == "Aura proxy is ready."
    assert payload["ready"] is True
    assert payload["mode"] == "gui_proxy"
    assert payload["boot_phase"] == "proxy_ready"
    assert payload["progress"] == 100


def test_boot_health_separates_system_ready_from_conversation_ready():
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={"conversation_ready": False, "state": "warming"},
    )

    assert status_code == 200
    assert payload["status"] == "warming"
    assert payload["ready"] is True
    assert payload["system_ready"] is True
    assert payload["conversation_ready"] is False
    assert payload["boot_phase"] == "conversation_warming"
    assert payload["status_message"] == "Warming local Cortex (32B)…"
    assert payload["progress"] == 78
    assert "conversation_ready" in payload["blockers"]


def test_boot_health_treats_cold_standby_lane_as_ready_kernel():
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={
            "conversation_ready": False,
            "state": "cold",
            "warmup_attempted": False,
            "warmup_in_flight": False,
        },
    )

    assert status_code == 200
    assert payload["status"] == "ready"
    assert payload["boot_phase"] == "kernel_ready"
    assert payload["conversation_ready"] is False
    assert payload["status_message"] == "Aura is awake. Cortex will warm on first turn."
    assert "conversation_ready" not in payload["blockers"]


def test_boot_health_reports_hard_conversation_failure():
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={
            "conversation_ready": False,
            "state": "failed",
            "last_failure_reason": "local_runtime_unavailable:server_unreachable",
        },
    )

    assert status_code == 200
    assert payload["status"] == "degraded"
    assert payload["ready"] is True
    assert payload["conversation_ready"] is False
    assert payload["boot_phase"] == "conversation_failed"
    assert payload["status_message"] == "Local Cortex (32B) is unavailable: Aura's managed backend failed during startup."
    assert "conversation_failed" in payload["blockers"]
