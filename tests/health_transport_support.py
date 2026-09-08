"""Publish existing probe/boot scenario fixtures at the shared transport boundary.

These fixtures predate the snapshot read model. Their probe and lane stubs
still describe the scenario; only their injection point has moved. No live
collector or worker is started by these helpers.
"""

from copy import deepcopy


def publish_probe_scenario(monkeypatch):
    from core.runtime import health_contract
    from interface import websocket_manager
    from interface.routes import system

    report = health_contract.runtime_health_report()
    probes = health_contract.required_probe_status(report)
    lane, ready = websocket_manager._conversation_lane_readiness()
    blockers = []
    if report.get("healthy") is not True:
        blockers = websocket_manager._runtime_report_blockers(report)
    snapshot = {
        "healthy": report.get("healthy") is True,
        "status": report.get("status"),
        "required_probes": probes,
        "runtime_probe_healthy": health_contract.required_probe_groups_pass(probes),
        "conversation_lane": lane,
        "conversation_ready": ready,
        "blockers": blockers,
    }
    monkeypatch.setattr(system, "read_runtime_health_snapshot", lambda: deepcopy(snapshot))


def publish_boot_scenario(monkeypatch, system):
    from core.health.conversation_lane import conversation_lane_is_busy
    from core.runtime.health_contract import required_probe_groups_pass

    lane = system._collect_conversation_lane_status_resilient()
    boot, status_code = system.build_boot_health_snapshot(None, {})
    integrity = system._runtime_integrity_public_payload(system._collect_runtime_integrity_report())
    ready = lane.get("conversation_ready") is True
    blockers = system._normalize_conversation_health_blockers(
        list(boot.get("blockers") or []),
        conversation_ready=ready,
        conversation_busy=conversation_lane_is_busy(lane),
    )
    probes = boot.get("required_probes") or {}
    snapshot = {
        "healthy": status_code in {200, 202} and boot.get("system_ready", boot.get("ready")) is True,
        "status": boot.get("status", "healthy"),
        "required_probes": probes,
        "runtime_probe_healthy": required_probe_groups_pass(probes),
        "conversation_lane": lane,
        "conversation_ready": ready,
        "blockers": blockers,
        "integrity": integrity,
        "proof_readiness_healthy": integrity.get("proof_readiness") is True,
        "integrity_blockers": integrity.get("proof_blockers", []),
        "boot": boot,
    }
    monkeypatch.setattr(system, "read_runtime_health_snapshot", lambda: deepcopy(snapshot))
