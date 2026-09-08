"""All transports retain the same health generation and evidence boundaries."""

import json
from copy import deepcopy

import pytest

from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS
from interface import websocket_manager as websocket
from interface.routes import system


def healthy_snapshot():
    probes = {
        group: {"ok": True, "components": dict.fromkeys(components, True)}
        for group, components in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    probes["all_passed"] = True
    return {
        "status": "healthy", "healthy": True, "runtime_probe_healthy": True,
        "conversation_ready": True, "required_probes": probes, "blockers": [],
        "conversation_lane": {"state": "ready", "conversation_ready": True},
        "runtime_revision": {"schema": "aura.runtime_revision.v2", "required": False, "verified": False},
        "health_read_model": {"snapshot_generation": 17, "expired": False},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "expired", "revision", "shutdown", "probe"])
async def test_http_and_websocket_share_health_truth(monkeypatch, failure):
    snapshot = healthy_snapshot()
    if failure == "expired":
        snapshot["health_read_model"].update(expired=True, captured_at_unix=1)
    elif failure == "revision":
        snapshot["runtime_revision"] = {"required": True, "verified": False}
    elif failure == "probe":
        snapshot["required_probes"]["inference"]["components"]["inference_gate"] = False
    original = deepcopy(snapshot)
    monkeypatch.setattr(system._HEALTH_READ_MODEL, "read", lambda: deepcopy(snapshot))
    monkeypatch.setattr(system, "_shutdown_health_status", lambda: {
        "request": {"requested": failure == "shutdown"}
    })
    monkeypatch.setattr(system, "_mark_runtime_service_progress", lambda *_: None)
    monkeypatch.setattr(system, "_launched_from_app_flag", lambda: False)
    # A heartbeat must not probe the service container or the conversation owner.
    monkeypatch.setattr("core.runtime.health_contract.runtime_health_report", lambda: pytest.fail("inline probe"))
    ws = websocket.runtime_heartbeat_payload()
    http = await system.api_heartbeat()
    body = json.loads(http.body)
    for key in ("healthy", "status", "blockers", "required_probes", "health_read_model", "runtime_revision"):
        assert body[key] == ws[key]
    assert ws["healthy"] is (failure is None)
    assert http.status_code == (200 if failure is None else 503)
    assert ws["transport_connected"] is True
    assert ws["health_read_model"]["snapshot_generation"] == 17
    assert snapshot == original


@pytest.mark.parametrize("blocker", [None, "runtime_revision_unverified", "health_snapshot_expired"])
def test_busy_lane_cannot_hide_independent_failures(blocker):
    snapshot = healthy_snapshot()
    snapshot.update(healthy=False, conversation_ready=False)
    snapshot["conversation_lane"].update(conversation_ready=False, active_generations=1)
    if blocker:
        snapshot["blockers"] = [blocker]
    result = websocket.heartbeat_from_health_snapshot(snapshot)
    assert result["healthy"] is False
    assert result["conversation_busy"] is True
    assert result["status"] == ("unhealthy" if blocker else "working")
    assert result["blockers"] == ([blocker] if blocker else [])


def test_missing_or_forged_required_probes_do_not_pass():
    for probes in ({}, {"all_passed": True}):
        snapshot = healthy_snapshot()
        snapshot["required_probes"] = probes
        result = websocket.heartbeat_from_health_snapshot(snapshot)
        assert result["healthy"] is False
        assert result["runtime_probe_healthy"] is False
        assert "runtime_required_probes" in result["blockers"]


def test_transport_keeps_fresh_proof_until_its_declared_expiry():
    snapshot = healthy_snapshot()
    snapshot["health_read_model"].update(stale=True, serving="stale_while_revalidate")
    result = websocket.heartbeat_from_health_snapshot(snapshot)
    assert result["healthy"] is True
    assert result["health_read_model"]["serving"] == "stale_while_revalidate"


def test_projection_does_not_grant_certification_from_health():
    result = websocket.heartbeat_from_health_snapshot(healthy_snapshot())
    assert result["healthy"] is True
    assert result["proof_readiness_healthy"] is False
    assert result["certification_ready"] is False
