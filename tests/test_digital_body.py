"""tests/test_digital_body.py
===========================
Unit and integration tests for core/embodiment/digital_body.py.
"""

from __future__ import annotations

import time
import pytest
from core.embodiment.digital_body import get_digital_body, DigitalBody


def test_digital_body_singleton() -> None:
    """Verifies that DigitalBody acts as a correct singleton."""
    body1 = get_digital_body()
    body2 = get_digital_body()
    assert body1 is body2
    assert isinstance(body1, DigitalBody)


def test_digital_body_telemetry_update() -> None:
    """Verifies that telemetry is refreshed with realistic values."""
    body = get_digital_body()
    
    # Force fresh update
    body.update_telemetry()
    state = body.get_state_dict()
    
    assert "resource_state" in state
    resources = state["resource_state"]
    
    # Assert bounds of CPU/memory/disk percent (0 to 100)
    assert 0.0 <= resources["cpu_percent"] <= 100.0
    assert 0.0 <= resources["memory_percent"] <= 100.0
    assert 0.0 <= resources["disk_percent"] <= 100.0
    assert 0.0 <= resources["latency_ms"]


def test_digital_body_commitments() -> None:
    """Verifies commitment registration, listing, and resolution lifecycle."""
    body = get_digital_body()
    
    # Register commitment
    c = {
        "goal": "Verify system integration",
        "action": "run_test",
        "deadline": time.time() + 10.0
    }
    body.register_commitment(c)
    
    state = body.get_state_dict()
    assert state["commitments_count"] >= 1
    
    # Locate active commitment
    active = state["active_commitments"]
    matching = [item for item in active if item.get("goal") == "Verify system integration"]
    assert len(matching) == 1
    commitment_id = matching[0]["id"]
    
    # Resolve commitment
    resolved = body.resolve_commitment(commitment_id, "completed")
    assert resolved is not None
    assert resolved["status"] == "completed"
    assert "resolved_at" in resolved
    
    # Verify no longer active
    state2 = body.get_state_dict()
    active2 = state2["active_commitments"]
    matching2 = [item for item in active2 if item.get("id") == commitment_id]
    assert len(matching2) == 0


def test_digital_body_degradation_flags() -> None:
    """Verifies registry and recovery from system degradations."""
    body = get_digital_body()
    
    # Mark a system as degraded
    body.mark_system_degraded("test_sensor_subsystem", degraded=True)
    state = body.get_state_dict()
    assert "test_sensor_subsystem" in state["degraded_systems"]
    
    # Recover it
    body.mark_system_degraded("test_sensor_subsystem", degraded=False)
    state2 = body.get_state_dict()
    assert "test_sensor_subsystem" not in state2["degraded_systems"]


def test_structured_commitment_ledger() -> None:
    """Verifies that registered commitments are structured with proper metadata fields."""
    body = get_digital_body()
    c = {
        "goal": "consolidate memory cache",
        "origin": "failed memory recall",
        "expected_value": 0.8,
        "risk": 0.05
    }
    body.register_commitment(c)
    state = body.get_state_dict()
    
    active = state["active_commitments"]
    matching = [item for item in active if item.get("goal") == "consolidate memory cache"]
    assert len(matching) == 1
    
    item = matching[0]
    assert item["origin"] == "failed memory recall"
    assert item["expected_value"] == 0.8
    assert item["risk"] == 0.05
    assert item["resource_budget"] == "low"
    assert "success_metric" in item
    assert "current_plan" in item
    
    # Resolve with evidence and postmortem
    body.resolve_commitment(item["id"], "completed", evidence="accuracy improved by 10%", postmortem="cache compacted successfully")
    assert item["evidence_of_completion"] == "accuracy improved by 10%"
    assert item["postmortem"] == "cache compacted successfully"


def test_action_authority_gating() -> None:
    """Verifies that actions are correctly gated based on safe, medium, high permissions."""
    body = get_digital_body()
    
    # Safe actions should always be authorized
    assert body.is_action_authorized("reflect") is True
    assert body.is_action_authorized("simulate plan") is True
    
    # Medium actions require specific permissions
    body.permissions["file_write"] = True
    assert body.is_action_authorized("file_write") is True
    
    body.permissions["file_write"] = False
    assert body.is_action_authorized("file_write") is False
    
    # High actions should be strictly gated or blocked
    assert body.is_action_authorized("commit_code") is True  # self_modification is True by default
    
    body.permissions["self_modification"] = False
    assert body.is_action_authorized("commit_code") is False
    
    # Untested high-risk features default to block
    assert body.is_action_authorized("write_production_code") is False

