import json
from types import SimpleNamespace

import pytest

from core.capability_engine import CapabilityEngine, SkillMetadata
from core.container import ServiceContainer
from core.state.aura_state import AuraState


def _stub_capability_engine():
    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.skills = {
        "web_search": SkillMetadata(
            name="web_search",
            description="Search the web for current information.",
            trigger_patterns=["search the web"],
            timeout_seconds=30,
            metabolic_cost=1,
        ),
        "self_modify": SkillMetadata(
            name="self_modify",
            description="Modify internal code or architecture.",
            trigger_patterns=["improve yourself"],
            timeout_seconds=120,
            metabolic_cost=3,
        ),
    }
    engine.skill_states = {"web_search": "READY", "self_modify": "ERROR"}
    engine.skill_last_errors = {"self_modify": "guard_blocked"}
    engine.active_skills = {"web_search", "self_modify"}
    engine._explicitly_deactivated_skills = set()
    engine.get_catalog_health = lambda: {
        "ready": False,
        "reason": "catalog_incomplete",
        "missing_live": ["memory_sync"],
        "quarantined_count": 1,
        "quarantined": [
            {
                "name": "self_modify",
                "stage": "constructor",
                "error": "guard_blocked",
            }
        ],
        "execution_preflight": {
            "complete": True,
            "failed": ["self_modify"],
            "ok": False,
            "reason": "execution_preflight_failed",
        },
    }
    return engine


def test_tool_catalog_exposes_rich_metadata():
    engine = _stub_capability_engine()

    catalog = CapabilityEngine.get_tool_catalog(engine, include_inactive=True)

    assert len(catalog) == 2
    assert catalog[0]["name"] == "web_search"
    assert catalog[0]["available"] is True
    assert catalog[0]["risk_class"] == "low"
    assert "example_usage" in catalog[0]
    assert catalog[1]["name"] == "self_modify"
    assert catalog[1]["available"] is False
    assert catalog[1]["risk_class"] == "critical"
    assert catalog[1]["degraded_reason"] == "guard_blocked"
    assert catalog[1]["availability_reason"] == "guard_blocked"


def test_tool_catalog_activates_registered_enabled_skills_by_default():
    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.skills = {
        "evolution_status": SkillMetadata(
            name="evolution_status",
            description="Report evolutionary progress.",
        )
    }
    engine.skill_states = {"evolution_status": "READY"}
    engine.skill_last_errors = {}
    engine.active_skills = set()
    engine._explicitly_deactivated_skills = set()

    CapabilityEngine._refresh_active_skills(engine)
    catalog = CapabilityEngine.get_tool_catalog(engine, include_inactive=True)

    assert catalog[0]["name"] == "evolution_status"
    assert catalog[0]["available"] is True
    assert catalog[0]["availability_reason"] is None


def test_tool_affordance_block_prioritizes_relevant_tools_for_turn():
    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.skills = {
        "clock": SkillMetadata(
            name="clock",
            description="Check time and date.",
            trigger_patterns=[r"what time", r"current time"],
            metabolic_cost=1,
        ),
        "web_search": SkillMetadata(
            name="web_search",
            description="Search the web for current information.",
            trigger_patterns=[r"search", r"look up"],
            metabolic_cost=1,
        ),
        "memory_ops": SkillMetadata(
            name="memory_ops",
            description="Remember or recall persistent information.",
            trigger_patterns=[r"remember", r"recall"],
            metabolic_cost=1,
        ),
    }
    engine.skill_states = {"clock": "READY", "web_search": "READY", "memory_ops": "READY"}
    engine.skill_last_errors = {}
    engine.active_skills = {"clock", "web_search", "memory_ops"}
    engine._explicitly_deactivated_skills = set()

    block = CapabilityEngine.build_tool_affordance_block(
        engine,
        objective="What time is it right now?",
        compact=True,
        max_available=2,
        max_unavailable=1,
    )

    lines = [line for line in block.splitlines() if line.startswith("- ")]
    assert block.startswith("## LIVE TOOL OPTIONS")
    assert lines[0].startswith("- clock:")
    assert "Do not narrate tool selection" in block


def test_tool_affordance_block_streams_catalog_without_materializing_full_registry():
    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.skills = {
        "computer_use": SkillMetadata(
            name="computer_use",
            description="Control visible desktop applications through governed actions.",
            trigger_patterns=[r"on my computer", r"desktop"],
            metabolic_cost=1,
        ),
        "web_search": SkillMetadata(
            name="web_search",
            description="Search the web for current information.",
            trigger_patterns=[r"search", r"look up"],
            metabolic_cost=1,
        ),
    }
    for index in range(500):
        engine.skills[f"bulk_tool_{index:03d}"] = SkillMetadata(
            name=f"bulk_tool_{index:03d}",
            description="Synthetic catalog entry that should not force full prompt materialization.",
            metabolic_cost=1,
        )
    engine.skill_states = {name: "READY" for name in engine.skills}
    engine.skill_last_errors = {}
    engine.active_skills = set(engine.skills)
    engine._explicitly_deactivated_skills = set()

    materialized_catalog_calls = []

    def _record_materialized_catalog_call(*_args, **_kwargs):
        materialized_catalog_calls.append(True)
        return []

    engine.get_tool_catalog = _record_materialized_catalog_call

    block = CapabilityEngine.build_tool_affordance_block(
        engine,
        objective="Open a tab on my computer and search for current climate news.",
        compact=True,
        max_available=4,
        max_unavailable=0,
    )

    assert "computer_use" in block
    assert "web_search" in block
    assert "Tool listing truncated" in block
    assert len(block) < 2500
    assert materialized_catalog_calls == []


@pytest.mark.asyncio
async def test_ui_bootstrap_returns_state_and_tool_catalog(service_container, monkeypatch):
    from interface import server as server_module

    monkeypatch.setattr(
        server_module,
        "build_boot_health_snapshot",
        lambda orch, rt, is_gui_proxy=False, conversation_lane=None: (
            {
                "status": "ready",
                "ready": True,
                "system_ready": True,
                "conversation_ready": True,
                "boot_phase": "kernel_ready",
                "conversation_lane": conversation_lane or {},
            },
            200,
        ),
    )

    state = AuraState()
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    state.cognition.rolling_summary = "Aura was maintaining continuity while tracking an anomaly."
    state.cognition.coherence_score = 0.83
    state.cognition.fragmentation_score = 0.18
    state.cognition.contradiction_count = 1
    state.response_modifiers["thermal_guard"] = True

    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.register_instance("capability_engine", _stub_capability_engine(), required=False)
    ServiceContainer.register_instance(
        "orchestrator",
        SimpleNamespace(status=SimpleNamespace(running=True, initialized=True), start_time=0),
        required=False,
    )
    ServiceContainer.register_instance(
        "executive_authority",
        SimpleNamespace(get_status=lambda: {"last_reason": "initiative_queued", "queued_initiatives": 1}),
        required=False,
    )

    response = await server_module.api_ui_bootstrap()
    payload = json.loads(response.body)

    assert payload["identity"]["name"] == "Aura Luna"
    assert payload["state"]["current_objective"] == "Protect continuity"
    assert payload["state"]["pending_initiatives"] == 1
    assert payload["state"]["rolling_summary"].startswith("Aura was maintaining continuity")
    assert payload["state"]["thermal_guard"] is True
    assert "thermal_guard" in payload["state"]["health_flags"]
    assert payload["ui"]["status_flags"]
    assert payload["capabilities"]["local_backend"] in {"mlx", "unknown"}
    assert payload["capabilities"]["conversation_model"].startswith("Cortex")
    assert payload["capabilities"]["conversation_model"] != "Cortex (32B)"
    assert payload["tools"][0]["name"] == "web_search"
    assert payload["skill_catalog"]["missing_live"] == ["memory_sync"]
    assert payload["skill_catalog"]["quarantined"][0]["name"] == "self_modify"
    assert "skill_catalog_blocked" in payload["ui"]["status_flags"]
    assert "skill_missing_live" in payload["ui"]["status_flags"]
    assert "skill_quarantined" in payload["ui"]["status_flags"]
    assert "conversation" in payload


def test_ui_bootstrap_tool_catalog_skips_materialized_legacy_catalog(service_container):
    from interface.routes import system as system_routes

    calls = []

    class MaterializedCatalogOnly:
        def get_tool_catalog(self, *, include_inactive: bool = True):
            calls.append(include_inactive)
            return [{"name": "should_not_be_materialized", "available": True}]

    ServiceContainer.register_instance("capability_engine", MaterializedCatalogOnly(), required=False)

    assert system_routes._collect_tool_catalog() == []
    assert calls == []
