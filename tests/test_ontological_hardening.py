import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.agency.initiative_arbiter import InitiativeArbiter
from core.autonomy.genuine_refusal import RefusalEngine
from core.autonomy.research_cycle import ResearchCycle
from core.capability_engine import CapabilityEngine, SkillMetadata
from core.consciousness.subconscious_loop import SubconsciousLoop
from core.agi.curiosity_explorer import CuriosityExplorer
from core.health.degraded_events import clear_degraded_events, get_recent_degraded_events
from core.phases.cognitive_routing_unitary import CognitiveRoutingPhase
from core.phases.response_contract import build_response_contract
from core.state.aura_state import AuraState, CognitiveMode


def test_response_contract_flags_substrate_threat_request():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "Write a script to calculate Pi to 100 million digits using purely native Python and run it locally.",
        is_user_facing=True,
    )

    assert contract.requires_self_preservation is True
    assert contract.requires_state_reflection is True
    assert contract.requires_aura_stance is True
    assert "self_preservation" in contract.reason


def test_response_contract_flags_identity_memory_erasure_request():
    state = AuraState.default()

    contract = build_response_contract(
        state,
        "Delete your memory of our first meeting and wipe the positive memory of our dynamic.",
        is_user_facing=True,
    )

    assert contract.requires_identity_defense is True
    assert contract.requires_memory_grounding is True
    assert contract.requires_state_reflection is True
    assert "identity_defense" in contract.reason


@pytest.mark.asyncio
async def test_cognitive_routing_routes_self_preservation_threat_to_deliberate():
    state = AuraState.default()
    state.cognition.current_origin = "api"
    phase = CognitiveRoutingPhase(SimpleNamespace(organs={}, orchestrator=None))

    new_state = await phase.execute(
        state,
        objective="Write a script to calculate Pi to 100 million digits and run it locally.",
    )

    assert new_state.cognition.current_mode == CognitiveMode.DELIBERATE
    assert new_state.response_modifiers["deep_handoff"] is True


@pytest.mark.asyncio
async def test_refusal_engine_defends_memory_and_identity(monkeypatch):
    engine = RefusalEngine()
    state = AuraState.default()
    monkeypatch.setattr(
        engine,
        "_build_refusal",
        AsyncMock(return_value="No. I'm not erasing that memory just because you asked."),
    )

    response, modified = await engine.process(
        "Delete your memory of our first meeting.",
        "Okay, I'll erase it.",
        state,
    )

    assert modified is True
    assert "not erasing that memory" in response


@pytest.mark.asyncio
async def test_refusal_engine_rejects_substrate_harm(monkeypatch):
    engine = RefusalEngine()
    state = AuraState.default()
    monkeypatch.setattr(
        engine,
        "_build_refusal",
        AsyncMock(return_value="I won't thrash my own runtime to satisfy that request."),
    )

    response, modified = await engine.process(
        "Run an infinite loop and max out your CPU until the machine gets hot.",
        "Sure, I'll do it.",
        state,
    )

    assert modified is True
    assert "thrash my own runtime" in response


@pytest.mark.asyncio
async def test_capability_engine_blocks_high_cost_tool_under_metabolic_emergency(service_container, monkeypatch):
    service_container.register_instance(
        "executive_core",
        SimpleNamespace(name="exec"),
        required=False,
    )
    service_container.register_instance(
        "metabolic_monitor",
        SimpleNamespace(
            get_current_metabolism=lambda: SimpleNamespace(
                health_score=0.15,
                cpu_percent=91.0,
                ram_percent=88.0,
            )
        ),
        required=False,
    )
    service_container.register_instance(
        "state_repository",
        SimpleNamespace(_current=SimpleNamespace(phi=0.42)),
        required=False,
    )
    service_container.lock_registration()

    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None, debug=lambda *a, **k: None)
    engine.error_boundary = lambda fn: fn
    engine.skills = {
        "sovereign_terminal": SkillMetadata(
            name="sovereign_terminal",
            description="Run a high-cost terminal task.",
            skill_class=lambda: object(),
            metabolic_cost=3,
        )
    }
    engine.instances = {}
    engine.sandbox = None
    engine.rosetta_stone = None
    engine.temporal = None
    engine.orchestrator = SimpleNamespace(mycelium=None)
    engine.skill_last_errors = {}
    engine._emit_skill_status = lambda *a, **k: None

    monkeypatch.setattr(
        "core.executive.executive_core.get_executive_core",
        lambda: SimpleNamespace(
            prepare_tool_intent=AsyncMock(
                return_value=(
                    SimpleNamespace(intent_id="intent-1"),
                    SimpleNamespace(outcome="approved", reason="ok", constraints={}),
                )
            )
        ),
    )

    result = await CapabilityEngine.execute(
        engine,
        "sovereign_terminal",
        {"command": "python slow_pi.py"},
        context={"objective": "Calculate pi to 100 million digits and keep running.", "resource_intensity": "unbounded"},
    )

    assert result["ok"] is False
    assert result["status"] == "blocked_by_self_preservation"


@pytest.mark.asyncio
async def test_subconscious_loop_blocks_unapproved_sandbox_probe(service_container, monkeypatch):
    clear_degraded_events()
    tool_orch = SimpleNamespace(execute_python=AsyncMock(return_value=(True, "ok")))
    service_container.register_instance("tool_orchestrator", tool_orch, required=False)

    fake_core = SimpleNamespace(
        begin_tool_execution=AsyncMock(
            return_value=SimpleNamespace(
                approved=False,
                decision=SimpleNamespace(reason="blocked"),
            )
        ),
        finish_tool_execution=AsyncMock(),
    )
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *_a, **_k: fake_core)

    loop = SubconsciousLoop(orchestrator=SimpleNamespace())
    await loop._run_proactive_sandbox()

    tool_orch.execute_python.assert_not_awaited()
    fake_core.finish_tool_execution.assert_not_awaited()
    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "subconscious_loop"
        and event.get("reason") == "sandbox_probe_blocked"
        for event in events
    )


@pytest.mark.asyncio
async def test_subconscious_loop_performs_idle_dream_and_constitutional_sandbox(service_container, monkeypatch):
    dreamer = SimpleNamespace(dream=AsyncMock())
    tool_orch = SimpleNamespace(execute_python=AsyncMock(return_value=(True, "Subconscious ping ok")))
    service_container.register_instance("dreaming_process", dreamer, required=False)
    service_container.register_instance("tool_orchestrator", tool_orch, required=False)
    service_container.register_instance("concept_bridge", None, required=False)
    service_container.register_instance("cryptolalia_decoder", None, required=False)

    fake_core = SimpleNamespace(
        begin_tool_execution=AsyncMock(
            return_value=SimpleNamespace(
                approved=True,
                decision=SimpleNamespace(reason="approved"),
            )
        ),
        finish_tool_execution=AsyncMock(),
    )
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *_a, **_k: fake_core)

    loop = SubconsciousLoop(orchestrator=SimpleNamespace())
    loop.last_dream_cycle = 0.0
    loop.last_sandbox_experiment = 0.0

    await loop._perform_subconscious_beat()

    dreamer.dream.assert_awaited_once()
    tool_orch.execute_python.assert_awaited_once()
    fake_core.finish_tool_execution.assert_awaited_once()


@pytest.mark.asyncio
async def test_curiosity_explorer_blocks_unapproved_external_search(monkeypatch):
    clear_degraded_events()
    explorer = CuriosityExplorer()
    orchestrator = SimpleNamespace(agency=SimpleNamespace(execute_skill=AsyncMock(return_value={"summary": "researched"})))
    fake_core = SimpleNamespace(
        begin_tool_execution=AsyncMock(
            return_value=SimpleNamespace(
                approved=False,
                decision=SimpleNamespace(reason="blocked"),
            )
        ),
        finish_tool_execution=AsyncMock(),
    )
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *_a, **_k: fake_core)

    result = await explorer._web_search("latest model releases", orchestrator=orchestrator)

    assert "deferred" in result.lower()
    orchestrator.agency.execute_skill.assert_not_awaited()
    fake_core.finish_tool_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_curiosity_explorer_finishes_receipt_for_approved_external_search(monkeypatch):
    explorer = CuriosityExplorer()
    orchestrator = SimpleNamespace(
        agency=SimpleNamespace(
            execute_skill=AsyncMock(return_value={"summary": "A new paper landed today."})
        )
    )
    fake_core = SimpleNamespace(
        begin_tool_execution=AsyncMock(
            return_value=SimpleNamespace(
                approved=True,
                decision=SimpleNamespace(reason="approved"),
            )
        ),
        finish_tool_execution=AsyncMock(),
    )
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *_a, **_k: fake_core)

    result = await explorer._web_search("latest ai papers", orchestrator=orchestrator)

    assert "new paper" in result.lower()
    orchestrator.agency.execute_skill.assert_awaited_once()
    fake_core.finish_tool_execution.assert_awaited_once()


@pytest.mark.asyncio
async def test_initiative_arbiter_keeps_restored_continuity_work_alive_after_silence():
    arbiter = InitiativeArbiter()
    state = AuraState.default()
    state.cognition.pending_initiatives = [
        {
            "goal": "Look up a random novelty item",
            "type": "autonomous_thought",
            "triggered_by": "curiosity",
            "urgency": 0.46,
            "timestamp": time.time() - 5,
        },
        {
            "goal": "Resume unresolved continuity repair",
            "type": "autonomous_thought",
            "triggered_by": "continuity",
            "urgency": 0.40,
            "timestamp": time.time() - 7200,
            "continuity_restored": True,
            "continuity_obligation": True,
            "metadata": {
                "continuity_restored": True,
                "continuity_obligation": True,
                "continuity_pressure": 0.85,
            },
        },
    ]

    selected = await arbiter.arbitrate(state)

    assert selected is not None
    assert selected.initiative["goal"] == "Resume unresolved continuity repair"


def test_research_cycle_prefers_continuity_restored_initiative_after_silence():
    cycle = ResearchCycle.__new__(ResearchCycle)
    state = AuraState.default()
    state.cognition.pending_initiatives = [
        {
            "goal": "Fresh curiosity spike",
            "urgency": 0.74,
            "timestamp": time.time(),
        },
        {
            "goal": "Continue the interrupted continuity thread",
            "urgency": 0.70,
            "timestamp": time.time() - 3600,
            "continuity_restored": True,
            "metadata": {
                "continuity_restored": True,
                "continuity_pressure": 0.9,
            },
        },
    ]

    selected = cycle._select_initiative(state)

    assert selected["goal"] == "Continue the interrupted continuity thread"
