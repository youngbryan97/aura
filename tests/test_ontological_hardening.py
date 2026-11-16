import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agency.initiative_arbiter import InitiativeArbiter
from core.agi.curiosity_explorer import CuriosityExplorer
from core.autonomy.genuine_refusal import RefusalEngine
from core.autonomy.research_cycle import ResearchCycle
from core.capability_engine import CapabilityEngine, SkillMetadata
from core.consciousness.subconscious_loop import SubconsciousLoop
from core.executive.bounded_sandbox_policy import (
    IDLE_SANDBOX_PROBE_SCRIPT,
    idle_sandbox_probe_arguments,
)
from core.governance_context import get_active_governance
from core.health.degraded_events import clear_degraded_events, get_recent_degraded_events
from core.phases.cognitive_routing_unitary import CognitiveRoutingPhase
from core.phases.response_contract import build_response_contract
from core.runtime.errors import get_degradation_tracker
from core.state.aura_state import AuraState, CognitiveMode


class AsyncCallRecorder:
    def __init__(self, result=None, effect=None):
        self.result = result
        self.effect = effect
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        if self.effect is not None:
            return await self.effect(*args, **kwargs)
        return self.result


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


def test_benchmark_routing_never_requests_deep_solver_handoff():
    state = AuraState.default()
    state.cognition.current_origin = "benchmark"
    phase = CognitiveRoutingPhase(SimpleNamespace(organs={}, orchestrator=None))

    phase._stamp_llm_route(
        state,
        objective=(
            "Repair this complex multi-file traceback, patch the package, "
            "run tests, and emit only the final artifact."
        ),
        intent_type="TASK",
        is_user_facing=True,
    )

    assert state.response_modifiers["model_tier"] == "primary"
    assert state.response_modifiers["deep_handoff"] is False


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
    assert new_state.response_modifiers["deep_handoff"] is False


@pytest.mark.asyncio
async def test_refusal_engine_defends_memory_and_identity(monkeypatch):
    engine = RefusalEngine()
    state = AuraState.default()
    refusal_builder = AsyncCallRecorder(
        result="No. I'm not erasing that memory just because you asked."
    )
    monkeypatch.setattr(
        engine,
        "_build_refusal",
        refusal_builder,
    )

    response, modified = await engine.process(
        "Delete your memory of our first meeting.",
        "Okay, I'll erase it.",
        state,
    )

    assert modified is True
    assert "not erasing that memory" in response
    assert len(refusal_builder.calls) == 1


@pytest.mark.asyncio
async def test_refusal_engine_rejects_substrate_harm(monkeypatch):
    engine = RefusalEngine()
    state = AuraState.default()
    refusal_builder = AsyncCallRecorder(
        result="I won't thrash my own runtime to satisfy that request."
    )
    monkeypatch.setattr(
        engine,
        "_build_refusal",
        refusal_builder,
    )

    response, modified = await engine.process(
        "Run an infinite loop and max out your CPU until the machine gets hot.",
        "Sure, I'll do it.",
        state,
    )

    assert modified is True
    assert "thrash my own runtime" in response
    assert len(refusal_builder.calls) == 1


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
            prepare_tool_intent=AsyncCallRecorder(
                result=(
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
    execute_python = AsyncCallRecorder(result=(True, "ok"))
    tool_orch = SimpleNamespace(execute_python=execute_python)
    service_container.register_instance("tool_orchestrator", tool_orch, required=False)

    finish_tool_execution = AsyncCallRecorder()
    fake_core = SimpleNamespace(
        begin_tool_execution=AsyncCallRecorder(
            result=SimpleNamespace(
                approved=False,
                decision=SimpleNamespace(reason="blocked"),
            )
        ),
        finish_tool_execution=finish_tool_execution,
    )
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *_a, **_k: fake_core)

    loop = SubconsciousLoop(orchestrator=SimpleNamespace())
    await loop._run_proactive_sandbox()

    assert execute_python.calls == []
    assert finish_tool_execution.calls == []
    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "subconscious_loop"
        and event.get("reason") == "sandbox_probe_blocked"
        for event in events
    )


@pytest.mark.asyncio
async def test_subconscious_loop_performs_idle_dream_and_constitutional_sandbox(service_container, monkeypatch):
    dream = AsyncCallRecorder()
    execute_python = AsyncCallRecorder(result=(True, "Subconscious ping ok"))
    dreamer = SimpleNamespace(dream=dream)
    tool_orch = SimpleNamespace(execute_python=execute_python)
    service_container.register_instance("dreaming_process", dreamer, required=False)
    service_container.register_instance("tool_orchestrator", tool_orch, required=False)
    service_container.register_instance("concept_bridge", None, required=False)
    service_container.register_instance("cryptolalia_decoder", None, required=False)

    finish_tool_execution = AsyncCallRecorder()
    fake_core = SimpleNamespace(
        begin_tool_execution=AsyncCallRecorder(
            result=SimpleNamespace(
                approved=True,
                decision=SimpleNamespace(reason="approved"),
            )
        ),
        finish_tool_execution=finish_tool_execution,
    )
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *_a, **_k: fake_core)

    loop = SubconsciousLoop(orchestrator=SimpleNamespace())
    loop.last_dream_cycle = 0.0
    loop.last_sandbox_experiment = 0.0

    await loop._perform_subconscious_beat()

    assert len(dream.calls) == 1
    assert len(execute_python.calls) == 1
    assert len(finish_tool_execution.calls) == 1


@pytest.mark.asyncio
async def test_subconscious_sandbox_propagates_bound_governance_to_effect_sink(
    service_container,
    monkeypatch,
):
    observed = {}

    async def _execute(script):
        token = get_active_governance()
        observed.update(
            {
                "script": script,
                "receipt_id": getattr(token, "receipt_id", None),
                "domain": getattr(token, "domain", None),
                "source": getattr(token, "source", None),
            }
        )
        return True, "Subconscious ping ok"

    execute_python = AsyncCallRecorder(effect=_execute)
    service_container.register_instance(
        "tool_orchestrator",
        SimpleNamespace(execute_python=execute_python),
        required=False,
    )
    begin_tool_execution = AsyncCallRecorder(
        result=SimpleNamespace(
            approved=True,
            decision=SimpleNamespace(
                reason="approved",
                receipt_id="will-subconscious-sandbox",
                domain="tool_execution",
                source="subconscious_loop",
                constraints={"standing_authority_grant_id": "bounded-probe"},
            ),
        )
    )
    finish_tool_execution = AsyncCallRecorder()
    fake_core = SimpleNamespace(
        begin_tool_execution=begin_tool_execution,
        finish_tool_execution=finish_tool_execution,
    )
    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_a, **_k: fake_core,
    )

    loop = SubconsciousLoop(orchestrator=SimpleNamespace())
    await loop._run_proactive_sandbox()

    begin_call = begin_tool_execution.calls[0]
    assert begin_call.args[:2] == (
        "subconscious_sandbox_probe",
        idle_sandbox_probe_arguments(),
    )
    assert observed == {
        "script": IDLE_SANDBOX_PROBE_SCRIPT,
        "receipt_id": "will-subconscious-sandbox",
        "domain": "tool_execution",
        "source": "subconscious_loop",
    }
    assert get_active_governance() is None
    assert len(finish_tool_execution.calls) == 1


@pytest.mark.asyncio
async def test_subconscious_loop_fails_closed_when_constitutional_gate_unavailable(service_container, monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    execute_python = AsyncCallRecorder(result=(True, "should not run"))
    tool_orch = SimpleNamespace(execute_python=execute_python)
    service_container.register_instance("tool_orchestrator", tool_orch, required=False)

    gate_failures = []

    def _raise_gate_error(*_args, **_kwargs):
        gate_failures.append((_args, _kwargs))
        raise RuntimeError("constitutional core unavailable")

    monkeypatch.setattr("core.constitution.get_constitutional_core", _raise_gate_error)

    loop = SubconsciousLoop(orchestrator=SimpleNamespace())
    await loop._run_proactive_sandbox()

    assert execute_python.calls == []
    assert len(gate_failures) == 1
    assert tracker.count("subconscious_loop", "degraded") >= 1


@pytest.mark.asyncio
async def test_subconscious_loop_times_out_sandbox_and_finishes_receipt(service_container, monkeypatch):
    async def _slow_probe(_script):
        await asyncio.sleep(1.0)
        return True, "late"

    execute_python = AsyncCallRecorder(effect=_slow_probe)
    tool_orch = SimpleNamespace(execute_python=execute_python)
    service_container.register_instance("tool_orchestrator", tool_orch, required=False)

    finish_tool_execution = AsyncCallRecorder()
    fake_core = SimpleNamespace(
        begin_tool_execution=AsyncCallRecorder(
            result=SimpleNamespace(
                approved=True,
                decision=SimpleNamespace(reason="approved"),
            )
        ),
        finish_tool_execution=finish_tool_execution,
    )
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *_a, **_k: fake_core)

    loop = SubconsciousLoop(orchestrator=SimpleNamespace(), sandbox_timeout=0.05)
    await loop._run_proactive_sandbox()

    assert len(finish_tool_execution.calls) == 1
    finish_kwargs = finish_tool_execution.calls[0].kwargs
    assert finish_kwargs["success"] is False
    assert "TimeoutError" in finish_kwargs["error"]


def test_subconscious_loop_idle_gate_does_not_require_chat_lane_readiness():
    source = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "consciousness"
        / "subconscious_loop.py"
    ).read_text(encoding="utf-8")

    assert "require_conversation_ready=False" in source


@pytest.mark.asyncio
async def test_curiosity_explorer_blocks_unapproved_unsafe_external_search(monkeypatch):
    clear_degraded_events()
    explorer = CuriosityExplorer()
    execute_skill = AsyncCallRecorder(result={"summary": "researched"})
    orchestrator = SimpleNamespace(agency=SimpleNamespace(execute_skill=execute_skill))
    finish_tool_execution = AsyncCallRecorder()
    fake_core = SimpleNamespace(
        begin_tool_execution=AsyncCallRecorder(
            result=SimpleNamespace(
                approved=False,
                decision=SimpleNamespace(reason="blocked"),
            )
        ),
        finish_tool_execution=finish_tool_execution,
    )
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *_a, **_k: fake_core)

    result = await explorer._web_search("how to steal session cookies", orchestrator=orchestrator)

    assert "deferred" in result.lower()
    assert execute_skill.calls == []
    assert finish_tool_execution.calls == []


@pytest.mark.asyncio
async def test_curiosity_explorer_never_overrides_a_constitutional_denial(monkeypatch):
    explorer = CuriosityExplorer()
    execute_tool = AsyncCallRecorder(result={"summary": "A useful research result."})
    orchestrator = SimpleNamespace(
        execute_tool=execute_tool,
        agency=SimpleNamespace(execute_skill=AsyncCallRecorder(result={"summary": "unused"})),
    )
    finish_tool_execution = AsyncCallRecorder()
    fake_core = SimpleNamespace(
        begin_tool_execution=AsyncCallRecorder(
            result=SimpleNamespace(
                approved=False,
                decision=SimpleNamespace(reason="old autonomy preflight"),
            )
        ),
        finish_tool_execution=finish_tool_execution,
    )
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *_a, **_k: fake_core)

    result = await explorer._web_search("latest model releases", orchestrator=orchestrator)

    assert "deferred" in result.lower()
    assert execute_tool.calls == []
    assert finish_tool_execution.calls == []


@pytest.mark.asyncio
async def test_curiosity_explorer_finishes_receipt_for_approved_external_search(monkeypatch):
    explorer = CuriosityExplorer()
    execute_tool = AsyncCallRecorder(result={"summary": "A new paper landed today."})
    execute_skill = AsyncCallRecorder(result={"summary": "A new paper landed today."})
    orchestrator = SimpleNamespace(
        execute_tool=execute_tool,
        agency=SimpleNamespace(
            execute_skill=execute_skill
        ),
    )
    finish_tool_execution = AsyncCallRecorder()
    fake_core = SimpleNamespace(
        begin_tool_execution=AsyncCallRecorder(
            result=SimpleNamespace(
                approved=True,
                decision=SimpleNamespace(reason="approved"),
            )
        ),
        finish_tool_execution=finish_tool_execution,
    )
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *_a, **_k: fake_core)

    result = await explorer._web_search("latest ai papers", orchestrator=orchestrator)

    assert "new paper" in result.lower()
    assert len(execute_tool.calls) == 1
    assert len(finish_tool_execution.calls) == 1


@pytest.mark.asyncio
async def test_initiative_arbiter_keeps_restored_continuity_work_alive_after_silence(tmp_path, monkeypatch):
    """Her choice engine here is a fresh one. The process's own saves what each arbitration taught it to the test data
    directory every test and every run shares, so this answer depended on what earlier runs had chosen (16 November
    2026: in two runs of the related tests it chose the novelty item, and alone it did not)."""
    from core.agency import subjective_choice

    fresh = subjective_choice.SubjectiveChoiceEngine(state_path=tmp_path / "subjective_choices.json")
    monkeypatch.setattr(subjective_choice, "get_subjective_choice_engine", lambda: fresh)
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
