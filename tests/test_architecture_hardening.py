import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import networkx as nx
import pytest

from core.utils import output_gate as output_gate_module
from core.container import ServiceContainer
from core.capability_engine import CapabilityEngine, SkillMetadata
from core.executive.executive_core import DecisionOutcome, DecisionRecord, ExecutiveCore
from core.executive.executive_ledger import ExecutiveLedger
from core.health.degraded_events import clear_degraded_events, get_recent_degraded_events
from core.memory import db_writer_queue as dbw_module
from core.memory.knowledge_graph import PersistentKnowledgeGraph
from core.memory.memory_facade import MemoryFacade
from core.proactive_communication import (
    EmotionalState,
    InterruptionUrgency,
    ProactiveCommunicationManager,
    ProactiveMessage,
)
from core.consciousness.executive_closure import ExecutiveClosureEngine
from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision, SubstrateAuthority
from core.kernel.aura_kernel import AuraKernel, KernelConfig
from core.kernel.bridge import LegacyPhase
from core.agency_core import AgencyCore, AgencyState
from core.autonomous_initiative_loop import AutonomousInitiativeLoop
from core.long_term_memory_engine import LongTermMemoryEngine
from core.motivation.goal_hierarchy import GoalHierarchy
from core.senses.continuous_perception import ContinuousPerceptionEngine
from core.social.social_imagination import SocialImagination
from core.state.aura_state import AuraState
from core.terminal_chat import TerminalFallbackChat
from core.utils.output_gate import AutonomousOutputGate
from core.world_model.acg import ActionConsequenceGraph
from core.world_model.belief_graph import BeliefGraph
from core.world_model.goal_beliefs import GoalBeliefManager
from core.brain.identity import IdentityService


class _FakeBeliefGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.self_node_id = "AURA_SELF"
        self.graph.add_node(self.self_node_id)

    def update_belief(self, source, relation, target, confidence_score, centrality, is_goal):
        self.graph.add_edge(
            source,
            target,
            relation=relation,
            confidence=confidence_score,
            centrality=centrality,
            is_goal=is_goal,
        )

    def _save(self):
        return None


@pytest.mark.asyncio
async def test_serialized_db_writer_keeps_db_paths_isolated(tmp_path):
    writer = dbw_module.SerializedDBWriter()
    try:
        db_a = tmp_path / "a.db"
        db_b = tmp_path / "b.db"

        await writer.execute(str(db_a), "CREATE TABLE IF NOT EXISTS t (value TEXT)")
        await writer.execute(str(db_b), "CREATE TABLE IF NOT EXISTS t (value TEXT)")
        await writer.execute(str(db_a), "INSERT INTO t (value) VALUES (?)", ("alpha",))
        await writer.execute(str(db_b), "INSERT INTO t (value) VALUES (?)", ("beta",))
        writer.flush_and_checkpoint()

        row_a = writer.fetchone(str(db_a), "SELECT value FROM t LIMIT 1")
        row_b = writer.fetchone(str(db_b), "SELECT value FROM t LIMIT 1")

        assert row_a["value"] == "alpha"
        assert row_b["value"] == "beta"
    finally:
        writer.shutdown()
        dbw_module._instance = None


def test_goal_belief_manager_dedupes_normalized_goal_content():
    belief_graph = _FakeBeliefGraph()
    belief_graph.graph.add_edge(
        belief_graph.self_node_id,
        "i should maintain continuity of experience and memory.",
        is_goal=True,
        confidence=0.9,
    )

    GoalBeliefManager(belief_graph)

    matching = [
        target
        for _source, target, data in belief_graph.graph.edges(data=True)
        if data.get("is_goal")
        and " ".join(str(target).strip().lower().split())
        == "i should maintain continuity of experience and memory."
    ]
    assert len(matching) == 1


def test_service_container_sovereignty_seal_detects_manifest_drift(tmp_path, monkeypatch):
    seal_path = tmp_path / "sovereignty_seal.json"
    monkeypatch.setattr(ServiceContainer, "_seal_path", classmethod(lambda cls: seal_path))

    ServiceContainer.clear()
    ServiceContainer.register_instance("alpha", SimpleNamespace(name="alpha"))
    payload = ServiceContainer.write_sovereignty_seal()

    assert seal_path.exists()
    assert payload["hash"]
    assert ServiceContainer.verify_sovereignty_seal() is True

    ServiceContainer.register_instance("beta", SimpleNamespace(name="beta"))
    assert ServiceContainer.verify_sovereignty_seal() is False

    stored = json.loads(seal_path.read_text())
    assert stored["service_count"] == 1


@pytest.mark.asyncio
async def test_executive_core_tracks_and_completes_tool_intents(tmp_path, monkeypatch):
    ledger = ExecutiveLedger(tmp_path / "executive_ledger.jsonl")
    monkeypatch.setattr(ExecutiveCore, "_get_ledger", lambda self: ledger)

    exec_core = ExecutiveCore()
    intent, record = await exec_core.prepare_tool_intent("clock", {}, source="capability_engine")

    assert record.outcome.value == "approved"
    assert len(exec_core.get_active_intents()) == 1

    exec_core.complete_intent(intent.intent_id, success=True)

    assert exec_core.get_active_intents() == []
    lines = (tmp_path / "executive_ledger.jsonl").read_text().strip().splitlines()
    assert any('"event": "decision"' in line for line in lines)
    assert any('"event": "intent_complete"' in line for line in lines)


@pytest.mark.asyncio
async def test_executive_core_convenience_tool_approval_does_not_leak_active_intents(tmp_path, monkeypatch):
    ledger = ExecutiveLedger(tmp_path / "executive_ledger.jsonl")
    monkeypatch.setattr(ExecutiveCore, "_get_ledger", lambda self: ledger)

    exec_core = ExecutiveCore()
    approved, reason, constraints = await exec_core.approve_tool("clock", {}, source="system")

    assert approved is True
    assert reason
    assert isinstance(constraints, dict)
    assert exec_core.get_active_intents() == []


def test_compact_skill_result_payload_preserves_clock_fields():
    from core.kernel.upgrades_10x import _compact_skill_result_payload

    payload = _compact_skill_result_payload({
        "ok": True,
        "summary": "It is currently Tuesday, April 07, 2026 06:40 PM.",
        "readable": "Tuesday, April 07, 2026 06:40 PM",
        "time": "2026-04-07T18:40:00",
        "message": "Clock skill completed.",
    })

    assert payload["readable"] == "Tuesday, April 07, 2026 06:40 PM"
    assert payload["time"] == "2026-04-07T18:40:00"
    assert payload["message"] == "Clock skill completed."


def test_godmode_normalizes_memory_ops_params():
    from core.kernel.upgrades_10x import GodModeToolPhase

    remember = GodModeToolPhase._normalize_skill_params(
        "memory_ops",
        "Remember for future sessions that my verification codename is glass orchard.",
        {},
    )
    recall = GodModeToolPhase._normalize_skill_params(
        "memory_ops",
        "What do you remember about my verification codename?",
        {},
    )

    assert remember["action"] == "remember"
    assert remember["content"] == "Remember for future sessions that my verification codename is glass orchard."
    assert recall["action"] == "recall"
    assert recall["query"] == "What do you remember about my verification codename?"


def test_capability_engine_preserves_top_level_fields_when_unwrapping_nested_params():
    params = CapabilityEngine._normalize_execution_params(
        {
            "params": {"action": "remember"},
            "content": "Remember for future sessions that my verification codename is glass orchard.",
        }
    )

    assert params["action"] == "remember"
    assert params["content"] == "Remember for future sessions that my verification codename is glass orchard."


def test_capability_engine_detects_memory_ops_for_future_session_requests():
    engine = CapabilityEngine()

    matched = engine.detect_intent("Remember for future sessions that my verification codename is glass orchard.")

    assert "memory_ops" in matched


def test_capability_engine_detects_memory_ops_for_common_remember_typo():
    engine = CapabilityEngine()

    matched = engine.detect_intent("What do you remeber about my verification codename?")

    assert "memory_ops" in matched


def test_substrate_authority_constrains_user_memory_write_during_cortisol_crisis():
    authority = SubstrateAuthority()
    authority._get_field_coherence = lambda: 0.8
    authority._get_somatic_state = lambda content, source, priority: (0.2, 0.9, True)
    authority._get_neurochemical_constraints = lambda category: ("cortisol_crisis", ["cortisol=0.91"])

    verdict = authority.authorize(
        content="memory:remember my verification codename",
        source="user",
        category=ActionCategory.MEMORY_WRITE,
        priority=0.8,
    )

    assert verdict.decision == AuthorizationDecision.CONSTRAIN
    assert any("user_facing_memory_write_constrained" in item for item in verdict.constraints)


def test_substrate_authority_still_blocks_autonomous_memory_write_during_cortisol_crisis():
    authority = SubstrateAuthority()
    authority._get_field_coherence = lambda: 0.8
    authority._get_somatic_state = lambda content, source, priority: (0.2, 0.9, True)
    authority._get_neurochemical_constraints = lambda category: ("cortisol_crisis", ["cortisol=0.91"])

    verdict = authority.authorize(
        content="memory:background consolidation",
        source="background_consolidation",
        category=ActionCategory.MEMORY_WRITE,
        priority=0.8,
    )

    assert verdict.decision == AuthorizationDecision.BLOCK
    assert "neurochemical_cortisol_crisis: category=MEMORY_WRITE blocked" == verdict.reason


def test_capability_engine_prefers_clock_for_time_queries():
    engine = CapabilityEngine()

    matched = engine.detect_intent("What time is it right now?")

    assert "clock" in matched
    assert "environment_info" not in matched


def test_capability_engine_detects_terminal_exec_prefix():
    engine = CapabilityEngine()

    matched = engine.detect_intent("execute: printf 'hello' > /tmp/aura-proof.txt")

    assert matched[0] == "sovereign_terminal"


def test_capability_engine_prefers_manifest_to_device_for_save_to_desktop_url():
    engine = CapabilityEngine()

    matched = engine.detect_intent("save to my desktop: https://httpbin.org/image/png")

    assert matched[0] == "manifest_to_device"
    assert "file_operation" not in matched


def test_capability_engine_detects_research_about_as_web_search():
    engine = CapabilityEngine()

    matched = engine.detect_intent("research about Python 3.12 release notes key improvements")

    assert matched[0] == "web_search"


def test_social_imagination_links_private_trouble_to_public_issue(tmp_path):
    engine = SocialImagination(tmp_path / "social_imagination.json")
    text = (
        "I'm working full time and still can't afford rent in this city. "
        "It feels like no matter how hard I work, housing keeps getting more expensive."
    )

    frame = engine.analyze_text(text)

    assert frame is not None
    assert any("housing" in issue.lower() for issue in frame.public_issues)
    assert any("labor market" in issue.lower() or "workplace" in issue.lower() for issue in frame.public_issues)
    assert "housing market" in " ".join(frame.institutions).lower()
    assert frame.personal_troubles
    assert frame.positive_possibilities
    assert "private struggle" in frame.reframing.lower() or "private" in frame.reframing.lower()

    block = engine.get_context_injection("bryan", current_text=text)
    assert "SOCIAL IMAGINATION" in block
    assert "Public issues in view" in block
    assert "Personal stakes in view" in block
    assert "Positive possibilities in view" in block


def test_social_imagination_can_relate_abstract_topics_personally(tmp_path):
    engine = SocialImagination(tmp_path / "social_imagination.json")
    text = "AI is reshaping work and education faster than institutions know how to adapt."

    frame = engine.analyze_text(text)

    assert frame is not None
    assert any("technological" in issue.lower() or "institutional" in issue.lower() for issue in frame.public_issues)
    assert any("work" in angle.lower() or "learning" in angle.lower() for angle in frame.personal_angles)
    assert any("creative" in item.lower() or "learning" in item.lower() for item in frame.positive_possibilities)

    block = engine.get_context_injection("bryan", current_text=text)
    assert "Personal stakes in view" in block
    assert "daily life" in block.lower() or "agency" in block.lower()


def test_social_imagination_holds_positive_feelings_as_socially_shaped(tmp_path):
    engine = SocialImagination(tmp_path / "social_imagination.json")
    text = "I'm excited that AI could help me learn faster and do more creative work."

    frame = engine.analyze_text(text)

    assert frame is not None
    assert frame.personal_troubles == []
    assert any("creative" in item.lower() or "learning" in item.lower() for item in frame.positive_possibilities)

    block = engine.get_context_injection("bryan", current_text=text)
    assert "Positive possibilities in view" in block
    assert "hope" in block.lower() or "delight" in block.lower()


def test_terminal_fallback_drops_autonomous_message_when_executive_rejects(monkeypatch):
    terminal = TerminalFallbackChat()

    class _RejectingExecutive:
        def request_approval_sync(self, _intent):
            return DecisionRecord(
                intent_id="reject",
                outcome=DecisionOutcome.REJECTED,
                reason="coherence_lockdown_0.10",
            )

    monkeypatch.setattr(
        "core.executive.executive_core.get_executive_core",
        lambda: _RejectingExecutive(),
    )

    terminal.queue_autonomous_message("Say something autonomous")

    assert list(terminal._pending) == []


def test_terminal_fallback_keeps_autonomous_message_when_executive_approves(monkeypatch):
    terminal = TerminalFallbackChat()

    class _ApprovingExecutive:
        def request_approval_sync(self, _intent):
            return DecisionRecord(
                intent_id="approve",
                outcome=DecisionOutcome.APPROVED,
                reason="approved",
            )

    monkeypatch.setattr(
        "core.executive.executive_core.get_executive_core",
        lambda: _ApprovingExecutive(),
    )

    terminal.queue_autonomous_message("Share the queued thought")

    assert list(terminal._pending) == ["Share the queued thought"]


def test_terminal_fallback_suppresses_when_executive_unavailable_in_live_runtime(monkeypatch):
    ServiceContainer.clear()
    clear_degraded_events()
    ServiceContainer.register_instance("executive_core", SimpleNamespace(name="exec"), required=False)
    ServiceContainer.lock_registration()

    monkeypatch.setattr(
        "core.executive.executive_core.get_executive_core",
        lambda: (_ for _ in ()).throw(RuntimeError("down")),
    )

    terminal = TerminalFallbackChat()
    terminal.queue_autonomous_message("Do not queue this")

    assert list(terminal._pending) == []
    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "terminal_fallback"
        and event.get("reason") == "executive_gate_unavailable"
        for event in events
    )


def test_enqueue_message_blocks_unapproved_background_injection(orchestrator, monkeypatch):
    ServiceContainer.clear()
    clear_degraded_events()
    ServiceContainer.register_instance("executive_core", SimpleNamespace(name="exec"), required=False)
    ServiceContainer.lock_registration()

    orchestrator.message_queue = asyncio.Queue(maxsize=10)

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_initiative_sync=lambda *args, **kwargs: (False, "blocked_by_test")
        ),
    )

    orchestrator.enqueue_message("Do not queue this", origin="background")

    assert orchestrator.message_queue.qsize() == 0
    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "message_queue"
        and event.get("reason") == "background_enqueue_blocked"
        for event in events
    )


def test_late_causal_service_registration_after_lock_is_reported():
    ServiceContainer.clear()
    clear_degraded_events()
    ServiceContainer.lock_registration()

    ServiceContainer.register_instance("agency_core", SimpleNamespace(name="agency"))

    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "service_container"
        and event.get("reason") == "late_causal_registration"
        and event.get("detail") == "agency_core"
        for event in events
    )


@pytest.mark.asyncio
async def test_output_gate_reroutes_unauthorized_autonomous_primary_output(service_container):
    gate = AutonomousOutputGate()

    await gate.emit(
        "Background urge that should not hit the primary channel directly.",
        origin="growth_ladder",
        target="primary",
        metadata={"spontaneous": True, "force_user": True},
    )

    item = await gate.secondary_queue.get()
    assert item["origin"] == "growth_ladder"
    assert item["metadata"]["authority_missing"] is True
    assert item["metadata"]["authority_rerouted"] is True


@pytest.mark.asyncio
async def test_output_gate_tracks_renderer_tasks_without_leaking(service_container):
    output_gate_module._background_tasks.clear()
    started = asyncio.Event()
    release = asyncio.Event()

    async def _render(_content, _metadata):
        started.set()
        await release.wait()

    service_container.register_instance(
        "multimodal_orchestrator",
        SimpleNamespace(render=_render),
        required=False,
    )

    gate = AutonomousOutputGate()
    await gate._send_to_primary(
        "Tracked multimodal output",
        origin="user",
        metadata={"suppress_bus": True, "voice": False},
    )

    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert output_gate_module._background_tasks
    task = next(iter(output_gate_module._background_tasks))
    assert getattr(task, "_aura_supervised", False) is True
    assert getattr(task, "_aura_task_tracker", "") == "OutputGate"

    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not output_gate_module._background_tasks


@pytest.mark.asyncio
async def test_proactive_manager_suppresses_legacy_fallback_when_constitutional_runtime_live(service_container):
    service_container.register_instance("executive_core", SimpleNamespace(name="exec"), required=False)
    service_container.lock_registration()

    callback = AsyncMock()
    manager = ProactiveCommunicationManager(notification_callback=callback)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "core.consciousness.executive_authority.get_executive_authority",
            lambda _orch=None: SimpleNamespace(release_expression=AsyncMock(side_effect=RuntimeError("authority down"))),
        )
        await manager._send_msg(
            ProactiveMessage(
                content="Something proactive",
                emotion=EmotionalState.CURIOUS,
                urgency=InterruptionUrgency.MEDIUM,
            )
        )

    callback.assert_not_awaited()
    assert manager.messages_sent_today == 0


@pytest.mark.asyncio
async def test_continuous_perception_blocks_unapproved_autonomous_injection(monkeypatch):
    ServiceContainer.clear()
    clear_degraded_events()

    orchestrator = SimpleNamespace(
        _agency_core=None,
        process_user_input=AsyncMock(),
        _handle_incoming_message=AsyncMock(),
    )
    engine = ContinuousPerceptionEngine(orchestrator)
    ServiceContainer.register_instance("executive_core", SimpleNamespace(name="exec"), required=False)
    ServiceContainer.lock_registration()

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_initiative=AsyncMock(return_value=(False, "blocked", None))
        ),
    )

    engine._dispatch_spontaneous_intent("A visual observation", source="vision_delta")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    orchestrator.process_user_input.assert_not_awaited()
    orchestrator._handle_incoming_message.assert_not_awaited()
    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "continuous_perception"
        and event.get("reason") == "autonomous_intent_blocked"
        for event in events
    )


@pytest.mark.asyncio
async def test_process_unprompted_stimulus_blocks_when_constitution_rejects(orchestrator, monkeypatch):
    clear_degraded_events()
    orchestrator.process_user_input_priority = AsyncMock()

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_initiative=AsyncMock(return_value=(False, "blocked", None))
        ),
    )

    await orchestrator.process_unprompted_stimulus("vision", {}, "sudden motion")

    orchestrator.process_user_input_priority.assert_not_awaited()
    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "sensory_motor"
        and event.get("reason") == "stimulus_blocked"
        for event in events
    )


@pytest.mark.asyncio
async def test_handle_impulse_blocks_when_constitution_rejects(orchestrator, monkeypatch):
    clear_degraded_events()
    orchestrator.process_user_input_priority = AsyncMock()

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_initiative=AsyncMock(return_value=(False, "blocked", None))
        ),
    )

    await orchestrator.handle_impulse("explore_knowledge")

    orchestrator.process_user_input_priority.assert_not_awaited()
    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "autonomy"
        and event.get("reason") == "impulse_processing_blocked"
        for event in events
    )


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_autonomous_initiative_loop_defers_when_identity_continuity_mismatch(tmp_path, monkeypatch):
    ServiceContainer.clear()
    continuity_module = __import__("core.continuity", fromlist=["_CONTINUITY_PATH", "_continuity"])
    monkeypatch.setattr(continuity_module, "_CONTINUITY_PATH", tmp_path / "continuity.json")
    continuity_module._continuity = None

    from core.continuity import ContinuityEngine

    engine = ContinuityEngine()
    engine.save(reason="graceful", belief_hash="persisted-self")

    state = AuraState()
    state.cognition.modifiers = {"continuity_obligations": {"identity_mismatch": True}}
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)

    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    gate = await loop._evaluate_initiative("novel topic")

    assert gate == {"allowed": False, "reason": "identity_continuity_mismatch"}


async def test_autonomous_initiative_loop_blocks_unapproved_browser_research(monkeypatch):
    ServiceContainer.clear()
    clear_degraded_events()

    sensory_motor = SimpleNamespace(actuate_browser=AsyncMock(return_value="researched"))
    ServiceContainer.register_instance("sensory_motor_cortex", sensory_motor, required=False)
    fake_core = SimpleNamespace(
        begin_tool_execution=AsyncMock(
            return_value=SimpleNamespace(
                approved=False,
                decision=SimpleNamespace(reason="blocked"),
            )
        ),
        finish_tool_execution=AsyncMock(),
    )

    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *_args, **_kwargs: fake_core)

    # Provide attributes that background_activity_allowed needs to proceed
    import time as _time
    fake_orch = SimpleNamespace(
        _last_user_interaction_time=_time.time() - 86400,  # 24h ago (well past idle threshold)
        is_busy=False,
        _suppress_unsolicited_proactivity_until=0.0,
        _foreground_user_quiet_until=0.0,
    )
    monkeypatch.setattr(
        "core.autonomous_initiative_loop._background_initiative_allowed",
        lambda *_a, **_kw: True,
    )

    loop = AutonomousInitiativeLoop(orchestrator=fake_orch)
    await loop.trigger_gap_search("novel topic")

    sensory_motor.actuate_browser.assert_not_awaited()
    fake_core.finish_tool_execution.assert_not_awaited()
    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "autonomous_initiative_loop"
        and event.get("reason") == "research_tool_blocked"
        for event in events
    )


@pytest.mark.asyncio
async def test_autonomous_initiative_loop_reports_missing_research_tool(monkeypatch):
    ServiceContainer.clear()
    clear_degraded_events()

    import time as _time
    fake_orch = SimpleNamespace(
        _last_user_interaction_time=_time.time() - 86400,
        is_busy=False,
        _suppress_unsolicited_proactivity_until=0.0,
        _foreground_user_quiet_until=0.0,
    )
    monkeypatch.setattr(
        "core.autonomous_initiative_loop._background_initiative_allowed",
        lambda *_a, **_kw: True,
    )

    loop = AutonomousInitiativeLoop(orchestrator=fake_orch)
    await loop.trigger_gap_search("novel topic")

    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "autonomous_initiative_loop"
        and event.get("reason") == "research_tool_unavailable"
        and event.get("detail") == "sensory_motor_cortex"
        for event in events
    )


def test_identity_service_blocks_unapproved_insight_write(monkeypatch, tmp_path):
    ServiceContainer.clear()
    ServiceContainer.register_instance("executive_core", SimpleNamespace(name="exec"), required=False)
    ServiceContainer.lock_registration()

    class _RejectingExecutive:
        def request_approval_sync(self, _intent):
            return DecisionRecord(
                intent_id="reject",
                outcome=DecisionOutcome.REJECTED,
                reason="identity_guard",
            )

    monkeypatch.setattr(
        "core.executive.executive_core.get_executive_core",
        lambda: _RejectingExecutive(),
    )

    identity = IdentityService()
    identity.data_path = tmp_path / "identity.json"
    identity.state.inner_insights.clear()

    identity.add_insight("This should not persist.", source="creative_synthesis")

    assert identity.state.inner_insights == []


def test_service_container_blocks_protected_overwrite_after_lock():
    ServiceContainer.clear()
    clear_degraded_events()
    original = SimpleNamespace(name="original")
    replacement = SimpleNamespace(name="replacement")

    ServiceContainer.register_instance("executive_core", original, required=False)
    ServiceContainer.lock_registration()
    ServiceContainer.register_instance("executive_core", replacement, required=False)

    assert ServiceContainer.get("executive_core") is original
    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "service_container"
        and event.get("reason") == "protected_service_overwrite_blocked"
        and event.get("detail") == "executive_core"
        for event in events
    )


@pytest.mark.asyncio
async def test_legacy_phase_is_quarantined_from_foreground_ticks():
    phase = LegacyPhase(SimpleNamespace())
    state = AuraState()
    state.cognition.current_origin = "user"
    state.cognition.last_response = "already have a response"

    result = await phase.execute(state, objective="hello", priority=True)

    assert result is state
    assert phase.legacy_orchestrator is None


def test_belief_graph_blocks_unapproved_belief_write(monkeypatch):
    ServiceContainer.clear()
    clear_degraded_events()
    ServiceContainer.register_instance("executive_core", SimpleNamespace(name="exec"), required=False)
    ServiceContainer.lock_registration()

    class _RejectingExecutive:
        def request_approval_sync(self, _intent):
            return DecisionRecord(
                intent_id="reject",
                outcome=DecisionOutcome.REJECTED,
                reason="belief_lockdown",
            )

    monkeypatch.setattr(
        "core.executive.executive_core.get_executive_core",
        lambda: _RejectingExecutive(),
    )

    graph = BeliefGraph()
    graph.update_belief("Aura", "believes", "forbidden", confidence_score=0.8)

    assert not graph.graph.has_edge("Aura", "forbidden")
    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "belief_graph"
        and event.get("reason") == "belief_update_blocked"
        for event in events
    )


@pytest.mark.asyncio
async def test_executive_closure_suppresses_direct_self_model_sync_when_runtime_live():
    ServiceContainer.clear()
    clear_degraded_events()
    ServiceContainer.register_instance("executive_core", SimpleNamespace(name="exec"), required=False)
    ServiceContainer.lock_registration()

    engine = ExecutiveClosureEngine()
    model = SimpleNamespace(beliefs={})

    await engine._sync_self_model_payload(model, {"closure_score": 0.9})

    assert model.beliefs == {}
    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "executive_closure"
        and event.get("reason") == "direct_self_model_sync_suppressed"
        for event in events
    )


@pytest.mark.asyncio
async def test_autonomous_initiative_loop_blocks_when_identity_continuity_mismatch():
    ServiceContainer.clear()
    state = AuraState()
    state.cognition.modifiers = {
        "continuity_obligations": {
            "identity_mismatch": True,
            "active_commitments": ["Protect continuity"],
        }
    }
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)

    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    decision = await loop._evaluate_initiative("novel topic")

    assert decision["allowed"] is False
    assert decision["reason"] == "identity_continuity_mismatch"


@pytest.mark.asyncio
async def test_autonomous_initiative_loop_defers_when_continuity_reentry_is_active():
    ServiceContainer.clear()
    state = AuraState()
    state.cognition.modifiers = {
        "continuity_obligations": {
            "continuity_reentry_required": True,
            "continuity_pressure": 0.82,
            "active_commitments": ["Protect continuity"],
        }
    }
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)

    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    decision = await loop._evaluate_initiative("novel topic")

    assert decision["allowed"] is False
    assert decision["reason"] == "continuity_reentry_required:0.82"


def test_goal_hierarchy_blocks_unapproved_goal_add(monkeypatch, tmp_path):
    ServiceContainer.clear()
    clear_degraded_events()
    ServiceContainer.register_instance("executive_core", SimpleNamespace(name="exec"), required=False)
    ServiceContainer.lock_registration()

    class _RejectingExecutive:
        def request_approval_sync(self, _intent):
            return DecisionRecord(
                intent_id="reject",
                outcome=DecisionOutcome.REJECTED,
                reason="goal_lockdown",
            )

    monkeypatch.setattr(
        "core.executive.executive_core.get_executive_core",
        lambda: _RejectingExecutive(),
    )

    hierarchy = GoalHierarchy(persist_path=str(tmp_path / "goals.json"))
    added = hierarchy.add_goal("Quietly create a new durable goal", priority=0.8)

    assert added == ""
    assert all("Quietly create a new durable goal" != goal.description for goal in hierarchy.goals.values())
    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "goal_hierarchy"
        and event.get("reason") == "goal_add_blocked"
        for event in events
    )


def test_agency_core_blocks_unapproved_pending_goal(monkeypatch):
    ServiceContainer.clear()
    clear_degraded_events()
    ServiceContainer.register_instance("executive_core", SimpleNamespace(name="exec"), required=False)
    ServiceContainer.lock_registration()

    class _RejectingExecutive:
        def request_approval_sync(self, _intent):
            return DecisionRecord(
                intent_id="reject",
                outcome=DecisionOutcome.REJECTED,
                reason="agency_lockdown",
            )

    monkeypatch.setattr(
        "core.executive.executive_core.get_executive_core",
        lambda: _RejectingExecutive(),
    )

    agency = AgencyCore.__new__(AgencyCore)
    agency.state = AgencyState()

    added = AgencyCore.add_goal(agency, {"description": "Blocked agency goal", "priority": 0.8})

    assert added is False
    assert agency.state.pending_goals == []
    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "agency_core"
        and event.get("reason") == "agency_state_mutation_blocked"
        for event in events
    )


@pytest.mark.asyncio
async def test_orchestrator_execute_tool_blocks_when_constitutional_gate_unavailable(orchestrator, monkeypatch):
    ServiceContainer.register_instance("executive_core", SimpleNamespace(name="exec"), required=False)
    ServiceContainer.lock_registration()

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("gate down")),
    )

    result = await orchestrator.execute_tool("notify_user", {"message": "hi"}, origin="user")

    assert result["ok"] is False
    assert "gate unavailable" in result["error"].lower()


@pytest.mark.asyncio
async def test_orchestrator_execute_tool_blocks_when_capability_token_missing(orchestrator, monkeypatch):
    ServiceContainer.register_instance("executive_core", object(), required=False)
    ServiceContainer.lock_registration()

    constitution = SimpleNamespace(
        begin_tool_execution=AsyncMock(
            return_value=SimpleNamespace(
                approved=True,
                capability_token_id=None,
                constraints={},
                decision=SimpleNamespace(reason="approved"),
                executive_intent_id="intent-1",
            )
        ),
        finish_tool_execution=AsyncMock(),
    )

    orchestrator.router = SimpleNamespace(
        skills={"notify_user": object()},
        execute=AsyncMock(return_value={"ok": True}),
    )

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: constitution,
    )

    result = await orchestrator.execute_tool("notify_user", {"message": "hi"}, origin="user")

    assert result["ok"] is False
    assert "capability token missing" in result["error"].lower()
    assert orchestrator.router.execute.await_count == 0
    assert constitution.finish_tool_execution.await_count == 1


def test_knowledge_graph_blocks_memory_write_when_constitutional_gate_rejects(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write_sync=lambda **_kwargs: (False, "blocked_by_test")
        ),
    )
    graph = PersistentKnowledgeGraph(str(tmp_path / "knowledge.db"))

    node_id = graph.add_knowledge("blocked fact", type="fact", source="test", confidence=0.7)

    assert node_id
    assert graph.count_nodes() == 0


@pytest.mark.asyncio
async def test_memory_facade_commit_interaction_blocks_when_constitutional_gate_rejects(monkeypatch):
    facade = MemoryFacade()
    facade._episodic = SimpleNamespace(record_episode_async=AsyncMock(return_value="episode-1"))

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncMock(return_value=(False, "blocked_by_test"))
        ),
    )

    result = await facade.commit_interaction(
        context="ctx",
        action="act",
        outcome="out",
        success=True,
        importance=0.8,
    )

    assert result is None
    assert facade._episodic.record_episode_async.await_count == 0


@pytest.mark.asyncio
async def test_long_term_memory_store_blocks_when_constitutional_gate_rejects(tmp_path, monkeypatch):
    engine = LongTermMemoryEngine()
    engine.db_path = tmp_path / "ltm.json"

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncMock(return_value=(False, "blocked_by_test"))
        ),
    )

    await engine.store("blocked memory", importance=0.9)

    assert engine.memories == []
    assert not engine.db_path.exists()


def test_acg_blocks_memory_write_when_constitutional_gate_rejects(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write_sync=lambda **_kwargs: (False, "blocked_by_test")
        ),
    )
    graph = ActionConsequenceGraph(str(tmp_path / "causal_graph.json"))

    graph.record_outcome("clock", "ctx", {"ok": True}, True)

    assert graph.links == []
    assert not Path(graph.persist_path).exists()


@pytest.mark.asyncio
async def test_agency_pulse_blocks_unapproved_autonomous_action(orchestrator, monkeypatch):
    clear_degraded_events()
    orchestrator.liquid_state = SimpleNamespace(get_arousal=lambda: 0.95)
    orchestrator._last_self_initiated_contact = 0.0
    orchestrator._agency_core = SimpleNamespace(
        state=SimpleNamespace(
            last_self_initiated_contact=0.0,
            last_observation_comment=0.0,
            unshared_observations=[],
        ),
        pulse=AsyncMock(
            return_value={
                "type": "autonomous_action",
                "skill": "clock",
                "params": {},
                "message": "Run clock",
                "source": "agency_core",
            }
        ),
    )

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_initiative=AsyncMock(return_value=(False, "blocked", None))
        ),
    )

    await orchestrator._pulse_agency_core()

    assert orchestrator.message_queue.qsize() == 0
    events = get_recent_degraded_events(limit=10)
    assert any(
        event.get("subsystem") == "orchestrator"
        and event.get("reason") == "agency_dispatch_blocked"
        and "autonomous_action" in str(event.get("detail", ""))
        for event in events
    )


@pytest.mark.asyncio
async def test_capability_engine_blocks_when_executive_gate_unavailable(service_container, monkeypatch):
    service_container.register_instance("executive_core", SimpleNamespace(name="exec"), required=False)
    service_container.lock_registration()

    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None, debug=lambda *a, **k: None)
    engine.error_boundary = lambda fn: fn
    engine.skills = {
        "clock": SkillMetadata(
            name="clock",
            description="Return the time.",
            skill_class=lambda: object(),
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
        lambda: (_ for _ in ()).throw(RuntimeError("down")),
    )

    result = await CapabilityEngine.execute(engine, "clock", {}, context={})

    assert result["ok"] is False
    assert result["status"] == "blocked_by_executive_gate_failure"


@pytest.mark.asyncio
async def test_kernel_load_initial_state_accepts_lightweight_vault():
    state = AuraState.default()
    kernel = AuraKernel(config=KernelConfig(), vault=SimpleNamespace(state=state))

    await kernel._load_initial_state()

    assert kernel.state is state
