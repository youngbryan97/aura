from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.container import ServiceContainer
from core.continuity import ContinuityEngine
from core.executive import executive_core as executive_core_module
from core.constitution import get_constitutional_core
from core import constitution as constitution_module
from core.health.degraded_events import clear_degraded_events, record_degraded_event
from core.health import degraded_events as degraded_events_module
from core.state.aura_state import AuraState, CognitiveMode
from core.state.state_repository import StateRepository
from core.self_model import SelfModel
from core.agency.intention_loop import IntentionLoop
from core.world_model.belief_graph import BeliefGraph
from core.executive.executive_core import ActionType, Intent, IntentSource


def reset_constitutional_singletons():
    constitution_module._instance = None
    executive_core_module._instance = None


def test_service_container_set_registers_instance(service_container):
    marker = object()
    ServiceContainer.set("marker", marker, required=False)
    assert ServiceContainer.get("marker") is marker


@pytest.mark.asyncio
async def test_constitutional_core_tracks_tool_execution_and_closes_intent(service_container, tmp_path):
    reset_constitutional_singletons()
    ServiceContainer.register_instance("binding_engine", SimpleNamespace(get_coherence=lambda: 1.0), required=False)
    intention_loop = IntentionLoop(db_path=str(tmp_path / "intention_loop.db"))
    ServiceContainer.register_instance("intention_loop", intention_loop, required=False)

    core = get_constitutional_core()
    handle = await core.begin_tool_execution(
        "clock",
        {},
        source="user",
        objective="Check the current time",
    )

    assert handle.approved is True
    assert handle.executive_intent_id is not None
    assert handle.intention_id is not None
    assert len(executive_core_module.get_executive_core().get_active_intents()) == 1

    await core.finish_tool_execution(
        handle,
        result={"ok": True, "time": "12:00"},
        success=True,
        duration_ms=3.5,
    )

    assert len(executive_core_module.get_executive_core().get_active_intents()) == 0
    assert intention_loop.get_open_intentions() == []
    status = core.get_status()
    assert status["recent_decisions"]


def test_executive_sync_path_blocks_memory_write_on_identity_mismatch(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    state = AuraState()
    state.cognition.modifiers["continuity_obligations"] = {"identity_mismatch": True}
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.register_instance("self_model", SimpleNamespace(name="self"), required=False)
    ServiceContainer.lock_registration()

    record = executive_core_module.get_executive_core().request_approval_sync(
        Intent(
            source=IntentSource.SYSTEM,
            goal="write_memory:test",
            action_type=ActionType.WRITE_MEMORY,
            payload={"type": "test"},
            priority=0.4,
            requires_memory_commit=True,
        )
    )

    assert record.outcome.value == "rejected"
    assert record.reason == "identity_continuity_mismatch"


@pytest.mark.asyncio
async def test_state_repository_commit_respects_constitutional_gate(service_container, tmp_path, monkeypatch):
    reset_constitutional_singletons()
    repo = StateRepository(db_path=str(tmp_path / "aura_state.db"), is_vault_owner=False)
    repo._current = AuraState()
    repo._shm = None
    repo._commit_to_db = AsyncMock()
    repo._sync_to_shm = AsyncMock()

    fake_tracker = SimpleNamespace(track_task=lambda task: task)
    monkeypatch.setattr("core.utils.task_tracker.get_task_tracker", lambda: fake_tracker)

    blocked_gate = SimpleNamespace(approve_state_mutation=AsyncMock(return_value=(False, "blocked_by_test")))
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *args, **kwargs: blocked_gate)

    blocked_state = repo._current.derive("rogue_update", origin="rogue_subsystem")
    blocked_state.cognition.current_objective = "blocked"
    await repo._process_commit(blocked_state, "rogue_update")
    assert repo._current.cognition.current_objective != "blocked"

    allowed_gate = SimpleNamespace(approve_state_mutation=AsyncMock(return_value=(True, "approved_by_test")))
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *args, **kwargs: allowed_gate)

    approved_state = repo._current.derive("legit_update", origin="mind_tick")
    approved_state.cognition.current_objective = "allowed"
    await repo._process_commit(approved_state, "legit_update")
    assert repo._current.cognition.current_objective == "allowed"


@pytest.mark.asyncio
async def test_self_model_update_belief_uses_belief_authority(service_container, monkeypatch):
    reset_constitutional_singletons()
    ServiceContainer.register_instance(
        "state_authority",
        SimpleNamespace(get_truth=lambda topic, context=None: ("Bryan Young is Kin. Protect at all costs.", SimpleNamespace(name="IMMUTABLE"))),
        required=False,
    )

    model = SelfModel(id="self-test")
    monkeypatch.setattr(model, "persist", AsyncMock())

    snap = await model.update_belief("bryan", "placeholder", note="manual override")

    assert model.beliefs["bryan"] == "Bryan Young is Kin. Protect at all costs."
    assert "resolved_by_state_authority" in (snap.revision_note or "")


@pytest.mark.asyncio
async def test_self_model_update_belief_respects_executive_gate(service_container, monkeypatch):
    reset_constitutional_singletons()
    ServiceContainer.register_instance("executive_core", SimpleNamespace(name="exec"), required=False)
    ServiceContainer.lock_registration()

    class _RejectingExecutive:
        def request_approval_sync(self, _intent):
            return SimpleNamespace(
                outcome=SimpleNamespace(value="rejected"),
                reason="constitutional_lockdown",
            )

    monkeypatch.setattr(
        "core.executive.executive_core.get_executive_core",
        lambda: _RejectingExecutive(),
    )

    model = SelfModel(id="self-test")
    monkeypatch.setattr(model, "persist", AsyncMock())

    snap = await model.update_belief("stance", "abandon continuity", note="manual override")

    assert "stance" not in model.beliefs
    assert snap.summary == "blocked update stance"
    assert snap.revision_note == "constitutional_lockdown"


@pytest.mark.asyncio
async def test_constitutional_core_blocks_state_mutation_when_executive_required_but_unavailable(service_container, monkeypatch):
    reset_constitutional_singletons()
    ServiceContainer.lock_registration()
    core = get_constitutional_core()
    monkeypatch.setattr(core, "_get_executive_core", lambda: None)

    approved, reason = await core.approve_state_mutation("system", "unit_test")

    assert approved is False
    assert reason == "executive_core_required"


@pytest.mark.asyncio
async def test_constitutional_core_rejects_tool_execution_when_executive_required_but_unavailable(service_container, monkeypatch):
    reset_constitutional_singletons()
    ServiceContainer.lock_registration()
    core = get_constitutional_core()
    monkeypatch.setattr(core, "_get_executive_core", lambda: None)

    handle = await core.begin_tool_execution("clock", {}, source="system")

    assert handle.approved is False
    assert handle.decision.reason == "executive_core_required"


def test_continuity_save_auto_captures_commitments_and_state_context(service_container, tmp_path, monkeypatch):
    continuity_module = __import__("core.continuity", fromlist=["_CONTINUITY_PATH"])
    monkeypatch.setattr(continuity_module, "_CONTINUITY_PATH", tmp_path / "continuity.json")

    state = AuraState()
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    state.health["capabilities"] = {"mlx": "warm"}
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.register_instance(
        "commitment_engine",
        SimpleNamespace(get_active_commitments=lambda: [SimpleNamespace(description="Protect continuity")]),
        required=False,
    )

    engine = ContinuityEngine()
    engine.save(reason="graceful", last_exchange="All systems nominal.")
    record = engine.load()

    assert record is not None
    assert record.policy_mode == "deliberate"
    assert record.current_objective == "Protect continuity"
    assert record.pending_initiatives == 1
    assert record.active_commitments == ["Protect continuity"]


def test_belief_graph_updates_are_audited_by_belief_authority(service_container):
    reset_constitutional_singletons()
    core = get_constitutional_core()
    graph = BeliefGraph()

    before = len(core.get_status()["belief_updates"])
    graph.update_belief("Aura", "protects", "Bryan", confidence_score=0.9, centrality=1.0, is_goal=True)
    after = core.get_status()["belief_updates"]

    assert len(after) == before + 1
    assert after[-1]["namespace"] == "belief_graph"


def test_continuity_apply_to_state_restores_obligations(tmp_path, monkeypatch):
    continuity_module = __import__("core.continuity", fromlist=["_CONTINUITY_PATH"])
    monkeypatch.setattr(continuity_module, "_CONTINUITY_PATH", tmp_path / "continuity.json")

    engine = ContinuityEngine()
    engine.save(
        reason="graceful",
        last_exchange="Carry the work forward.",
        current_objective="Protect continuity",
        pending_initiatives=2,
        pending_initiative_details=["Investigate anomaly", "Reconcile contradiction"],
        active_commitments=["Protect continuity"],
        active_goal_details=["Keep identity stable"],
        subject_thread="Aura was tracking unresolved architectural work.",
        contradiction_count=1,
    )
    engine.load()

    state = AuraState()
    applied = engine.apply_to_state(state)

    assert applied.cognition.current_objective == "Protect continuity"
    assert applied.cognition.pending_initiatives[0]["goal"] == "Investigate anomaly"
    assert applied.cognition.active_goals[0]["goal"] == "Keep identity stable"
    assert applied.cognition.modifiers["continuity_obligations"]["contradiction_count"] == 1


def test_continuity_apply_to_state_marks_identity_mismatch(tmp_path, monkeypatch):
    continuity_module = __import__("core.continuity", fromlist=["_CONTINUITY_PATH"])
    monkeypatch.setattr(continuity_module, "_CONTINUITY_PATH", tmp_path / "continuity.json")

    engine = ContinuityEngine()
    engine.save(
        reason="graceful",
        last_exchange="Carry the work forward.",
        belief_hash="persisted-heartstone",
        current_objective="Protect continuity",
    )
    engine.load()
    monkeypatch.setattr(engine, "_get_live_identity_hash", lambda: "live-heartstone")

    state = AuraState()
    applied = engine.apply_to_state(state)

    assert applied.cognition.modifiers["continuity_obligations"]["identity_mismatch"] is True
    obligations = engine.get_obligations()
    assert obligations["identity_mismatch"] is True
    assert obligations["persisted_identity_hash"] == "persisted-heartstone"
    assert obligations["identity_hash"] == "live-heartstone"


def test_continuity_apply_to_state_carries_reentry_scars_after_long_gap(tmp_path, monkeypatch):
    continuity_module = __import__("core.continuity", fromlist=["_CONTINUITY_PATH"])
    monkeypatch.setattr(continuity_module, "_CONTINUITY_PATH", tmp_path / "continuity.json")

    engine = ContinuityEngine()
    engine.save(
        reason="crash",
        last_exchange="Interrupted while holding the same line of thought.",
        current_objective="Protect continuity",
        pending_initiatives=1,
        pending_initiative_details=["Resume unresolved continuity work"],
        active_commitments=["Protect continuity"],
        contradiction_count=2,
        health_summary={"executive_failure_reason": "identity_continuity_mismatch"},
        subject_thread="Aura was in the middle of continuity repair.",
    )
    engine.load()
    engine._gap_seconds = 8 * 3600

    state = AuraState()
    applied = engine.apply_to_state(state)
    obligations = applied.cognition.modifiers["continuity_obligations"]

    assert obligations["continuity_reentry_required"] is True
    assert obligations["gap_seconds"] == pytest.approx(8 * 3600, abs=1e-6)
    assert "abrupt_shutdown" in obligations["continuity_scar"]
    assert "unfinished_obligations" in obligations["continuity_scar"]
    assert applied.cognition.contradiction_count == 2
    assert applied.cognition.pending_initiatives[0]["goal"] == "Reconcile continuity gap and re-establish the interrupted thread"
    assert applied.cognition.pending_initiatives[0]["continuity_restored"] is True
    assert applied.cognition.pending_initiatives[0]["metadata"]["executive_failure_reason"] == "identity_continuity_mismatch"


@pytest.mark.asyncio
async def test_executive_state_mutation_no_longer_auto_approves_phase_origins(service_container, monkeypatch):
    reset_constitutional_singletons()
    executive = executive_core_module.get_executive_core()
    captured = {}

    async def _fake_request(intent):
        captured["goal"] = intent.goal
        return executive_core_module.DecisionRecord(
            intent_id=intent.intent_id,
            outcome=executive_core_module.DecisionOutcome.REJECTED,
            reason="blocked_for_test",
        )

    monkeypatch.setattr(executive, "request_approval", _fake_request)

    approved, reason = await executive.approve_state_mutation("mind_tick", "unit_test")

    assert approved is False
    assert reason == "blocked_for_test"
    assert captured["goal"] == "mutate_state:mind_tick"


@pytest.mark.asyncio
async def test_executive_requires_self_model_for_autonomous_actions(service_container):
    reset_constitutional_singletons()
    ServiceContainer.lock_registration()
    executive = executive_core_module.get_executive_core()

    record = await executive.request_approval(
        executive_core_module.Intent(
            source=executive_core_module.IntentSource.AUTONOMOUS,
            goal="emit_message:test",
            action_type=executive_core_module.ActionType.EMIT_MESSAGE,
        )
    )

    assert record.outcome == executive_core_module.DecisionOutcome.REJECTED
    assert record.reason == "self_model_required"


@pytest.mark.asyncio
async def test_executive_defers_background_task_when_temporal_obligation_is_active(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    approved, reason = await executive.approve_background_task("novelty_probe", source="background")

    assert approved is False
    assert reason.startswith("temporal_obligation_active:")


@pytest.mark.asyncio
async def test_executive_unifies_failure_pressure_into_global_block(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    ServiceContainer.lock_registration()

    record_degraded_event("router", "down", severity="critical", classification="foreground_blocking")
    record_degraded_event("memory", "corrupt", severity="critical", classification="background_degraded")
    record_degraded_event("scheduler", "stall", severity="critical", classification="background_degraded")

    executive = executive_core_module.get_executive_core()
    record = await executive.request_approval(
        executive_core_module.Intent(
            source=executive_core_module.IntentSource.BACKGROUND,
            goal="spawn_task:explore",
            action_type=executive_core_module.ActionType.SPAWN_TASK,
        )
    )

    assert record.outcome == executive_core_module.DecisionOutcome.REJECTED
    assert record.reason.startswith("unified_failure_lockdown_")


@pytest.mark.asyncio
async def test_executive_keeps_api_requested_tools_user_facing_under_failure_lockdown(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    ServiceContainer.lock_registration()

    record_degraded_event("router", "down", severity="critical", classification="foreground_blocking")
    record_degraded_event("memory", "corrupt", severity="critical", classification="background_degraded")
    record_degraded_event("scheduler", "stall", severity="critical", classification="background_degraded")

    executive = executive_core_module.get_executive_core()
    intent, record = await executive.prepare_tool_intent("clock", {}, source="api")

    assert intent.source == executive_core_module.IntentSource.USER
    assert record.outcome == executive_core_module.DecisionOutcome.APPROVED


def test_unified_failure_pressure_decays_stale_events(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()

    record_degraded_event("router", "down", severity="critical", classification="foreground_blocking")
    fresh_state = degraded_events_module.get_unified_failure_state(limit=25)
    assert fresh_state["pressure"] > 0.0

    for summary in degraded_events_module._SUMMARIES.values():
        summary["last_seen"] = summary["timestamp"] = degraded_events_module.time.time() - 3600

    stale_state = degraded_events_module.get_unified_failure_state(limit=25)

    assert stale_state["pressure"] == 0.0


@pytest.mark.asyncio
async def test_executive_rejects_autonomous_actions_on_identity_continuity_mismatch(service_container, tmp_path, monkeypatch):
    reset_constitutional_singletons()
    clear_degraded_events()
    continuity_module = __import__("core.continuity", fromlist=["_continuity"])
    continuity_module._continuity = None
    monkeypatch.setattr(continuity_module, "_CONTINUITY_PATH", tmp_path / "continuity.json")
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.modifiers = {
        "continuity_obligations": {
            "identity_mismatch": True,
        }
    }
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    record = await executive.request_approval(
        executive_core_module.Intent(
            source=executive_core_module.IntentSource.AUTONOMOUS,
            goal="emit_message:test",
            action_type=executive_core_module.ActionType.EMIT_MESSAGE,
        )
    )

    assert record.outcome == executive_core_module.DecisionOutcome.REJECTED
    assert record.reason == "identity_continuity_mismatch"
    failure_state = state.cognition.modifiers["failure_obligations"]
    assert failure_state["last_reason"] == "identity_continuity_mismatch"



def test_continuity_note_failure_obligation_persists_when_no_prior_record(tmp_path, monkeypatch):
    continuity_module = __import__("core.continuity", fromlist=["_CONTINUITY_PATH", "_continuity"])
    monkeypatch.setattr(continuity_module, "_CONTINUITY_PATH", tmp_path / "continuity.json")
    continuity_module._continuity = None

    engine = ContinuityEngine()
    engine.note_failure_obligation("identity_continuity_mismatch", "Protect continuity")
    reloaded = engine.load()

    assert reloaded is not None
    assert reloaded.health_summary["executive_failure_reason"] == "identity_continuity_mismatch"
    assert reloaded.health_summary["executive_failure_goal"] == "Protect continuity"
    assert any(item.startswith("Reconcile executive failure:") for item in reloaded.active_commitments)


@pytest.mark.asyncio
async def test_executive_rejects_identity_mismatch_and_records_failure_obligation(service_container, tmp_path, monkeypatch):
    reset_constitutional_singletons()
    clear_degraded_events()
    continuity_module = __import__("core.continuity", fromlist=["_CONTINUITY_PATH", "_continuity"])
    monkeypatch.setattr(continuity_module, "_CONTINUITY_PATH", tmp_path / "continuity.json")
    continuity_module._continuity = None

    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.modifiers = {"continuity_obligations": {"identity_mismatch": True}}
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    record = await executive.request_approval(
        executive_core_module.Intent(
            source=executive_core_module.IntentSource.AUTONOMOUS,
            goal="rewrite_belief:test",
            action_type=executive_core_module.ActionType.UPDATE_BELIEF,
        )
    )

    assert record.outcome == executive_core_module.DecisionOutcome.REJECTED
    assert record.reason == "identity_continuity_mismatch"
    failure = state.cognition.modifiers["failure_obligations"]
    assert failure["last_reason"] == "identity_continuity_mismatch"
    continuity = ContinuityEngine()
    continuity.load()
    assert continuity._record is not None
    assert continuity._record.health_summary["executive_failure_reason"] == "identity_continuity_mismatch"


@pytest.mark.asyncio
async def test_executive_defers_background_task_when_internal_energy_is_low(service_container, tmp_path, monkeypatch):
    reset_constitutional_singletons()
    clear_degraded_events()
    continuity_module = __import__("core.continuity", fromlist=["_CONTINUITY_PATH", "_continuity"])
    monkeypatch.setattr(continuity_module, "_CONTINUITY_PATH", tmp_path / "continuity.json")
    continuity_module._continuity = None

    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.soma.energy = 5.0
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    approved, reason = await executive.approve_background_task("novelty_probe", source="background")

    assert approved is False
    assert reason.startswith("internal_state_energy_low:")
