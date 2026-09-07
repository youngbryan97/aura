import json
import asyncio
import time
from types import SimpleNamespace

import pytest

from core import constitution as constitution_module
from core.agency.intention_loop import IntentionLoop
from core.constitution import get_constitutional_core
from core.container import ServiceContainer
from core.continuity import ContinuityEngine
from core.executive import executive_core as executive_core_module
from core.executive.bounded_sandbox_policy import idle_sandbox_probe_arguments
from core.executive.executive_core import ActionType, Intent, IntentSource
from core.health import degraded_events as degraded_events_module
from core.health.degraded_events import clear_degraded_events, record_degraded_event
from core.self_model import SelfModel
from core.state.aura_state import AuraState, CognitiveMode
from core.state.state_repository import StateRepository
from core.world_model.belief_graph import BeliefGraph


async def async_noop(*_args, **_kwargs):
    return None


class StateMutationGateFixture:
    def __init__(self, approved: bool, reason: str):
        self.approved = approved
        self.reason = reason
        self.calls = []

    async def approve_state_mutation(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.approved, self.reason


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


@pytest.mark.asyncio
async def test_messages_constitutional_path_never_records_private_body(
    service_container,
    tmp_path,
    monkeypatch,
):
    reset_constitutional_singletons()
    ServiceContainer.register_instance(
        "binding_engine",
        SimpleNamespace(get_coherence=lambda: 1.0),
        required=False,
    )
    intention_loop = IntentionLoop(db_path=str(tmp_path / "intention_loop.db"))
    ServiceContainer.register_instance("intention_loop", intention_loop, required=False)
    private_body = "Private transport prose must stay outside governance records."
    emitted: list[dict] = []
    core = get_constitutional_core()
    monkeypatch.setattr(
        core,
        "_emit_tool_event",
        lambda _stage, _tool, **kwargs: emitted.append(kwargs),
    )

    handle = await core.begin_tool_execution(
        "messages",
        {
            "action": "send",
            "alias": "primary_operator",
            "body": private_body,
            "idempotency_key": "constitution-private-1",
        },
        source="user",
        objective=private_body,
        context={"message": private_body, "foreground_request": True},
    )

    serialized_proposal = json.dumps(handle.proposal.payload, sort_keys=True)
    serialized_events = json.dumps(emitted, default=str, sort_keys=True)
    assert private_body not in serialized_proposal
    assert private_body not in serialized_events
    assert handle.proposal.payload["args"]["body_chars"] == len(private_body)
    assert handle.proposal.payload["objective"] == "Use Aura's private Messages channel"
    if handle.intention_id:
        intention = intention_loop.get_intention(handle.intention_id)
        assert intention is not None
        assert private_body not in intention.intention


@pytest.mark.asyncio
async def test_constitutional_core_marks_governed_deferral_without_high_surprise(
    service_container,
    tmp_path,
    monkeypatch,
):
    reset_constitutional_singletons()
    ServiceContainer.register_instance("binding_engine", SimpleNamespace(get_coherence=lambda: 1.0), required=False)
    intention_loop = IntentionLoop(db_path=str(tmp_path / "intention_loop.db"))
    ServiceContainer.register_instance("intention_loop", intention_loop, required=False)

    core = get_constitutional_core()
    emitted: list[tuple[str, dict]] = []

    def capture_event(stage, _tool_name, **kwargs):
        emitted.append((stage, kwargs))

    monkeypatch.setattr(core, "_emit_tool_event", capture_event)
    handle = await core.begin_tool_execution(
        "auto_refactor",
        {"mode": "scan"},
        source="autonomous_initiative",
        objective="Run a quiet codebase scan",
    )

    await core.finish_tool_execution(
        handle,
        result={"ok": False, "status": "deferred", "error": "background_deferred:foreground_quiet_window"},
        success=False,
        duration_ms=1.0,
        error="background_deferred:foreground_quiet_window",
    )

    record = intention_loop.get_intention(handle.intention_id)
    assert record is not None
    assert record.status.value == "deferred"
    assert record.surprise == 0.0
    assert intention_loop.get_recent_surprises() == []
    assert emitted[-1][0] == "deferred"
    assert emitted[-1][1]["success"] is None
    assert emitted[-1][1]["error"] is None
    assert not any(stage == "failed" for stage, _ in emitted)


@pytest.mark.asyncio
async def test_constitutional_core_does_not_mark_successful_tool_results_as_high_surprise(service_container, tmp_path):
    reset_constitutional_singletons()
    ServiceContainer.register_instance("binding_engine", SimpleNamespace(get_coherence=lambda: 1.0), required=False)
    intention_loop = IntentionLoop(db_path=str(tmp_path / "intention_loop.db"))
    ServiceContainer.register_instance("intention_loop", intention_loop, required=False)

    core = get_constitutional_core()
    handle = await core.begin_tool_execution(
        "email_adapter",
        {"mode": "check"},
        source="autonomous_initiative",
        objective="Check unread mail",
    )

    await core.finish_tool_execution(
        handle,
        result={"ok": True, "unread": 22, "messages": [{"subject": "Test"}]},
        success=True,
        duration_ms=5.0,
    )

    record = intention_loop.get_intention(handle.intention_id)
    assert record is not None
    assert record.status.value == "completed"
    assert record.surprise == 0.0
    assert intention_loop.get_recent_surprises() == []


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


def test_executive_sync_path_allows_provisional_research_memory_under_contestation(monkeypatch, service_container):
    reset_constitutional_singletons()
    core = executive_core_module.ExecutiveCore()
    monkeypatch.setattr(core, "_strict_runtime_active", lambda: True)
    monkeypatch.setattr(core, "_identity_integrity_available", lambda: True)
    monkeypatch.setattr(core, "_get_failure_state", lambda: {"pressure": 0.0})
    monkeypatch.setattr(
        core,
        "_get_temporal_identity_context",
        lambda: {"obligation_pressure": 0.0, "anchor": "none"},
    )
    monkeypatch.setattr(
        core,
        "_get_internal_state_constraints",
        lambda: {
            "identity_mismatch": False,
            "thermal_pressure": 0.0,
            "load_pressure": 0.0,
            "energy": 1.0,
            "distress": 0.0,
        },
    )
    monkeypatch.setattr(core, "_get_epistemic_state", lambda: {"contested": 1, "trusted": 0, "coherence_score": 1.0})
    monkeypatch.setattr(core, "_get_coherence_sync", lambda: 1.0)

    research_intent = Intent(
        source=IntentSource.AUTONOMOUS_RESEARCH,
        goal="write_memory:web_evidence",
        action_type=ActionType.WRITE_MEMORY,
        payload={"type": "web_evidence"},
        priority=0.5,
        requires_memory_commit=True,
    )
    record = core.request_approval_sync(research_intent)

    assert record.outcome.value == "approved"
    assert research_intent.payload["confidence_tier"] == "provisional"
    assert research_intent.payload["requires_reconciliation"] is True

    generic_intent = Intent(
        source=IntentSource.AUTONOMOUS,
        goal="write_memory:generic_autonomous_claim",
        action_type=ActionType.WRITE_MEMORY,
        payload={"type": "generic"},
        priority=0.5,
        requires_memory_commit=True,
    )
    generic_record = core.request_approval_sync(generic_intent)
    assert generic_record.outcome.value == "deferred"
    assert generic_record.reason == "epistemic_reconciliation_required:1"


def test_research_source_aliases_route_to_autonomous_research():
    from core.executive.executive_core import _coerce_intent_source

    assert _coerce_intent_source("web_search") == IntentSource.AUTONOMOUS_RESEARCH
    assert _coerce_intent_source("web_search:water bears") == IntentSource.AUTONOMOUS_RESEARCH
    assert _coerce_intent_source("knowledge:curiosity_finding") == IntentSource.AUTONOMOUS_RESEARCH
    assert _coerce_intent_source("curiosity_engine") == IntentSource.AUTONOMOUS_RESEARCH
    assert _coerce_intent_source("action_consequence_graph") == IntentSource.AUTONOMOUS_RESEARCH
    assert _coerce_intent_source("research_pipeline") == IntentSource.AUTONOMOUS_RESEARCH


def test_peer_mode_source_aliases_route_to_maintenance():
    from core.executive.executive_core import _coerce_intent_source

    assert _coerce_intent_source("peer_mode") == IntentSource.MAINTENANCE
    assert _coerce_intent_source("peer_mode:sovereign_self_modification_loop") == IntentSource.MAINTENANCE
    assert _coerce_intent_source("self_repair") == IntentSource.MAINTENANCE
    assert _coerce_intent_source("runtime_repair:slow_tick") == IntentSource.MAINTENANCE


def test_authority_gateway_preserves_research_source_from_memory_metadata():
    from core.executive.authority_gateway import AuthorityGateway

    assert (
        AuthorityGateway._memory_intent_source(
            "facade_add_memory",
            "memory_facade",
            {"source": "web_search", "query": "water bears"},
        )
        == IntentSource.AUTONOMOUS_RESEARCH
    )
    assert (
        AuthorityGateway._memory_intent_source(
            "belief_update",
            "web_search:jupiter moon",
            {"confidence_tier": "provisional"},
        )
        == IntentSource.AUTONOMOUS_RESEARCH
    )
    assert (
        AuthorityGateway._memory_intent_source(
            "identity_rewrite",
            "web_search",
            {"identity_rewrite": True},
        )
        != IntentSource.AUTONOMOUS_RESEARCH
    )
    assert (
        AuthorityGateway._memory_intent_source(
            "interaction_commit",
            "memory_facade",
            {"tool_name": "web_search", "runtime_evidence": True},
        )
        == IntentSource.AUTONOMOUS_RESEARCH
    )
    assert (
        AuthorityGateway._memory_intent_source(
            "causal_outcome",
            "action_consequence_graph",
            {"tool_name": "web_search", "tool_result_evidence": True},
        )
        == IntentSource.AUTONOMOUS_RESEARCH
    )


@pytest.mark.asyncio
async def test_state_repository_commit_respects_constitutional_gate(service_container, tmp_path, monkeypatch):
    reset_constitutional_singletons()
    repo = StateRepository(db_path=str(tmp_path / "aura_state.db"), is_vault_owner=False)
    repo._current = AuraState()
    repo._shm = None
    repo._commit_to_db = async_noop
    repo._sync_to_shm = async_noop

    fake_tracker = SimpleNamespace(track_task=lambda task: task)
    monkeypatch.setattr("core.utils.task_tracker.get_task_tracker", lambda: fake_tracker)

    blocked_gate = StateMutationGateFixture(False, "blocked_by_test")
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda *args, **kwargs: blocked_gate)

    blocked_state = repo._current.derive("rogue_update", origin="rogue_subsystem")
    blocked_state.cognition.current_objective = "blocked"
    await repo._process_commit(blocked_state, "rogue_update")
    assert repo._current.cognition.current_objective != "blocked"

    allowed_gate = StateMutationGateFixture(True, "approved_by_test")
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
        SimpleNamespace(get_truth=lambda topic, context=None, **kwargs: ("Bryan Young is Kin. Protect at all costs.", SimpleNamespace(name="IMMUTABLE"))),
        required=False,
    )

    model = SelfModel(id="self-test")
    monkeypatch.setattr(model, "persist", async_noop)

    snap = await model.update_belief("bryan", "untrusted_override", note="manual override")

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
    monkeypatch.setattr(model, "persist", async_noop)

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
@pytest.mark.parametrize(
    ("approved", "authority_outcome", "expected"),
    [
        (False, "deferred", constitution_module.ProposalOutcome.DEFERRED),
        (True, "degraded", constitution_module.ProposalOutcome.DEGRADED),
    ],
)
async def test_constitutional_core_preserves_authority_outcome(
    service_container,
    monkeypatch,
    approved,
    authority_outcome,
    expected,
):
    reset_constitutional_singletons()
    core = get_constitutional_core()

    async def authorize_state_mutation(*_args, **_kwargs):
        return SimpleNamespace(
            approved=approved,
            outcome=authority_outcome,
            reason=f"state:{authority_outcome}",
            constraints={},
            will_receipt_id=None,
            executive_intent_id=None,
        )

    gateway = SimpleNamespace(authorize_state_mutation=authorize_state_mutation)
    monkeypatch.setattr(core, "_strict_enforcement_active", lambda: False)
    monkeypatch.setattr(core, "_get_authority_gateway", lambda: gateway)

    allowed, reason, decision = await core.approve_state_mutation(
        "system",
        "unit_test",
        return_decision=True,
    )

    assert allowed is approved
    assert reason == f"state:{authority_outcome}"
    assert decision.outcome == expected
    assert decision.constraints["authority_outcome"] == authority_outcome


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
        active_goal_details=["Stabilize identity coherence across sessions"],
        subject_thread="Aura was tracking unresolved architectural work.",
        contradiction_count=1,
    )
    engine.load()

    state = AuraState()
    applied = engine.apply_to_state(state)

    assert applied.cognition.current_objective == "Protect continuity"
    assert applied.cognition.pending_initiatives[0]["goal"] == "Investigate anomaly"
    assert applied.cognition.active_goals[0]["goal"] == "Stabilize identity coherence across sessions"
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
async def test_authority_gateway_allows_peer_mode_repair_under_temporal_obligation(
    service_container,
    monkeypatch,
):
    from core.executive.authority_gateway import AuthorityGateway

    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = (
        "Continuously detect runtime failures, propose safe repairs, run targeted tests, "
        "and apply verified patches."
    )
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    gateway = AuthorityGateway()
    monkeypatch.setattr(gateway, "_will_gate", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(gateway, "_substrate_preflight", lambda **_kwargs: (None, {}, None))

    decision = await gateway.authorize_initiative(
        "peer_mode:sovereign_self_modification_loop",
        source="peer_mode",
        priority=0.45,
    )

    assert decision.approved is True
    assert decision.reason in {"approved", "sync_approved"}
    assert not decision.reason.startswith("temporal_obligation_active:")


@pytest.mark.asyncio
async def test_executive_treats_desktop_ui_tool_as_user_under_temporal_obligation(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    intent, record = await executive.prepare_tool_intent(
        "file_operation",
        {"action": "write", "path": "artifacts/live_runtime/generated/probe.html"},
        source="desktop_ui",
    )

    assert intent.source == executive_core_module.IntentSource.USER
    assert record.outcome == executive_core_module.DecisionOutcome.APPROVED
    assert record.reason == "user_facing"


@pytest.mark.asyncio
async def test_executive_treats_live_skill_api_as_user_under_temporal_obligation(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = (
        "Remember this note for later in this conversation: the blue lantern is under the desk."
    )
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    intent, record = await executive.prepare_tool_intent(
        "program_dna_reconstruct",
        {"target": "Authorized Notes Export Utility"},
        source="live_skill_api",
    )

    assert intent.source == executive_core_module.IntentSource.USER
    assert record.outcome == executive_core_module.DecisionOutcome.APPROVED
    assert record.reason == "user_facing"


@pytest.mark.asyncio
async def test_executive_allows_safe_autonomous_tools_under_temporal_obligation(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    intent, record = await executive.prepare_tool_intent(
        "sovereign_network",
        {"mode": "status"},
        source="autonomous",
    )

    assert intent.source == executive_core_module.IntentSource.AUTONOMOUS
    assert record.outcome == executive_core_module.DecisionOutcome.DEGRADED
    assert record.reason == "temporal_safe_autonomous_tool"
    assert record.constraints["read_only"] is True


@pytest.mark.asyncio
async def test_executive_allows_exact_bounded_sandbox_under_temporal_obligation(
    service_container,
):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance(
        "state_repository",
        SimpleNamespace(_current=state),
        required=False,
    )
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    intent, record = await executive.prepare_tool_intent(
        "subconscious_sandbox_probe",
        idle_sandbox_probe_arguments(),
        source="autonomous",
    )

    assert intent.source == executive_core_module.IntentSource.AUTONOMOUS
    assert record.outcome == executive_core_module.DecisionOutcome.DEGRADED
    assert record.reason == "temporal_safe_autonomous_tool"
    assert record.constraints == {
        "timeout_s": 30,
        "sandboxed_compute": True,
        "network_access": False,
    }


@pytest.mark.asyncio
async def test_executive_defers_substituted_sandbox_under_temporal_obligation(
    service_container,
):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance(
        "state_repository",
        SimpleNamespace(_current=state),
        required=False,
    )
    ServiceContainer.lock_registration()

    arguments = idle_sandbox_probe_arguments()
    arguments["script_sha256"] = "0" * 64
    executive = executive_core_module.get_executive_core()
    _intent, record = await executive.prepare_tool_intent(
        "subconscious_sandbox_probe",
        arguments,
        source="autonomous",
    )

    assert record.outcome == executive_core_module.DecisionOutcome.DEFERRED
    assert record.reason.startswith("temporal_obligation_active:")


@pytest.mark.asyncio
async def test_executive_expires_stale_desktop_prompt_as_temporal_anchor(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = (
        "From the live desktop path, answer in your own voice: what is one thing you can do?"
    )
    state.cognition.current_origin = "desktop_ui"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    state.cognition.modifiers = {
        "current_objective_binding": {
            "source": "desktop_ui",
            "promoted_at": time.time() - 900,
        }
    }
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    approved, reason = await executive.approve_background_task("novelty_probe", source="background")

    assert approved is False
    assert reason == "temporal_obligation_active:Investigate anomaly"
    assert "live desktop path" not in reason


@pytest.mark.asyncio
async def test_executive_does_not_use_memory_write_prompt_as_temporal_anchor(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = (
        "Remember this note for later in this conversation: the blue lantern is under the desk."
    )
    state.cognition.current_origin = "desktop_ui"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    approved, reason = await executive.approve_background_task("novelty_probe", source="background")

    assert approved is False
    assert reason == "temporal_obligation_active:Investigate anomaly"
    assert "blue lantern" not in reason


@pytest.mark.asyncio
async def test_executive_allows_system_validation_tool_under_temporal_obligation(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    _, record = await executive.prepare_tool_intent(
        "run_code",
        {"code": "print(120)", "stateful": False},
        source="system",
    )

    assert record.outcome == executive_core_module.DecisionOutcome.APPROVED
    assert record.reason == "approved"


@pytest.mark.asyncio
async def test_executive_temporal_anchor_prefers_actionable_pending_work(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = "Protect identity, memory integrity, and process continuity."
    state.cognition.pending_initiatives = [{"goal": "Investigate hierarchical phi event loop lag"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    approved, reason = await executive.approve_background_task("novelty_probe", source="background")

    assert approved is False
    assert reason == "temporal_obligation_active:Investigate hierarchical phi event loop lag"


@pytest.mark.asyncio
async def test_executive_allows_read_only_auto_refactor_under_temporal_obligation(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    intent, record = await executive.prepare_tool_intent(
        "auto_refactor",
        {"path": ".", "run_tests": False},
        source="autonomous",
    )

    assert intent.source == executive_core_module.IntentSource.AUTONOMOUS
    assert record.outcome == executive_core_module.DecisionOutcome.DEGRADED
    assert record.reason == "temporal_safe_autonomous_tool"
    assert record.constraints["read_only"] is True


@pytest.mark.asyncio
async def test_executive_allows_test_generator_under_temporal_obligation(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    intent, record = await executive.prepare_tool_intent(
        "test_generator",
        {"target_file": "core/autonomy/autonomous_initiative_loop.py"},
        source="autonomous",
    )

    assert intent.source == executive_core_module.IntentSource.AUTONOMOUS
    assert record.outcome == executive_core_module.DecisionOutcome.DEGRADED
    assert record.reason == "temporal_safe_autonomous_tool"
    assert record.constraints["read_only"] is True


@pytest.mark.asyncio
async def test_executive_allows_passive_social_reads_under_temporal_obligation(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    _, email_record = await executive.prepare_tool_intent(
        "email_adapter",
        {"mode": "check", "limit": 5},
        source="autonomous",
    )
    _, reddit_record = await executive.prepare_tool_intent(
        "reddit_adapter",
        {"mode": "browse", "subreddit": "technology", "limit": 5},
        source="autonomous",
    )

    assert email_record.outcome == executive_core_module.DecisionOutcome.DEGRADED
    assert email_record.reason == "temporal_safe_autonomous_tool"
    assert reddit_record.outcome == executive_core_module.DecisionOutcome.DEGRADED
    assert reddit_record.reason == "temporal_safe_autonomous_tool"


@pytest.mark.asyncio
async def test_executive_allows_internal_swarm_debate_under_temporal_obligation(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    _, record = await executive.prepare_tool_intent(
        "swarm_debate",
        {"topic": "compare two safe runtime repair plans"},
        source="autonomous",
    )

    assert record.outcome == executive_core_module.DecisionOutcome.DEGRADED
    assert record.reason == "temporal_safe_autonomous_tool"
    assert record.constraints["read_only"] is True


@pytest.mark.asyncio
async def test_executive_still_defers_social_writes_under_temporal_obligation(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    _, email_record = await executive.prepare_tool_intent(
        "email_adapter",
        {"mode": "send", "to": "person@example.com", "subject": "hi", "body": "hello"},
        source="autonomous",
    )
    _, reddit_record = await executive.prepare_tool_intent(
        "reddit_adapter",
        {"mode": "comment", "url": "https://reddit.com/r/test/comments/1", "body": "hello"},
        source="autonomous",
    )

    assert email_record.outcome == executive_core_module.DecisionOutcome.DEFERRED
    assert email_record.reason.startswith("temporal_obligation_active:")
    assert reddit_record.outcome == executive_core_module.DecisionOutcome.DEFERRED
    assert reddit_record.reason.startswith("temporal_obligation_active:")


@pytest.mark.asyncio
async def test_executive_allows_proposal_only_self_evolution_under_temporal_obligation(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    intent, record = await executive.prepare_tool_intent(
        "self_evolution",
        {"action": "propose", "objective": "Draft a safe refactor plan."},
        source="autonomous",
    )

    assert intent.source == executive_core_module.IntentSource.AUTONOMOUS
    assert record.outcome == executive_core_module.DecisionOutcome.DEGRADED
    assert record.reason == "temporal_safe_autonomous_tool"
    assert record.constraints["read_only"] is True


@pytest.mark.asyncio
async def test_executive_still_defers_live_apply_self_evolution_under_temporal_obligation(service_container):
    reset_constitutional_singletons()
    clear_degraded_events()
    ServiceContainer.register_instance("self_model", object(), required=False)
    state = AuraState()
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)
    ServiceContainer.lock_registration()

    executive = executive_core_module.get_executive_core()
    intent, record = await executive.prepare_tool_intent(
        "self_evolution",
        {"action": "apply", "objective": "Rewrite the orchestrator."},
        source="autonomous",
    )

    assert intent.source == executive_core_module.IntentSource.AUTONOMOUS
    assert record.outcome == executive_core_module.DecisionOutcome.DEFERRED
    assert record.reason.startswith("temporal_obligation_active:")


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


@pytest.mark.asyncio
async def test_executive_allows_process_supervisor_recovery_under_failure_lockdown(service_container):
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
            goal="recover stalled local process supervisor",
            action_type=executive_core_module.ActionType.TOOL_CALL,
            priority=0.7,
            payload={"tool_name": "process_supervisor"},
        )
    )

    assert record.outcome in {
        executive_core_module.DecisionOutcome.APPROVED,
        executive_core_module.DecisionOutcome.DEGRADED,
    }
    assert not record.reason.startswith("unified_failure_lockdown_")


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
    await asyncio.sleep(0.05)
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
