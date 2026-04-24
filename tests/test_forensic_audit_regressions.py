import asyncio
import importlib.util
import os
import signal
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import core.constitution as constitution_module
import core.executive.authority_gateway as authority_gateway_module
import core.executive.executive_core as executive_core_module
from core.adaptation.epistemic_humility import EpistemicHumility, FailureEvent
from core.agency.capability_system import get_capability_manager
from core.adaptation.heuristic_synthesizer import HeuristicSynthesizer
from core.agency.self_play import ContinuousSelfPlay
from core.consciousness.substrate_authority import (
    AuthorizationDecision,
    SubstrateVerdict,
)
from core.constitution import get_constitutional_core
from core.container import ServiceContainer
from core.data.project_store import ProjectStore
from core.executive.authority_gateway import get_authority_gateway
from core.global_workspace import GlobalWorkspace, WorkItem
from core.goals.goal_engine import GoalEngine
from core.health.degraded_events import clear_degraded_events
from core.health.degraded_events import get_recent_degraded_events
from core.health.degraded_events import record_degraded_event
from core.cognitive.router import IntentRouter, Intent
from core.orchestrator.main import RobustOrchestrator
from core.plugin_loader import PluginManager
from core.runtime.impulse_governance import run_governed_impulse
from core.runtime.proposal_governance import (
    propose_governed_initiative_to_state,
    queue_governed_initiative,
)
from core.runtime.organism_status import get_organism_status
from core.runtime.service_access import resolve_identity_prompt_surface
from core.senses.sensory_client import SensoryLocalClient
from core.senses.sensory_instincts import SensoryInstincts
from core.skills.base_skill import BaseSkill
from core.state.aura_state import AuraState
from core.systems.metabolism import MetabolismEngine


def _reset_authority_runtime() -> None:
    ServiceContainer.clear()
    ServiceContainer._registration_locked = False
    constitution_module._instance = None
    authority_gateway_module._instance = None
    executive_core_module._instance = None
    get_capability_manager()._tokens.clear()
    clear_degraded_events()


def test_project_store_transaction_rolls_back_on_error(tmp_path):
    db_path = tmp_path / "projects.db"
    store = ProjectStore(str(db_path))

    with pytest.raises(RuntimeError):
        with store.transaction() as conn:
            project = store.create_project("Alpha", "Goal", conn=conn)
            store.add_task(project.id, "step-1", conn=conn)
            raise RuntimeError("force rollback")

    assert store.get_active_projects() == []


def test_metabolism_dir_size_returns_total_for_success_path(tmp_path):
    target = tmp_path / "cache"
    target.mkdir()
    (target / "a.bin").write_bytes(b"a" * 3)
    nested = target / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"b" * 5)

    assert MetabolismEngine._dir_size(target) == 8


def test_plugin_manager_allows_normal_dunder_init_and_blocks_dangerous_dunder_access(tmp_path):
    manager = PluginManager(plugin_dir=str(tmp_path))

    ok_file = tmp_path / "ok.py"
    ok_file.write_text(
        "class MyPlugin:\n"
        "    def __init__(self):\n"
        "        self.value = 1\n",
        encoding="utf-8",
    )
    assert manager.validate_plugin(str(ok_file)) is True

    bad_file = tmp_path / "bad.py"
    bad_file.write_text(
        "class MyPlugin:\n"
        "    def leak(self, obj):\n"
        "        return obj.__subclasses__()\n",
        encoding="utf-8",
    )
    assert manager.validate_plugin(str(bad_file)) is False


def test_orchestrator_services_reports_missing_critical_service_once():
    _reset_authority_runtime()
    ServiceContainer.register_instance("executive_core", object(), required=False)
    ServiceContainer.lock_registration()

    orchestrator = RobustOrchestrator.__new__(RobustOrchestrator)

    assert orchestrator.knowledge_graph is None
    assert orchestrator.knowledge_graph is None

    events = [
        event for event in get_recent_degraded_events(limit=10)
        if event.get("subsystem") == "orchestrator_services"
        and event.get("reason") == "critical_service_missing"
        and event.get("detail") == "knowledge_graph"
    ]
    assert len(events) == 1


def test_orchestrator_package_exports_async_agent_alias():
    import core.orchestrator as orchestrator_package

    assert orchestrator_package.AsyncAgentOrchestrator is RobustOrchestrator


@pytest.mark.asyncio
async def test_intent_router_records_missing_llm_as_noncritical_fallback():
    _reset_authority_runtime()

    router = IntentRouter()

    result = await router.classify("Tell me something interesting about yourself right now.")

    assert result == Intent.CHAT
    events = [
        event for event in get_recent_degraded_events(limit=10)
        if event.get("subsystem") == "intent_router"
        and event.get("reason") == "llm_router_missing"
    ]
    assert len(events) == 1


def test_legacy_orchestrator_module_is_a_package_shim():
    shim_path = Path(__file__).resolve().parents[1] / "core" / "orchestrator.py"
    spec = importlib.util.spec_from_file_location("legacy_orchestrator_shim", shim_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import core.orchestrator as orchestrator_package

    assert module.RobustOrchestrator is orchestrator_package.RobustOrchestrator
    assert module.AsyncAgentOrchestrator is orchestrator_package.AsyncAgentOrchestrator
    assert module.create_orchestrator is orchestrator_package.create_orchestrator
    assert module.SovereignOrchestrator is orchestrator_package.SovereignOrchestrator
    assert module.Orchestrator is orchestrator_package.Orchestrator
    assert module.SystemStatus is orchestrator_package.SystemStatus


def test_legacy_orchestrator_boot_module_is_a_package_shim():
    shim_path = Path(__file__).resolve().parents[1] / "core" / "orchestrator_boot.py"
    spec = importlib.util.spec_from_file_location("legacy_orchestrator_boot_shim", shim_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import core.orchestrator.boot as boot_package

    assert module.OrchestratorBootMixin is boot_package.OrchestratorBootMixin


def test_legacy_identity_module_is_a_package_shim():
    shim_path = Path(__file__).resolve().parents[1] / "core" / "identity.py"
    spec = importlib.util.spec_from_file_location("legacy_identity_shim", shim_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import core.identity as identity_package

    assert module.IdentityCore is identity_package.IdentityCore
    assert module.IdentitySystem is identity_package.IdentitySystem
    assert module.get_identity_system is identity_package.get_identity_system
    assert module.identity_manager is identity_package.identity_manager


def test_legacy_goals_module_is_a_package_shim():
    shim_path = Path(__file__).resolve().parents[1] / "core" / "goals.py"
    spec = importlib.util.spec_from_file_location("legacy_goals_shim", shim_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import core.goals as goals_package

    assert module.GoalEngine is goals_package.GoalEngine
    assert module.Goal("Protect continuity").objective == "Protect continuity"


def test_epistemic_humility_tie_break_prefers_most_recent_source():
    humility = EpistemicHumility(orchestrator=None)
    failures = [
        FailureEvent(source="vision", error_msg="a", context=""),
        FailureEvent(source="audio", error_msg="b", context=""),
        FailureEvent(source="vision", error_msg="c", context=""),
        FailureEvent(source="audio", error_msg="d", context=""),
    ]

    assert humility._select_domain(failures) == "audio"


def test_heuristic_synthesizer_trim_prefers_proven_rules(tmp_path):
    synth = HeuristicSynthesizer(heuristics_path=str(tmp_path / "heuristics.json"))
    synth._active_heuristics = [
        {
            "rule": f"new-rule-{idx}",
            "created_at": float(idx),
            "hits": 0,
            "survival_count": 0,
        }
        for idx in range(20)
    ]
    synth._active_heuristics.append(
        {
            "rule": "battle-tested",
            "created_at": 0.0,
            "hits": 4,
            "survival_count": 9,
        }
    )

    synth._trim_heuristics()

    rules = {item["rule"] for item in synth._active_heuristics}
    assert "battle-tested" in rules
    assert len(synth._active_heuristics) == 20


def test_self_play_quality_gate_rejects_long_but_unstructured_answer():
    self_play = ContinuousSelfPlay()
    weak_solution = " ".join(["answer"] * 60)

    succeeded, confidence, reason = self_play._evaluate_solution_quality(weak_solution)

    assert succeeded is False
    assert confidence < 0.55
    assert "missing_reasoning_markers" in reason


def test_self_play_quality_gate_accepts_structured_reasoning():
    self_play = ContinuousSelfPlay()
    strong_solution = (
        "1. First, identify the resource constraint because the timeline and budget conflict. "
        "2. Second, compare the trade-off if we optimize for speed versus reliability. "
        "Therefore the best resolution is to stage the rollout, validate the critical path, "
        "and keep the risky branch isolated until the dependencies stabilize.\n\n"
        "Final answer: execute the staged plan, verify each checkpoint, then widen scope."
    )

    succeeded, confidence, reason = self_play._evaluate_solution_quality(strong_solution)

    assert succeeded is True
    assert confidence >= 0.55
    assert reason == "structured"


@pytest.mark.asyncio
async def test_base_skill_stats_are_per_instance_and_count_failures():
    class DemoSkill(BaseSkill):
        name = "demo"

        def __init__(self, should_fail: bool = False):
            self.should_fail = should_fail

        async def execute(self, params, context):
            if self.should_fail:
                raise RuntimeError("boom")
            return {"ok": True, "summary": "done"}

    ok_skill = DemoSkill()
    fail_skill = DemoSkill(should_fail=True)

    ok_result = await ok_skill.safe_execute({})
    fail_result = await fail_skill.safe_execute({})

    assert ok_result["ok"] is True
    assert fail_result["ok"] is False
    assert ok_skill.get_stats()["executions"] == 1
    assert ok_skill.get_stats()["failures"] == 0
    assert fail_skill.get_stats()["executions"] == 1
    assert fail_skill.get_stats()["failures"] == 1


@pytest.mark.asyncio
async def test_core_base_skill_preserves_error_dict_without_forcing_ok_true():
    class ErrorOnlySkill(BaseSkill):
        name = "error_only"

        async def execute(self, params, context):
            return {"error": "nope"}

    result = await ErrorOnlySkill().safe_execute({})

    assert result["ok"] is False
    assert result["error"] == "nope"


@pytest.mark.asyncio
async def test_legacy_base_skill_uses_to_thread_for_sync_execute_and_preserves_errors(monkeypatch):
    from infrastructure.base_skill import BaseSkill as LegacyBaseSkill

    calls = []

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append(fn.__name__)
        return fn(*args, **kwargs)

    monkeypatch.setattr("infrastructure.base_skill.asyncio.to_thread", fake_to_thread)

    class SyncErrorSkill(LegacyBaseSkill):
        name = "sync_error"

        def execute(self, goal, context):
            return {"error": "blocked"}

    result = await SyncErrorSkill().safe_execute({})

    assert calls == ["execute"]
    assert result["ok"] is False
    assert result["error"] == "blocked"


@pytest.mark.asyncio
async def test_filesystem_reality_shortcut_is_disabled_for_user_facing_requests():
    from core.orchestrator.mixins.incoming_logic import IncomingLogicMixin

    class Probe(IncomingLogicMixin):
        output_gate = SimpleNamespace(emit=AsyncMock())

    probe = Probe()

    handled = await probe._handle_filesystem_reality_check(
        "check if /tmp/example.txt exists",
        "user",
    )

    assert handled is False
    probe.output_gate.emit.assert_not_called()


@pytest.mark.asyncio
async def test_graceful_shutdown_signal_path_does_not_raise_system_exit(monkeypatch):
    from core.graceful_shutdown import GracefulShutdown

    hook = AsyncMock()
    container = SimpleNamespace(shutdown=AsyncMock())

    GracefulShutdown._hooks = [hook]
    GracefulShutdown._is_shutting_down = False
    GracefulShutdown._shutdown_event = asyncio.Event()

    monkeypatch.setattr("core.container.get_container", lambda: container)

    await GracefulShutdown.trigger_shutdown(signal.SIGTERM)

    hook.assert_awaited_once()
    container.shutdown.assert_awaited_once()
    assert GracefulShutdown._shutdown_event.is_set()


@pytest.mark.asyncio
async def test_graceful_shutdown_signal_handlers_are_task_tracked(monkeypatch):
    from core.graceful_shutdown import GracefulShutdown

    handlers = []
    scheduled = {}
    container = SimpleNamespace(shutdown=AsyncMock())

    class _Loop:
        def add_signal_handler(self, sig, callback):
            handlers.append((sig, callback))

    class _Tracker:
        def create_task(self, coro, name=None):
            task = asyncio.create_task(coro, name=name)
            scheduled["name"] = name
            scheduled["task"] = task
            return task

    GracefulShutdown._hooks = []
    GracefulShutdown._is_shutting_down = False
    GracefulShutdown._shutdown_event = None

    monkeypatch.setattr("asyncio.get_running_loop", lambda: _Loop())
    monkeypatch.setattr("core.graceful_shutdown.get_task_tracker", lambda: _Tracker())
    monkeypatch.setattr("core.container.get_container", lambda: container)

    GracefulShutdown.setup_signals()

    assert handlers
    handlers[0][1]()
    await scheduled["task"]

    assert scheduled["name"].startswith("graceful_shutdown.")
    container.shutdown.assert_awaited_once()


def test_file_operation_no_longer_allows_desktop_agency_test_escape(tmp_path):
    from core.skills.file_operation import FileOperationSkill

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "agency_test" / "proof.txt"
    outside.parent.mkdir()

    skill = FileOperationSkill()
    skill.root_dir = os.path.realpath(str(workspace))

    with pytest.raises(PermissionError):
        skill._safe_resolve(str(outside))


@pytest.mark.asyncio
async def test_sensory_gate_run_always_closes_browser_and_bus(monkeypatch):
    from core.actors import sensory_gate as sensory_gate_module

    events = []
    actor = None

    class FakeBus:
        def __init__(self, *args, **kwargs):
            pass

        def register_handler(self, *_args, **_kwargs):
            return None

        def start(self):
            actor._is_active = False

        async def stop(self):
            events.append("bus.stop")

        async def send(self, *_args, **_kwargs):
            events.append("bus.send")

    class FakeBrowser:
        def __init__(self, *args, **kwargs):
            events.append("browser.init")

        async def close(self):
            events.append("browser.close")

    monkeypatch.setattr(sensory_gate_module, "LocalPipeBus", FakeBus)
    monkeypatch.setattr(sensory_gate_module, "PhantomBrowser", FakeBrowser)

    actor = sensory_gate_module.SensoryGateActor(connection=object())
    await actor.run()

    assert "browser.close" in events
    assert "bus.stop" in events


def test_sensory_client_defers_async_lock_creation():
    client = SensoryLocalClient()

    assert client._lock is None
    assert client._start_lock is None


def test_sensory_instincts_respects_substrate_authority_block(monkeypatch):
    class BlockingAuthority:
        def authorize(self, **_kwargs):
            from core.consciousness.substrate_authority import (
                AuthorizationDecision,
                SubstrateVerdict,
            )

            return SubstrateVerdict(
                decision=AuthorizationDecision.BLOCK,
                reason="blocked",
                field_coherence=0.1,
                somatic_approach=-0.8,
                somatic_confidence=0.9,
                neurochemical_state="cortisol_crisis",
                body_budget_available=False,
                constraints=[],
            )

    class LiquidState:
        def __init__(self):
            self.called = False

        def update(self, **_kwargs):
            self.called = True

    liquid_state = LiquidState()

    def fake_get(name, default=None):
        if name == "substrate_authority":
            return BlockingAuthority()
        if name == "liquid_state":
            return liquid_state
        return default

    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(fake_get))

    instincts = SensoryInstincts(orchestrator=None)
    applied = instincts.trigger_spike("vision", 0.9, emotion="curiosity")

    assert applied is False
    assert liquid_state.called is False


def test_goal_engine_requires_stronger_phrase_alignment():
    engine = GoalEngine()
    objective = "understand machine learning"
    misleading_message = "I do not understand why this machine broke while learning new skills."

    assert engine._shows_goal_progress(objective, [misleading_message]) is False
    assert engine._shows_goal_progress(objective, ["We now understand machine learning fundamentals clearly."]) is True


def test_global_workspace_history_buffer_is_bounded_and_sliceable():
    workspace = GlobalWorkspace()

    for idx in range(600):
        workspace.history.append(
            WorkItem(
                priority=1.0,
                ts=float(idx),
                id=str(idx),
                source="test",
                payload={"idx": idx},
                reason=None,
            )
        )

    assert len(workspace.history) == workspace.max_history
    tail = workspace.history[-3:]
    assert isinstance(tail, list)
    assert [item.payload["idx"] for item in tail] == [597, 598, 599]


def test_service_container_require_fails_loudly_for_missing_service():
    _reset_authority_runtime()

    with pytest.raises(Exception):
        ServiceContainer.require("missing_service")


@pytest.mark.asyncio
async def test_authority_gateway_generates_and_revokes_tool_capability_token():
    _reset_authority_runtime()

    gateway = get_authority_gateway()
    decision = await gateway.authorize_tool_execution(
        "demo_tool",
        {"value": 1},
        source="user",
    )

    assert decision.approved is True
    assert decision.executive_intent_id
    assert decision.capability_token_id
    assert gateway.verify_tool_access("demo_tool", decision.capability_token_id) is True

    gateway.finalize_tool_execution(
        executive_intent_id=decision.executive_intent_id,
        capability_token_id=decision.capability_token_id,
        success=True,
    )

    assert gateway.verify_tool_access("demo_tool", decision.capability_token_id) is False


@pytest.mark.asyncio
async def test_constitution_tool_handle_carries_capability_token_and_revokes_it():
    _reset_authority_runtime()

    constitution = get_constitutional_core()
    gateway = get_authority_gateway()

    handle = await constitution.begin_tool_execution(
        "demo_tool",
        {"value": 1},
        source="user",
        objective="Run a safe demo tool",
    )

    assert handle.approved is True
    assert handle.capability_token_id
    assert gateway.verify_tool_access("demo_tool", handle.capability_token_id) is True

    await constitution.finish_tool_execution(
        handle,
        result={"ok": True},
        success=True,
        duration_ms=1.0,
    )

    assert gateway.verify_tool_access("demo_tool", handle.capability_token_id) is False


@pytest.mark.asyncio
async def test_constitution_initiative_gate_requires_substrate_in_strict_runtime():
    _reset_authority_runtime()
    ServiceContainer._registration_locked = True

    approved, reason, _decision = await get_constitutional_core().approve_initiative(
        "peer_mode:permanent_swarm_debate",
        source="peer_mode",
        urgency=0.4,
    )

    assert approved is False
    assert reason.startswith("substrate_authority_required:")


@pytest.mark.asyncio
async def test_constitution_response_gate_preserves_substrate_receipt():
    _reset_authority_runtime()

    class ConstrainingAuthority:
        def authorize(self, **_kwargs):
            return SubstrateVerdict(
                decision=AuthorizationDecision.CONSTRAIN,
                reason="field_warning",
                field_coherence=0.33,
                somatic_approach=-0.1,
                somatic_confidence=0.7,
                neurochemical_state="normal",
                body_budget_available=True,
                constraints=["field_warning"],
                receipt_id="receipt-1",
            )

    ServiceContainer.register_instance("substrate_authority", ConstrainingAuthority(), required=False)

    approved, reason, decision = await get_constitutional_core().approve_response(
        "hello there",
        source="user",
        urgency=0.4,
    )

    assert approved is True
    assert reason in {"user_facing", "essential_action"}
    assert decision is not None
    assert decision.substrate_receipt_id == "receipt-1"
    assert decision.constraints["substrate_constrained"] is True


def test_organism_status_aggregates_canonical_self_and_failure_pressure():
    _reset_authority_runtime()
    state = AuraState.default()
    state.version = 12
    state.cognition.current_objective = "Protect continuity"
    state.cognition.pending_initiatives = [{"goal": "Investigate anomaly"}]
    state.cognition.active_goals = [{"goal": "Stay coherent"}]
    ServiceContainer.register_instance("state_repository", type("Repo", (), {"_current": state})(), required=False)
    ServiceContainer.register_instance(
        "canonical_self",
        type("Canonical", (), {"version": 7, "current_intention": "stabilize continuity"})(),
        required=False,
    )
    record_degraded_event("memory", "wal_pressure", severity="warning")

    organism = get_organism_status()

    assert organism["canonical_self_version"] == 7
    assert organism["state_version"] == 12
    assert organism["current_objective"] == "Protect continuity"
    assert organism["current_intention"] == "stabilize continuity"
    assert organism["pending_initiatives"] == 1
    assert organism["active_goals"] == 1
    assert organism["failure_pressure"] > 0.0
    assert organism["identity_surface"] == "canonical_self"
    assert organism["identity_name"]


@pytest.mark.asyncio
async def test_run_governed_impulse_requires_initiative_approval(monkeypatch):
    _reset_authority_runtime()

    class Queue:
        def full(self):
            return False

    class LiquidState:
        def __init__(self):
            self.calls = []

        async def update(self, **kwargs):
            self.calls.append(kwargs)

    orchestrator = type(
        "Orchestrator",
        (),
        {
            "message_queue": Queue(),
            "liquid_state": LiquidState(),
            "state_repo": type("Repo", (), {"_current": AuraState.default()})(),
        },
    )()
    emitted = []

    def enqueue_message(message, priority=20, origin="background"):
        emitted.append((message, priority, origin))

    orchestrator.enqueue_message = enqueue_message

    class BlockedConstitution:
        async def approve_initiative(self, *args, **kwargs):
            return False, "blocked", None

        async def approve_state_mutation(self, *args, **kwargs):
            return True, "approved"

    monkeypatch.setattr(
        "core.runtime.impulse_governance.get_constitutional_core",
        lambda orch=None: BlockedConstitution(),
    )

    applied = await run_governed_impulse(
        orchestrator,
        source="autonomy",
        summary="blocked impulse",
        message="Impulse: blocked.",
        state_cause="blocked_shift",
        state_update={"delta_curiosity": 0.3},
    )

    assert applied is False
    assert emitted == []
    assert orchestrator.liquid_state.calls == []


@pytest.mark.asyncio
async def test_run_governed_impulse_updates_state_and_enqueues_on_success(monkeypatch):
    _reset_authority_runtime()

    class Queue:
        def full(self):
            return False

    class LiquidState:
        def __init__(self):
            self.calls = []

        async def update(self, **kwargs):
            self.calls.append(kwargs)

    orchestrator = type(
        "Orchestrator",
        (),
        {
            "message_queue": Queue(),
            "liquid_state": LiquidState(),
            "state_repo": type("Repo", (), {"_current": AuraState.default()})(),
        },
    )()
    emitted = []

    def enqueue_message(message, priority=20, origin="background"):
        emitted.append((message, priority, origin))

    orchestrator.enqueue_message = enqueue_message

    class AllowedConstitution:
        async def approve_initiative(self, *args, **kwargs):
            return True, "approved", None

        async def approve_state_mutation(self, *args, **kwargs):
            return True, "approved"

    monkeypatch.setattr(
        "core.runtime.impulse_governance.get_constitutional_core",
        lambda orch=None: AllowedConstitution(),
    )

    applied = await run_governed_impulse(
        orchestrator,
        source="autonomy",
        summary="allowed impulse",
        message="Impulse: allowed.",
        state_cause="allowed_shift",
        state_update={"delta_curiosity": 0.3},
        enqueue_priority=42,
    )

    assert applied is True
    assert emitted == [("Impulse: allowed.", 42, "autonomy")]
    assert orchestrator.liquid_state.calls == [{"_caller": "autonomy", "delta_curiosity": 0.3}]


@pytest.mark.asyncio
async def test_proposal_governance_blocks_before_executive_queue(monkeypatch):
    _reset_authority_runtime()

    state = AuraState.default()
    queued = []

    class BlockingConstitution:
        async def approve_initiative(self, *args, **kwargs):
            return False, "blocked_by_test", None

    class FakeAuthority:
        async def propose_initiative_to_state(self, *args, **kwargs):
            queued.append(True)
            return state, {"action": "queued", "reason": "should_not_happen"}

    monkeypatch.setattr(
        "core.runtime.proposal_governance.get_constitutional_core",
        lambda orch=None: BlockingConstitution(),
    )
    monkeypatch.setattr(
        "core.runtime.proposal_governance.get_executive_authority",
        lambda orch=None: FakeAuthority(),
    )

    new_state, decision = await propose_governed_initiative_to_state(
        state,
        "Investigate anomaly",
        source="test_source",
        kind="test",
        urgency=0.7,
    )

    assert new_state is state
    assert decision["action"] == "blocked"
    assert queued == []


@pytest.mark.asyncio
async def test_queue_governed_initiative_does_not_commit_when_blocked(monkeypatch):
    _reset_authority_runtime()

    state = AuraState.default()
    commits = []

    class Repo:
        def get_current(self):
            return state

        async def commit(self, *_args, **_kwargs):
            commits.append(True)

    class BlockingConstitution:
        async def approve_initiative(self, *args, **kwargs):
            return False, "blocked_by_test", None

    ServiceContainer.register_instance("state_repository", Repo(), required=False)
    monkeypatch.setattr(
        "core.runtime.proposal_governance.get_constitutional_core",
        lambda orch=None: BlockingConstitution(),
    )

    decision = await queue_governed_initiative(
        "Investigate anomaly",
        source="test_source",
        kind="test",
        urgency=0.7,
    )

    assert decision["action"] == "blocked"
    assert commits == []


def test_runtime_state_prefers_canonical_self_over_fallback_stub():
    _reset_authority_runtime()
    ServiceContainer.register_instance(
        "canonical_self",
        type(
            "Canonical",
            (),
            {
                "version": 12,
                "current_intention": "stabilize continuity",
                "identity": type("Identity", (), {"name": "Aura Luna"})(),
            },
        )(),
        required=False,
    )
    ServiceContainer.register_instance(
        "self_model",
        type("Stub", (), {"get_status": lambda self: {"version": 1, "name": "stub"}})(),
        required=False,
    )

    from core.runtime_tools import get_runtime_state

    state = get_runtime_state()["state"]["self_model"]

    assert state["version"] == 12
    assert state["name"] == "Aura Luna"
    assert state["current_intention"] == "stabilize continuity"


def test_core_self_package_self_model_is_compatibility_shim():
    import core.self as self_package
    from core.self_model import SelfModel as RootSelfModel

    assert self_package.SelfModel is RootSelfModel


def test_identity_prompt_surface_prefers_package_identity_over_promptless_stub():
    _reset_authority_runtime()
    ServiceContainer.register_instance("identity", type("Stub", (), {"name": "stub"})(), required=False)

    surface = resolve_identity_prompt_surface(default=None)

    assert surface is not None
    assert hasattr(surface, "get_full_system_prompt")
