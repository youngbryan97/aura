from types import SimpleNamespace

import pytest

import core.agency_core as agency_module
from core.agency_core import AgencyCore, SovereignSwarm, _schedule_agency_task
from core.container import ServiceContainer


@pytest.fixture(autouse=True)
def clean_container():
    ServiceContainer.reset()
    yield
    ServiceContainer.reset()


class ClosingAwaitable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def __await__(self):
        if False:
            yield None
        return None


class FailingTracker:
    def create_task(self, _awaitable, *, name=None):
        self.last_name = name
        raise RuntimeError(f"{name}: loop unavailable")


class NoneReturningTracker:
    def create_task(self, _awaitable, *, name=None):
        self.last_name = name
        return None


class ViabilityBehavior:
    initiative_budget_per_min = 10.0


class ViabilityState:
    value = "healthy"


class Viability:
    state = ViabilityState()

    def behavior(self):
        return ViabilityBehavior()


class Bus:
    def __init__(self, allowed):
        self.allowed = allowed
        self.submitted = []

    def submit(self, payload):
        self.submitted.append(dict(payload))
        return self.allowed


class QuietSelfPlay:
    async def trigger_cycle(self, _timestamp):
        return None


class QuietPhenomenology:
    async def reflect(self, _pad, _events):
        return None


class QuietReporter:
    def get_affect_description(self):
        return {"valence": 0.0, "arousal": 0.5}


def test_agency_scheduler_closes_unscheduled_awaitable():
    awaitable = ClosingAwaitable()

    task = _schedule_agency_task(awaitable, name="agency.contract", tracker=FailingTracker())

    assert task is None
    assert awaitable.closed is True


def test_agency_scheduler_resets_owner_state_when_tracker_returns_none():
    awaitable = ClosingAwaitable()
    owner = SimpleNamespace(pending=True)

    task = _schedule_agency_task(
        awaitable,
        name="agency.none-returning-tracker",
        tracker=NoneReturningTracker(),
        on_unscheduled=lambda: setattr(owner, "pending", False),
    )

    assert task is None
    assert awaitable.closed is True
    assert owner.pending is False


def test_agency_scheduler_resets_owner_state_when_loop_unavailable():
    awaitable = ClosingAwaitable()
    owner = SimpleNamespace(pending=True)

    task = _schedule_agency_task(
        awaitable,
        name="agency.loop-unavailable",
        tracker=FailingTracker(),
        on_unscheduled=lambda: setattr(owner, "pending", False),
    )

    assert task is None
    assert awaitable.closed is True
    assert owner.pending is False


@pytest.mark.asyncio
async def test_swarm_spawn_does_not_keep_untracked_shards(monkeypatch):
    monkeypatch.setenv("AURA_SWARM_SHARDS", "1")
    monkeypatch.setattr(agency_module, "get_task_tracker", lambda: FailingTracker())
    monkeypatch.setattr("core.runtime.background_policy.background_activity_reason", lambda *args, **kwargs: "")

    swarm = SovereignSwarm(SimpleNamespace(cognitive_engine=object()))
    spawned = await swarm.spawn_shard("inspect continuity", "runtime contract")

    assert spawned is False
    assert swarm.active_shards == {}


@pytest.mark.asyncio
async def test_swarm_registry_guard_exists_and_resets_when_update_unscheduled(monkeypatch):
    monkeypatch.setenv("AURA_SWARM_SHARDS", "1")
    degradations = []
    monkeypatch.setattr(agency_module, "get_task_tracker", lambda: FailingTracker())
    monkeypatch.setattr(
        agency_module,
        "_record_agency_degradation",
        lambda error, **_kwargs: degradations.append(error),
    )
    monkeypatch.setattr("core.runtime.background_policy.background_activity_reason", lambda *args, **kwargs: "")

    swarm = SovereignSwarm(SimpleNamespace(cognitive_engine=object()))
    spawned = await swarm.spawn_shard("inspect shutdown recovery", "live proof regression")

    assert spawned is False
    assert swarm._registry_shards_update_pending is False
    assert not [
        error
        for error in degradations
        if isinstance(error, AttributeError) and "_registry_shards_update_pending" in str(error)
    ]


@pytest.mark.asyncio
async def test_agency_initialize_resets_self_play_pending_when_schedule_deferred(monkeypatch):
    async def _fake_consciousness_coordinator():
        return object()

    monkeypatch.setattr(
        "core.consciousness.coordinator.get_consciousness_coordinator",
        _fake_consciousness_coordinator,
    )
    monkeypatch.setattr(agency_module, "get_task_tracker", lambda: FailingTracker())

    agency = AgencyCore(orchestrator=None)
    agency.meta_cognition = None

    await agency.initialize()

    assert agency._self_play_pulse_pending is False


def test_phenomenal_pulse_resets_pending_when_schedule_deferred(monkeypatch):
    async def _fake_phenomenal_integrator():
        return object()

    monkeypatch.setattr(
        "core.affect.phenomenal_integration.get_phenomenal_integrator",
        _fake_phenomenal_integrator,
    )
    monkeypatch.setattr(agency_module, "get_task_tracker", lambda: FailingTracker())

    agency = AgencyCore(orchestrator=None)
    agency._trigger_phenomenological_pulse()

    assert agency._phenomenal_pulse_pending is False


def test_agency_registers_one_dedicated_cognitive_loop_hook_slot():
    agency = AgencyCore(orchestrator=None)

    assert "cognitive_loop" in agency._pathway_registry
    assert agency._pathway_cognitive_loop(0.0, 0.0) is None


def test_agency_status_exposes_cognitive_loop_runtime_receipt():
    agency = AgencyCore(orchestrator=None)
    agency._pathway_hooks["cognitive_loop"] = [object()]
    agency._cognitive_loop_last_run = 1234.5
    agency._last_cognitive_loop_receipt = {
        "verified": False,
        "attempts": 1,
        "workspace": {
            "admitted": True,
            "reason": "accepted_for_competition",
        },
    }

    status = agency.get_status()["cognitive_loop"]

    assert status == {
        "registered": True,
        "running": False,
        "last_run": 1234.5,
        "last_verified": False,
        "last_attempts": 1,
        "workspace_admitted": True,
        "workspace_reason": "accepted_for_competition",
    }


@pytest.mark.asyncio
async def test_pulse_commits_visible_side_effects_only_after_bus_approval(monkeypatch):
    monkeypatch.setattr("core.organism.viability.get_viability", lambda: Viability())
    monkeypatch.setattr("core.runtime.background_policy.background_activity_reason", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "core.consciousness.self_report.SelfReportEngine",
        lambda: QuietReporter(),
    )

    agency = AgencyCore(orchestrator=None)
    agency.self_play_engine = QuietSelfPlay()
    agency.phenomenology = QuietPhenomenology()
    agency.state.unshared_observations = ["screen changed"]
    agency.state.topics_to_discuss = ["runtime honesty"]
    agency.state.last_self_initiated_contact = 0.0
    agency._pathway_registry = {
        "contract_probe": lambda _now, _idle: {
            "type": "initiate_conversation",
            "message": "I noticed something.",
            "source": "contract_probe",
            "priority": 0.9,
            "modality": "chat",
            "_consume_observation": True,
            "_consume_topic": True,
        }
    }

    blocked_bus = Bus(False)
    monkeypatch.setattr(agency_module.AgencyBus, "get", lambda: blocked_bus)

    blocked = await agency.pulse()

    assert blocked is None
    assert agency.state.unshared_observations == ["screen changed"]
    assert agency.state.topics_to_discuss == ["runtime honesty"]
    assert agency.state.last_self_initiated_contact == 0.0
    assert blocked_bus.submitted

    allowed_bus = Bus(True)
    monkeypatch.setattr(agency_module.AgencyBus, "get", lambda: allowed_bus)

    approved = await agency.pulse()

    assert approved is not None
    assert agency.state.unshared_observations == []
    assert agency.state.topics_to_discuss == []
    assert agency.state.last_self_initiated_contact > 0.0
    assert allowed_bus.submitted


@pytest.mark.asyncio
async def test_pulse_fails_closed_when_viability_gate_errors(monkeypatch):
    monkeypatch.setattr(
        "core.organism.viability.get_viability",
        lambda: (_ for _ in ()).throw(RuntimeError("viability offline")),
    )

    agency = AgencyCore(orchestrator=None)
    evaluated = False

    def _pathway(_now, _idle):
        nonlocal evaluated
        evaluated = True
        return {"type": "internal_reflection", "thought": "should not emit", "priority": 1.0}

    agency._pathway_registry = {"must_not_run": _pathway}

    result = await agency.pulse()
    status = agency.get_status()

    assert result is None
    assert evaluated is False
    assert status["status"] == "degraded"
    assert status["alive"] is False
    assert "viability offline" in status["last_viability_error"]


