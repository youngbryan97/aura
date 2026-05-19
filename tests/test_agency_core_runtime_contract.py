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


@pytest.mark.asyncio
async def test_swarm_spawn_does_not_keep_untracked_shards(monkeypatch):
    monkeypatch.setattr(agency_module, "get_task_tracker", lambda: FailingTracker())
    monkeypatch.setattr("core.runtime.background_policy.background_activity_reason", lambda *args, **kwargs: "")

    swarm = SovereignSwarm(SimpleNamespace(cognitive_engine=object()))
    spawned = await swarm.spawn_shard("inspect continuity", "runtime contract")

    assert spawned is False
    assert swarm.active_shards == {}


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
