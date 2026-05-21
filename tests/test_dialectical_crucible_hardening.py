import asyncio
import time
from types import SimpleNamespace

import pytest

from core.adaptation.dialectics import DialecticalCrucible
from core.container import ServiceContainer


@pytest.fixture(autouse=True)
def clear_services():
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()


class QuietOrchestrator:
    def __init__(self):
        now = time.time()
        self.start_time = now - 1000.0
        self._last_user_interaction_time = now - 1000.0
        self._suppress_unsolicited_proactivity_until = 0.0
        self._foreground_user_quiet_until = 0.0
        self.is_busy = False


class RecordingBeliefs:
    def __init__(self):
        self.claims = []

    async def process_new_claim(self, **kwargs):
        self.claims.append(kwargs)


class RecordingHypha:
    def __init__(self):
        self.pulses = []

    def pulse(self, **kwargs):
        self.pulses.append(kwargs)


class RecordingMycelium:
    def __init__(self):
        self.hypha = RecordingHypha()

    def get_hypha(self, source, target):
        assert (source, target) == ("adaptation", "cognition")
        return self.hypha


class ScriptedEngine:
    def __init__(self):
        self.calls = []

    async def think(self, *, objective, mode, priority):
        self.calls.append((objective, mode, priority))
        if "ANTAGONIST" in objective:
            return SimpleNamespace(content="The belief ignores boundary conditions.")
        if "DEFENDER" in objective:
            return SimpleNamespace(content="The core claim survives with narrower scope.")
        return SimpleNamespace(content="Synthesis: care is durable when bounded by evidence.")


def _register_quiet_runtime():
    ServiceContainer.register_instance("orchestrator", QuietOrchestrator(), required=False)


@pytest.mark.asyncio
async def test_crucible_commits_synthesis_and_pulses_success():
    _register_quiet_runtime()
    engine = ScriptedEngine()
    beliefs = RecordingBeliefs()
    mycelium = RecordingMycelium()
    ServiceContainer.register_instance("cognitive_engine", engine, required=False)
    ServiceContainer.register_instance("belief_revision_engine", beliefs, required=False)
    ServiceContainer.register_instance("mycelial_network", mycelium, required=False)

    result = await DialecticalCrucible(stage_timeout_s=1.0).run_crucible(
        "Care should guide action", context="ethics"
    )

    assert result["ok"] is True
    assert result["belief_committed"] is True
    assert result["pulse_sent"] is True
    assert beliefs.claims[0]["claim"].startswith("Synthesis:")
    assert beliefs.claims[0]["source"] == "dialectical_crucible"
    assert len(engine.calls) == 3
    assert mycelium.hypha.pulses == [{"success": True}]


@pytest.mark.asyncio
async def test_crucible_fails_closed_when_background_policy_is_unavailable(monkeypatch):
    _register_quiet_runtime()
    engine = ScriptedEngine()
    ServiceContainer.register_instance("cognitive_engine", engine, required=False)

    import core.runtime.background_policy as background_policy

    def broken_policy(*_args, **_kwargs):
        message = "policy unavailable"
        raise RuntimeError(message)

    monkeypatch.setattr(background_policy, "background_activity_reason", broken_policy)

    result = await DialecticalCrucible(stage_timeout_s=1.0).run_crucible("Risky belief")

    assert result == {"ok": False, "reason": "background_policy_unavailable"}
    assert engine.calls == []


@pytest.mark.asyncio
async def test_crucible_stage_timeout_releases_capacity():
    _register_quiet_runtime()

    class HangingEngine:
        async def think(self, *, objective, mode, priority):
            await asyncio.sleep(10)

    crucible = DialecticalCrucible(max_concurrent_debates=1, stage_timeout_s=0.01)
    ServiceContainer.register_instance("cognitive_engine", HangingEngine(), required=False)

    result = await crucible.run_crucible("A belief that cannot be checked")

    assert result == {"ok": False, "reason": "antithesis_failed"}
    assert crucible._active_debates == 0
