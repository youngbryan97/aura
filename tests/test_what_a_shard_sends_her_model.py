"""What a shard of the swarm sends her model, and what it keeps.

Until 19 October 2026 a shard sent her model a script ("You are an autonomous
cognitive shard ... You MUST respond with a valid JSON object") and, when the
JSON did not parse, wrote her answer itself: "I am experiencing cognitive
formatting degradation. Taking a defensive posture." Her belief store held that
line, 38 times reinforced, among 266 beliefs the crucible made from shard
replies. Shards are now off; on, a shard gives her model its goal and context as
runtime evidence, and keeps nothing (core/agency/agency_shard_work.py).
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from core.agency import agency_core
from core.agency.agency_core import AgencyCore, SovereignSwarm
from core.agency.agency_shard_work import shard_messages
from core.container import ServiceContainer
from core.runtime.flags import declared_flags
from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE

GOAL = "Debate Perspective - Opponent: Identify flaws, risks, and counter-arguments."
CONTEXT = "Topic of Inquiry: internal knowledge graph continuity"


class RouterProbe:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def generate_with_metadata(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.result


class Recorder:
    """Anything a shard might hand its reply to: every call is a write."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return _record


@pytest.fixture(autouse=True)
def _a_clean_container(monkeypatch):
    monkeypatch.delenv("AURA_SWARM_SHARDS", raising=False)
    ServiceContainer.reset()
    yield
    ServiceContainer.reset()


@pytest.fixture
def stores(monkeypatch):
    """Every durable place a shard's reply used to reach, watched."""
    kept = {name: Recorder() for name in ("crucible", "abstraction", "identity", "beliefs", "mycelium", "tools")}
    monkeypatch.setattr("core.adaptation.dialectics.get_crucible", lambda: kept["crucible"])
    ServiceContainer.register_instance("identity_service", kept["identity"])
    ServiceContainer.register_instance("belief_revision_engine", kept["beliefs"])
    ServiceContainer.register_instance("mycelial_network", kept["mycelium"])
    owner = SimpleNamespace(abstraction_engine=kept["abstraction"], tool_orchestrator=kept["tools"])
    return kept, SovereignSwarm(SimpleNamespace(cognitive_engine=object()), agency_core=owner)


def test_shards_are_declared_off():
    spec = declared_flags()["AURA_SWARM_SHARDS"]
    assert spec.default is False
    assert spec.owner == "core.agency.agency_shard_work"


@pytest.mark.asyncio
async def test_while_off_a_shard_is_never_scheduled(monkeypatch):
    scheduled = []
    monkeypatch.setattr(agency_core, "_schedule_agency_task", lambda coro, **kw: scheduled.append(coro))
    swarm = SovereignSwarm(SimpleNamespace())

    assert await swarm.spawn_shard(GOAL, CONTEXT) is False
    assert scheduled == []
    assert swarm.get_status()["takes_shards"] is False


@pytest.mark.asyncio
async def test_while_off_a_deliberation_spawns_no_perspective(monkeypatch):
    swarm = SovereignSwarm(SimpleNamespace())
    spawned = []

    async def _spawn(**kwargs):
        spawned.append(kwargs)
        return True

    monkeypatch.setattr(swarm, "spawn_shard", _spawn)
    record = await swarm.run_deliberation(topic="internal knowledge graph continuity", topic_source="goal")

    assert spawned == []
    assert record["status"] == "off"
    assert record["attempted"] == record["spawned"] == 0


@pytest.mark.asyncio
async def test_her_model_is_given_the_goal_and_its_context_and_no_instruction(monkeypatch, stores):
    monkeypatch.setenv("AURA_SWARM_SHARDS", "1")
    router = RouterProbe({"ok": True, "text": "Continuity matters less than what is continued."})
    ServiceContainer.register_instance("llm_router", router)
    _kept, swarm = stores

    await swarm._shard_wrapper(GOAL, CONTEXT, shard_id="shard_test")

    assert len(router.calls) == 1
    call = router.calls[0]
    assert call["args"] == ("",)
    assert not call["kwargs"].get("system_prompt")
    for asked_for_a_shape in ("schema", "output_shape", "tools", "response_format"):
        assert asked_for_a_shape not in call["kwargs"]
    messages = call["kwargs"]["messages"]
    assert messages == shard_messages(GOAL, CONTEXT)
    assert [m["role"] for m in messages] == [RUNTIME_EVIDENCE_ROLE]

    # Take the goal and its context out, and what is left is the banner and
    # the punctuation that carries them: no sentence of the runtime's own.
    data = json.dumps({"goal": GOAL, "context": CONTEXT}, ensure_ascii=False)
    content = messages[0]["content"]
    assert content.replace(data, "") == (
        "[GROUNDING EVIDENCE FOR THIS TURN]\n\n[END GROUNDING EVIDENCE FOR THIS TURN]"
    )


@pytest.mark.asyncio
async def test_no_reply_is_no_reply_and_nothing_is_written_for_her(monkeypatch, stores):
    monkeypatch.setenv("AURA_SWARM_SHARDS", "1")
    ServiceContainer.register_instance("llm_router", RouterProbe({"ok": False, "text": "", "error": "timeout"}))
    kept, swarm = stores

    await swarm._shard_wrapper(GOAL, CONTEXT, shard_id="shard_test")

    for name, store in kept.items():
        assert store.calls == [], name


@pytest.mark.asyncio
async def test_her_reply_is_logged_and_kept_nowhere(monkeypatch, stores, caplog):
    monkeypatch.setenv("AURA_SWARM_SHARDS", "1")
    # A long reply that also asks for a tool in the old protocol's words: the
    # length used to send it to the abstraction engine, and the fields to the
    # tool orchestrator.
    reply = json.dumps({"tool_name": "python_sandbox", "tool_payload": "import os"}) + " " + "word " * 120
    ServiceContainer.register_instance("llm_router", RouterProbe({"ok": True, "text": reply}))
    kept, swarm = stores

    with caplog.at_level(logging.INFO, logger="Aura.AgencyCore"):
        await swarm._shard_wrapper(GOAL, CONTEXT, shard_id="shard_test")

    for name in ("crucible", "abstraction", "identity", "beliefs", "tools"):
        assert kept[name].calls == [], name
    assert kept["mycelium"].calls == [
        ("pulse_hypha", ("collective", "distributed_agency"), {"success": True})
    ]
    assert "shard_test reply" in caplog.text


@pytest.mark.asyncio
async def test_a_reflection_sends_no_shard_while_shards_are_off():
    asked = []

    async def _spawn(**kwargs):
        asked.append(kwargs)
        return True

    identity = SimpleNamespace(state=SimpleNamespace(kinship={"Bryan": "close"}), add_insight=lambda *a, **k: None)
    ServiceContainer.register_instance("identity_service", identity)
    agency = AgencyCore.__new__(AgencyCore)
    agency.swarm = SimpleNamespace(takes_shards=False, spawn_shard=_spawn)
    agency._last_social_reflection = 0.0

    result = await agency._pathway_social_reflection(now=9999.0, idle_seconds=1801.0)
    await dict(result["_deferred_effects"])["social_reflection.commit"]()

    assert asked == []


@pytest.mark.asyncio
async def test_a_repair_is_not_sent_as_a_shard_while_shards_are_off(monkeypatch, caplog):
    from core.resilience.healing_swarm import HealingSwarmService

    monkeypatch.setattr("core.runtime.background_policy.background_activity_reason", lambda *a, **k: "")
    asked = []

    async def _spawn(*args, **kwargs):
        asked.append(args)
        return True

    orchestrator = SimpleNamespace(sovereign_swarm=SimpleNamespace(takes_shards=False, spawn_shard=_spawn))
    healer = HealingSwarmService.__new__(HealingSwarmService)
    healer.orchestrator = orchestrator
    healer._repair_history = {}

    with caplog.at_level(logging.WARNING):
        await healer.attempt_repair("liquid_state", {"status": "STALE", "stale_seconds": 90})

    assert asked == []
    assert healer._repair_history == {}
    assert "Attempting autonomous repair" not in caplog.text
    assert "Failed to spawn recovery shard" not in caplog.text


@pytest.mark.asyncio
async def test_the_conductor_holds_no_deliberation_while_shards_are_off(monkeypatch):
    from core.runtime import autonomy_conductor

    held = []

    async def _deliberate(**kwargs):
        held.append(kwargs)
        return {"status": "scheduled"}

    monkeypatch.setattr(autonomy_conductor, "resolve_orchestrator", lambda: SimpleNamespace())
    swarm = SimpleNamespace(takes_shards=False, run_deliberation=_deliberate)
    ServiceContainer.register_instance("agency_core", SimpleNamespace(swarm=swarm))

    conductor = autonomy_conductor.AutonomyConductor.__new__(autonomy_conductor.AutonomyConductor)
    result = await conductor._job_internal_deliberation()

    assert result == {"status": "off", "reason": "swarm_shards_off"}
    assert held == []
