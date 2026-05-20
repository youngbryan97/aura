from types import SimpleNamespace

import pytest

from core.collective.delegator import AgentDelegator, SwarmAgent
from core.runtime.errors import get_degradation_tracker


def _mute_container(monkeypatch):
    monkeypatch.setattr(
        "core.collective.delegator.ServiceContainer.get",
        lambda _name, default=None: default,
    )


@pytest.mark.asyncio
async def test_callback_failure_does_not_poison_completed_agent(monkeypatch):
    _mute_container(monkeypatch)
    get_degradation_tracker().reset()

    class Brain:
        async def think(self, *_args, **_kwargs):
            return SimpleNamespace(content="shard complete")

    async def broken_callback(**_kwargs):
        raise RuntimeError("callback failed")

    delegator = AgentDelegator(SimpleNamespace(cognitive_engine=Brain()))
    agent_id = await delegator.delegate("critic", "review this", callback=broken_callback)
    agent = delegator.active_agents[agent_id]

    await agent.done_event.wait()

    assert agent.status == "COMPLETED"
    assert agent.result == "shard complete"
    assert any(
        record.action == "preserved agent result after callback failed"
        for record in get_degradation_tracker().recent(subsystem="delegator", limit=5)
    )


@pytest.mark.asyncio
async def test_empty_shard_output_fails_closed_with_explicit_result(monkeypatch):
    _mute_container(monkeypatch)

    class Brain:
        async def think(self, *_args, **_kwargs):
            return SimpleNamespace(content="  ")

    delegator = AgentDelegator(SimpleNamespace(cognitive_engine=Brain()))
    agent_id = await delegator.delegate("researcher", "find the gap")
    agent = delegator.active_agents[agent_id]

    await agent.done_event.wait()

    assert agent.status == "FAILED"
    assert "empty output" in str(agent.result)


@pytest.mark.asyncio
async def test_debate_timeout_marks_agents_failed_and_releases_capacity(monkeypatch):
    _mute_container(monkeypatch)

    class FakeTask:
        def __init__(self):
            self.cancelled = False

        def done(self):
            return False

        def cancel(self):
            self.cancelled = True

    delegator = AgentDelegator(SimpleNamespace(cognitive_engine=None))

    async def stuck_delegate(specialty, _prompt, **_kwargs):
        agent_id = f"stuck-{specialty}"
        agent = SwarmAgent(agent_id, specialty)
        agent.status = "BUSY"
        agent.task = FakeTask()
        delegator.active_agents[agent_id] = agent
        return agent_id

    monkeypatch.setattr(delegator, "delegate", stuck_delegate)

    result = await delegator.delegate_debate("sealed task", roles=["architect"], timeout=0.05)
    agent = delegator.active_agents["stuck-architect"]

    assert "failed to produce" in result
    assert agent.status == "FAILED"
    assert agent.done_event.is_set()
    assert agent.task.cancelled is True
    assert delegator._busy_count() == 0


@pytest.mark.asyncio
async def test_join_all_ignores_retained_completed_agents(monkeypatch):
    _mute_container(monkeypatch)
    delegator = AgentDelegator(SimpleNamespace(cognitive_engine=None))
    agent = SwarmAgent("done", "critic")
    agent.status = "COMPLETED"
    agent.done_event.set()
    delegator.active_agents[agent.id] = agent

    assert await delegator.join_all(timeout=0.05) is True


@pytest.mark.asyncio
async def test_orchestrator_tool_adapter_preserves_origin_outside_payload(monkeypatch):
    _mute_container(monkeypatch)

    class Engine:
        def __init__(self):
            self.tools = {}

        def register_tool(self, name, fn):
            self.tools[name] = fn

    class Orchestrator:
        capability_engine = SimpleNamespace(skills={"web_search": object()})

        async def execute_tool(self, tool_name, args, **kwargs):
            return {"tool_name": tool_name, "args": args, "kwargs": kwargs}

    engine = Engine()
    delegator = AgentDelegator(SimpleNamespace(cognitive_engine=None))

    assert delegator._register_orchestrator_tools(engine, Orchestrator()) == 1

    result = await engine.tools["web_search"](query="aura", origin="swarm_agent:test")

    assert result == {
        "tool_name": "web_search",
        "args": {"query": "aura"},
        "kwargs": {"origin": "swarm_agent:test"},
    }


def test_deterministic_consensus_tolerates_invalid_confidence(monkeypatch):
    _mute_container(monkeypatch)
    delegator = AgentDelegator(SimpleNamespace(cognitive_engine=None))

    result = delegator._deterministic_consensus(
        "ship readiness",
        ['{"claim":"ready with caveats","confidence":"unknown","flaws":["needs eval"]}'],
    )

    assert "ready with caveats" in result
    assert "Confidence: 0.50" in result
    assert "needs eval" in result
