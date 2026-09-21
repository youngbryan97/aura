import time
from types import SimpleNamespace

import pytest

from core.collective.belief_sync import BeliefSync
from core.collective.delegator import AgentDelegator, SwarmAgent


class _Thought:
    def __init__(self, content: str):
        self.content = content


class _RecordingCognitiveEngine:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    async def think(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return _Thought(self.response)


class _PendingTask:
    def __init__(self):
        self.cancelled_flag = False

    def cancel(self):
        self.cancelled_flag = True

    def cancelled(self):
        return self.cancelled_flag

    def done(self):
        return False

    def exception(self):
        return None


class _ClosingTaskTracker:
    def __init__(self):
        self.created = []

    def create_task(self, awaitable, name=None):
        self.created.append(name)
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        return _PendingTask()


class _CapabilityExecutionRecorder:
    def __init__(self):
        self.calls = []

    async def execute(self, name, payload):
        self.calls.append({"name": name, "payload": payload})
        return {"ok": True, "peers": []}


@pytest.mark.asyncio
async def test_agent_delegator_concurrency(monkeypatch):
    orchestrator = SimpleNamespace(cognitive_engine=_RecordingCognitiveEngine("result"))
    delegator = AgentDelegator(orchestrator)
    delegator.max_parallel = 2

    tracker = _ClosingTaskTracker()
    monkeypatch.setattr("core.collective.delegator.get_task_tracker", lambda: tracker)
    monkeypatch.setattr(delegator, "effective_max_parallel", lambda: delegator.max_parallel)

    first_id = await delegator.delegate("critic", "Task 1")
    second_id = await delegator.delegate("critic", "Task 2")
    third_id = await delegator.delegate("critic", "Task 3")

    assert first_id
    assert second_id
    assert third_id == ""
    assert len(delegator.active_agents) == 2
    assert {agent.status for agent in delegator.active_agents.values()} == {"BUSY"}
    assert tracker.created == [f"AgentDelegator.{first_id}", f"AgentDelegator.{second_id}"]


@pytest.mark.asyncio
async def test_agent_delegator_debate_synthesis(monkeypatch):
    """The synthesis is one request for words, through the router.

    It used to go through the cognitive engine, and that made it a
    cognitive TURN: memory retrieval searched for "You are the Master
    Synthesizer" (5.2s, over budget) and the kernel answered it as
    conversation until the loop detector fired, four times in an hour
    (LIVE 2026-09-19). This test stubbed the engine and went on asserting
    against a seam the synthesis no longer uses.
    """
    router = _RecordingCognitiveEngine("Consensus result")
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: router if name == "llm_router" else default),
        raising=False,
    )
    delegator = AgentDelegator(SimpleNamespace(cognitive_engine=object()))

    async def complete_delegate(specialty, prompt, **kwargs):
        agent_id = f"role-{specialty}"
        agent = SwarmAgent(agent_id, specialty)
        agent.status = "COMPLETED"
        agent.result = f"Result for {specialty}"
        agent.done_event.set()
        delegator.active_agents[agent_id] = agent
        return agent_id

    delegator.delegate = complete_delegate

    result = await delegator.delegate_debate("Test topic", roles=["architect", "critic"])

    assert "Consensus result" in result
    assert len(router.calls) == 1
    call = router.calls[0]
    prompt = call["prompt"] or call["kwargs"].get("prompt", "")
    assert "Result for architect" in prompt
    assert "Result for critic" in prompt
    assert "Test topic" in prompt


@pytest.mark.asyncio
async def test_agent_delegator_debate_without_a_router(monkeypatch):
    """The null. With no router the debate still returns every claim."""
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: default),
        raising=False,
    )
    delegator = AgentDelegator(SimpleNamespace(cognitive_engine=object()))

    async def complete_delegate(specialty, prompt, **kwargs):
        agent_id = f"role-{specialty}"
        agent = SwarmAgent(agent_id, specialty)
        agent.status = "COMPLETED"
        agent.result = f"Result for {specialty}"
        agent.done_event.set()
        delegator.active_agents[agent_id] = agent
        return agent_id

    delegator.delegate = complete_delegate

    result = await delegator.delegate_debate("Test topic", roles=["architect", "critic"])

    assert "Test topic" in result
    assert "Result for architect" in result
    assert "Result for critic" in result


@pytest.mark.asyncio
async def test_belief_sync_rpc_validation():
    orchestrator = SimpleNamespace(peers={})
    belief_sync = BeliefSync(orchestrator)

    assert await belief_sync.handle_rpc_request("delete_all_files", {}) == {"error": "Method not allowed"}
    assert await belief_sync.handle_rpc_request("query_beliefs", {}) == {"error": "Invalid entity parameter"}
    assert await belief_sync.handle_rpc_request("query_beliefs", {"entity": 123}) == {
        "error": "Invalid entity parameter"
    }


@pytest.mark.asyncio
async def test_belief_sync_discovery_defers_when_background_policy_blocks(monkeypatch):
    orchestrator = SimpleNamespace(peers={}, _last_user_interaction_time=time.time())
    belief_sync = BeliefSync(orchestrator)
    belief_sync.running = True
    belief_sync.discovery_interval = 300.0

    engine = _CapabilityExecutionRecorder()

    monkeypatch.setattr(
        "core.collective.belief_sync.ServiceContainer.get",
        lambda name, default=None: engine if name == "capability_engine" else default,
    )
    monkeypatch.setattr(
        "core.collective.belief_sync.background_policy.background_activity_reason",
        lambda *args, **kwargs: "recent_user_5",
    )

    sleep_calls = {"count": 0}

    async def fake_sleep(_seconds):
        sleep_calls["count"] += 1
        belief_sync.running = False

    monkeypatch.setattr("core.collective.belief_sync.asyncio.sleep", fake_sleep)

    await belief_sync._discovery_loop()

    assert engine.calls == []
    assert sleep_calls["count"] == 1


@pytest.mark.asyncio
async def test_agent_delegator_liveness(monkeypatch):
    tracker = _ClosingTaskTracker()
    orchestrator = SimpleNamespace()

    monkeypatch.setattr("core.collective.delegator.AgentDelegator._scavenger_loop", lambda self: None)
    monkeypatch.setattr("core.collective.delegator.get_task_tracker", lambda: tracker)

    delegator = AgentDelegator(orchestrator)
    assert not delegator.is_alive()

    await delegator.start()
    assert delegator.is_alive()
    assert tracker.created == ["AgentDelegator.scavenger"]

    await delegator.stop()
    assert not delegator.is_alive()


def test_agent_delegator_health_contract_evaluation():
    from core.container import ServiceContainer
    from core.runtime.health_contract import evaluate_health

    delegator = AgentDelegator(SimpleNamespace())
    ServiceContainer.register_instance("agent_delegator", delegator)

    try:
        verdict = evaluate_health()
        status = next(s for s in verdict.services if s.requirement.container_key == "agent_delegator")
        assert status.present
        assert status.liveness_ok is False

        delegator.running = True
        verdict = evaluate_health()
        status = next(s for s in verdict.services if s.requirement.container_key == "agent_delegator")
        assert status.present
        assert status.liveness_ok is True
    finally:
        ServiceContainer._services.pop("agent_delegator", None)
