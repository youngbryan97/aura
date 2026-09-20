"""A swarm shard asks the language organ; it does not become her objective.

LIVE 2026-09-15: "TaskEngine: planning for '[SWARM PROTOCOL: You are 'The
Architect'. Des..." and "subjective preference override: [SWARM PROTOCOL:
...". The shard's persona prompt went through the cognitive engine as an
OBJECTIVE, the initiative arbiter ranked it against her own goals and the
task engine tried to plan it. A shard is one perspective on a prompt, asked
of the router with the perspective as data.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.collective.delegator import AgentDelegator, SwarmAgent
from core.container import ServiceContainer


class _Router:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return "one perspective"


@pytest.mark.asyncio
async def test_the_shard_reaches_the_router_with_its_perspective_as_data(monkeypatch):
    ServiceContainer.clear()
    router = _Router()
    ServiceContainer.register_instance("llm_router", router, required=False)
    # No cognitive engine registered on purpose: the shard must not need one.

    delegator = AgentDelegator(orchestrator=SimpleNamespace(cognitive_engine=None))
    agent = SwarmAgent("agent-1", "architect")
    await delegator._run_agent(agent, "Design a cache for the resident model.", None)

    assert agent.result == "one perspective"
    assert len(router.calls) == 1
    call = router.calls[0]
    assert call["origin"] == "swarm:architect"

    # A caller that names its own origin and tier keeps them: passing both
    # raised "got multiple values for keyword argument 'origin'" live.
    other = SwarmAgent("agent-2", "critic")
    await delegator._run_agent(
        other, "Find the flaw.", None, origin="debate:critic", prefer_tier="primary"
    )
    assert router.calls[1]["origin"] == "debate:critic"
    assert router.calls[1]["prefer_tier"] == "primary"
    assert "Design a cache for the resident model." in call["prompt"]
    assert "patterns, resilience, scalability" in call["prompt"]
    assert "You are" not in call["prompt"]
    assert "SWARM PROTOCOL" not in call["prompt"]
    ServiceContainer.clear()


@pytest.mark.asyncio
async def test_the_synthesis_reaches_the_router_and_never_the_engine(monkeypatch):
    """LIVE 2026-09-19: "Unitary Tick Initiated: 'You are the Master
    Synthesizer…'" — the synthesis prompt went through the cognitive engine
    as an objective; memory retrieval searched for the phrase (5.2s, over
    its budget) and the loop detector fired on the kernel's conversational
    answers, four times in an hour. A synthesis is one request for words."""
    ServiceContainer.clear()
    router = _Router()
    ServiceContainer.register_instance("llm_router", router, required=False)

    class _Engine:
        async def think(self, *args, **kwargs):
            raise AssertionError("the synthesis reached the cognitive engine")

    delegator = AgentDelegator(orchestrator=SimpleNamespace(cognitive_engine=_Engine()))
    result = await delegator.synthesize_consensus(
        "Design a cache for the resident model.", ["Claim: shard one.", "Claim: shard two."]
    )

    assert result == "one perspective"
    assert len(router.calls) == 1
    call = router.calls[0]
    assert call["origin"] == "swarm:synthesis" and call["is_background"] is True
    assert "Design a cache for the resident model." in call["prompt"]
    assert "shard two" in call["prompt"]
    ServiceContainer.clear()


@pytest.mark.asyncio
async def test_without_a_router_the_synthesis_is_the_deterministic_one():
    ServiceContainer.clear()
    delegator = AgentDelegator(orchestrator=SimpleNamespace(cognitive_engine=None))
    result = await delegator.synthesize_consensus("topic", ["Claim: a.", "Claim: b."])
    assert result == delegator._deterministic_consensus("topic", ["Claim: a.", "Claim: b."])
    ServiceContainer.clear()
