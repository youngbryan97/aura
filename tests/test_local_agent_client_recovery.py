from __future__ import annotations

import pytest

from core.brain.llm.local_agent_client import LocalAgentClient
from core.container import ServiceContainer
from core.runtime.errors import get_degradation_tracker


class _Client(LocalAgentClient):
    def __init__(self, responses, *, adapter=None):
        super().__init__(model="test-local-agent", adapter=adapter)
        self.responses = list(responses)
        self.prompts = []

    def generate_stream(self, *args, **kwargs):
        return iter(())

    async def generate(self, prompt: str, **kwargs):
        self.prompts.append((prompt, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _FailingAdapter:
    async def execute_tool(self, tool_name, tool_args):
        raise RuntimeError("tool bridge offline")


class _Adapter:
    async def execute_tool(self, tool_name, tool_args):
        return f"{tool_name} ok with {tool_args}"


@pytest.fixture(autouse=True)
def isolated_local_agent_state(monkeypatch):
    ServiceContainer.clear()
    get_degradation_tracker().reset()
    monkeypatch.setattr(
        "core.brain.llm.local_agent_client._emit_agent_event",
        lambda *args, **kwargs: True,
    )
    yield
    ServiceContainer.clear()
    get_degradation_tracker().reset()


@pytest.mark.asyncio
async def test_tool_execution_failure_becomes_react_observation():
    client = _Client(
        [
            '{"tool": "web_search", "args": {"query": "Aura"}}',
            "Final answer after observing the failed tool.",
        ],
        adapter=_FailingAdapter(),
    )

    result = await client.think_and_act("search", "system", max_turns=2, context={})

    assert result["content"] == "Final answer after observing the failed tool."
    assert "Tool web_search failed" in client.prompts[1][0]
    actions = [
        record.action for record in get_degradation_tracker().recent(subsystem="local_agent_client")
    ]
    assert (
        "converted tool execution failure into an observation and continued the ReAct loop"
        in actions
    )


@pytest.mark.asyncio
async def test_local_model_generation_failure_fails_closed():
    client = _Client([RuntimeError("model offline")])

    result = await client.think_and_act("hello", "system", max_turns=1, context={})

    assert result["confidence"] == 0.0
    assert "local model failed" in result["content"]
    last = get_degradation_tracker().recent(subsystem="local_agent_client")[-1]
    assert last.severity == "critical"
    assert (
        last.action == "failed closed before tool execution because local model generation failed"
    )


@pytest.mark.asyncio
async def test_max_tool_turn_returns_last_observation_instead_of_generic_timeout():
    client = _Client(
        ['{"tool": "clock", "args": {"timezone": "UTC"}}'],
        adapter=_Adapter(),
    )

    result = await client.think_and_act("time", "system", max_turns=1, context={})

    assert result["confidence"] == 0.4
    assert "tool-turn limit" in result["content"]
    assert "clock ok" in result["content"]
