from pathlib import Path

import pytest

from core.brain.llm.llm_router import IntelligentLLMRouter, LLMEndpoint, LLMTier
from tools.audit_degradation import analyze_file


class _FailingAsyncStream:
    def __init__(self) -> None:
        self.next_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.next_calls += 1
        raise RuntimeError("provider unavailable")


class _FailingStreamClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_text_stream_async(self, *_args, **_kwargs):
        self.calls += 1
        self.stream = _FailingAsyncStream()
        return self.stream


class _EmptyStreamClient:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_text_stream_async(self, *_args, **_kwargs):
        self.calls += 1
        yield {"type": "metadata", "provider": "empty"}
        yield "   "


class _StreamingClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def generate_text_stream_async(self, *_args, **_kwargs):
        self.calls += 1
        yield self.text


class _NonAgenticClient:
    def __init__(self) -> None:
        self.calls = 0

    async def think(self, *_args, **_kwargs):
        self.calls += 1
        return True, "ungrounded tool-shaped answer", {}


class _CapturingTextClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def generate_text_async(self, prompt: str, **kwargs):
        self.calls.append((prompt, kwargs))
        return "captured"


def test_legacy_llm_router_degradation_audit_is_clean():
    assert analyze_file(Path("core/brain/llm/llm_router.py")) == []


def test_legacy_router_tier_aliases_are_consistent_between_stream_and_think():
    assert IntelligentLLMRouter._resolve_tier("api_fast") is LLMTier.PRIMARY
    assert IntelligentLLMRouter._resolve_tier("local") is LLMTier.PRIMARY
    assert IntelligentLLMRouter._resolve_tier("api_deep") is LLMTier.SECONDARY
    assert IntelligentLLMRouter._resolve_tier("local_fast") is LLMTier.TERTIARY


@pytest.mark.asyncio
async def test_stream_provider_unavailable_fails_over_to_next_endpoint():
    router = IntelligentLLMRouter()
    failing = _FailingStreamClient()
    backup = _StreamingClient("backup stream")
    router.register_endpoint(
        LLMEndpoint(name="Primary-Test", tier=LLMTier.PRIMARY, model_name="primary", client=failing)
    )
    router.register_endpoint(
        LLMEndpoint(name="Secondary-Test", tier=LLMTier.SECONDARY, model_name="secondary", client=backup)
    )

    chunks = [event.content async for event in router.generate_stream("hello", origin="user")]

    assert chunks == ["backup stream"]
    assert failing.calls == 1
    assert failing.stream.next_calls == 1
    assert backup.calls == 1
    assert router.health_monitor.failure_counts["Primary-Test"] == 1


@pytest.mark.asyncio
async def test_empty_stream_is_not_treated_as_success_or_emitted():
    router = IntelligentLLMRouter()
    empty = _EmptyStreamClient()
    backup = _StreamingClient("recovered stream")
    router.register_endpoint(
        LLMEndpoint(name="Primary-Test", tier=LLMTier.PRIMARY, model_name="primary", client=empty)
    )
    router.register_endpoint(
        LLMEndpoint(name="Secondary-Test", tier=LLMTier.SECONDARY, model_name="secondary", client=backup)
    )

    chunks = [event.content async for event in router.generate_stream("hello", origin="user")]

    assert chunks == ["recovered stream"]
    assert empty.calls == 1
    assert backup.calls == 1
    assert router.health_monitor.failure_counts["Primary-Test"] == 1


@pytest.mark.asyncio
async def test_tool_required_route_blocks_plain_llm_fallback_without_agentic_endpoint():
    router = IntelligentLLMRouter()
    non_agentic = _NonAgenticClient()
    router.register_endpoint(
        LLMEndpoint(name="Primary-Test", tier=LLMTier.PRIMARY, model_name="primary", client=non_agentic)
    )

    result = await router.think_and_act(
        "Search the web and save a summary.",
        tools={"web_search": object()},
        origin="user",
    )

    assert result["error"] == "no_agentic_endpoint"
    assert result["content"] == ""
    assert non_agentic.calls == 0


@pytest.mark.asyncio
async def test_structured_system_messages_reach_client_once_and_nothing_is_written_before_them():
    router = IntelligentLLMRouter()
    client = _CapturingTextClient()
    router.register_endpoint(
        LLMEndpoint(
            name="Primary-Test",
            tier=LLMTier.PRIMARY,
            model_name="primary",
            client=client,
        )
    )
    messages = [
        {"role": "system", "content": "SYSTEM_MARKER_ALPHA"},
        {"role": "system", "content": "SYSTEM_MARKER_BETA"},
        {"role": "user", "content": "Summarize the two markers."},
    ]

    result = await router.think(messages=messages, origin="user")

    assert result == "captured"
    assert len(client.calls) == 1
    _, kwargs = client.calls[0]
    assert not kwargs.get("system_prompt")
    serialized = "\n".join(
        str(message.get("content") or "")
        for message in kwargs["messages"]
        if isinstance(message, dict)
    )
    assert serialized.count("SYSTEM_MARKER_ALPHA") == 1
    assert serialized.count("SYSTEM_MARKER_BETA") == 1
    # Who she is lives in her cortex's weights (F06): the router writes no persona in front of what it was given.
    assert serialized.count("You are Aura") == 0
