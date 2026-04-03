import pytest

from core.brain.llm.llm_router import IntelligentLLMRouter, LLMEndpoint, LLMTier
from core.brain.llm_health_router import EndpointHealth, HealthAwareLLMRouter


class _TimeoutRecordingClient:
    def __init__(self):
        self.calls = []

    async def think(self, prompt: str, system_prompt: str = "", **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "timeout": kwargs.get("timeout"),
            }
        )
        return "ready"


class _PromptOnlyClient:
    def __init__(self):
        self.calls = []

    async def think(self, prompt: str, system_prompt: str = "", timeout: float = 0.0):
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "timeout": timeout,
            }
        )
        return "ready"


@pytest.mark.asyncio
async def test_direct_client_think_receives_timeout_budget():
    router = HealthAwareLLMRouter()
    client = _TimeoutRecordingClient()
    endpoint = EndpointHealth(
        name="Cortex",
        url="internal",
        model="test",
        is_local=True,
        tier="local",
        client=client,
    )

    result = await router._call_endpoint(
        endpoint,
        "With me?",
        "Be helpful",
        timeout=67.0,
    )

    assert result["ok"] is True
    assert result["text"] == "ready"
    assert client.calls[0]["timeout"] == 67.0


@pytest.mark.asyncio
async def test_public_think_serializes_full_messages_for_clients_without_kwargs():
    router = HealthAwareLLMRouter()
    client = _PromptOnlyClient()
    router.register(
        name="Gemini-Fast",
        url="cloud",
        model="gemini-test",
        is_local=False,
        tier="api_fast",
        client=client,
    )

    result = await router.think(
        messages=[
            {"role": "system", "content": "Speak as Aura."},
            {"role": "user", "content": "Earlier context about Bryan and music."},
            {"role": "assistant", "content": "I remember that thread."},
            {"role": "user", "content": "Search the web for the song and tell me what it means."},
        ],
        origin="user",
        prefer_tier="api_fast",
    )

    assert result == "ready"
    assert "Earlier context about Bryan and music." in client.calls[0]["prompt"]
    assert "I remember that thread." in client.calls[0]["prompt"]
    assert "Search the web for the song" in client.calls[0]["prompt"]
    assert client.calls[0]["system_prompt"] == "Speak as Aura."


def test_missing_origin_defaults_to_background_when_purpose_is_not_user_facing():
    assert HealthAwareLLMRouter._is_background_request(
        origin=None,
        purpose=None,
        explicit_background=False,
    ) is True
    assert HealthAwareLLMRouter._is_background_request(
        origin=None,
        purpose="expression",
        explicit_background=False,
    ) is False


def test_unknown_internal_origin_defaults_to_background():
    assert HealthAwareLLMRouter._is_background_request(
        origin="kernel",
        purpose=None,
        explicit_background=False,
    ) is True
    assert HealthAwareLLMRouter._is_background_request(
        origin="inner_monologue",
        purpose=None,
        explicit_background=False,
    ) is True
    assert HealthAwareLLMRouter._is_background_request(
        origin="api",
        purpose=None,
        explicit_background=False,
    ) is False


class _LegacyRecordingClient:
    def __init__(self):
        self.calls = []

    async def think(self, prompt: str, system_prompt: str = "", **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "messages": kwargs.get("messages"),
            }
        )
        return True, "ready", {}


@pytest.mark.asyncio
async def test_legacy_router_preserves_full_messages_for_adapter_calls():
    router = IntelligentLLMRouter()
    client = _LegacyRecordingClient()
    router.register_endpoint(
        LLMEndpoint(
            name="Local-Test",
            tier=LLMTier.PRIMARY,
            model_name="test",
            client=client,
        )
    )

    result = await router.think(
        messages=[
            {"role": "system", "content": "Speak as Aura."},
            {"role": "user", "content": "Bryan mentioned the song already."},
            {"role": "assistant", "content": "I remember that thread."},
            {"role": "user", "content": "Search the web and tell me the author."},
        ],
        origin="user",
        prefer_endpoint="Local-Test",
    )

    assert result == "ready"
    assert client.calls[0]["messages"] is not None
    assert client.calls[0]["messages"][-1]["content"] == "Search the web and tell me the author."


class _NeverCalledClient:
    def __init__(self):
        self.calls = 0

    async def think(self, prompt: str, system_prompt: str = "", **kwargs):
        self.calls += 1
        return "should-not-run"


@pytest.mark.asyncio
async def test_background_requests_are_suppressed_during_router_high_pressure_mode():
    router = HealthAwareLLMRouter()
    client = _NeverCalledClient()
    router.register(
        name="Brainstem",
        url="internal",
        model="brainstem-test",
        is_local=True,
        tier="local_fast",
        client=client,
    )
    router.high_pressure_mode = True

    result = await router.generate_with_metadata(
        "background reflection",
        origin="system",
    )

    assert result["ok"] is False
    assert result["error"] == "background_deferred:memory_pressure"
    assert client.calls == 0


class _FailedLocalClient:
    def __init__(self):
        self.calls = 0

    def get_lane_status(self):
        return {
            "state": "failed",
            "last_error": "local_runtime_unavailable:server_unreachable",
        }

    async def think(self, prompt: str, system_prompt: str = "", **kwargs):
        self.calls += 1
        return "should-not-run"


@pytest.mark.asyncio
async def test_router_surfaces_hard_local_lane_failure_without_calling_client():
    router = HealthAwareLLMRouter()
    client = _FailedLocalClient()
    endpoint = EndpointHealth(
        name="Cortex",
        url="internal",
        model="test",
        is_local=True,
        tier="local",
        client=client,
    )

    result = await router._call_endpoint(
        endpoint,
        "With me?",
        "Be helpful",
        timeout=10.0,
    )

    assert result["ok"] is False
    assert result["error"] == "local_runtime_unavailable:server_unreachable"
    assert client.calls == 0


def test_background_quiet_error_treats_local_runtime_unavailable_as_non_user_noise():
    from core.brain.llm_health_router import _background_error_is_quiet

    assert _background_error_is_quiet("local_runtime_unavailable:server_unreachable") is True
