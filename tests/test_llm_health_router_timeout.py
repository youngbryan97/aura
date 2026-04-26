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


class _EmptyClient:
    async def think(self, prompt: str, system_prompt: str = "", **kwargs):
        return ""


class _DualGenerateClient:
    def __init__(self):
        self.calls = []

    async def generate(self, prompt: str, system_prompt: str = "", **kwargs):
        self.calls.append(
            {
                "method": "generate",
                "prompt": prompt,
                "system_prompt": system_prompt,
                "messages": kwargs.get("messages"),
            }
        )
        return "wrong-path"

    async def generate_text_async(self, prompt: str, system_prompt: str = "", **kwargs):
        self.calls.append(
            {
                "method": "generate_text_async",
                "prompt": prompt,
                "system_prompt": system_prompt,
                "messages": kwargs.get("messages"),
            }
        )
        return "right-path"


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


@pytest.mark.asyncio
async def test_call_endpoint_prefers_generate_text_async_over_generate_for_dual_clients():
    router = HealthAwareLLMRouter()
    client = _DualGenerateClient()
    endpoint = EndpointHealth(
        name="Cortex",
        url="internal",
        model="local-test",
        is_local=True,
        tier="local",
        client=client,
    )

    result = await router._call_endpoint(
        endpoint,
        "Reply cleanly.",
        "Speak as Aura.",
        timeout=30.0,
        messages=[
            {"role": "system", "content": "Speak as Aura."},
            {"role": "user", "content": "Reply cleanly."},
        ],
    )

    assert result["ok"] is True
    assert result["text"] == "right-path"
    assert [call["method"] for call in client.calls] == ["generate_text_async"]
    assert client.calls[0]["messages"][-1]["content"] == "Reply cleanly."


@pytest.mark.asyncio
async def test_router_think_failsofts_when_client_returns_no_text():
    router = HealthAwareLLMRouter()
    router.register(
        name="Gemini-Fast",
        url="cloud",
        model="gemini-test",
        is_local=False,
        tier="api_fast",
        client=_EmptyClient(),
    )

    result = await router.think(
        prompt="With me?",
        origin="user",
        prefer_tier="api_fast",
    )

    assert result == "I lost the reply lane for a moment. Ask that again and I'll answer cleanly."


@pytest.mark.asyncio
async def test_router_recovers_to_cloud_when_foreground_local_lane_returns_no_text():
    router = HealthAwareLLMRouter()
    cloud = _TimeoutRecordingClient()
    router.register(
        name="Cortex",
        url="internal",
        model="local-test",
        is_local=True,
        tier="local",
        client=_EmptyClient(),
    )
    router.register(
        name="Gemini-Fast",
        url="cloud",
        model="gemini-test",
        is_local=False,
        tier="api_fast",
        client=cloud,
    )

    result = await router.think(
        prompt="With me?",
        origin="user",
    )

    assert result == "ready"
    assert cloud.calls


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


class _FailedInferenceGateClient:
    def __init__(self):
        self.calls = 0

    def get_conversation_status(self):
        return {
            "state": "failed",
            "last_failure_reason": "mlx_runtime_unavailable:metal_device_enumeration_crash",
            "conversation_ready": False,
        }

    async def think(self, prompt: str, system_prompt: str = "", **kwargs):
        self.calls += 1
        return None


@pytest.mark.asyncio
async def test_router_reads_failed_inference_gate_lane_without_calling_client():
    router = HealthAwareLLMRouter()
    client = _FailedInferenceGateClient()
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
    assert result["error"] == "mlx_runtime_unavailable:metal_device_enumeration_crash"
    assert client.calls == 0


class _HeartbeatStalledClient:
    def __init__(self):
        self.calls = 0

    def get_conversation_status(self):
        return {
            "state": "failed",
            "last_failure_reason": "heartbeat_stalled_during_generation",
            "conversation_ready": False,
        }

    async def think(self, prompt: str, system_prompt: str = "", **kwargs):
        self.calls += 1
        return None


@pytest.mark.asyncio
async def test_router_treats_heartbeat_stall_as_transient_cooldown(monkeypatch):
    router = HealthAwareLLMRouter()
    client = _HeartbeatStalledClient()
    endpoint = EndpointHealth(
        name="Brainstem",
        url="internal",
        model="test",
        is_local=True,
        tier="local_fast",
        client=client,
    )

    result = await router._call_endpoint(
        endpoint,
        "With me?",
        "Be helpful",
        timeout=10.0,
    )

    assert result["ok"] is False
    assert result["error"] == "heartbeat_stalled_during_generation"
    assert client.calls == 0
    assert endpoint.failure_count == 0
    assert endpoint.state.value == "open"


def test_background_quiet_error_treats_local_runtime_unavailable_as_non_user_noise():
    from core.brain.llm_health_router import _background_error_is_quiet

    assert _background_error_is_quiet("local_runtime_unavailable:server_unreachable") is True
