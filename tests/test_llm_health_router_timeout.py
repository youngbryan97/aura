import asyncio
import time
from pathlib import Path

import pytest

from core.brain.llm.llm_router import IntelligentLLMRouter, LLMEndpoint, LLMTier
from core.brain.llm_health_router import (
    EndpointHealth,
    HealthAwareLLMRouter,
    _endpoint_call_budgets,
    _endpoint_call_timeout,
    generation_concurrency_limit,
)
from tools.audit_degradation import analyze_file


def test_generation_concurrency_limit_serializes_desktop_safe_boot():
    assert generation_concurrency_limit(
        {
            "AURA_SAFE_BOOT_DESKTOP": "1",
            "AURA_MAX_CONCURRENT_GENERATIONS": "8",
        }
    ) == 1
    assert generation_concurrency_limit(
        {
            "AURA_SAFE_BOOT_DESKTOP": "1",
            "AURA_MAX_CONCURRENT_GENERATIONS": "3",
            "AURA_ALLOW_CONCURRENT_DESKTOP_GENERATIONS": "1",
        }
    ) == 3
    assert generation_concurrency_limit(
        {
            "AURA_SAFE_BOOT_DESKTOP": "0",
            "AURA_MAX_CONCURRENT_GENERATIONS": "invalid",
        }
    ) == 2


def test_llm_health_router_degradation_audit_is_clean():
    assert analyze_file(Path("core/brain/llm_health_router.py")) == []


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

    async def think(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout: float = 0.0,  # noqa: ASYNC109 - fake client verifies timeout forwarding.
    ):
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


class _KwargRecordingGenerateClient:
    def __init__(self):
        self.calls = []

    async def generate_text_async(self, prompt: str, system_prompt: str = "", **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                **kwargs,
            }
        )
        return "ready"


class _ContextRecordingGenerateClient:
    def __init__(self):
        self.calls = []

    async def generate(self, prompt: str, *, context=None, **kwargs):
        self.calls.append({"prompt": prompt, "context": dict(context or {}), **kwargs})
        return "ready"


class _HangingAbortableClient:
    def __init__(self):
        self.abort_reasons = []

    async def think(self, prompt: str, system_prompt: str = "", **kwargs):
        await asyncio.sleep(60.0)
        return "late"

    def force_abort_active_generation(self, *, reason: str):
        self.abort_reasons.append(reason)
        return True


class _DeferredWarmupClient:
    async def warmup(self):
        return False

    def get_lane_status(self):
        return {
            "state": "recovering",
            "last_error": "warmup_deferred",
            "conversation_ready": False,
        }


def _block_event_loop_for(seconds: float) -> None:
    time.sleep(seconds)


class _NonCooperativeBlockingClient:
    def __init__(self):
        self.abort_reasons = []

    async def think(self, prompt: str, system_prompt: str = "", **kwargs):
        _block_event_loop_for(0.08)
        return "late success after blocked event loop"

    def force_abort_active_generation(self, *, reason: str):
        self.abort_reasons.append(reason)
        return True


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
async def test_deep_handoff_restore_does_not_claim_deferred_primary_is_ready(caplog):
    router = HealthAwareLLMRouter()
    router.register(
        name="Cortex",
        url="internal",
        model="test",
        is_local=True,
        tier="local",
        client=_DeferredWarmupClient(),
    )

    with caplog.at_level("INFO"):
        await router._restore_primary_after_deep_handoff()

    assert "restored Cortex after deep handoff" not in caplog.text
    assert "restore remained unavailable" in caplog.text


@pytest.mark.asyncio
async def test_router_outer_watchdog_aborts_hung_endpoint():
    router = HealthAwareLLMRouter()
    client = _HangingAbortableClient()
    router.register(
        name="Cortex",
        url="internal",
        model="test",
        is_local=True,
        tier="local",
        client=client,
    )

    result = await router.generate_with_metadata(
        "probe",
        timeout=0.01,
        prefer_tier="primary",
        origin="proof",
        purpose="proof_model_lane_probe",
        foreground_request=True,
        health_probe=True,
        skip_runtime_payload=True,
        allow_cloud_fallback=False,
    )

    assert result["ok"] is False
    assert result["endpoint"] == "all_failed"
    assert result["error"].startswith("endpoint_timeout:Cortex:")
    assert client.abort_reasons
    assert client.abort_reasons[0].startswith("endpoint_timeout:Cortex:")


def test_foreground_local_endpoint_budget_stays_inside_desktop_request_envelope():
    cooperative, wall = _endpoint_call_budgets(
        180.0,
        foreground_local=True,
        prompt_chars=6300,
        max_tokens=512,
    )

    assert wall == 105.0
    assert cooperative == 103.0


def test_proof_and_health_endpoint_budgets_keep_explicit_timeout_contract():
    proof_cooperative, proof_wall = _endpoint_call_budgets(
        180.0,
        foreground_local=True,
        prompt_chars=6300,
        max_tokens=512,
        proof_evaluation_contract=True,
    )
    health_cooperative, health_wall = _endpoint_call_budgets(
        180.0,
        foreground_local=True,
        prompt_chars=6300,
        max_tokens=512,
        health_probe=True,
    )

    assert proof_cooperative == 180.0
    assert proof_wall == _endpoint_call_timeout(180.0)
    assert health_cooperative == 180.0
    assert health_wall == _endpoint_call_timeout(180.0)


def test_long_form_foreground_endpoint_keeps_owning_desktop_deadline():
    cooperative, wall = _endpoint_call_budgets(
        356.0,
        foreground_local=True,
        prompt_chars=7600,
        max_tokens=2560,
    )

    assert cooperative == 356.0
    assert wall == _endpoint_call_timeout(356.0)


@pytest.mark.asyncio
async def test_foreground_local_router_forwards_bounded_cooperative_timeout():
    router = HealthAwareLLMRouter()
    client = _TimeoutRecordingClient()
    router.register(
        name="Cortex",
        url="internal",
        model="test",
        is_local=True,
        tier="local",
        client=client,
    )

    result = await router.generate_with_metadata(
        "continue the ordinary live desktop conversation",
        timeout=180.0,
        prefer_tier="primary",
        origin="user",
        purpose="chat",
        foreground_request=True,
        max_tokens=512,
        skip_runtime_payload=True,
        allow_cloud_fallback=False,
    )

    assert result["ok"] is True
    assert client.calls[0]["timeout"] == 103.0


@pytest.mark.asyncio
async def test_router_preserves_user_surface_completion_floor_in_generate_context():
    router = HealthAwareLLMRouter()
    client = _ContextRecordingGenerateClient()
    router.register(
        name="Cortex",
        url="internal",
        model="test",
        is_local=True,
        tier="local",
        client=client,
    )

    result = await router.generate_with_metadata(
        "answer this foreground turn completely",
        timeout=30.0,
        prefer_tier="primary",
        origin="user",
        purpose="chat",
        foreground_request=True,
        max_tokens=512,
        user_surface_completion_floor=512,
        clean_user_surface_contract=True,
        skip_runtime_payload=True,
        allow_cloud_fallback=False,
    )

    assert result["ok"] is True
    assert client.calls[0]["context"]["user_surface_completion_floor"] == 512


@pytest.mark.asyncio
async def test_router_wall_clock_watchdog_rejects_noncooperative_late_success(monkeypatch):
    router = HealthAwareLLMRouter()
    client = _NonCooperativeBlockingClient()
    router.register(
        name="Cortex",
        url="internal",
        model="test",
        is_local=True,
        tier="local",
        client=client,
    )
    monkeypatch.setattr("core.brain.llm_health_router._endpoint_call_timeout", lambda _timeout: 0.02)

    result = await router.generate_with_metadata(
        "probe",
        timeout=0.01,
        prefer_tier="primary",
        origin="proof",
        purpose="proof_model_lane_probe",
        foreground_request=True,
        health_probe=True,
        skip_runtime_payload=True,
        allow_cloud_fallback=False,
    )

    assert result["ok"] is False
    assert result["endpoint"] == "all_failed"
    assert result["error"].startswith("endpoint_timeout:Cortex:")
    assert client.abort_reasons


@pytest.mark.asyncio
async def test_public_think_serializes_full_messages_for_clients_without_kwargs():
    router = HealthAwareLLMRouter()
    client = _PromptOnlyClient()
    router.register(
        name="Cortex",
        url="internal",
        model="local-test",
        is_local=True,
        tier="local",
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
        prefer_tier="primary",
    )

    assert result == "ready"
    assert "Earlier context about Bryan and music." in client.calls[0]["prompt"]
    assert "I remember that thread." in client.calls[0]["prompt"]
    assert "Search the web for the song" in client.calls[0]["prompt"]
    assert client.calls[0]["system_prompt"].startswith("Speak as Aura.")
    assert "COGNITION & REASONING" in client.calls[0]["system_prompt"]


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
        name="Cortex",
        url="internal",
        model="local-test",
        is_local=True,
        tier="local",
        client=_EmptyClient(),
    )

    result = await router.think(
        prompt="With me?",
        origin="user",
        prefer_tier="primary",
    )

    assert result == "I lost the reply lane for a moment. Ask that again and I'll answer cleanly."


@pytest.mark.asyncio
async def test_quality_rejection_does_not_trip_healthy_local_endpoint():
    class _QualityRejectedClient:
        async def think(self, *_args, **_kwargs):
            return None

        @staticmethod
        def get_last_generation_metadata():
            return {
                "error": "surface_quality_rejected",
                "failure_reasons": ["missing_requested_word_count"],
                "surface_control_receipt": {
                    "surface_quality_gate_enabled": True,
                    "surface_quality_gate_passed": False,
                    "surface_quality_gate_attempts": 3,
                },
            }

    router = HealthAwareLLMRouter()
    endpoint = EndpointHealth(
        name="Cortex",
        url="internal",
        model="local-test",
        is_local=True,
        tier="local",
        client=_QualityRejectedClient(),
    )

    result = await router._call_endpoint(
        endpoint,
        "In exactly five words, state why checksums matter.",
        "Speak as Aura.",
        timeout=30.0,
    )

    assert result["ok"] is False
    assert result["error"] == "surface_quality_rejected"
    assert result["failure_reasons"] == ["missing_requested_word_count"]
    assert endpoint.failure_count == 0
    assert endpoint.state.value == "closed"


@pytest.mark.asyncio
async def test_direct_surface_receipt_prevents_false_endpoint_failure():
    class _ReceiptOnlyQualityRejectedClient:
        async def think(self, *_args, **_kwargs):
            return None

        @staticmethod
        def get_last_generation_metadata():
            return {}

        @staticmethod
        def get_last_surface_control_receipt():
            return {
                "surface_quality_gate_enabled": True,
                "surface_quality_gate_passed": False,
                "surface_quality_gate_reasons": ["corrupted_language"],
            }

    router = HealthAwareLLMRouter()
    endpoint = EndpointHealth(
        name="Cortex",
        url="internal",
        model="local-test",
        is_local=True,
        tier="local",
        client=_ReceiptOnlyQualityRejectedClient(),
    )

    result = await router._call_endpoint(
        endpoint,
        "Explain the result completely.",
        "Speak as Aura.",
        timeout=30.0,
    )

    assert result["ok"] is False
    assert result["error"] == "surface_quality_rejected"
    assert endpoint.failure_count == 0
    assert endpoint.state.value == "closed"


@pytest.mark.asyncio
async def test_router_does_not_recover_to_a_removed_remote_provider():
    router = HealthAwareLLMRouter()
    router.register(
        name="Cortex",
        url="internal",
        model="local-test",
        is_local=True,
        tier="local",
        client=_EmptyClient(),
    )

    result = await router.think(
        prompt="With me?",
        origin="user",
    )

    assert result == "I lost the reply lane for a moment. Ask that again and I'll answer cleanly."


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


@pytest.mark.asyncio
async def test_router_stamps_inferred_background_for_local_runtime_client(monkeypatch):
    router = HealthAwareLLMRouter()
    client = _KwargRecordingGenerateClient()
    router.register(
        name="Brainstem",
        url="internal",
        model="local-fast-test",
        is_local=True,
        tier="local_fast",
        client=client,
    )

    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "0")
    result = await router.think(
        prompt="Compress this internal affect appraisal.",
        origin="affect_engine",
        prefer_tier="tertiary",
    )

    assert result == "ready"
    assert client.calls
    assert client.calls[0]["is_background"] is True


@pytest.mark.asyncio
async def test_router_stamps_inferred_foreground_for_user_local_runtime_client(monkeypatch):
    router = HealthAwareLLMRouter()
    client = _KwargRecordingGenerateClient()
    router.register(
        name="Cortex",
        url="internal",
        model="local-primary-test",
        is_local=True,
        tier="local",
        client=client,
    )

    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "0")
    result = await router.think(
        prompt="Answer the user directly.",
        origin="user",
        prefer_tier="primary",
    )

    assert result == "ready"
    assert client.calls
    assert client.calls[0]["is_background"] is False
    assert client.calls[0]["foreground_request"] is True


@pytest.mark.asyncio
async def test_router_defers_background_local_runtime_during_foreground_quiet_window():
    from core.brain.llm import deferral_record

    deferral_record.reset_for_test()
    router = HealthAwareLLMRouter()
    client = _KwargRecordingGenerateClient()
    router.register(
        name="Brainstem",
        url="internal",
        model="local-fast-test",
        is_local=True,
        tier="local_fast",
        client=client,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(HealthAwareLLMRouter, "_foreground_quiet_window_active", classmethod(lambda cls: True))
        result = await router.think(
            prompt="Compress this internal affect appraisal.",
            origin="affect_engine",
            prefer_tier="tertiary",
        )

    assert result is None
    assert client.calls == []
    recorded = deferral_record.take_deferral(
        origin="affect_engine",
        not_before=0.0,
    )
    assert recorded is not None
    assert recorded.reason == "foreground_quiet_window"
    deferral_record.reset_for_test()


def test_desktop_background_headroom_defers_brainstem_before_memory_spike(monkeypatch):
    from types import SimpleNamespace

    import core.brain.llm_health_router as router_module

    monkeypatch.setattr(
        router_module,
        "desktop_resource_guard_enabled",
        lambda env=None: True,
    )
    # Genuine pre-spike headroom: high pressure AND low available. The old values
    # here (58% / 26.5GB) were actually the desktop STEADY STATE with the 32B
    # resident — deferring there meant background cognition could never run, so
    # mind_tick never completed a successful tick → false-death → the launcher
    # respawned a duplicate 32B (self-sustaining respawn loop, 2026-07). The gate
    # is now calibrated to admit at steady state (see
    # tests/test_brainstem_background_headroom_calibration.py) while STILL
    # deferring as real pressure builds toward a spike, which this asserts.
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            pressure_pct=68.0,
            available_gb=20.0,
            process_rss_gb=20.0,
            process_rss_limit_gb=40.0,
        ),
    )
    ep = EndpointHealth(
        name="Brainstem",
        url="internal",
        model="Qwen2.5-7B-Instruct-4bit",
        is_local=True,
        tier="local_fast",
    )

    # Pin the KERNEL's own reading. Without this the test asks the host, and
    # the answer changes what the assertion means.
    #
    # A kernel-pressure escape was added 2026-08-17: when the OS reports no
    # pressure, the derived percentage does not get to veto the small models,
    # because psutil's macOS accounting counts reclaimable cache as consumed
    # and was deferring every fallback on a machine with room. That is correct,
    # and it silently made this test depend on ambient state — on a healthy
    # host the kernel says "normal", the escape fires, and the deferral this
    # asserts never happens.
    #
    # Both branches are pinned below, because the policy is the pair: the
    # derived reading defers only when the kernel agrees there is pressure.
    monkeypatch.setattr(
        "core.utils.memory_monitor.kernel_memory_pressure_level", lambda: "warn"
    )

    reason = HealthAwareLLMRouter._desktop_background_endpoint_deferral_reason(ep)

    assert reason is not None
    assert reason.startswith("desktop_background_headroom:Brainstem:")

    # And the escape itself: same numbers, kernel reporting no pressure, so the
    # derived percentage must NOT keep the small models out.
    monkeypatch.setattr(
        "core.utils.memory_monitor.kernel_memory_pressure_level", lambda: "normal"
    )

    assert HealthAwareLLMRouter._desktop_background_endpoint_deferral_reason(ep) is None


def test_desktop_background_headroom_allows_reflex_with_moderate_headroom(monkeypatch):
    from types import SimpleNamespace

    import core.brain.llm_health_router as router_module

    monkeypatch.setattr(router_module, "desktop_resource_guard_enabled", lambda env=None: True)
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            pressure_pct=58.0,
            available_gb=26.5,
            process_rss_gb=20.0,
            process_rss_limit_gb=40.0,
        ),
    )
    ep = EndpointHealth(
        name="Reflex",
        url="internal",
        model="Qwen2.5-1.5B-Instruct-4bit",
        is_local=True,
        tier="emergency",
    )

    assert HealthAwareLLMRouter._desktop_background_endpoint_deferral_reason(ep) is None


def test_shared_background_admission_reports_every_closed_lane(monkeypatch):
    from types import SimpleNamespace

    import core.brain.llm_health_router as router_module

    monkeypatch.setattr(router_module, "desktop_resource_guard_enabled", lambda env=None: True)
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            pressure_pct=77.0,
            available_gb=14.0,
            process_rss_gb=20.0,
            process_rss_limit_gb=42.0,
        ),
    )
    monkeypatch.setattr(
        "core.utils.memory_monitor.kernel_memory_pressure_level", lambda: "warn"
    )

    reasons = router_module.desktop_background_endpoint_deferral_reasons(
        ("Brainstem", "Reflex")
    )

    assert set(reasons) == {"Brainstem", "Reflex"}
    assert reasons["Brainstem"].startswith("desktop_background_headroom:Brainstem:")
    assert reasons["Reflex"].startswith("desktop_background_headroom:Reflex:")


@pytest.mark.asyncio
async def test_router_deduplicates_repeated_background_deferral_logs(caplog):
    router = HealthAwareLLMRouter()

    with caplog.at_level("INFO", logger="Brain.HealthRouter"):
        router._log_background_deferral(
            scope="local_endpoint",
            origin="affect_engine",
            reason="foreground_quiet_window",
            endpoint="Brainstem",
        )
        router._log_background_deferral(
            scope="local_endpoint",
            origin="affect_engine",
            reason="foreground_quiet_window",
            endpoint="Brainstem",
        )

    matching = [
        record
        for record in caplog.records
        if "Deferring background local endpoint Brainstem" in record.getMessage()
    ]
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_router_defers_background_local_runtime_during_safe_boot_guard(monkeypatch):
    router = HealthAwareLLMRouter()
    router._created_at = time.monotonic()
    client = _KwargRecordingGenerateClient()
    router.register(
        name="Brainstem",
        url="internal",
        model="local-fast-test",
        is_local=True,
        tier="local_fast",
        client=client,
    )

    monkeypatch.setattr(
        "core.brain.llm_health_router.desktop_resource_guard_enabled",
        lambda: True,
    )
    monkeypatch.setenv("AURA_SAFE_BOOT_BACKGROUND_GUARD_SECS", "180")

    result = await router.think(
        prompt="Compress this internal affect appraisal.",
        origin="affect_engine",
        prefer_tier="tertiary",
    )

    assert result is None
    assert client.calls == []


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


def test_generation_gate_refuses_to_stack_beyond_bound():
    """Round-9 finding: nine concurrent generations stacked for one turn,
    allocating ~2GB/s until macOS executed the process at 78GB
    phys_footprint. The gate makes stacking impossible: saturated
    callers get a truthful failure, never a queue-up."""
    import core.brain.llm_health_router as router_mod

    gate = router_mod._GENERATION_GATE
    held = []
    try:
        # Drain every slot.
        while gate.acquire(blocking=False):
            held.append(True)
        assert held, "gate must have at least one slot"

        # A saturated acquire with a tiny wait must fail, not block long.
        import time as _time

        start = _time.monotonic()
        got = gate.acquire(True, 0.2)
        elapsed = _time.monotonic() - start
        assert got is False
        assert elapsed < 5.0
        assert router_mod._GATE_SATURATION_RESULT["ok"] is False
        assert "saturated" in router_mod._GATE_SATURATION_RESULT["endpoint"]
    finally:
        for _ in held:
            gate.release()
