import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.brain.inference_gate import InferenceGate


class _FakeClient:
    def __init__(self, text: str):
        self.text = text
        self.generate_text_async = AsyncMock(return_value=(True, text, {}))


class _RecordingClient:
    def __init__(self, text: str):
        self.text = text
        self.deadlines = []
        self.prompts = []
        self.kwargs = []

    async def generate_text_async(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        self.deadlines.append(kwargs.get("deadline"))
        return self.text


class _NoTextClient:
    def __init__(self):
        self.generate_text_async = AsyncMock(return_value=(False, "", {}))


class _LaneWarmupClient:
    def __init__(self):
        self.warmup = AsyncMock(side_effect=self._finish_warmup)
        self.state = "cold"

    async def _finish_warmup(self):
        self.state = "ready"

    def get_lane_status(self):
        return {
            "state": self.state,
            "last_error": "",
            "conversation_ready": self.state == "ready",
            "warmup_attempted": self.state != "cold",
            "warmup_in_flight": False,
            "last_transition_at": 1.0,
        }

    def note_lane_recovering(self, reason):
        self.state = "recovering"

    def note_lane_failed(self, reason):
        self.state = "failed"


@pytest.mark.asyncio
async def test_background_requests_stay_off_cortex():
    gate = InferenceGate()
    cortex = _FakeClient("cortex")
    brainstem = _FakeClient("brainstem")
    cpu = _FakeClient("cpu")
    gate._mlx_client = cortex
    gate._ensure_cortex_recovery = AsyncMock()

    clients = {
        "/models/brainstem": brainstem,
        "/models/fallback": cpu,
    }

    def _fake_get_mlx_client(model_path=None, **kwargs):
        return clients[model_path]

    with patch.object(InferenceGate, "_background_local_deferral_reason", return_value=None):
        with patch("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with patch("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                with patch("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                    result = await gate.generate(
                        "background reflection",
                        context={"prefer_tier": "primary", "origin": "system"},
                    )

    assert result == "brainstem"
    cortex.generate_text_async.assert_not_called()
    brainstem.generate_text_async.assert_awaited()
    gate._ensure_cortex_recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_requests_wait_while_cortex_quiet_window_is_active():
    gate = InferenceGate()
    gate._mlx_client = _LaneWarmupClient()
    gate._ensure_cortex_recovery = AsyncMock()

    with patch.object(InferenceGate, "_foreground_quiet_window_active", return_value=True):
        with patch.object(
            InferenceGate,
            "get_conversation_status",
            return_value={
                "conversation_ready": False,
                "state": "warming",
                "warmup_in_flight": True,
            },
        ):
            result = await gate.generate(
                "background reflection",
                context={"prefer_tier": "primary", "origin": "system"},
            )

    assert result is None
    gate._ensure_cortex_recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_requests_wait_when_cortex_has_failed():
    gate = InferenceGate()
    failed_lane = _LaneWarmupClient()
    failed_lane.state = "failed"
    gate._mlx_client = failed_lane
    gate._ensure_cortex_recovery = AsyncMock()

    result = await gate.generate(
        "background reflection",
        context={"prefer_tier": "primary", "origin": "system"},
    )

    assert result is None
    gate._ensure_cortex_recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_deep_handoff_uses_solver_then_returns_response():
    gate = InferenceGate()
    cortex = _FakeClient("cortex")
    solver = _FakeClient("solver")
    gate._mlx_client = cortex
    gate._restore_primary_after_deep_handoff = AsyncMock()

    def _fake_get_mlx_client(model_path=None, **kwargs):
        if model_path == "/models/deep":
            return solver
        if model_path == "/models/active":
            return cortex
        raise AssertionError(f"Unexpected model path: {model_path}")

    scheduled = []

    def _capture_task(coro):
        scheduled.append(coro)
        return MagicMock(name="task")

    with patch("asyncio.create_task", side_effect=_capture_task):
        with patch("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with patch("core.brain.llm.model_registry.get_deep_model_path", return_value="/models/deep"):
                with patch("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                    with patch("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                        result = await gate.generate(
                            "perform a flagship architecture deep dive",
                            context={"prefer_tier": "secondary", "deep_handoff": True},
                        )

    for coro in scheduled:
        await coro

    assert result == "solver"
    solver.generate_text_async.assert_awaited()
    cortex.generate_text_async.assert_not_called()
    gate._restore_primary_after_deep_handoff.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_facing_primary_uses_conversational_budget_and_chatml():
    gate = InferenceGate()
    cortex = _RecordingClient("hello")
    gate._mlx_client = cortex

    with patch("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with patch("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with patch("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "Say hi.",
                    context={"origin": "user", "prefer_tier": "primary", "history": []},
                )

    assert result == "hello"
    assert cortex.deadlines
    expected_total = InferenceGate._default_timeout_for_request(
        "user",
        "primary",
        deep_handoff=False,
        is_background=False,
    )
    expected_primary, _ = InferenceGate._split_attempt_timeouts(expected_total, "primary")
    assert cortex.deadlines[0]._timeout == expected_primary
    expected_tokens = InferenceGate._default_max_tokens_for_request(
        "user",
        "primary",
        deep_handoff=False,
        is_background=False,
    )
    assert cortex.kwargs[0]["max_tokens"] == expected_tokens
    assert cortex.prompts[0].startswith("<|im_start|>")
    assert "<|im_start|>assistant" in cortex.prompts[0]
    assert "<|SYSTEM|>" not in cortex.prompts[0]


@pytest.mark.asyncio
async def test_user_facing_primary_uses_compact_foreground_context_builders():
    gate = InferenceGate()
    cortex = _RecordingClient("hello")
    gate._mlx_client = cortex
    gate._build_compact_system_prompt = MagicMock(return_value="compact-system")
    gate._build_compact_living_mind_context = AsyncMock(return_value="compact-live")
    gate._build_system_prompt = MagicMock(side_effect=AssertionError("full system prompt should not be used"))
    gate._build_living_mind_context = AsyncMock(side_effect=AssertionError("full living context should not be used"))

    with patch("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with patch("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with patch("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "With me?",
                    context={"origin": "api", "prefer_tier": "primary", "history": []},
                )

    assert result == "hello"
    gate._build_compact_system_prompt.assert_called_once()
    gate._build_compact_living_mind_context.assert_awaited_once()
    assert "compact-system" in cortex.prompts[0]
    assert "compact-live" in cortex.prompts[0]


@pytest.mark.asyncio
async def test_user_facing_primary_preserves_prebuilt_messages_for_local_mlx():
    gate = InferenceGate()
    cortex = _RecordingClient("hello")
    gate._mlx_client = cortex
    gate._build_compact_system_prompt = MagicMock(side_effect=AssertionError("prebuilt messages should bypass prompt rebuild"))
    gate._build_compact_living_mind_context = AsyncMock(side_effect=AssertionError("prebuilt messages should bypass prompt rebuild"))
    gate._build_messages = MagicMock(side_effect=AssertionError("prebuilt messages should bypass history assembly"))
    gate._build_compact_messages = MagicMock(side_effect=AssertionError("prebuilt messages should bypass history assembly"))

    messages = [
        {"role": "system", "content": "You are Aura."},
        {"role": "user", "content": "Say exactly: 32B lane online."},
    ]

    with patch("core.brain.llm.mlx_client.get_mlx_client", return_value=_FakeClient("fallback")):
        with patch("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with patch("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "Say exactly: 32B lane online.",
                    context={"origin": "api", "prefer_tier": "primary", "messages": messages},
                )

    assert result == "hello"
    assert "32B lane online" in cortex.prompts[0]
    assert "Aura" in cortex.prompts[0]
    assert "conversation history" not in cortex.prompts[0].lower()


@pytest.mark.asyncio
async def test_background_primary_downgrades_timeout_and_tier():
    gate = InferenceGate()
    cortex = _RecordingClient("cortex")
    brainstem = _RecordingClient("brainstem")
    cpu = _RecordingClient("cpu")
    gate._mlx_client = cortex

    clients = {
        "/models/brainstem": brainstem,
        "/models/fallback": cpu,
    }

    def _fake_get_mlx_client(model_path=None, **kwargs):
        return clients[model_path]

    with patch.object(InferenceGate, "_background_local_deferral_reason", return_value=None):
        with patch("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
            with patch("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
                with patch("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                    result = await gate.generate(
                        "background reflection",
                        context={"origin": "system", "prefer_tier": "primary"},
                    )

    assert result == "brainstem"
    assert not cortex.deadlines
    assert brainstem.deadlines
    expected_total = InferenceGate._default_timeout_for_request(
        "system",
        "tertiary",
        deep_handoff=False,
        is_background=True,
    )
    expected_primary, _ = InferenceGate._split_attempt_timeouts(expected_total, "tertiary")
    assert brainstem.deadlines[0]._timeout == expected_primary
    expected_tokens = InferenceGate._default_max_tokens_for_request(
        "system",
        "tertiary",
        deep_handoff=False,
        is_background=True,
    )
    assert brainstem.kwargs[0]["max_tokens"] == expected_tokens


def test_routing_user_origin_is_treated_as_human_input():
    assert InferenceGate._origin_is_user_facing("user") is True
    assert InferenceGate._origin_is_user_facing("voice_command") is True
    assert InferenceGate._origin_is_user_facing("routing_user") is True
    assert InferenceGate._origin_is_user_facing("routing_voice_command") is True


def test_user_facing_primary_budget_allows_32b_cold_start():
    total = InferenceGate._default_timeout_for_request(
        "user",
        "primary",
        deep_handoff=False,
        is_background=False,
    )
    primary, fallback = InferenceGate._split_attempt_timeouts(total, "primary")
    assert total == 75.0
    assert primary >= 60.0
    assert fallback >= 5.0


@pytest.mark.asyncio
async def test_user_facing_primary_falls_back_to_brainstem_when_cortex_fails_without_cloud():
    gate = InferenceGate()
    class _FailedNoTextClient(_NoTextClient):
        def get_lane_status(self):
            return {
                "state": "failed",
                "last_error": "worker_failed",
                "conversation_ready": False,
                "warmup_attempted": True,
                "warmup_in_flight": False,
                "last_transition_at": 1.0,
            }

    cortex = _FailedNoTextClient()
    brainstem = _RecordingClient("brainstem")
    cpu = _RecordingClient("cpu")
    gate._mlx_client = cortex

    clients = {
        "/models/brainstem": brainstem,
        "/models/fallback": cpu,
    }

    def _fake_get_mlx_client(model_path=None, **kwargs):
        return clients[model_path]

    with patch("core.brain.llm.mlx_client.get_mlx_client", side_effect=_fake_get_mlx_client):
        with patch("core.brain.llm.model_registry.get_brainstem_path", return_value="/models/brainstem"):
            with patch("core.brain.llm.model_registry.get_fallback_path", return_value="/models/fallback"):
                result = await gate.generate(
                    "You able to speak?",
                    context={"origin": "user", "prefer_tier": "primary", "allow_cloud_fallback": False},
                )

    assert result == "brainstem"
    assert brainstem.deadlines
    assert brainstem.kwargs[0]["foreground_request"] is True
    assert not cpu.deadlines


def test_conversation_status_is_not_ready_after_timeout_mark():
    gate = InferenceGate()

    class _LaneClient:
        def __init__(self):
            self.reason = ""

        def note_lane_recovering(self, reason):
            self.reason = reason

        def get_lane_status(self):
            return {
                "state": "recovering",
                "last_error": self.reason,
                "conversation_ready": False,
            }

    gate._mlx_client = _LaneClient()
    gate.note_foreground_timeout("foreground_timeout")
    lane = gate.get_conversation_status()

    assert lane["state"] == "recovering"
    assert lane["conversation_ready"] is False
    assert lane["last_failure_reason"] == "foreground_timeout"


def test_conversation_status_respects_ready_lane_even_without_recent_generation():
    gate = InferenceGate()
    gate._last_successful_generation_at = time.time() - 600.0

    class _ReadyLane:
        def get_lane_status(self):
            return {
                "state": "ready",
                "last_error": "",
                "conversation_ready": True,
                "last_ready_at": time.time() - 45.0,
                "last_progress_at": time.time() - 45.0,
                "warmup_attempted": True,
                "warmup_in_flight": False,
            }

    gate._mlx_client = _ReadyLane()

    lane = gate.get_conversation_status()

    assert lane["state"] == "ready"
    assert lane["conversation_ready"] is True


def test_note_foreground_timeout_schedules_fast_reprewarm(monkeypatch):
    gate = InferenceGate()
    scheduled = {}

    def _record_schedule(delay=12.0):
        scheduled["delay"] = delay

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: object())
    gate._schedule_background_cortex_prewarm = _record_schedule
    gate.note_foreground_timeout("foreground_timeout")

    assert scheduled["delay"] == 2.0


@pytest.mark.asyncio
async def test_ensure_foreground_ready_warms_cold_lane_once():
    gate = InferenceGate()
    client = _LaneWarmupClient()
    gate._mlx_client = client

    lane = await gate.ensure_foreground_ready(timeout=10.0)

    client.warmup.assert_awaited_once()
    assert lane["conversation_ready"] is True
    assert lane["state"] == "ready"


@pytest.mark.asyncio
async def test_think_forwards_explicit_timeout_to_generate():
    gate = InferenceGate()
    gate.generate = AsyncMock(return_value="hello")

    result = await gate.think(
        "With me?",
        system_prompt="Be helpful",
        origin="api",
        prefer_tier="primary",
        timeout=67.0,
    )

    assert result == "hello"
    gate.generate.assert_awaited_once()
    assert gate.generate.await_args.kwargs["timeout"] == 67.0


@pytest.mark.asyncio
async def test_initialize_defers_eager_warmup_when_explicitly_disabled():
    gate = InferenceGate()
    client = MagicMock()
    client.warmup = AsyncMock()

    with patch.dict(os.environ, {"AURA_EAGER_CORTEX_WARMUP": "0"}, clear=False):
        with patch("core.brain.llm.mlx_client.get_mlx_client", return_value=client):
            with patch("core.brain.llm.model_registry.get_model_path", return_value="/models/active"):
                with patch("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                    await gate.initialize()

    client.warmup.assert_not_awaited()
    assert gate._initialized is True


@pytest.mark.asyncio
async def test_initialize_auto_warms_on_high_memory_desktop():
    gate = InferenceGate()
    client = MagicMock()
    client.warmup = AsyncMock()
    vm = MagicMock(total=64 * 1024 ** 3, available=40 * 1024 ** 3, percent=37.0)

    with patch.dict(os.environ, {"AURA_EAGER_CORTEX_WARMUP": "auto"}, clear=False):
        with patch("core.brain.inference_gate.psutil.virtual_memory", return_value=vm):
            with patch("core.brain.llm.mlx_client.get_mlx_client", return_value=client):
                with patch("core.brain.llm.model_registry.get_model_path", return_value="/models/active"):
                    with patch("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                        await gate.initialize()

    client.warmup.assert_awaited_once()
    assert gate._prewarm_task is not None


@pytest.mark.asyncio
async def test_initialize_allows_opt_in_eager_warmup():
    gate = InferenceGate()
    client = MagicMock()
    client.warmup = AsyncMock()

    with patch.dict(os.environ, {"AURA_EAGER_CORTEX_WARMUP": "1"}, clear=False):
        with patch("core.brain.llm.mlx_client.get_mlx_client", return_value=client):
            with patch("core.brain.llm.model_registry.get_model_path", return_value="/models/active"):
                with patch("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                    await gate.initialize()

    client.warmup.assert_awaited_once()


@pytest.mark.asyncio
async def test_background_requests_defer_under_memory_pressure_when_cortex_is_ready():
    gate = InferenceGate()
    gate._mlx_client = _LaneWarmupClient()
    gate._mlx_client.state = "ready"
    gate._ensure_cortex_recovery = AsyncMock()

    with patch.object(InferenceGate, "_background_memory_pressure_active", return_value=True):
        with patch.object(
            InferenceGate,
            "get_conversation_status",
            return_value={
                "conversation_ready": True,
                "state": "ready",
                "warmup_in_flight": False,
            },
        ):
            result = await gate.generate(
                "background reflection",
                context={"prefer_tier": "primary", "origin": "system"},
            )

    assert result is None
    gate._ensure_cortex_recovery.assert_not_awaited()


def test_desktop_safe_boot_still_schedules_deferred_cortex_prewarm(monkeypatch):
    monkeypatch.delenv("AURA_DEFERRED_CORTEX_PREWARM", raising=False)
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))

    assert InferenceGate._boot_should_schedule_deferred_prewarm() is False


def test_background_local_deferral_protects_cold_cortex_during_safe_boot(monkeypatch):
    gate = InferenceGate()
    gate._created_at = time.monotonic()
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))
    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False))
    monkeypatch.setattr(InferenceGate, "_foreground_owner_active", staticmethod(lambda: False))
    monkeypatch.setattr(gate, "_should_quiet_background_for_cortex_startup", lambda: False)
    monkeypatch.setattr(gate, "_background_memory_pressure_active", lambda: False)
    monkeypatch.setattr(
        gate,
        "get_conversation_status",
        lambda: {"conversation_ready": False, "state": "cold", "warmup_in_flight": False},
    )

    assert gate._background_local_deferral_reason(origin="system") == "cortex_startup_quiet"
