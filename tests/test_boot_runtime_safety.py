from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.brain.inference_gate import InferenceGate
from core.brain.llm_health_router import build_router_from_config
from core.container import ServiceContainer
from core.config import PROJECT_ROOT, config
from core.runtime.boot_safety import main_process_camera_policy, uvloop_allowed
from core.runtime.desktop_boot_safety import (
    compute_mlx_cache_limit,
    desktop_safe_boot_enabled,
    inprocess_mlx_metal_enabled,
)
from core.senses.continuous_vision import ContinuousSensoryBuffer
from core.sensory_motor_cortex import SensoryMotorCortex
from core.utils.memory_monitor import AppleSiliconMemoryMonitor


def test_config_exports_project_root_alias():
    assert PROJECT_ROOT == config.paths.project_root


def test_uvloop_disabled_by_default_on_darwin(monkeypatch):
    monkeypatch.delenv("AURA_ENABLE_UVLOOP", raising=False)
    assert uvloop_allowed(platform="darwin") is False


def test_uvloop_can_be_forced_on_darwin(monkeypatch):
    monkeypatch.setenv("AURA_ENABLE_UVLOOP", "1")
    assert uvloop_allowed(platform="darwin") is True


def test_main_process_camera_policy_blocks_darwin_without_override(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MAIN_PROCESS_CAMERA", raising=False)
    enabled, reason = main_process_camera_policy(True, platform="darwin")
    assert enabled is False
    assert "cv2/PyAV" in reason


def test_continuous_vision_blocks_forced_camera_on_darwin(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_CAMERA", "1")
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MAIN_PROCESS_CAMERA", raising=False)

    with patch("core.runtime.boot_safety.sys.platform", "darwin"):
        buffer = ContinuousSensoryBuffer(Path("/tmp/aura-test"))

    assert buffer.camera_enabled is False


def test_sensory_motor_cortex_blocks_forced_camera_on_darwin(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_CAMERA", "1")
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MAIN_PROCESS_CAMERA", raising=False)

    with patch("core.runtime.boot_safety.sys.platform", "darwin"):
        cortex = SensoryMotorCortex()

    assert cortex.camera_enabled is False


def test_sensory_motor_cortex_syncs_user_activity_before_idle_trigger():
    orchestrator = SimpleNamespace(
        _last_user_interaction_time=200.0,
        status=SimpleNamespace(is_processing=False),
        _current_thought_task=None,
    )
    cortex = SensoryMotorCortex(orchestrator=orchestrator, config={"boredom_threshold": 120})
    cortex.last_interaction_time = 0.0

    assert cortex._should_trigger_volition(now=250.0) is False
    assert cortex.last_interaction_time == 200.0


def test_sensory_motor_cortex_skips_volition_while_processing():
    orchestrator = SimpleNamespace(
        _last_user_interaction_time=0.0,
        status=SimpleNamespace(is_processing=True),
        _current_thought_task=None,
    )
    cortex = SensoryMotorCortex(orchestrator=orchestrator, config={"boredom_threshold": 120})
    cortex.last_interaction_time = 0.0

    assert cortex._should_trigger_volition(now=500.0) is False
    assert cortex.last_interaction_time == 500.0


@pytest.mark.asyncio
async def test_sensory_motor_cortex_routes_idle_volition_into_autonomy():
    orchestrator = SimpleNamespace(
        _trigger_autonomous_thought=AsyncMock(),
        generate_autonomous_thought=AsyncMock(),
        emit_spontaneous_message=AsyncMock(),
    )
    cortex = SensoryMotorCortex(orchestrator=orchestrator)

    await cortex._dispatch_idle_volition(reason="idle_timeout")

    orchestrator._trigger_autonomous_thought.assert_awaited_once_with(False)
    orchestrator.generate_autonomous_thought.assert_not_called()
    orchestrator.emit_spontaneous_message.assert_not_called()


def test_memory_monitor_uses_psutil_pressure_sample(monkeypatch):
    monitor = AppleSiliconMemoryMonitor()
    monkeypatch.setattr(
        "core.utils.memory_monitor.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=57.8),
    )

    assert monitor._get_pressure_sysctl() == 57


def test_health_router_prefers_existing_inference_gate(monkeypatch):
    sentinel_gate = object()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default="_SENTINEL": sentinel_gate if name == "inference_gate" else default),
    )

    router = build_router_from_config(config)

    from core.brain.llm.model_registry import PRIMARY_ENDPOINT
    assert router.endpoints[PRIMARY_ENDPOINT].client is sentinel_gate


def test_desktop_safe_boot_tracks_app_launch_context(monkeypatch):
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")

    assert desktop_safe_boot_enabled() is True


def test_inference_gate_disables_boot_prewarm_under_safe_desktop_boot(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")

    assert InferenceGate._boot_should_eager_warmup() is False
    assert InferenceGate._boot_should_schedule_deferred_prewarm() is False


def test_compute_mlx_cache_limit_uses_safer_cap_for_desktop_safe_boot(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_LAUNCHED_FROM_APP", raising=False)

    total = 64 * 1024 ** 3
    limit = compute_mlx_cache_limit(total)

    assert limit == int(total * 0.56)
    assert limit < 36 * 1024 ** 3


def test_compute_mlx_cache_limit_defaults_to_standard_ratio_when_not_safe(monkeypatch):
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_LAUNCHED_FROM_APP", raising=False)

    limit = compute_mlx_cache_limit(64 * 1024 ** 3)

    assert limit == int(64 * 1024 ** 3 * 0.75)


def test_inprocess_mlx_metal_disabled_during_safe_boot(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_FORCE_INPROCESS_MLX_METAL", raising=False)
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_INPROCESS_MLX_METAL", raising=False)
    monkeypatch.delenv("AURA_DISABLE_INPROCESS_MLX_METAL", raising=False)

    enabled, reason = inprocess_mlx_metal_enabled(
        platform_name="darwin",
        mac_version="26.4",
    )

    assert enabled is False
    assert reason == "desktop_safe_boot"


def test_inprocess_mlx_metal_disabled_on_macos26_by_default(monkeypatch):
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_LAUNCHED_FROM_APP", raising=False)
    monkeypatch.delenv("AURA_FORCE_INPROCESS_MLX_METAL", raising=False)
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_INPROCESS_MLX_METAL", raising=False)
    monkeypatch.delenv("AURA_DISABLE_INPROCESS_MLX_METAL", raising=False)

    enabled, reason = inprocess_mlx_metal_enabled(
        platform_name="darwin",
        mac_version="26.4",
    )

    assert enabled is False
    assert reason == "macos26_guard"


def test_inprocess_mlx_metal_can_be_forced_for_debugging(monkeypatch):
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_LAUNCHED_FROM_APP", raising=False)
    monkeypatch.setenv("AURA_FORCE_INPROCESS_MLX_METAL", "1")
    monkeypatch.delenv("AURA_DISABLE_INPROCESS_MLX_METAL", raising=False)

    enabled, reason = inprocess_mlx_metal_enabled(
        platform_name="darwin",
        mac_version="26.4",
    )

    assert enabled is True
    assert reason == "forced"


import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_continuous_vision_defers_screen_backend_without_permission(monkeypatch):
    class _FakeMSSModule:
        def mss(self):
            raise AssertionError("mss() should not be called without active permission")

    guard = MagicMock()
    guard.check_permission = AsyncMock(return_value={"granted": False, "status": "deferred"})

    monkeypatch.setitem(sys.modules, "mss", _FakeMSSModule())

    with patch("core.container.ServiceContainer.get", return_value=guard):
        buffer = ContinuousSensoryBuffer(Path("/tmp/aura-test"))
        ready = await buffer._ensure_screen_backend()

    assert ready is False
    assert buffer.sct is None
    assert buffer.monitor is None
