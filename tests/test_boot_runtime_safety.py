import asyncio
import builtins
import contextlib
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.runtime.boot_safety as boot_safety_module
from core.brain.inference_gate import InferenceGate
from core.brain.llm_health_router import build_router_from_config
from core.config import PROJECT_ROOT, config
from core.container import ServiceContainer
from core.runtime.boot_safety import main_process_camera_policy, uvloop_allowed
from core.runtime.desktop_boot_safety import (
    compute_mlx_cache_limit,
    compute_mlx_memory_limit,
    compute_process_rss_limit,
    desktop_resource_guard_enabled,
    desktop_safe_boot_enabled,
    inprocess_mlx_metal_enabled,
)
from core.senses.continuous_vision import ContinuousSensoryBuffer
from core.somatic.sensory_motor_cortex import SensoryMotorCortex
from core.utils.memory_monitor import AppleSiliconMemoryMonitor


@pytest.fixture(autouse=True)
def _restore_process_environment():
    """Put os.environ back after every test in this file.

    ORDER-DEPENDENCE DEFECT, 2026-07-25.
    ``test_inference_gate_disables_boot_prewarm_under_safe_desktop_boot``
    passed alone and failed under roughly half of pytest-randomly's seeds.

    The cause is not a test that forgot to clean up. This file is the only
    one that invokes a real boot entry point — ``aura_main.main()`` — and
    that entry point's JOB is to configure the process environment for the
    desktop runtime. ``monkeypatch`` undoes what monkeypatch set; it knows
    nothing about the twenty-four variables production code writes directly
    with ``os.environ[...] = ...`` while it runs.

    ``AURA_DEFERRED_CORTEX_PREWARM=1`` was the one that showed up, because
    the prewarm test asserts a False that becomes True the moment that
    variable is set. The rest leaked silently and are worse company:
    ``AURA_SECURITY_PROFILE=owner_autonomous``,
    ``AURA_ALLOW_NETWORK_ACCESS=1``, the memory-governor thresholds, the RSS
    caps, ``AURA_SERVER_PORT``. Any test running afterwards in the same
    process inherited a fully configured owner-autonomous desktop runtime,
    which can make a security or admission test pass for a reason that has
    nothing to do with what it claims to check.

    Restoring the whole environment is the fix that matches the cause: the
    boot path is allowed to write env — that is what it is for — and a test
    process is not allowed to keep the result.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        for key in [k for k in os.environ if k not in snapshot]:
            del os.environ[key]
        for key, value in snapshot.items():
            if os.environ.get(key) != value:
                os.environ[key] = value

VISION_TEST_ROOT = Path(tempfile.gettempdir()) / "aura-test"


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


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


def test_aura_main_does_not_eager_import_cv2_in_primary_process():
    source = (config.paths.project_root / "aura_main.py").read_text(encoding="utf-8")

    assert "import cv2 as _cv2" not in source
    assert "import cv2  # noqa" not in source
    assert "install_main_process_cv2_guard()" in source


def test_media_safe_imports_blocks_cv2_in_primary_darwin(monkeypatch):
    import core.media.safe_imports as safe_imports

    monkeypatch.setattr(safe_imports.sys, "platform", "darwin")
    monkeypatch.delenv("AURA_MEDIA_SIDECAR_PROCESS", raising=False)
    monkeypatch.delenv("AURA_ALLOW_INPROCESS_CV2_WITH_STT", raising=False)

    assert safe_imports.cv2_main_process_blocked() is True


def test_media_safe_imports_allows_cv2_in_sidecar_darwin(monkeypatch):
    import core.media.safe_imports as safe_imports

    monkeypatch.setattr(safe_imports.sys, "platform", "darwin")
    monkeypatch.setenv("AURA_MEDIA_SIDECAR_PROCESS", "1")
    monkeypatch.delenv("AURA_ALLOW_INPROCESS_CV2_WITH_STT", raising=False)

    assert safe_imports.cv2_main_process_blocked() is False


def test_media_safe_imports_blocks_torchcodec_in_primary_darwin(monkeypatch):
    import core.media.safe_imports as safe_imports

    monkeypatch.setattr(safe_imports.sys, "platform", "darwin")
    monkeypatch.delenv("AURA_MEDIA_SIDECAR_PROCESS", raising=False)
    monkeypatch.delenv("AURA_ALLOW_INPROCESS_TORCHCODEC_WITH_STT", raising=False)

    assert safe_imports.torchcodec_main_process_blocked() is True
    assert safe_imports.blocked_native_media_import("torchcodec.decoders") == "torchcodec"


def test_media_safe_imports_allows_torchcodec_only_at_safe_boundaries(monkeypatch):
    import core.media.safe_imports as safe_imports

    monkeypatch.setattr(safe_imports.sys, "platform", "darwin")
    monkeypatch.setenv("AURA_MEDIA_SIDECAR_PROCESS", "1")
    assert safe_imports.torchcodec_main_process_blocked() is False

    monkeypatch.delenv("AURA_MEDIA_SIDECAR_PROCESS", raising=False)
    monkeypatch.setenv("AURA_ALLOW_INPROCESS_TORCHCODEC_WITH_STT", "1")
    assert safe_imports.torchcodec_main_process_blocked() is False


def test_media_guard_rejects_torchcodec_without_importing_it(monkeypatch):
    import core.media.safe_imports as safe_imports

    monkeypatch.setattr(safe_imports.sys, "platform", "darwin")
    monkeypatch.delenv("AURA_MEDIA_SIDECAR_PROCESS", raising=False)
    monkeypatch.delenv("AURA_ALLOW_INPROCESS_TORCHCODEC_WITH_STT", raising=False)
    monkeypatch.setattr(safe_imports, "_CV2_IMPORT_GUARD_INSTALLED", False)
    original = builtins.__import__
    try:
        safe_imports.install_main_process_cv2_guard()
        with pytest.raises(ImportError, match="TorchCodec import is blocked"):
            builtins.__import__("torchcodec.decoders")
    finally:
        builtins.__import__ = original
        safe_imports._CV2_IMPORT_GUARD_INSTALLED = False


def test_sensory_sidecar_marks_media_process_boundary(monkeypatch):
    """The child must be marked as the media sidecar, however it is spawned.

    This asserted the literal line ``os.environ["AURA_MEDIA_SIDECAR_PROCESS"]
    = "1"`` in the client source. Child-process spawning was then centralised
    behind the subprocess gateway, which takes the marker as
    ``environment_overrides`` — a strictly better arrangement — and the test
    failed for the improvement. A test that punishes a correct refactor is
    worse than no test: it teaches people to route around it.

    What actually matters is the property: the spawned worker gets the
    marker, and it is the marker ``safe_imports`` reads. Both are checked
    against behaviour now, so the client is free to spawn however it likes.
    """
    from core.media import safe_imports
    from core.senses import sensory_client

    # The client hands the marker to whatever spawns the child.
    captured: dict[str, object] = {}

    class _Gateway:
        def spawn_python_process(self, spec, context=None):
            captured["overrides"] = dict(getattr(spec, "environment_overrides", {}) or {})
            raise RuntimeError("spawn intercepted: the spec is what is under test")

    monkeypatch.setattr(sensory_client, "get_subprocess_gateway", _Gateway)
    client = sensory_client.SensoryLocalClient()
    with contextlib.suppress(RuntimeError):
        asyncio.run(client.start())

    assert captured.get("overrides", {}).get("AURA_MEDIA_SIDECAR_PROCESS") == "1", (
        "the sensory worker is spawned without the media-sidecar marker; "
        "cv2 would be blocked in the child that exists to run it"
    )

    # The worker sets it for itself too, so a directly-launched worker is
    # marked as well — and it is the same key safe_imports consults.
    worker_source = (config.paths.project_root / "core/senses/sensory_worker.py").read_text(
        encoding="utf-8"
    )
    assert "AURA_MEDIA_SIDECAR_PROCESS" in worker_source

    monkeypatch.setattr(safe_imports.sys, "platform", "darwin")
    monkeypatch.setenv("AURA_MEDIA_SIDECAR_PROCESS", "1")
    monkeypatch.delenv("AURA_ALLOW_INPROCESS_CV2_WITH_STT", raising=False)
    assert safe_imports.cv2_main_process_blocked() is False


@pytest.mark.asyncio
async def test_voice_engine_mic_stream_start_is_timeout_bounded(monkeypatch, tmp_path):
    from core.senses import voice_engine
    from core.senses.voice_engine import SovereignVoiceEngine

    stream_closed = threading.Event()

    class BlockingSoundDevice:
        class InputStream:
            def __init__(self, *args, **kwargs):
                time.sleep(0.3)

            def start(self):
                return None

            def stop(self):
                return None

            def close(self):
                stream_closed.set()

    monkeypatch.setenv("AURA_MIC_START_TIMEOUT_S", "0.05")
    monkeypatch.setattr(voice_engine, "sd", BlockingSoundDevice)

    engine = SovereignVoiceEngine(data_dir=str(tmp_path))
    engine.microphone_enabled = True
    engine._stt_initialized = True
    engine.stt_model = object()
    engine._pulse_hypha = lambda *args, **kwargs: None
    engine._signal_mycelium = lambda *args, **kwargs: None
    recoveries: list[str] = []
    engine._schedule_microphone_recovery = recoveries.append

    started = await engine.start_listening()

    assert started is False
    assert engine._mic_listening is False
    assert engine._is_feeding is False
    assert engine._mic_lease is not None
    assert await asyncio.to_thread(stream_closed.wait, 0.5)
    for _ in range(20):
        if engine._mic_start_task is None and engine._mic_lease is None:
            break
        await asyncio.sleep(0.01)
    assert engine._mic_start_task is None
    assert engine._mic_lease is None
    assert recoveries == ["startup_timeout"]


def test_continuous_vision_blocks_forced_camera_on_darwin(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_CAMERA", "1")
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MAIN_PROCESS_CAMERA", raising=False)
    monkeypatch.setattr(boot_safety_module.sys, "platform", "darwin")

    buffer = ContinuousSensoryBuffer(VISION_TEST_ROOT)

    assert buffer.camera_enabled is False


def test_sensory_motor_cortex_blocks_forced_camera_on_darwin(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_CAMERA", "1")
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MAIN_PROCESS_CAMERA", raising=False)
    monkeypatch.setattr(boot_safety_module.sys, "platform", "darwin")

    cortex = SensoryMotorCortex()

    assert cortex.camera_enabled is False


def test_body_schema_camera_discovery_does_not_import_cv2(monkeypatch):
    from core.somatic.body_schema import BodySchema

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "cv2":
            raise AssertionError("BodySchema must not import cv2 during discovery")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    body = BodySchema()

    assert body.get_limb("camera") is not None


def test_capability_discovery_sensor_scan_does_not_import_optional_libraries(monkeypatch):
    from core.somatic.body_schema import BodySchema
    from core.somatic.capability_discovery import CapabilityDiscoveryDaemon

    real_import = builtins.__import__

    tracked = set(CapabilityDiscoveryDaemon.TRACKED_SENSORS)
    body = BodySchema()

    def guarded_import(name, *args, **kwargs):
        if name in tracked:
            raise AssertionError(
                f"CapabilityDiscovery must not import {name} during sensor scan"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    daemon = CapabilityDiscoveryDaemon(interval=999.0)

    discoveries, losses = daemon._scan_sensors(body)

    assert isinstance(discoveries, list)
    assert isinstance(losses, list)


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
    trigger_autonomous_thought = AsyncCallRecorder()
    generate_autonomous_thought = AsyncCallRecorder()
    emit_spontaneous_message = AsyncCallRecorder()
    orchestrator = SimpleNamespace(
        _trigger_autonomous_thought=trigger_autonomous_thought,
        generate_autonomous_thought=generate_autonomous_thought,
        emit_spontaneous_message=emit_spontaneous_message,
    )
    cortex = SensoryMotorCortex(orchestrator=orchestrator)

    await cortex._dispatch_idle_volition(reason="idle_timeout")

    assert trigger_autonomous_thought.calls == [((False,), {})]
    assert generate_autonomous_thought.calls == []
    assert emit_spontaneous_message.calls == []


def test_memory_monitor_uses_resource_observer_pressure_sample(resource_observer):
    monitor = AppleSiliconMemoryMonitor()
    resource_observer.configure_memory(percent=57.8)

    assert monitor._get_pressure_sysctl() == 57


def test_health_router_prefers_existing_inference_gate(monkeypatch):
    sentinel_gate = object()
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default="_SENTINEL": sentinel_gate if name == "inference_gate" else default),
    )

    router = build_router_from_config(config)

    from core.brain.llm.model_registry import PRIMARY_ENDPOINT
    assert router.endpoints[PRIMARY_ENDPOINT].client is sentinel_gate


@pytest.mark.asyncio
async def test_lazy_local_client_initializes_off_event_loop(monkeypatch):
    sentinel_gate = object()
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default="_SENTINEL": sentinel_gate if name == "inference_gate" else default),
    )

    router = build_router_from_config(config)

    from core.brain.llm.model_registry import BRAINSTEM_ENDPOINT

    client = router.endpoints[BRAINSTEM_ENDPOINT].client
    generate_text_async = AsyncCallRecorder("ok")
    downstream = SimpleNamespace(generate_text_async=generate_text_async)
    offloads = []

    async def fake_to_thread(fn):
        offloads.append(fn)
        return fn()

    monkeypatch.setattr(client, "_get_client", lambda: downstream)
    monkeypatch.setattr("core.brain.llm_health_router.asyncio.to_thread", fake_to_thread)

    assert await client.generate_text_async("hello") == "ok"
    assert len(offloads) == 1
    assert generate_text_async.calls == [(("hello",), {})]


def test_health_router_exposes_only_cortex_during_primary_proof(monkeypatch):
    sentinel_gate = object()
    monkeypatch.setattr("core.brain.llm_health_router.proof_run_active", lambda **_kwargs: True)
    monkeypatch.setattr("core.brain.llm_health_router.proof_model_tier", lambda: "primary")
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default="_SENTINEL": sentinel_gate if name == "inference_gate" else default),
    )

    router = build_router_from_config(config)

    from core.brain.llm.model_registry import PRIMARY_ENDPOINT

    assert list(router.endpoints) == [PRIMARY_ENDPOINT]
    assert router.endpoints[PRIMARY_ENDPOINT].client is sentinel_gate


def test_desktop_app_launch_uses_resource_guard_without_reduced_safe_boot(monkeypatch):
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_DESKTOP_RESOURCE_GUARD", raising=False)
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")

    assert desktop_safe_boot_enabled() is False
    assert desktop_resource_guard_enabled() is True
    assert InferenceGate._desktop_safe_boot_enabled() is False
    assert InferenceGate._desktop_resource_guard_enabled() is True


def test_desktop_resource_guard_does_not_disable_background_local_cognition(monkeypatch):
    gate = InferenceGate.__new__(InferenceGate)
    gate._created_at = 0.0

    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM", raising=False)
    monkeypatch.setattr(gate, "_foreground_user_turn_active", lambda: False)
    monkeypatch.setattr(gate, "_foreground_owner_active", lambda: False)
    monkeypatch.setattr(gate, "_foreground_headroom_reserved", lambda _tier: False)
    monkeypatch.setattr(gate, "_should_quiet_background_for_cortex_startup", lambda: False)
    monkeypatch.setattr(gate, "_foreground_quiet_window_active", lambda: False)
    monkeypatch.setattr(gate, "get_conversation_status", lambda: {
        "conversation_ready": True,
        "state": "ready",
        "warmup_in_flight": False,
    })
    monkeypatch.setattr(gate, "_background_memory_pressure_active", lambda: False)
    monkeypatch.setattr("core.runtime.proof_policy.proof_run_active", lambda *args, **kwargs: False)
    monkeypatch.setattr("core.brain.llm.model_registry.get_local_backend", lambda: "mlx")

    assert gate._background_local_deferral_reason(origin="autonomous_initiative") is None


def test_inference_gate_disables_boot_prewarm_under_safe_desktop_boot(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_AUTO_PREWARM_CORTEX", raising=False)
    # Safe boot only skips the IMPLICIT prewarm; an explicit setting is
    # honoured on purpose. State it, so the test cannot be decided by
    # whatever ran before it.
    monkeypatch.delenv("AURA_DEFERRED_CORTEX_PREWARM", raising=False)

    assert InferenceGate._boot_should_eager_warmup() is False
    assert InferenceGate._boot_should_schedule_deferred_prewarm() is False


def test_compute_mlx_cache_limit_uses_safer_cap_for_desktop_safe_boot(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_LAUNCHED_FROM_APP", raising=False)

    total = 64 * 1024 ** 3
    limit = compute_mlx_cache_limit(total)

    assert limit == 10 * 1024 ** 3


def test_compute_mlx_memory_limit_uses_desktop_safe_active_memory_ceiling(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_MLX_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    total = 64 * 1024 ** 3
    limit = compute_mlx_memory_limit(total)

    assert limit == 34 * 1024 ** 3


def test_compute_process_rss_limit_uses_desktop_safe_guard_ceiling(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_PROCESS_RSS_LIMIT_GB", raising=False)
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    total = 64 * 1024 ** 3
    limit = compute_process_rss_limit(total)

    assert limit == int(total * 0.81)


def test_desktop_process_limit_preserves_host_reserve_for_resident_32b(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_PROCESS_RSS_LIMIT_GB", raising=False)
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    total = 64 * 1024**3
    limit = compute_process_rss_limit(total)

    assert 51 * 1024**3 < limit < 52 * 1024**3
    assert total - limit > 12 * 1024**3


def test_desktop_safe_boot_clamps_unsafe_inherited_model_limits(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.setenv("AURA_MLX_MEMORY_LIMIT_GB", "96")
    monkeypatch.setenv("AURA_PROCESS_RSS_LIMIT_GB", "120")
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    total = 64 * 1024 ** 3

    assert compute_mlx_memory_limit(total) == 34 * 1024 ** 3
    assert compute_process_rss_limit(total) == int(total * 0.81)


def test_desktop_safe_boot_allows_explicit_unsafe_memory_override(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.setenv("AURA_MLX_MEMORY_LIMIT_GB", "40")
    monkeypatch.setenv("AURA_PROCESS_RSS_LIMIT_GB", "42")
    monkeypatch.setenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", "1")

    total = 64 * 1024 ** 3

    assert compute_mlx_memory_limit(total) == 40 * 1024 ** 3
    assert compute_process_rss_limit(total) == 42 * 1024 ** 3


def test_desktop_safe_boot_clamps_stale_floor_overrides(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.setenv("AURA_SAFE_BOOT_METAL_CACHE_FLOOR_GB", "80")
    monkeypatch.setenv("AURA_SAFE_BOOT_MLX_MEMORY_FLOOR_GB", "80")
    monkeypatch.setenv("AURA_SAFE_BOOT_PROCESS_RSS_FLOOR_GB", "80")
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    total = 64 * 1024 ** 3

    assert compute_mlx_cache_limit(total) == 10 * 1024 ** 3
    assert compute_mlx_memory_limit(total) == 34 * 1024 ** 3
    assert compute_process_rss_limit(total) == int(total * 0.81)


def test_live_boot_proof_inherits_safe_desktop_mlx_limits(resource_observer):
    from tools.live_boot_proof import build_safe_boot_env, live_proof_rss_abort_mb

    resource_observer.configure_memory(total_bytes=64 * 1024**3)

    env = build_safe_boot_env({}, observer=resource_observer)

    assert env["AURA_LOCAL_BACKEND"] == "mlx"
    assert env["AURA_SAFE_BOOT_DESKTOP"] == "1"
    assert env["AURA_HEADLESS"] == "1"
    assert env["AURA_DEFERRED_CORTEX_PREWARM"] == "1"
    assert env["AURA_LOCAL_RUNTIME_SINGLETON"] == "1"
    assert env["AURA_LOCAL_PARALLEL_SLOTS"] == "1"
    assert env["AURA_GOVERNANCE_MODE"] == "production"
    assert env["AURA_CONTRACTS_ENFORCE"] == "1"
    assert env["AURA_EAGER_LOCAL_SENSORY_BOOT"] == "0"
    assert env["AURA_ENABLE_PROACTIVE_VISION"] == "0"
    assert env["AURA_DESKTOP_METAL_CACHE_RATIO"] == "0.16"
    assert env["AURA_DESKTOP_METAL_CACHE_CAP_GB"] == "10"
    assert env["AURA_FOREGROUND_CHAT_MAX_TOKENS"] == "2048"
    assert env["AURA_MLX_MEMORY_LIMIT_GB"] == "34"
    assert env["AURA_PROCESS_RSS_LIMIT_GB"] == "52"
    assert env["AURA_MEMWATCH_SOFT_MB"] == "46714"
    assert env["AURA_MEMWATCH_HARD_MB"] == "53084"
    assert env["AURA_MEMWATCH_LETHAL_MB"] == "57180"
    assert env["AURA_MEMORY_SENTINEL_INTERVAL_S"] == "0.5"
    assert env["AURA_GOVERNOR_PRUNE_MB"] == "46714"
    assert env["AURA_GOVERNOR_UNLOAD_MB"] == "49368"
    assert env["AURA_GOVERNOR_CRITICAL_MB"] == "51492"
    assert [
        int(env["AURA_GOVERNOR_PRUNE_MB"]),
        int(env["AURA_MEMWATCH_SOFT_MB"]),
        int(env["AURA_GOVERNOR_UNLOAD_MB"]),
        int(env["AURA_GOVERNOR_CRITICAL_MB"]),
        int(env["AURA_MEMWATCH_HARD_MB"]),
        int(env["AURA_MEMWATCH_LETHAL_MB"]),
    ] == [46714, 46714, 49368, 51492, 53084, 57180]
    assert env["AURA_ENABLE_LOCAL_DEEP_SOLVER"] == "0"
    assert env["AURA_MLX_32B_PROJECTED_FOOTPRINT_GB"] == "auto"
    assert env["AURA_MLX_32B_PROCESS_RESERVE_GB"] == "3"
    assert env["AURA_MLX_72B_PROJECTED_FOOTPRINT_GB"] == "auto"
    assert env["AURA_MLX_72B_PROCESS_RESERVE_GB"] == "5"
    assert live_proof_rss_abort_mb(env) == 57_344.0


def test_live_boot_proof_desktop_mode_does_not_impersonate_packaged_launcher(resource_observer):
    from tools.live_boot_proof import build_safe_boot_env

    resource_observer.configure_memory(total_bytes=64 * 1024**3)

    env = build_safe_boot_env({}, mode="desktop", observer=resource_observer)

    assert env["AURA_LOCAL_BACKEND"] == "mlx"
    assert env["AURA_SAFE_BOOT_DESKTOP"] == "0"
    assert env["AURA_DESKTOP_RESOURCE_GUARD"] == "1"
    assert env["AURA_HEADLESS"] == "0"
    assert env["AURA_LAUNCHED_FROM_APP"] == "0"
    assert env["AURA_EXTERNAL_GUI_OWNER"] == "0"
    assert env["AURA_GOVERNANCE_MODE"] == "production"
    assert env["AURA_CONTRACTS_ENFORCE"] == "1"
    assert env["AURA_EAGER_LOCAL_SENSORY_BOOT"] == "1"
    assert "AURA_AUTO_LISTEN" not in env
    assert env["AURA_EAGER_CORTEX_WARMUP"] == "0"
    assert env["AURA_DEFERRED_CORTEX_PREWARM"] == "1"
    assert env["AURA_ENABLE_LOCAL_DEEP_SOLVER"] == "0"
    assert env["AURA_AMBIENT_STREAM_INTERVAL_S"] == "5"
    assert env["AURA_AUTONOMIC_REFLECTION_INTERVAL_S"] == "30"


def test_live_boot_proof_preserves_operator_mlx_limit(resource_observer):
    from tools.live_boot_proof import build_safe_boot_env

    resource_observer.configure_memory(total_bytes=64 * 1024**3)

    env = build_safe_boot_env(
        {"AURA_MLX_MEMORY_LIMIT_GB": "28"},
        observer=resource_observer,
    )

    assert env["AURA_SAFE_BOOT_DESKTOP"] == "1"
    assert env["AURA_MLX_MEMORY_LIMIT_GB"] == "28"


def test_live_boot_proof_clamps_unsafe_parent_memory_limits(resource_observer):
    from tools.live_boot_proof import build_safe_boot_env, live_proof_rss_abort_mb

    resource_observer.configure_memory(total_bytes=64 * 1024**3)

    env = build_safe_boot_env(
        {
            "AURA_MLX_MEMORY_LIMIT_GB": "96",
            "AURA_PROCESS_RSS_LIMIT_GB": "120",
            "AURA_LIVE_PROOF_RSS_ABORT_MB": "90000",
        },
        observer=resource_observer,
    )

    assert env["AURA_MLX_MEMORY_LIMIT_GB"] == "34"
    assert env["AURA_PROCESS_RSS_LIMIT_GB"] == "52"
    assert live_proof_rss_abort_mb(env) == 57_344.0


def test_live_boot_proof_uses_readiness_heartbeat_contract():
    source = (PROJECT_ROOT / "tools" / "live_boot_proof.py").read_text()

    assert "resolve_launch_python()" in source
    assert "self.launch_python" in source
    assert "/api/health/heartbeat" in source
    assert "required_probes" in source
    assert "runtime_probe_healthy" in source
    assert "system_ready" in source
    assert "/api/health" in source
    assert "LIVE_DESKTOP_FULL_RUNTIME_COMPONENTS" in source
    assert "full_runtime_ready" in source
    assert "screen_perception" in source
    assert "perceptual_pump" in source
    assert "cognitive_situation" in source
    assert "imagination_engine" in source
    assert "ambient_developer_stream" in source
    assert "autonomic_reflection_loop" in source
    assert "exercise_cognitive_organ_participation" in source
    assert "started_monotonic" in source
    assert "duration_budget_s" in source
    assert "within_budget" in source
    assert "exercise_capability_inventory_turn" in source
    assert "X-Aura-Require-CognitiveEngine" in source


def test_desktop_boot_sets_hardened_governance_defaults():
    source = (PROJECT_ROOT / "aura_main.py").read_text()

    assert "if args.desktop or args.headless:" in source
    assert (
        'if (args.desktop or args.headless) and "AURA_DESKTOP_RESOURCE_GUARD" not in os.environ:'
        not in source
    )
    assert 'os.environ["AURA_GOVERNANCE_MODE"] = "production"' in source
    assert 'os.environ["AURA_CONTRACTS_ENFORCE"] = "1"' in source


@pytest.mark.asyncio
async def test_verifier_foundry_restore_runs_off_event_loop(monkeypatch):
    import aura_main
    import core.brain.verifiers.foundry as foundry_module

    foundry = SimpleNamespace(status=lambda: {"cells": {}, "pending_verdicts": 0})
    calls: list[object] = []

    async def _to_thread(func, *args, **kwargs):
        calls.append(func)
        return func(*args, **kwargs)

    monkeypatch.setattr(aura_main.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(foundry_module, "boot_verifier_foundry", lambda: foundry)

    loaded, status = await aura_main._load_verifier_foundry_off_loop()

    assert loaded is foundry
    assert status["pending_verdicts"] == 0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_ulysses_boot_remains_reachable(monkeypatch):
    import aura_main
    import core.sovereignty.ulysses as ulysses_module

    expected = {
        "active_contracts": 1,
        "hard": 1,
        "integrity": 1.0,
        "chain_length": 1,
    }
    covenant = SimpleNamespace(status=lambda: expected)

    async def _to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(aura_main, "_env_flag", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(aura_main.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(ulysses_module, "boot_ulysses_covenant", lambda: covenant)

    assert await aura_main._activate_ulysses_covenant_for_boot() == expected


def test_desktop_boot_overrides_inherited_weak_governance_when_guard_is_preconfigured(
    monkeypatch,
):
    import aura_main

    launchagent_calls = []
    monkeypatch.setattr(sys, "argv", ["aura_main.py", "--desktop", "--stop"])
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "research")
    monkeypatch.setenv("AURA_CONTRACTS_ENFORCE", "0")
    monkeypatch.setattr(aura_main, "_maybe_relaunch_with_preferred_python", lambda: None)
    monkeypatch.setattr(aura_main, "_ensure_reaper_manifest_env", lambda: None)
    monkeypatch.setattr(
        aura_main,
        "_disable_legacy_launchagent",
        lambda **kwargs: launchagent_calls.append(kwargs),
    )
    monkeypatch.setattr(aura_main, "stop_aura", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        aura_main.main()

    assert exc_info.value.code == 0
    assert aura_main.os.environ["AURA_GOVERNANCE_MODE"] == "production"
    assert aura_main.os.environ["AURA_CONTRACTS_ENFORCE"] == "1"
    assert aura_main.os.environ["AURA_LOCAL_BACKEND"] == "mlx"
    assert launchagent_calls == [
        {"quarantine_obsolete": True, "reason": "modern_desktop_launch"}
    ]


def test_live_boot_proof_requires_cognitive_organ_participation(monkeypatch, tmp_path):
    import tools.live_boot_proof as live_boot_proof

    payload = {
        "full_runtime": {
            "components": {
                "cognitive_situation": {
                    "running": True,
                    "frames_built": 4,
                    "latest": {"frame_id": "situation-4"},
                },
                "imagination_engine": {
                    "running": True,
                    "frames_built": 4,
                    "latest": {"frame_id": "imagination-4"},
                },
                "timescale_bridge": {
                    "running": True,
                    "observations": 3,
                    "frames_ingested": 3,
                    "latest_observation": {"source": "perceptual_pump"},
                    "last_reconciliation": {
                        "idle_gap_s": 0.2,
                        "summary": "recent apps: Aura Zenith",
                        "directives": [
                            "Anchor the reply to the user's current message and verified recent conversation."
                        ],
                    },
                },
                "ambient_developer_stream": {
                    "running": True,
                    "frames": 1,
                    "latest_frame": {"summary": "ambient developer stream observed no material changes"},
                },
                "autonomic_reflection_loop": {
                    "running": True,
                    "reflections_written": 0,
                    "errors": 0,
                },
            }
        }
    }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.timeout = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, _url):
            return SimpleNamespace(status_code=200, text="", json=lambda: payload)

    monkeypatch.setattr(live_boot_proof.httpx, "Client", FakeClient)
    proof = live_boot_proof.LiveProof(
        port=8999,
        mode="desktop",
        boot_timeout_s=1.0,
        skip_desktop=True,
        restart_continuity=False,
        conversation_soak_turns=0,
        proof_dir=tmp_path,
    )

    assert proof.exercise_cognitive_organ_participation() is True
    assert proof.steps[-1]["organs"]["cognitive_situation"]["participated"] is True
    assert proof.steps[-1]["organs"]["timescale_bridge"]["participated"] is True
    assert proof.steps[-1]["organs"]["ambient_developer_stream"]["participated"] is True
    assert proof.steps[-1]["organs"]["autonomic_reflection_loop"]["participated"] is True

    payload["full_runtime"]["components"]["imagination_engine"]["frames_built"] = 0
    payload["full_runtime"]["components"]["imagination_engine"]["latest"] = None

    assert proof.exercise_cognitive_organ_participation() is False
    assert proof.steps[-1]["blockers"] == ["imagination_engine"]

    payload["full_runtime"]["components"]["imagination_engine"]["frames_built"] = 4
    payload["full_runtime"]["components"]["imagination_engine"]["latest"] = {
        "frame_id": "imagination-4"
    }
    payload["full_runtime"]["components"]["timescale_bridge"]["last_reconciliation"] = None

    assert proof.exercise_cognitive_organ_participation() is False
    assert proof.steps[-1]["blockers"] == ["timescale_bridge"]

    payload["full_runtime"]["components"]["timescale_bridge"]["last_reconciliation"] = {
        "idle_gap_s": 0.2,
        "summary": "recent apps: Aura Zenith",
        "directives": [
            "Anchor the reply to the user's current message and verified recent conversation."
        ],
    }
    payload["full_runtime"]["components"]["ambient_developer_stream"]["latest_frame"] = None

    assert proof.exercise_cognitive_organ_participation() is False
    assert proof.steps[-1]["blockers"] == ["ambient_developer_stream"]


def test_live_boot_proof_runtime_stream_scan_fails_failure_markers(monkeypatch, tmp_path):
    import tools.live_boot_proof as live_boot_proof

    monkeypatch.setattr(live_boot_proof, "PROOF_DIR", tmp_path)
    proof = live_boot_proof.LiveProof(
        port=8999,
        mode="desktop",
        boot_timeout_s=1.0,
        skip_desktop=True,
        restart_continuity=False,
        conversation_soak_turns=0,
    )
    proof.stdout_path.write_text(
        "Cortex Warming...\n"
        "Traceback (most recent call last):\n"
        "Runtime: DEGRADED\n"
        "Dialogue contract deterministic repair still failed before retry: initial=ungrounded_live_voice\n"
        "cortex route blocked\n",
        encoding="utf-8",
    )

    assert proof.scan_runtime_stream() is False
    step = proof.steps[-1]
    assert step["step"] == "runtime_stream_scan"
    assert "Cortex Warming" in step["markers"]
    assert "Traceback" in step["markers"]
    assert "Runtime: DEGRADED" in step["markers"]
    assert "Dialogue contract deterministic repair still failed before retry" in step["markers"]
    assert "Cortex route blocked" in step["markers"]


def test_live_boot_proof_stream_scan_ignores_non_log_level_error_words(monkeypatch, tmp_path):
    import tools.live_boot_proof as live_boot_proof

    monkeypatch.setattr(live_boot_proof, "PROOF_DIR", tmp_path)
    proof = live_boot_proof.LiveProof(
        port=8999,
        mode="desktop",
        boot_timeout_s=1.0,
        skip_desktop=True,
        restart_continuity=False,
        conversation_soak_turns=0,
    )
    proof.stdout_path.write_text(
        "StructuredErrorLogger initialized at data/error_logs\n"
        "HEALTH CONTRACT: All critical + important services online\n"
        "CriticalityRegulator initialized\n"
        "🧠 MemoryRetrieval: Searching for context: [SILENT AUTO-FIX] Investigate a timeout. Error: co...\n",
        encoding="utf-8",
    )

    assert proof.scan_runtime_stream() is True
    step = proof.steps[-1]
    assert step["step"] == "runtime_stream_scan"
    assert step["markers"] == {}


def test_live_boot_proof_verdict_records_commit_and_end_metadata():
    source = (PROJECT_ROOT / "tools" / "live_boot_proof.py").read_text()

    assert '"ended_at": finished_at' in source
    assert '"git_commit": git_commit' in source
    assert '"git_dirty": git_dirty' in source
    assert '"stdout_log": artifact_display_path(self.stdout_path)' in source
    assert "current_git_commit()" in source
    assert "current_git_dirty()" in source


def test_live_boot_proof_supports_stable_output_directory():
    source = (PROJECT_ROOT / "tools" / "live_boot_proof.py").read_text()

    assert "--out-dir" in source
    assert "self.latest_verdict_path" in source
    assert "LATEST_VERDICT.json" in source


def test_live_boot_proof_preflight_ignores_shell_mentions_of_aura_main():
    from tools.live_boot_proof import LiveProof

    shell_probe = [
        "/bin/zsh",
        "-c",
        "pgrep -fl \"aura_main.py --desktop\" | head -2",
    ]
    actual_runtime = [
        "/opt/homebrew/bin/python3.12",
        "aura_main.py",
        "--desktop",
        "--port",
        "8034",
    ]

    assert LiveProof._is_aura_main_process(shell_probe) is False
    assert LiveProof._is_aura_main_process(actual_runtime) is True


def test_service_registration_uses_canonical_meta_cognition_import():
    source = (PROJECT_ROOT / "core/service_registration.py").read_text()

    assert "from core.cognition.meta_cognition import MetaEvolutionEngine" in source
    assert "from .meta_cognition import MetaEvolutionEngine" not in source


def test_sensory_gate_actor_uses_supported_restart_policy():
    source = (PROJECT_ROOT / "core/orchestrator/main.py").read_text()

    assert 'name="SensoryGate", target=start_sensory_gate, args=(), restart_policy="transient"' in source
    assert 'restart_policy="one_for_one"' not in source


def test_compute_mlx_cache_limit_defaults_to_standard_ratio_when_not_safe(monkeypatch):
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_LAUNCHED_FROM_APP", raising=False)

    limit = compute_mlx_cache_limit(64 * 1024 ** 3)

    assert limit == int(64 * 1024 ** 3 * 0.75)


def test_rsi_lab_creates_data_dir_without_runtime_globals(monkeypatch, tmp_path):
    from research.meta_learning_loop import RSILab

    monkeypatch.setattr(type(config.paths), "_runtime_home_cache", tmp_path)

    lab = RSILab()

    assert lab.lab_dir == tmp_path / "data" / "rsi_lab"
    assert lab.lab_dir.exists()


@pytest.mark.asyncio
async def test_rsi_lab_requires_validation_evidence_for_promotion(monkeypatch, tmp_path):
    from research.meta_learning_loop import RSILab

    monkeypatch.setattr(type(config.paths), "_runtime_home_cache", tmp_path)
    lab = RSILab()
    weak_id = lab.submit_candidate(
        "heuristic",
        "always do the clever thing",
        "Too vague to promote because it has no validation or rollback evidence.",
    )
    strong_id = lab.submit_candidate(
        "skill",
        {
            "steps": ["inspect inputs", "run verifier", "emit receipt"],
            "tool_contract": {"input": "objective", "output": "verified_plan"},
            "evidence": {
                "provenance": "unit-test",
                "validation_command": "pytest tests/test_boot_runtime_safety.py",
                "validation_passed": True,
                "rollback_plan": "remove skill registration",
                "receipt_id": "receipt_rsi_validation_001",
                "risk": {"level": "bounded", "blast_radius": "skill registry only"},
            },
        },
        "Promote because the skill has explicit validation, provenance, rollback, and bounded risk.",
    )

    assert await lab.evaluate_pending_candidates() == 2

    assert lab.candidates[weak_id].status == "failed"
    assert "validation_passed" in lab.candidates[weak_id].evaluation_report["blocking_failures"]
    assert lab.candidates[strong_id].status == "passed"
    assert lab.candidates[strong_id].evaluation_report["checks"]["receipt_present"] is True
    assert lab.promote(weak_id) is False
    assert lab.candidates[weak_id].status == "failed"
    assert lab.promote("missing-candidate") is False
    assert lab.promote(strong_id) is True
    assert lab.candidates[strong_id].status == "promoted"


def test_rsi_lab_loads_valid_candidates_while_skipping_corrupt_records(monkeypatch, tmp_path):
    from research.meta_learning_loop import RSILab

    monkeypatch.setattr(type(config.paths), "_runtime_home_cache", tmp_path)
    lab_dir = tmp_path / "data" / "rsi_lab"
    lab_dir.mkdir(parents=True)
    (lab_dir / "candidates.json").write_text(
        json.dumps(
            {
                "valid": {
                    "id": "valid",
                    "artifact_type": "heuristic",
                    "content": {"rule": "prefer verified changes because they are reversible"},
                    "rationale": "Keep only candidates with enough evidence because promotion is risky.",
                    "status": "pending_eval",
                    "score": 0.0,
                    "evaluation_report": {},
                    "created_at": 1.0,
                },
                "corrupt": ["not", "a", "candidate"],
            }
        ),
        encoding="utf-8",
    )

    lab = RSILab()

    assert list(lab.candidates) == ["valid"]
    assert lab.candidates["valid"].artifact_type == "heuristic"


@pytest.mark.asyncio
async def test_chaos_fill_disk_creates_bounded_pressure_file(monkeypatch, tmp_path):
    from tools.chaos import injector

    scheduled = []

    class Tracker:
        def create_task(self, coro, name=None):
            scheduled.append((coro, name))
            return SimpleNamespace(done=lambda: False)

    monkeypatch.setenv("AURA_CHAOS_DISK_TARGET_DIR", str(tmp_path))
    monkeypatch.setenv("AURA_CHAOS_DISK_MAX_MB", "1")
    monkeypatch.setenv("AURA_CHAOS_DISK_RESTORE_SECONDS", "0")
    monkeypatch.setattr(injector, "get_task_tracker", lambda: Tracker())

    result = await injector._fill_disk()

    pressure_file = Path(result["target"])
    assert result["applied"] is True
    assert result["bytes_written"] == 1024 * 1024
    assert await asyncio.to_thread(pressure_file.exists)
    assert scheduled and scheduled[0][1] == "chaos.fill_disk.restore_pressure_file"

    await scheduled[0][0]

    assert not await asyncio.to_thread(pressure_file.exists)
    assert not await asyncio.to_thread(pressure_file.parent.exists)


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
    assert reason == "desktop_resource_guard"


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


def test_live_learner_autorun_training_requires_explicit_operator_policy(monkeypatch):
    from core.learning.live_learner import LiveLearner, TrainingPolicy

    monkeypatch.delenv("AURA_SELF_TRAIN_AUTORUN", raising=False)
    assert TrainingPolicy.from_env().autorun_enabled is False

    learner = LiveLearner.__new__(LiveLearner)
    learner._policy = TrainingPolicy(autorun_enabled=False)
    learner._training_in_progress = False
    learner._model_path = "aura-model"
    learner._buffer = [{} for _ in range(LiveLearner.MIN_EXAMPLES_FOR_TRAINING)]
    learner._last_train_time = 0.0

    assert learner._should_train() is False

    learner._policy = TrainingPolicy(autorun_enabled=True)
    assert learner._should_train() is True


def test_voice_engine_imports_current_data_dir_path(monkeypatch):
    """Voice boot must not regress to the historical core.common.paths DATA_DIR import."""
    monkeypatch.setenv("AURA_AUTO_LISTEN", "0")

    from core.self_model import DATA_FILE
    from core.senses.voice_engine import SovereignVoiceEngine
    from core.utils.paths import DATA_DIR

    engine = SovereignVoiceEngine()

    assert DATA_FILE == DATA_DIR / "self_model.json"
    assert str(engine.data_dir).endswith("voice_models")


@pytest.mark.asyncio
async def test_continuous_vision_defers_screen_backend_without_permission(monkeypatch):
    class _FakeMSSModule:
        def __init__(self):
            self.mss_calls = 0

        def mss(self):
            self.mss_calls += 1
            raise AssertionError("mss() should not be called without active permission")

    check_permission = AsyncCallRecorder({"granted": False, "status": "deferred"})
    guard = SimpleNamespace(check_permission=check_permission)
    privacy_admission = AsyncCallRecorder(SimpleNamespace(allowed=True))

    fake_mss = _FakeMSSModule()
    monkeypatch.setitem(sys.modules, "mss", fake_mss)
    monkeypatch.setattr(
        "core.senses.continuous_vision.evaluate_screen_capture_admission_async",
        privacy_admission,
    )
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None: guard if name == "permission_guard" else default),
    )

    buffer = ContinuousSensoryBuffer(VISION_TEST_ROOT)
    ready = await buffer._ensure_screen_backend()

    assert ready is False
    assert len(privacy_admission.calls) == 1
    assert len(check_permission.calls) == 1
    assert fake_mss.mss_calls == 0
    assert buffer.sct is None
    assert buffer.monitor is None
