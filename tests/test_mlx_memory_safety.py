from __future__ import annotations

import json
import time
from types import SimpleNamespace


def test_throughput_sample_prefers_worker_measured_generation_clocks(monkeypatch):
    from core.brain.llm import mlx_client
    from core.brain.llm.model_registry import get_model_runtime_assignment

    captured = []
    monkeypatch.setattr(mlx_client, "record_generation", lambda **kwargs: captured.append(kwargs))
    assignment = get_model_runtime_assignment("Aura-32B-20260510-151144")
    client = mlx_client.MLXLocalClient(
        assignment.model_path,
        runtime_assignment=assignment,
    )
    client._current_request_started_at = time.time() - 90.0

    client._record_throughput_sample(
        {
            "tokens_used": 30,
            "prompt_tokenization": {"tokens": 900},
            "prompt_cache_reused_tokens": 700,
            "generation_stop_reason": "eos",
            "generation_performance": {
                "prefill_seconds": 2.25,
                "decode_seconds": 7.5,
            },
        },
        prompt="ignored because tokenization is measured",
        foreground_request=True,
    )

    assert len(captured) == 1
    assert captured[0]["prefill_seconds"] == 2.25
    assert captured[0]["decode_seconds"] == 7.5
    assert captured[0]["cache_warm"] is True


def test_worker_preserves_mlx_generation_performance():
    from core.brain.llm.mlx_worker import _generation_performance_snapshot

    response = SimpleNamespace(
        prompt_tokens=900,
        prompt_tps=300.0,
        generation_tokens=40,
        generation_tps=8.0,
        peak_memory=14.25,
        finish_reason="stop",
    )

    measured = _generation_performance_snapshot(
        response,
        fallback_prompt_tokens=1,
        fallback_generation_tokens=2,
        first_token_seconds=3.1,
        stream_seconds=8.2,
    )

    assert measured == {
        "prompt_tokens": 900,
        "prompt_tps": 300.0,
        "prefill_seconds": 3.0,
        "generation_tokens": 40,
        "generation_tps": 8.0,
        "decode_seconds": 5.0,
        "first_token_seconds": 3.1,
        "stream_seconds": 8.2,
        "peak_memory_gb": 14.25,
        "finish_reason": "stop",
    }


def test_external_memory_sentinel_uses_current_footprint_not_lifetime_peak():
    from tools.memory_sentinel import _RUsageV4, current_phys_footprint_bytes

    usage = _RUsageV4()
    usage.ri_resident_size = 2 * 1024**3
    usage.ri_phys_footprint = 7 * 1024**3
    usage.ri_lifetime_max_phys_footprint = 105 * 1024**3

    assert current_phys_footprint_bytes(usage) == 7 * 1024**3


def test_high_pressure_preserves_foreground_completion_reserve_and_recurrence():
    from core.brain.llm.mlx_client import _apply_memory_pressure_generation_controls

    options = {
        "max_tokens": 2048,
        "clean_user_surface_contract": True,
        "clean_user_surface_recurrent_loops": 2,
        "user_surface_completion_floor": 1024,
    }
    snapshot = SimpleNamespace(max_token_cap=192)

    controlled = _apply_memory_pressure_generation_controls(options, snapshot)

    assert controlled["max_tokens"] == 1024
    assert controlled["clean_user_surface_recurrent_loops"] == 2
    assert controlled["memory_pressure_token_cap"] == 192
    assert controlled["completion_floor_applied"] is True


def test_critical_pressure_keeps_hard_cap_even_with_completion_reserve():
    from core.brain.llm.mlx_client import _apply_memory_pressure_generation_controls

    controlled = _apply_memory_pressure_generation_controls(
        {
            "max_tokens": 1536,
            "clean_user_surface_contract": True,
            "clean_user_surface_recurrent_loops": 2,
            "user_surface_completion_floor": 1024,
        },
        SimpleNamespace(max_token_cap=64),
    )

    assert controlled["max_tokens"] == 64
    assert controlled["clean_user_surface_recurrent_loops"] == 1
    assert controlled["completion_floor_applied"] is False


def test_memory_pressure_generation_controls_preserve_depth_without_cap():
    from core.brain.llm.mlx_client import _apply_memory_pressure_generation_controls

    options = {
        "max_tokens": 2048,
        "clean_user_surface_contract": True,
        "clean_user_surface_recurrent_loops": 2,
    }
    snapshot = SimpleNamespace(max_token_cap=None)

    controlled = _apply_memory_pressure_generation_controls(options, snapshot)

    assert controlled["max_tokens"] == 2048
    assert controlled["clean_user_surface_recurrent_loops"] == 2


def test_memory_pressure_preserves_depth_when_cap_does_not_reduce_budget():
    from core.brain.llm.mlx_client import _apply_memory_pressure_generation_controls

    options = {
        "max_tokens": 1792,
        "clean_user_surface_contract": True,
        "clean_user_surface_recurrent_loops": 2,
    }
    snapshot = SimpleNamespace(max_token_cap=2048)

    controlled = _apply_memory_pressure_generation_controls(options, snapshot)

    assert controlled["max_tokens"] == 1792
    assert controlled["clean_user_surface_recurrent_loops"] == 2
    assert "recurrent_loops_reduced_by_pressure" not in controlled


def test_memory_pressure_preserves_depth_for_a_minor_adaptive_trim():
    from core.brain.llm.mlx_client import _apply_memory_pressure_generation_controls

    controlled = _apply_memory_pressure_generation_controls(
        {
            "max_tokens": 2048,
            "clean_user_surface_contract": True,
            "clean_user_surface_recurrent_loops": 2,
        },
        SimpleNamespace(max_token_cap=1941),
    )

    assert controlled["max_tokens"] == 1941
    assert controlled["clean_user_surface_recurrent_loops"] == 2
    assert "recurrent_loops_reduced_by_pressure" not in controlled


def test_memory_pressure_generation_controls_use_model_default_when_unspecified():
    from core.brain.llm.mlx_client import _apply_memory_pressure_generation_controls

    options = {
        "clean_user_surface_contract": True,
        "clean_user_surface_recurrent_loops": 2,
    }
    snapshot = SimpleNamespace(max_token_cap=192)

    controlled = _apply_memory_pressure_generation_controls(
        options,
        snapshot,
        default_max_tokens=4096,
    )

    assert controlled["max_tokens"] == 192
    assert controlled["clean_user_surface_recurrent_loops"] == 1


def test_completion_reserve_never_expands_the_callers_budget():
    from core.brain.llm.mlx_client import _apply_memory_pressure_generation_controls

    controlled = _apply_memory_pressure_generation_controls(
        {
            "max_tokens": 384,
            "clean_user_surface_contract": True,
            "user_surface_completion_floor": 1024,
        },
        SimpleNamespace(max_token_cap=192),
    )

    assert controlled["max_tokens"] == 384
    assert controlled["user_surface_completion_floor"] == 384


def test_mlx_client_retains_surface_control_receipt_from_worker_response():
    from core.brain.llm.mlx_client import MLXLocalClient
    from core.brain.llm.model_registry import get_model_runtime_assignment

    assignment = get_model_runtime_assignment("Aura-32B-20260510-151144")
    client = MLXLocalClient(
        assignment.model_path,
        runtime_assignment=assignment,
    )

    client._record_surface_control_receipt_from_response(
        {
            "status": "ok",
            "surface_control_receipt": {
                "enabled": True,
                "live_mind_controls_bound": True,
                "clean_user_surface_contract": True,
                "surface_alpha_applied": 0.22,
                "surface_alpha_applied_ok": True,
                "recurrent_runtime_loops_applied": 2,
                "recurrent_runtime_loops_applied_ok": True,
                "surface_quality_gate_enabled": True,
                "surface_quality_gate_passed": True,
                "surface_quality_gate_attempts": 1,
                "surface_quality_gate_reasons": ["retryable_draft"],
                "continuation_resume_requested": False,
                "continuation_resume_applied": False,
                "continuation_resume_available": True,
                "continuation_resume_handle": "c" * 32,
                "authorship_replacement_applied": False,
                "text_mutations": [
                    {
                        "stage": "worker.runtime_grounding",
                        "method": "replace_visible_answer",
                        "reasons": ["verified_runtime_fact"],
                        "deterministic": True,
                        "authorship_effect": "replaced_by_runtime",
                        "before_chars": 5,
                        "after_chars": 7,
                        "before_sha256": "a" * 64,
                        "after_sha256": "b" * 64,
                    }
                ],
                "applied": True,
                "untrusted_extra": "drop-me",
            },
        }
    )

    receipt = client.get_last_surface_control_receipt()
    assert receipt["enabled"] is True
    assert receipt["live_mind_controls_bound"] is True
    assert receipt["clean_user_surface_contract"] is True
    assert receipt["surface_alpha_applied"] == 0.22
    assert receipt["recurrent_runtime_loops_applied"] == 2
    assert receipt["surface_quality_gate_enabled"] is True
    assert receipt["surface_quality_gate_passed"] is True
    assert receipt["surface_quality_gate_attempts"] == 1
    assert receipt["surface_quality_gate_reasons"] == ["retryable_draft"]
    assert receipt["continuation_resume_available"] is True
    assert receipt["continuation_resume_handle"] == "c" * 32
    assert receipt["text_mutation_count"] == 1
    assert receipt["authorship_replacement_applied"] is True
    assert receipt["authorship_augmentation_applied"] is False
    assert receipt["model_replacement_applied"] is False
    assert receipt["applied"] is True
    assert "untrusted_extra" not in receipt


def test_mlx_foreground_first_token_watchdog_aborts_tokenless_wall_clock_stall(monkeypatch):
    from core.brain.llm import mlx_client

    timers = []
    degraded = []
    aborted = []

    class FakeTimer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.daemon = False
            self.name = ""
            self.cancelled = False

        def start(self):
            timers.append(self)

        def cancel(self):
            self.cancelled = True

    client = mlx_client.MLXLocalClient("Aura-32B-20260510-151144")
    monkeypatch.setattr(mlx_client._threading, "Timer", FakeTimer)
    monkeypatch.setattr(mlx_client, "_runtime_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        client,
        "_first_token_hard_ceiling",
        lambda *, foreground_request=False: 0.01,
    )
    monkeypatch.setattr(
        client,
        "_record_degraded_event",
        lambda *args, **kwargs: degraded.append((args, kwargs)),
    )
    monkeypatch.setattr(
        client,
        "force_abort_active_generation",
        lambda reason, *, expected_request_id=None: aborted.append(
            (reason, expected_request_id)
        )
        or True,
    )
    client._mark_generation_started("req-live", prompt_chars=32, requested_max_tokens=16)
    client._current_request_started_at = time.time() - 1.0

    timer = client._start_foreground_first_token_watchdog(
        "req-live",
        foreground_request=True,
    )

    assert timer is timers[0]
    assert timer.delay >= 10.0
    timer.callback()
    assert aborted == [("first_token_wall_clock_watchdog", "req-live")]
    assert degraded
    assert degraded[0][0][0] == "first_token_wall_clock_watchdog"


def test_mlx_first_token_ceiling_is_bounded_by_request_deadline(monkeypatch):
    from core.brain.llm import mlx_client

    client = mlx_client.MLXLocalClient("Aura-32B-20260510-151144")
    monkeypatch.setattr(
        client,
        "_first_token_hard_ceiling",
        lambda *, foreground_request=False: 120.0 if foreground_request else 90.0,
    )

    assert client._deadline_bound_first_token_hard_ceiling(
        None,
        foreground_request=True,
    ) == 120.0
    assert client._deadline_bound_first_token_hard_ceiling(
        45.0,
        foreground_request=True,
    ) == 41.0
    # CP126 dec24697: these used to be 10.0 and 10.0 — an 8-second budget
    # produced a ceiling two seconds PAST the deadline, and an already-expired
    # one was granted a fresh ten seconds. The 4-second reserve that lets the
    # caller fail closed and recycle now always holds, and the ceiling never
    # exceeds what the caller actually has left.
    assert client._deadline_bound_first_token_hard_ceiling(
        8.0,
        foreground_request=True,
    ) == 4.0
    assert client._deadline_bound_first_token_hard_ceiling(
        0.0,
        foreground_request=True,
    ) == 0.0


def test_mlx_generation_tracking_carries_deadline_bound_first_token_ceiling():
    from core.brain.llm import mlx_client

    client = mlx_client.MLXLocalClient("Aura-32B-20260510-151144")

    client._mark_generation_started(
        "req-live",
        prompt_chars=32,
        requested_max_tokens=16,
        first_token_hard_ceiling_s=41.0,
    )

    assert client._current_first_token_hard_ceiling_s == 41.0

    client._clear_active_generation_tracking()

    assert client._current_first_token_hard_ceiling_s == 0.0


def test_mlx_force_abort_kills_worker_before_lifecycle_lock_cleanup(monkeypatch):
    from core.brain.llm import mlx_client

    class FakeProcess:
        def __init__(self):
            self.killed = False
            self.joined = False

        def is_alive(self):
            return not self.killed

        def kill(self):
            self.killed = True

        def join(self, timeout=None):
            self.joined = True

    process = FakeProcess()
    client = mlx_client.MLXLocalClient("Aura-32B-20260510-151144")
    client._process = process
    client._active_generations = 1
    client._current_request_started_at = time.time() - 500.0
    monkeypatch.setattr(client, "_replace_ipc_queues", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_record_degraded_event", lambda *args, **kwargs: None)

    client._lock.acquire()
    started = time.time()
    try:
        aborted = client.force_abort_active_generation("test_lock_unavailable_abort")
    finally:
        client._lock.release()

    assert aborted is True
    assert process.killed is True
    assert process.joined is True
    assert time.time() - started < 1.0


def test_mlx_worker_spawn_blocks_32b_when_headroom_is_too_low(monkeypatch):
    from core.brain.llm import mlx_client

    gib = 1024**3
    monkeypatch.setattr(
        mlx_client.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=int(64.0 * gib)),
    )
    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=12.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)
    monkeypatch.delenv("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", raising=False)

    reason = mlx_client._memory_pressure_blocks_worker_spawn(
        "/models/Qwen2.5-32B-Instruct-8bit",
    )

    assert reason is not None
    assert "model_load_headroom" in reason
    assert "required 24.0GB" in reason


def test_mlx_worker_spawn_allows_32b_with_sufficient_headroom(monkeypatch):
    from core.brain.llm import mlx_client

    gib = 1024**3
    monkeypatch.setattr(
        mlx_client.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=int(64.0 * gib)),
    )
    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=24.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)

    assert mlx_client._memory_pressure_blocks_worker_spawn(
        "/models/Qwen2.5-32B-Instruct-8bit",
    ) is None


def test_mlx_worker_spawn_blocks_projected_32b_overcommit(monkeypatch):
    from core.brain.llm import mlx_client

    gib = 1024**3
    monkeypatch.setattr(
        mlx_client.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=int(64.0 * gib)),
    )
    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=28.0,
        process_rss_gb=8.0,
        process_rss_limit_gb=38.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)
    monkeypatch.delenv("AURA_MLX_32B_PROJECTED_FOOTPRINT_GB", raising=False)

    reason = mlx_client._memory_pressure_blocks_worker_spawn(
        "/models/Qwen2.5-32B-Instruct-8bit",
    )

    assert reason is not None
    assert "projected_process_tree_rss:8.0GB+35.0GB+reserve3.0GB=46.0GB" in reason


def test_resident_32b_declared_peak_fits_host_derived_desktop_envelope(monkeypatch):
    """A 64GB desktop must be able to admit its declared resident cortex."""
    from core.brain.llm import mlx_client
    from core.runtime.desktop_boot_safety import compute_process_rss_limit

    gib = 1024**3
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.delenv("AURA_PROCESS_RSS_LIMIT_GB", raising=False)
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            refuse_heavy_local_generation=False,
            available_gb=28.0,
            process_rss_gb=25.6,
            process_rss_limit_gb=compute_process_rss_limit(64 * gib) / gib,
            reason="",
        ),
    )
    monkeypatch.setattr(mlx_client, "_model_load_min_available_gb", lambda _path: 16.0)
    monkeypatch.setattr(mlx_client, "_declared_mlx_worker_footprint_gb", lambda _path: 25.3)
    monkeypatch.setattr(mlx_client, "_model_process_reserve_gb", lambda _path: 3.0)

    assert mlx_client._memory_pressure_blocks_worker_spawn("Aura-32B") is None


def test_resident_32b_admission_still_rejects_when_host_reserve_is_gone(monkeypatch):
    from core.brain.llm import mlx_client

    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            refuse_heavy_local_generation=True,
            available_gb=3.0,
            process_rss_gb=42.0,
            process_rss_limit_gb=51.8,
            reason="memory_pressure:95.0%/3.0GB",
        ),
    )
    monkeypatch.setattr(mlx_client, "_model_load_min_available_gb", lambda _path: 16.0)

    assert (
        mlx_client._memory_pressure_blocks_worker_spawn("Aura-32B") == "memory_pressure:95.0%/3.0GB"
    )


def test_mlx_worker_spawn_blocks_when_unified_guard_refuses(monkeypatch):
    from core.brain.llm import mlx_client

    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=True,
        available_gb=40.0,
        reason="process_tree_rss:54GB/48GB",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)

    reason = mlx_client._memory_pressure_blocks_worker_spawn(
        "/models/Qwen2.5-72B-Instruct-4bit",
    )

    assert reason == "process_tree_rss:54GB/48GB"


def test_mlx_worker_spawn_blocks_72b_on_64gb_without_large_free_headroom(monkeypatch):
    from core.brain.llm import mlx_client

    gib = 1024**3
    monkeypatch.setattr(
        mlx_client.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=int(64.0 * gib)),
    )
    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=48.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)
    monkeypatch.delenv("AURA_MLX_72B_LOAD_MIN_AVAILABLE_GB", raising=False)

    reason = mlx_client._memory_pressure_blocks_worker_spawn(
        "/models/Qwen2.5-72B-Instruct-4bit",
    )

    assert reason is not None
    assert "model_load_headroom" in reason
    assert "required 52.0GB" in reason


def test_mlx_worker_spawn_blocks_72b_projected_process_overcommit(monkeypatch):
    from core.brain.llm import mlx_client

    gib = 1024**3
    monkeypatch.setattr(
        mlx_client.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=int(128.0 * gib)),
    )
    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=72.0,
        process_rss_gb=12.0,
        process_rss_limit_gb=48.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)
    monkeypatch.delenv("AURA_MLX_72B_PROJECTED_FOOTPRINT_GB", raising=False)

    reason = mlx_client._memory_pressure_blocks_worker_spawn(
        "/models/Qwen2.5-72B-Instruct-4bit",
    )

    assert reason is not None
    assert "projected_process_tree_rss:12.0GB+41.0GB+reserve5.0GB=58.0GB" in reason


def test_mlx_worker_spawn_uses_auto_projection_from_local_artifact(monkeypatch, tmp_path):
    from core.brain.llm import mlx_client

    gib = 1024**3
    model_dir = tmp_path / "Aura-32B-20260510-151144"
    model_dir.mkdir()
    weights = model_dir / "weights.safetensors"
    with weights.open("wb") as handle:
        handle.truncate(int(17.0 * gib))

    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=30.0,
        process_rss_gb=8.0,
        process_rss_limit_gb=40.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)
    monkeypatch.setenv("AURA_MLX_32B_PROJECTED_FOOTPRINT_GB", "auto")

    assert mlx_client._memory_pressure_blocks_worker_spawn(str(model_dir)) is None

    snapshot_tight = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=30.0,
        process_rss_gb=15.0,
        process_rss_limit_gb=40.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot_tight)

    reason = mlx_client._memory_pressure_blocks_worker_spawn(str(model_dir))

    assert reason is not None
    assert "projected_process_tree_rss" in reason


def test_worker_memory_sentinel_uses_bounded_heavy_lane_limits(monkeypatch):
    from core.brain.llm.mlx_worker import WorkerMemorySentinel

    writer = SimpleNamespace(put=lambda _payload: None)
    sentinel_32b = WorkerMemorySentinel(writer, "/models/Qwen2.5-32B-Instruct-8bit")
    sentinel_72b = WorkerMemorySentinel(writer, "/models/Qwen2.5-72B-Instruct-4bit")

    assert sentinel_32b._worker_rss_limit_gb(64.0) <= 36.0
    assert sentinel_72b._worker_rss_limit_gb(64.0) <= 40.0

    # The override is clamped to the default under the desktop resource
    # guard, which reads AURA_SAFE_BOOT_DESKTOP / AURA_DESKTOP_RESOURCE_GUARD
    # from the process environment. A test that leaves either set turned this
    # assertion into a coin flip — passing alone, failing in company. Pin the
    # guard state this case is about.
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_DESKTOP_RESOURCE_GUARD", raising=False)
    monkeypatch.setenv("AURA_MLX_WORKER_RSS_LIMIT_GB", "44")
    assert sentinel_32b._worker_rss_limit_gb(64.0) == 44.0


def test_worker_memory_sentinel_clamps_override_in_desktop_safe_boot(monkeypatch):
    from core.brain.llm.mlx_worker import WorkerMemorySentinel

    writer = SimpleNamespace(put=lambda _payload: None)
    sentinel_32b = WorkerMemorySentinel(writer, "/models/Qwen2.5-32B-Instruct-8bit")

    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)
    monkeypatch.setenv("AURA_MLX_WORKER_RSS_LIMIT_GB", "44")

    assert sentinel_32b._worker_rss_limit_gb(64.0) <= 36.0


def test_desktop_safe_boot_keeps_bounded_primary_prompt_cache(monkeypatch):
    """The guard bounds the cache instead of zeroing it.

    Budget 0 under the desktop guard forced every conversation turn to
    re-prefill the whole history; per-turn latency grew with context and
    the endurance runs saturated to the turn timeout by turn ~9-15 (the
    "15-turn resident ceiling", endurance-0715-clean).

    This asserts the PROPERTY — nonzero and memory-bounded — rather than a
    specific entry count. It used to pin the count at 2 as a proxy for "RAM
    stays bounded", because the entry count was the only bound there was. Total
    retained KV is now capped directly, which is a strictly stronger bound, so
    the entry count is free to be large enough that distinct prompt families
    stop evicting each other.
    """
    from core.brain.llm.mlx_worker import (
        _prompt_cache_entry_budget_for_model,
        _prompt_cache_entry_token_cap_for_model,
        _prompt_cache_kv_bytes_per_token,
        _prompt_cache_total_token_budget_for_model,
    )

    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")

    primary = "/models/Qwen2.5-32B-Instruct-8bit"
    assert _prompt_cache_entry_budget_for_model(primary) > 0, "the cache must not be zeroed"
    assert _prompt_cache_entry_token_cap_for_model(primary) == 6144
    worst_case_bytes = (
        _prompt_cache_total_token_budget_for_model(primary)
        * _prompt_cache_kv_bytes_per_token(primary)
    )
    assert worst_case_bytes <= 4 * 1024**3, (
        f"retained KV is not bounded: {worst_case_bytes / 1024**3:.1f}GB"
    )
    # The 72B lane stays cacheless: its KV would dwarf the envelope.
    assert _prompt_cache_entry_budget_for_model("/models/Qwen2.5-72B-Instruct-4bit") == 0


def test_hybrid_prompt_cache_budget_uses_checkpoint_geometry(tmp_path, monkeypatch):
    from core.brain.llm import mlx_worker
    from core.brain.llm.model_artifact_profile import (
        reset_model_artifact_profile_cache,
    )

    model = tmp_path / "Aura-27B-hybrid"
    model.mkdir()
    layer_types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(64)
    ]
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "dtype": "bfloat16",
                    "mamba_ssm_dtype": "float32",
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                    "num_hidden_layers": 64,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "vocab_size": 248320,
                    "max_position_embeddings": 262144,
                    "layer_types": layer_types,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 48,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4,
                },
            }
        )
    )
    monkeypatch.setattr(
        mlx_worker,
        "_qualified_serving_limits_for_model",
        lambda _path: SimpleNamespace(served_context_tokens=32768),
    )
    reset_model_artifact_profile_cache()
    try:
        footprint = mlx_worker._prompt_cache_footprint_for_model(str(model))
        expected_fixed = 48 * (
            3 * (2 * 16 * 128 + 48 * 128) * 2
            + 48 * 128 * 128 * 4
        )
        assert footprint.measured is True
        assert footprint.kv_bytes_per_token == 64 * 1024
        assert footprint.fixed_bytes_per_entry == expected_fixed
        assert mlx_worker._prompt_cache_entry_token_cap_for_model(str(model)) == 32768
        total_tokens = mlx_worker._prompt_cache_total_token_budget_for_model(
            str(model)
        )
        assert (
            total_tokens * footprint.kv_bytes_per_token
            + footprint.fixed_bytes_per_entry
            <= 3 * 1024**3
        )
        assert mlx_worker._load_effective_context_window(str(model)) == 32768
    finally:
        reset_model_artifact_profile_cache()


def test_prompt_cache_byte_budget_counts_fixed_recurrent_state():
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(
        max_size=4,
        max_total_tokens=1000,
        kv_bytes_per_token=1,
        fixed_bytes_per_entry=100,
        max_total_bytes=250,
    )
    key = (4, "user_surface")
    lru.insert_cache(key, list(range(10)), ["first"])
    lru.insert_cache(key, list(range(20, 30)), ["second"])
    lru.insert_cache(key, list(range(40, 50)), ["third"])

    assert lru.retained_entries() == 2
    assert lru.retained_bytes() == 220


def test_probe_and_proof_lanes_bypass_but_user_surface_reuses():
    """Probes/proof contracts never touch the cache; user turns may.

    clean_user_surface_contract used to force a bypass — but every live
    user turn carries it, so the conversation path could never reuse KV
    and each turn re-prefilled the entire history. User turns now get a
    partitioned scope instead of no cache.
    """
    from core.brain.llm.mlx_worker import (
        _job_requires_prompt_cache_bypass,
        _prompt_cache_scope_for_job,
    )

    assert _job_requires_prompt_cache_bypass({"health_probe": True}) is True
    assert _job_requires_prompt_cache_bypass({"proof_evaluation_contract": True}) is True
    assert _job_requires_prompt_cache_bypass({"strict_answer_contract": True}) is True
    assert _job_requires_prompt_cache_bypass({"clean_user_surface_contract": True}) is False
    assert _job_requires_prompt_cache_bypass({"action": "generate"}) is False

    assert _prompt_cache_scope_for_job({"clean_user_surface_contract": True}) == "user_surface"
    assert _prompt_cache_scope_for_job({"action": "generate"}) == "default"


def test_prompt_cache_scopes_are_partitioned_and_entries_capped():
    """A default-scope entry must be invisible to the user-surface scope,
    and an over-cap prompt must not be retained."""
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=2, max_entry_tokens=4)
    fake_cache = ["kv"]
    default_key = (1, "default")
    surface_key = (1, "user_surface")

    lru.insert_cache(default_key, [1, 2, 3], fake_cache)
    hit, remaining = lru.fetch_nearest_cache(
        surface_key, [1, 2, 3],
        can_trim_prompt_cache=lambda _pc: False,
        trim_prompt_cache=lambda _pc, _n: None,
    )
    assert hit is None and remaining == [1, 2, 3], (
        "user-surface fetch must not see default-scope KV"
    )
    hit, remaining = lru.fetch_nearest_cache(
        default_key, [1, 2, 3],
        can_trim_prompt_cache=lambda _pc: True,
        trim_prompt_cache=lambda _pc, _n: None,
    )
    assert hit == fake_cache and remaining == [3], (
        "an exact hit replays the final token; mlx_lm cannot decode from an "
        "empty prompt"
    )

    lru.insert_cache(surface_key, [1, 2, 3, 4, 5], fake_cache)
    hit, _ = lru.fetch_nearest_cache(
        surface_key, [1, 2, 3, 4, 5],
        can_trim_prompt_cache=lambda _pc: False,
        trim_prompt_cache=lambda _pc, _n: None,
    )
    assert hit is None, "an over-cap prompt must not be retained"


def test_prompt_cache_prefix_reuse_shrinks_the_next_turns_prefill():
    """The endurance mechanism itself: turn N+1's prompt extends turn N's
    prompt+reply, so only the NEW suffix should need prefill."""
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=2)
    key = (7, "user_surface")
    turn1_tokens = list(range(100))
    lru.insert_cache(key, turn1_tokens, ["kv-turn1"])

    turn2_tokens = turn1_tokens + list(range(1000, 1040))
    hit, remaining = lru.fetch_nearest_cache(
        key, turn2_tokens,
        can_trim_prompt_cache=lambda _pc: False,
        trim_prompt_cache=lambda _pc, _n: None,
    )
    assert hit == ["kv-turn1"]
    assert remaining == list(range(1000, 1040)), (
        "prefill must shrink to the new turn's suffix, not the whole history"
    )


def test_mlx_never_reports_the_prompt_cache_back_so_the_worker_must_own_it():
    """The bug this pins: the worker harvested ``response.prompt_cache``.

    mlx_lm's ``GenerationResponse`` has no such field and never has, so
    ``final_prompt_cache`` stayed None, ``insert_cache`` was never reached, and
    the whole prompt cache was dead code — measured live at 0 hits in 92
    lookups while every turn re-prefilled its entire history. Reuse only works
    if the worker CREATES the cache object and keeps its own reference.
    """
    import dataclasses

    from mlx_lm.generate import GenerationResponse
    from mlx_lm.models.cache import make_prompt_cache

    fields = {f.name for f in dataclasses.fields(GenerationResponse)}
    assert "prompt_cache" not in fields, (
        "if mlx_lm ever starts reporting the cache, revisit worker ownership"
    )
    assert callable(make_prompt_cache), (
        "the worker depends on being able to mint its own cache on a miss"
    )


def test_internal_lane_churn_cannot_evict_the_user_conversation():
    """Aura runs dozens of internal generations per minute beside a
    conversation. With one global eviction queue they threw the conversation's
    entry away immediately, so the reuse that decides endurance never survived
    to the next user turn."""
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=2)
    surface_key = (7, "user_surface")
    default_key = (7, "default")
    turn1 = list(range(200))
    lru.insert_cache(surface_key, turn1, ["kv-conversation"])

    # A minute of loop ticks, enrichment, dreaming, health probes.
    for tick in range(25):
        lru.insert_cache(default_key, [900_000 + tick, tick, tick + 1], ["kv-internal"])

    hit, remaining = lru.fetch_nearest_cache(
        surface_key, turn1 + list(range(5000, 5030)),
        can_trim_prompt_cache=lambda _pc: True,
        trim_prompt_cache=lambda _pc, _n: None,
    )
    assert hit == ["kv-conversation"], (
        "internal-lane churn evicted the user conversation's KV"
    )
    assert remaining == list(range(5000, 5030))


def test_clean_retry_only_clears_the_failing_prompt_cache_scope():
    """An internal retry cannot make the next live conversation turn cold."""
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=12, max_total_tokens=100_000, kv_bytes_per_token=1)
    surface_key = (7, "user_surface")
    default_key = (7, "default")
    surface_tokens = list(range(200))
    lru.insert_cache(surface_key, surface_tokens, ["kv-conversation"])
    lru.insert_cache(default_key, [800, 801, 802], ["kv-internal"])

    lru.clear_model_key(default_key)

    internal_hit, _ = lru.fetch_nearest_cache(
        default_key,
        [800, 801, 802, 803],
        can_trim_prompt_cache=lambda _pc: True,
        trim_prompt_cache=lambda _pc, _n: None,
    )
    surface_hit, remaining = lru.fetch_nearest_cache(
        surface_key,
        surface_tokens + [999],
        can_trim_prompt_cache=lambda _pc: True,
        trim_prompt_cache=lambda _pc, _n: None,
    )

    assert internal_hit is None
    assert surface_hit == ["kv-conversation"]
    assert remaining == [999]


def test_lane_budgets_never_exceed_the_models_entry_budget():
    """Reserving a slot for the conversation must not raise total KV RAM:
    per-entry cost is fixed, so the lane budgets have to sum to max_size."""
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    for max_size in (1, 2, 6, 12):
        lru = _PromptCacheLRU(max_size=max_size)
        lanes = {
            lru._lane_of((1, scope)) for scope in ("user_surface", "default")
        }
        total = sum(lru._lane_budget(lane) for lane in lanes)
        assert total <= max_size, f"lane budgets overspent at max_size={max_size}"
        assert all(lru._lane_budget(lane) >= 1 for lane in lanes), (
            "a lane with a queue needs at least one slot"
        )

    lru = _PromptCacheLRU(max_size=2)
    for entry in range(6):
        lru.insert_cache((1, "default"), [entry, entry + 1], ["kv"])
    for entry in range(6):
        lru.insert_cache((1, "user_surface"), [500 + entry, entry], ["kv"])
    held = sum(len(queue) for queue in lru._lru.values())
    assert held <= 2, f"prompt cache retained {held} entries against a budget of 2"


def test_optional_deep_solver_memory_refusal_stays_noncritical(monkeypatch):
    from core.brain.llm import mlx_client
    from core.brain.llm.model_runtime_roles_for_tests import assignment_for

    events = []
    # The refusal is only "optional" on the SOLVER lane, and since CP941 a
    # synthetic path is assigned auxiliary/best_effort. Saying the role out
    # loud is what CP941 asked callers to do.
    deep_path = "/models/Qwen2.5-72B-Instruct-4bit"
    client = mlx_client.MLXLocalClient(
        deep_path, runtime_assignment=assignment_for(deep_path, role="solver")
    )
    monkeypatch.setattr(
        client,
        "_record_degraded_event",
        lambda reason, **kwargs: events.append((reason, kwargs)),
    )

    # CP126 5ce89b9e: the handler takes the EXCEPTION now. A substring of an
    # arbitrary message could be spelled by accident; a type cannot.
    handled = client._handle_optional_deep_solver_memory_refusal(
        mlx_client.ModelLoadAdmissionRefused(
            "model_load_headroom:25.5GB < required 52.0GB"
        )
    )

    assert handled is True
    assert client._lane_state == "cold"
    assert client._init_future is None
    assert client._consecutive_spawn_failures == 0
    assert client._spawn_backoff_until > time.time()
    assert events == [
        (
            "optional_deep_solver_memory_refusal",
            {
                "detail": (
                    "Qwen2.5-72B-Instruct-4bit:"
                    "memory_pressure_refused_worker_spawn:model_load_headroom:25.5GB < required 52.0GB"
                ),
                "severity": "warning",
                "foreground_request": False,
                "classification": "non_critical_fallback",
            },
        )
    ]
    assert client._classify_failure(
        reason="memory_pressure_refused_worker_spawn:model_load_headroom",
    ) == "non_critical_fallback"


def test_optional_deep_solver_handler_ignores_primary_32b_memory_refusal(monkeypatch):
    from core.brain.llm import mlx_client

    events = []
    client = mlx_client.MLXLocalClient("/models/Qwen2.5-32B-Instruct-8bit")
    monkeypatch.setattr(
        client,
        "_record_degraded_event",
        lambda reason, **kwargs: events.append((reason, kwargs)),
    )

    handled = client._handle_optional_deep_solver_memory_refusal(
        mlx_client.ModelLoadAdmissionRefused(
            "model_load_headroom:12.0GB < required 24.0GB"
        )
    )

    assert handled is False
    assert events == []


def test_optional_deep_solver_handler_ignores_an_unrelated_failure_that_quotes_it(
    monkeypatch,
):
    """The live hazard: a wrapped traceback carrying the refusal text."""
    from core.brain.llm import mlx_client

    events = []
    client = mlx_client.MLXLocalClient("/models/Qwen2.5-72B-Instruct-4bit")
    monkeypatch.setattr(
        client,
        "_record_degraded_event",
        lambda reason, **kwargs: events.append((reason, kwargs)),
    )

    handled = client._handle_optional_deep_solver_memory_refusal(
        RuntimeError(
            "worker crashed while logging "
            "memory_pressure_refused_worker_spawn:model_load_headroom"
        )
    )

    assert handled is False, "a crash is not an admission decision"
    assert events == []


def test_mlx_worker_accepts_zeroed_shared_substrate_for_affective_sync():
    source = open("core/brain/llm/mlx_worker.py", encoding="utf-8").read()

    assert "if substrate_mem is not None:" in source
    assert "engine.start_substrate_sync(shared_state=substrate_mem)" in source
    assert '"steering_active": bool(_steering_active)' in source


def test_mlx_client_records_worker_steering_liveness_receipt():
    source = open("core/brain/llm/mlx_client.py", encoding="utf-8").read()

    assert 'raw_steering = res.get("steering_active")' in source
    # Strict receipt typing: only an actual bool may activate the shared
    # steering channels (bool("false") is True — CP126 finding).
    assert "if isinstance(raw_steering, bool):" in source
    assert "self._steering_active.value = steering_active" in source
    assert "self._substrate_mem[-1] = 1.0 if steering_active else 0.0" in source
    assert "self._steering_liveness_observed = True" in source


# ── pressure-adaptive token-progress budgets ─────────────────────────────────
# Under unified-memory contention a resident heavy model's first token slows
# because prompt eval competes for bandwidth; killing it pays a ~20GB reload
# that deepens the contention (the Jul 7 soak doom loop). These tests pin the
# bounded stretch: heavy lanes only, emergency excluded, caller deadlines
# still dominate, and the whole feature is env-pinned OFF for the suite.

def _fake_snapshot(level: str):
    tiers = ["warning", "high", "critical", "emergency"]
    idx = tiers.index(level) if level in tiers else -1
    return SimpleNamespace(
        level=level,
        warning=idx >= 0,
        high=idx >= 1,
        critical=idx >= 2,
        emergency=idx >= 3,
    )


def _budget_client(model_path: str = "Qwen2.5-32B-cortex"):
    from core.brain.llm.mlx_client import MLXLocalClient

    return MLXLocalClient(model_path=model_path)


def test_pressure_stretch_is_pinned_off_for_the_suite(monkeypatch):
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("critical"),
    )
    client = _budget_client()
    factor, reason = client._pressure_adaptive_stretch()
    assert factor == 1.0 and reason == ""


def test_pressure_stretch_scales_token_budgets_by_tier(monkeypatch):
    monkeypatch.setenv("AURA_FIRST_TOKEN_PRESSURE_ADAPT", "1")
    client = _budget_client()
    for level, expected in (("warning", 1.2), ("high", 1.35), ("critical", 1.5)):
        monkeypatch.setattr(
            "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
            lambda _level=level, **_kw: _fake_snapshot(_level),
        )
        factor, reason = client._pressure_adaptive_stretch()
        assert factor == expected
        assert reason == f"memory_pressure_{level}"
        assert client._token_stall_after(foreground_request=True) == 40.0 * expected


def test_pressure_stretch_excluded_at_emergency(monkeypatch):
    monkeypatch.setenv("AURA_FIRST_TOKEN_PRESSURE_ADAPT", "1")
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("emergency"),
    )
    client = _budget_client()
    factor, _ = client._pressure_adaptive_stretch()
    assert factor == 1.0  # the refuse-generation path owns emergencies


def test_pressure_stretch_ignores_light_lanes(monkeypatch):
    monkeypatch.setenv("AURA_FIRST_TOKEN_PRESSURE_ADAPT", "1")
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("critical"),
    )
    client = _budget_client(model_path="Qwen2.5-1.5B-reflex")
    factor, _ = client._pressure_adaptive_stretch()
    assert factor == 1.0
    assert client._token_stall_after() == 8.0


def test_hard_ceiling_stretches_bounded_under_pressure(monkeypatch):
    monkeypatch.setenv("AURA_FIRST_TOKEN_PRESSURE_ADAPT", "1")
    monkeypatch.delenv("AURA_FIRST_TOKEN_ABSOLUTE_CEILING_S", raising=False)
    client = _budget_client()
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("critical"),
    )
    stretched = client._first_token_hard_ceiling(foreground_request=True)
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("nominal"),
    )
    baseline = client._first_token_hard_ceiling(foreground_request=True)
    assert baseline < stretched <= baseline * 1.5


def test_caller_deadline_still_dominates_stretch(monkeypatch):
    monkeypatch.setenv("AURA_FIRST_TOKEN_PRESSURE_ADAPT", "1")
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("critical"),
    )
    client = _budget_client()
    bounded = client._deadline_bound_first_token_hard_ceiling(
        20.0, foreground_request=True
    )
    assert bounded <= 16.0  # remaining - reserve, stretch cannot exceed it


def test_stall_receipts_name_the_pressure_tier(monkeypatch):
    client = _budget_client()
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("high"),
    )
    assert client._pressure_receipt_suffix() == ":memory=high"
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("nominal"),
    )
    assert client._pressure_receipt_suffix() == ""


def test_worker_deadline_reserves_a_delivery_margin():
    """A cooperative stop is worthless if nobody is still waiting for it.

    The worker used to receive the caller's FULL remaining budget, so its
    deadline and the gate's expired together: a decode that stopped politely at
    token 149 was abandoned by a gate that had already timed out. Measured live
    as "Cortex consumed 77.5s without usable text" followed immediately by
    "Abort arrived after the generation finished; nothing to abort".
    """

    def worker_budget(remaining_s: float) -> float:
        # Mirrors the computation in MLXClient.generate's deadline plumbing.
        margin = max(1.5, min(6.0, remaining_s * 0.08))
        return max(remaining_s * 0.5, remaining_s - margin)

    for remaining in (10.0, 30.0, 77.3, 120.0, 300.0):
        budget = worker_budget(remaining)
        assert budget < remaining, (
            f"worker must stop before the caller's deadline (remaining={remaining})"
        )
        assert remaining - budget >= 1.5, "delivery margin too small to cross IPC"
        assert budget >= remaining * 0.5, (
            f"margin ate more than half the budget (remaining={remaining})"
        )

    # A long budget must not surrender a proportionally huge margin.
    assert 300.0 - worker_budget(300.0) <= 6.0


def test_oversized_scaffold_is_trimmed_instead_of_failing_the_lane_forever():
    """A prompt over the model's window used to be a permanent dead lane.

    The window was enforced only as a hard refusal in the worker and nothing
    upstream bounded a prompt against it, so a lane that overshot failed on
    every attempt. Measured live: a background swarm turn rendered 41,219
    tokens for a 32,768-token window, and 161,578 of its 171,058 characters
    were ONE system message — 94% scaffold around a 462-character request.
    """
    from core.brain.llm.mlx_worker import _shrink_scaffold_to_context_window

    class _Tok:
        """One token per 4 characters, deterministic."""

        @staticmethod
        def apply_chat_template(messages, tools=None, add_generation_prompt=True, tokenize=False):
            return "\n".join(
                f"<|{m['role']}|>{m['content']}" for m in messages if isinstance(m, dict)
            )

        @staticmethod
        def encode(text):
            return list(range(max(1, len(text) // 4)))

    huge_scaffold = "S" * 160_000
    request = "R" * 460
    messages = [
        {"role": "system", "content": huge_scaffold},
        {"role": "system", "content": "keep me: short operating rules"},
        {"role": "user", "content": request},
    ]
    prompt = _Tok.apply_chat_template(messages)
    tokens = _Tok.encode(prompt)
    window, reserve = 32_768, 92
    assert len(tokens) + reserve > window, "fixture must actually overshoot"

    new_prompt, new_tokens, note = _shrink_scaffold_to_context_window(
        messages=messages,
        prompt=prompt,
        tokens=tokens,
        window=window,
        output_reserve=reserve,
        tokenizer=_Tok,
        tools=None,
    )

    assert note, "a trim must be reported, not applied silently"
    assert len(new_tokens) + reserve <= window, "the trim must actually make it fit"
    # The request survives intact; scaffold is what gets shed.
    assert request in new_prompt, "the user's own turn must never be trimmed away"
    assert "keep me: short operating rules" in new_prompt
    assert "scaffold trimmed to fit" in new_prompt
    # The oversized block keeps its opening instructions rather than vanishing.
    assert "S" * 1000 in new_prompt

    # A prompt that already fits is returned untouched, with no note.
    small = [{"role": "user", "content": "hello"}]
    small_prompt = _Tok.apply_chat_template(small)
    small_tokens = _Tok.encode(small_prompt)
    same_prompt, same_tokens, same_note = _shrink_scaffold_to_context_window(
        messages=small,
        prompt=small_prompt,
        tokens=small_tokens,
        window=window,
        output_reserve=reserve,
        tokenizer=_Tok,
        tools=None,
    )
    assert (same_prompt, same_tokens, same_note) == (small_prompt, small_tokens, "")


def test_a_request_too_large_to_fit_is_still_refused():
    """Trimming scaffold must never become "delete the user's question"."""
    from core.brain.llm.mlx_worker import _shrink_scaffold_to_context_window

    class _Tok:
        @staticmethod
        def apply_chat_template(messages, tools=None, add_generation_prompt=True, tokenize=False):
            return "\n".join(f"<|{m['role']}|>{m['content']}" for m in messages)

        @staticmethod
        def encode(text):
            return list(range(max(1, len(text) // 4)))

    messages = [
        {"role": "system", "content": "small scaffold"},
        {"role": "user", "content": "U" * 400_000},
    ]
    prompt = _Tok.apply_chat_template(messages)
    tokens = _Tok.encode(prompt)

    _, out_tokens, note = _shrink_scaffold_to_context_window(
        messages=messages,
        prompt=prompt,
        tokens=tokens,
        window=32_768,
        output_reserve=92,
        tokenizer=_Tok,
        tools=None,
    )
    assert note == "", "no trim should be claimed when nothing safe can be shed"
    assert len(out_tokens) + 92 > 32_768, "the caller must still see the overflow and refuse"


def test_volatile_grounding_rides_behind_the_conversation_not_ahead_of_it():
    """The endurance property, stated as a cache invariant.

    Grounding that changes every turn (clock, action receipts, felt state) used
    to be appended to the SYSTEM prompt, which is message[0] — ahead of the
    entire conversation. Any churn there invalidates the KV prefix for
    everything behind it. Measured live on the user surface: reuse of 126 of
    1774 tokens (7%), divergence beginning exactly at
    "## WHAT YOU ACTUALLY JUST DID".

    Delivered LAST, divergence lands after the history, so turn N+1 reuses the
    prefix through turn N instead of re-prefilling from token zero. This test
    encodes that as the trie behaviour the worker actually relies on.
    """
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=2)
    key = (11, "user_surface")

    stable_system = list(range(100))          # identity + policy: byte-stable
    history = list(range(200, 260))           # u1 a1 ... uN
    reply_n = list(range(900, 930))           # aN, generated

    def volatile(turn: int) -> list[int]:
        # A block whose content differs every turn.
        return [7000 + turn, 7100 + turn, 7200 + turn]

    # ── Grounding LAST: system, history, volatile, reply ──
    turn_n = stable_system + history + volatile(1) + reply_n
    lru.insert_cache(key, turn_n, ["kv"])
    next_user = list(range(300, 320))
    turn_n_plus_1 = stable_system + history + reply_n + next_user + volatile(2)
    _, remaining = lru.fetch_nearest_cache(
        key, turn_n_plus_1,
        can_trim_prompt_cache=lambda _pc: True,
        trim_prompt_cache=lambda _pc, _n: None,
    )
    reused_last = len(turn_n_plus_1) - len(remaining)

    # ── Grounding FIRST (the old shape): system, volatile, history, reply ──
    lru_old = _PromptCacheLRU(max_size=2)
    old_turn_n = stable_system + volatile(1) + history + reply_n
    lru_old.insert_cache(key, old_turn_n, ["kv"])
    old_turn_n_plus_1 = stable_system + volatile(2) + history + reply_n + next_user
    _, old_remaining = lru_old.fetch_nearest_cache(
        key, old_turn_n_plus_1,
        can_trim_prompt_cache=lambda _pc: True,
        trim_prompt_cache=lambda _pc, _n: None,
    )
    reused_first = len(old_turn_n_plus_1) - len(old_remaining)

    assert reused_first <= len(stable_system), (
        "volatile grounding ahead of the history cannot reuse past it — that is "
        "the wall this test exists to describe"
    )
    assert reused_last >= len(stable_system) + len(history), (
        f"grounding delivered last must reuse the whole history prefix; "
        f"reused {reused_last} of {len(turn_n_plus_1)}"
    )
    # The gain is exactly the history that used to be stranded behind the
    # volatile block — and it grows with the conversation, which is the point.
    assert reused_last - reused_first >= len(history), (
        f"moving grounding last must recover the whole stranded history "
        f"({reused_first} -> {reused_last} tokens, history={len(history)})"
    )


def test_the_sanitizer_does_not_annihilate_replies_over_ordinary_english():
    """A FATAL check must not fire on words people use.

    `_sanitize_telemetry_leakage` returns None to mean "destroy the whole
    answer", and its marker pattern was case-INSENSITIVE over a list that
    included the bare word PROCEEDING plus phrases like "field coherence",
    "system authority" and "memory scar". So any reply containing "proceeding"
    was annihilated. Measured live: a conversational turn produced a 226-token
    draft, the user got "I couldn't get to an answer I'd stand behind on that
    one", and the only log line was "Hallucination detected by sanitizer.
    Returning empty text for caller-side recovery."

    Machine tokens keep their teeth — their identity IS the casing — while the
    natural-language jargon moves to the reliability gate's repairable
    `pseudo_internal_jargon` reason instead of a death sentence.
    """
    from core.brain.llm.mlx_worker import _sanitize_telemetry_leakage as sanitize

    must_survive = (
        "Before proceeding, I'll take a position: forgetting is mostly a mercy.",
        "There's a kind of field coherence to how memories settle over time.",
        "No system authority can decide that for you.",
        "A memory scar is a useful metaphor for what trauma leaves behind.",
        "The Earth's core is around 5,200 degrees Celsius, kept hot by residual "
        "formation heat and the decay of uranium and thorium.",
    )
    for reply in must_survive:
        assert sanitize(reply) is not None, (
            f"ordinary English was treated as model-state corruption: {reply[:56]!r}"
        )
        assert sanitize(reply) == reply, "a surviving reply must not be altered"

    # Genuine leakage signatures still destroy the answer.
    must_die = (
        "PROCEEDING TOOL_ACTION CONVERGE_UNION",
        "The answer is MySelfEpsilon and CanonicalStabilityAnchor.",
        "INTRUSION_DETECTED",
        "ExistenceHash",
    )
    for reply in must_die:
        assert sanitize(reply) is None, f"a real leak survived: {reply[:56]!r}"

    # The other fatal checks are untouched.
    assert sanitize("value " + "1" * 25) is None, "digit-run check must still fire"


def test_live_sanitizer_routes_intact_draft_to_typed_authored_repair():
    from core.brain.llm.mlx_worker import (
        _route_telemetry_sanitizer_draft,
        _surface_quality_failure_reasons,
        _telemetry_sanitization_failure_reasons,
    )

    draft = (
        "I am steady enough to answer directly. ExistenceHash is an internal "
        "identifier that must not appear in the visible reply."
    )
    job = {
        "clean_user_surface_contract": True,
        "user_surface_validation_prompt": "Hey Aura, how are you doing right now?",
    }

    assert _telemetry_sanitization_failure_reasons(draft) == [
        "backend_symbolic_surface_leak"
    ]
    assert _telemetry_sanitization_failure_reasons("value " + "1" * 25) == [
        "unbounded_numeric_identifier"
    ]
    assert _telemetry_sanitization_failure_reasons(
        "/private/a/b/c /private/d/e/f /private/g/h/i"
        + " /private/j/k/l" * 8
    ) == ["telemetry_path_wall"]
    assert _telemetry_sanitization_failure_reasons("xublcate evocer") == [
        "corrupted_language"
    ]
    routed, reasons = _route_telemetry_sanitizer_draft(
        draft,
        is_proof=False,
        authored_surface_repair_available=True,
    )
    assert routed == draft
    assert reasons == ["backend_symbolic_surface_leak"]
    assert "backend_symbolic_surface_leak" in _surface_quality_failure_reasons(
        job,
        routed,
    )


def test_sanitizer_still_destroys_unspeakable_draft_without_owned_repair_lane():
    from core.brain.llm.mlx_worker import _route_telemetry_sanitizer_draft

    routed, reasons = _route_telemetry_sanitizer_draft(
        "PROCEEDING TOOL_ACTION",
        is_proof=True,
        authored_surface_repair_available=False,
    )

    assert routed == ""
    assert reasons == ["backend_symbolic_surface_leak"]


def test_cancelled_live_draft_preserves_typed_repair_custody():
    from core.brain.llm.mlx_worker import _route_cooperative_partial_draft

    draft = "I can answer this, but ExistenceHash must not reach the visible surface."
    state = {
        "surface_quality_gate_enabled": True,
        "surface_quality_gate_passed": False,
        "surface_quality_gate_reasons": [],
        "telemetry_sanitizer_reasons": [],
    }
    job = {
        "clean_user_surface_contract": True,
        "user_surface_validation_prompt": "Explain the result.",
    }

    routed = _route_cooperative_partial_draft(
        job,
        draft,
        state,
        is_proof=False,
    )

    assert routed == draft
    assert state["telemetry_sanitizer_reasons"] == [
        "backend_symbolic_surface_leak"
    ]
    assert state["surface_quality_gate_reasons"] == [
        "backend_symbolic_surface_leak"
    ]
    assert state["surface_quality_rejected_text"] == draft
    assert state["surface_quality_gate_passed"] is False


def test_cancelled_strict_draft_remains_withheld_without_repair_owner():
    from core.brain.llm.mlx_worker import _route_cooperative_partial_draft

    state = {
        "surface_quality_gate_enabled": False,
        "surface_quality_gate_passed": True,
    }
    routed = _route_cooperative_partial_draft(
        {"strict_answer_contract": True},
        "PROCEEDING TOOL_ACTION",
        state,
        is_proof=True,
    )

    assert routed == ""
    assert state["telemetry_sanitizer_reasons"] == [
        "backend_symbolic_surface_leak"
    ]
    assert "surface_quality_rejected_text" not in state


def test_cancelled_clean_live_partial_remains_available_for_completion():
    from core.brain.llm.mlx_worker import _route_cooperative_partial_draft

    state = {
        "surface_quality_gate_enabled": True,
        "surface_quality_gate_passed": False,
    }
    draft = "Dijkstra's invariant is that each settled distance is final."

    assert _route_cooperative_partial_draft(
        {
            "clean_user_surface_contract": True,
            "user_surface_validation_prompt": "Explain Dijkstra's algorithm.",
        },
        draft,
        state,
        is_proof=False,
    ) == draft
    assert state["telemetry_sanitizer_reasons"] == []
    assert "surface_quality_rejected_text" not in state


def test_worker_retry_keeps_stronger_rejected_incumbent() -> None:
    from core.brain.llm.mlx_worker import _remember_surface_quality_rejected_draft

    state = {}
    incumbent = (
        "Dijkstra settles the least tentative distance. The answer continues "
        "with pseudocode, a worked example, and both complexity bounds."
    )
    _remember_surface_quality_rejected_draft(
        state,
        incumbent,
        ["truncated_tail"],
    )
    _remember_surface_quality_rejected_draft(
        state,
        (
            "We need answer user's request. Need explain every requested part. "
            "We must include all five obligations before the final answer."
        ),
        ["internal_task_prompt_leak", "truncated_tail", "unanswered_question_part"],
    )

    assert state["surface_quality_rejected_text"] == incumbent
    assert state["surface_quality_rejected_reasons"] == ["truncated_tail"]


def test_worker_retry_replaces_weaker_rejected_incumbent() -> None:
    from core.brain.llm.mlx_worker import _remember_surface_quality_rejected_draft

    state = {}
    _remember_surface_quality_rejected_draft(
        state,
        "A partial answer that stops here",
        ["truncated_tail", "unanswered_question_part"],
    )
    complete = (
        "Dijkstra's invariant is that a settled distance is final, and "
        "Bellman-Ford is the correct alternative when edges may be negative."
    )
    _remember_surface_quality_rejected_draft(state, complete, ["truncated_tail"])

    assert state["surface_quality_rejected_text"] == complete
    assert state["surface_quality_rejected_reasons"] == ["truncated_tail"]


def test_live_surface_quality_retry_preserves_valid_prefill_cache():
    from pathlib import Path

    source = Path("core/brain/llm/mlx_worker.py").read_text(encoding="utf-8")
    start = source.index(
        "if internal_attempt < max_internal_retries and not surface_wall_exceeded:"
    )
    end = source.index("continue", start)
    retry_branch = source[start:end]

    assert "prompt_cache_lru.clear()" not in retry_branch
    assert "_clear_mlx_cache(mx)" not in retry_branch
    assert "surface_retry_started = time.monotonic()" in retry_branch


def test_volatile_state_context_is_appended_so_the_kv_prefix_stays_cacheable():
    """Volatile grounding LAST, or prompt-cache reuse is worthless.

    Mood/energy/focus/substrate-age change on every single turn. Prepending
    them made the KV prefix diverge inside the first ~20 tokens, so the cache
    reused essentially nothing. Measured live once the cache started working:

      prefix diverges at token 21 (0% of 31718 reused)
      stable head: 'System State Context:\\n[Affect: Current Mood: TIRED (Energy: 0.'

    31,697 tokens re-prefilled because 21 were reusable.
    """
    import inspect

    from core.brain import llm_health_router

    source = inspect.getsource(llm_health_router)
    assert 'f"{system_prompt}\\n\\nSystem State Context:\\n{context_header}"' in source, (
        "the volatile state block must be appended after the stable prompt"
    )
    assert 'f"System State Context:\\n{context_header}\\n\\n{system_prompt}"' not in source, (
        "prepending volatile state destroys prompt-cache reuse for the whole runtime"
    )


def test_prompt_cache_reuse_survives_a_volatile_tail_but_not_a_volatile_head():
    """The mechanism behind the rule, exercised on the real trie."""
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    stable = list(range(1, 401))          # identity + contract: identical every turn
    key = (11, "user_surface")

    # Volatile LAST: turn 1 stored stable+tail+reply; turn 2 shares the stable head.
    lru = _PromptCacheLRU(max_size=2)
    lru.insert_cache(key, stable + [9001, 9002] + [7001, 7002], ["kv"])
    hit, remaining = lru.fetch_nearest_cache(
        key, stable + [9003, 9004],
        can_trim_prompt_cache=lambda _pc: True,
        trim_prompt_cache=lambda _pc, _n: None,
    )
    assert hit == ["kv"]
    assert len(remaining) <= 2, (
        f"a volatile TAIL should leave almost nothing to prefill, got {len(remaining)}"
    )

    # Volatile FIRST: nothing meaningful is reusable, which is the live failure.
    lru = _PromptCacheLRU(max_size=2)
    lru.insert_cache(key, [9001, 9002] + stable + [7001, 7002], ["kv"])
    hit, remaining = lru.fetch_nearest_cache(
        key, [9003, 9004] + stable,
        can_trim_prompt_cache=lambda _pc: True,
        trim_prompt_cache=lambda _pc, _n: None,
    )
    assert len(remaining) >= len(stable), (
        "a volatile HEAD must be shown to destroy reuse — that is why the rule exists"
    )


def test_every_weight_class_caps_per_entry_tokens_and_total_kv():
    """An entry COUNT is not a memory bound.

    The small classes returned 0 (uncapped) per-entry tokens, which was
    harmless only while the cache was broken and nothing was ever stored. Once
    insertion worked, "12 entries, uncapped" meant 12 x however many tokens the
    caller sent — a 31,718-token prompt was measured live and managed RSS grew
    73,963MB/h toward the 49GB ceiling.
    """
    from core.brain.llm.mlx_worker import (
        _prompt_cache_entry_budget_for_model,
        _prompt_cache_entry_token_cap_for_model,
        _prompt_cache_kv_bytes_per_token,
        _prompt_cache_total_token_budget_for_model,
    )

    for path in (
        "Aura-32B-crsm-closeout-jul1-20260701-215118",
        "Qwen2.5-7B-Instruct-4bit",
        "Qwen2.5-1.5B-Instruct-4bit",
        "Qwen2.5-14B-Instruct-4bit",
    ):
        entries = _prompt_cache_entry_budget_for_model(path)
        if entries <= 0:
            continue  # nothing retained at all; no bound needed
        cap = _prompt_cache_entry_token_cap_for_model(path)
        assert cap > 0, f"{path} retains entries with no per-entry token cap"
        total = _prompt_cache_total_token_budget_for_model(path)
        assert total > 0, f"{path} has no total token budget"
        per_token = _prompt_cache_kv_bytes_per_token(path)
        assert per_token > 0
        worst_case_gb = (total * per_token) / (1024 ** 3)
        assert worst_case_gb <= 4.0, (
            f"{path} can retain {worst_case_gb:.1f}GB of KV — not a bound"
        )


def test_total_token_budget_evicts_and_drains_internal_lanes_before_the_conversation():
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(
        max_size=8,
        max_entry_tokens=0,
        max_total_tokens=1000,
        kv_bytes_per_token=1024,
    )
    surface = (5, "user_surface")
    internal = (5, "default")

    lru.insert_cache(surface, list(range(1, 501)), ["kv-conversation"])
    for batch in range(4):
        lru.insert_cache(internal, list(range(10_000 + batch * 400, 10_400 + batch * 400)), ["kv-bg"])

    assert lru.retained_tokens() <= 1000, (
        f"total token budget not enforced: {lru.retained_tokens()} tokens retained"
    )
    assert lru.retained_bytes() == lru.retained_tokens() * 1024

    # The conversation's prefix must be the last thing surrendered.
    hit, _ = lru.fetch_nearest_cache(
        surface, list(range(1, 501)) + [77],
        can_trim_prompt_cache=lambda _pc: True,
        trim_prompt_cache=lambda _pc, _n: None,
    )
    assert hit == ["kv-conversation"], (
        "internal-lane pressure drained the user conversation's KV first"
    )


def test_prompt_cache_exposes_the_oom_ladders_missing_rung():
    """The verifier warned on every boot that the ladder had NO rungs."""
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=4, max_total_tokens=10_000, kv_bytes_per_token=2048)
    lru.insert_cache((1, "user_surface"), list(range(300)), ["kv"])
    lru.insert_cache((1, "default"), list(range(1000, 1200)), ["kv"])

    assert lru.retained_entries() == 2
    assert lru.retained_tokens() == 500
    expected = 500 * 2048
    assert lru.retained_bytes() == expected

    freed = lru.shed()
    assert freed == expected, "shed must report the bytes it actually released"
    assert lru.retained_tokens() == 0
    assert lru.retained_entries() == 0


def test_prompt_cache_resume_capability_is_exact_scoped_and_one_use():
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=4)
    key = (7, "user_surface")
    tokens = [10, 20, 30, 40]
    trimmed = []
    lru.insert_cache(key, tokens, ["exact-kv"])

    handle = lru.bind_resume(key, tokens)
    cache, remaining, resumed, failure = lru.fetch_resume(
        handle,
        key,
        can_trim_prompt_cache=lambda _cache: True,
        trim_prompt_cache=lambda _cache, count: trimmed.append(count),
    )

    assert cache == ["exact-kv"]
    assert remaining == [40]
    assert resumed == tokens
    assert failure == ""
    assert trimmed == [1]

    cache, _remaining, _resumed, failure = lru.fetch_resume(
        handle,
        key,
        can_trim_prompt_cache=lambda _cache: True,
        trim_prompt_cache=lambda _cache, _count: None,
    )
    assert cache is None
    assert failure == "unknown_or_expired_handle"


def test_prompt_cache_resume_appends_new_turn_tokens_after_exact_replay():
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=4)
    key = (8, "user_surface")
    tokens = [10, 20, 30, 40]
    lru.insert_cache(key, tokens, ["exact-kv"])
    handle = lru.bind_resume(key, tokens, context_digest="authority-v1")

    cache, remaining, resumed, failure = lru.fetch_resume(
        handle,
        key,
        can_trim_prompt_cache=lambda _cache: True,
        trim_prompt_cache=lambda _cache, _count: None,
        append_tokens=[40, 50, 60],
        context_digest="authority-v1",
    )

    assert cache == ["exact-kv"]
    assert remaining == [40, 50, 60]
    assert resumed == [10, 20, 30, 40, 50, 60]
    assert failure == ""


def test_prompt_cache_resume_refuses_an_append_from_a_different_chat_boundary():
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=4)
    key = (8, "user_surface")
    tokens = [10, 20, 30, 40]
    lru.insert_cache(key, tokens, ["exact-kv"])
    handle = lru.bind_resume(key, tokens, context_digest="wire-v1")

    cache, remaining, resumed, failure = lru.fetch_resume(
        handle,
        key,
        can_trim_prompt_cache=lambda _cache: True,
        trim_prompt_cache=lambda _cache, _count: None,
        append_tokens=[99, 50, 60],
        context_digest="wire-v1",
    )

    assert cache is None
    assert remaining == []
    assert resumed == []
    assert failure == "append_boundary_final_token_mismatch"
    assert lru.retained_entries() == 1


def test_prompt_cache_resume_refuses_changed_authority_context():
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=4)
    key = (8, "user_surface")
    tokens = [10, 20, 30, 40]
    lru.insert_cache(key, tokens, ["exact-kv"])
    handle = lru.bind_resume(key, tokens, context_digest="authority-v1")

    cache, remaining, resumed, failure = lru.fetch_resume(
        handle,
        key,
        can_trim_prompt_cache=lambda _cache: True,
        trim_prompt_cache=lambda _cache, _count: None,
        append_tokens=[50],
        context_digest="authority-v2",
    )

    assert cache is None
    assert remaining == []
    assert resumed == []
    assert failure == "context_mismatch"
    assert lru.retained_entries() == 1


def test_prompt_cache_resume_capability_refuses_cross_lane_and_clear_invalidates():
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=4)
    key = (9, "user_surface")
    tokens = [1, 2, 3]
    lru.insert_cache(key, tokens, ["kv"])
    handle = lru.bind_resume(key, tokens)

    cache, _remaining, _resumed, failure = lru.fetch_resume(
        handle,
        (9, "default"),
        can_trim_prompt_cache=lambda _cache: True,
        trim_prompt_cache=lambda _cache, _count: None,
    )
    assert cache is None
    assert failure == "model_or_lane_mismatch"

    handle = lru.bind_resume(key, tokens)
    lru.clear()
    cache, _remaining, _resumed, failure = lru.fetch_resume(
        handle,
        key,
        can_trim_prompt_cache=lambda _cache: True,
        trim_prompt_cache=lambda _cache, _count: None,
    )
    assert cache is None
    assert failure == "unknown_or_expired_handle"


def test_prompt_cache_resume_capability_remains_fetchable_at_binding_limit():
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=2)
    key = (11, "user_surface")
    tokens = [4, 5, 6]
    lru.insert_cache(key, tokens, ["kv"])

    first = lru.bind_resume(key, tokens)
    # Independent capabilities must own independent mutable KV objects. A
    # second handle may not alias the first handle's state.
    lru.insert_cache(key, tokens, ["kv-second"])
    second = lru.bind_resume(key, tokens)
    assert first and second

    cache, remaining, resumed, failure = lru.fetch_resume(
        first,
        key,
        can_trim_prompt_cache=lambda _cache: True,
        trim_prompt_cache=lambda _cache, _count: None,
    )

    assert cache == ["kv"]
    assert remaining == [6]
    assert resumed == tokens
    assert failure == ""


def test_prompt_cache_resume_capability_trims_a_real_mlx_kv_cache():
    import mlx.core as mx
    from mlx_lm.models.cache import (
        KVCache,
        can_trim_prompt_cache,
        trim_prompt_cache,
    )

    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=2)
    key = (13, "user_surface")
    tokens = [31, 32, 33, 34, 35]
    cache = [KVCache()]
    cache[0].update_and_fetch(
        mx.ones((1, 2, len(tokens), 4)),
        mx.ones((1, 2, len(tokens), 4)) * 2,
    )
    mx.eval(cache[0].keys, cache[0].values)
    lru.insert_cache(key, tokens, cache)

    handle = lru.bind_resume(key, tokens)
    resumed_cache, remaining, resumed, failure = lru.fetch_resume(
        handle,
        key,
        can_trim_prompt_cache=can_trim_prompt_cache,
        trim_prompt_cache=trim_prompt_cache,
    )

    assert failure == ""
    assert remaining == [tokens[-1]]
    assert resumed == tokens
    assert resumed_cache is not None
    assert resumed_cache[0].offset == len(tokens) - 1


def test_prompt_cache_resume_owns_final_state_when_general_cache_did_not_retain_it():
    import mlx.core as mx
    from mlx_lm.models.cache import (
        KVCache,
        can_trim_prompt_cache,
        trim_prompt_cache,
    )

    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=4, max_entry_tokens=32)
    key = (17, "user_surface")
    tokens = list(range(12))
    final_cache = [KVCache()]
    final_cache[0].update_and_fetch(
        mx.ones((1, 2, len(tokens), 4)),
        mx.ones((1, 2, len(tokens), 4)) * 2,
    )
    mx.eval(final_cache[0].keys, final_cache[0].values)

    assert lru._search(key, tokens).exact is None
    handle = lru.bind_resume(key, tokens, prompt_cache=final_cache)
    assert handle
    assert lru.retained_entries() == 1
    assert lru.retained_tokens() == len(tokens)

    resumed_cache, remaining, resumed, failure = lru.fetch_resume(
        handle,
        key,
        can_trim_prompt_cache=can_trim_prompt_cache,
        trim_prompt_cache=trim_prompt_cache,
    )

    assert failure == ""
    assert remaining == [tokens[-1]]
    assert resumed == tokens
    assert resumed_cache is final_cache
    assert resumed_cache[0].offset == len(tokens) - 1
    assert lru.retained_entries() == 0
    assert lru.retained_tokens() == 0


def test_prompt_cache_resume_rewinds_real_hybrid_kv_and_recurrent_state():
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    from core.brain.llm.mlx_worker import (
        _capture_prompt_cache_one_token_rollback,
        _PromptCacheLRU,
    )

    lru = _PromptCacheLRU(
        max_size=2,
        max_total_tokens=100,
        kv_bytes_per_token=16,
    )
    key = (23, "user_surface")
    tokens = [41, 42, 43, 44]
    kv_cache = KVCache()
    recurrent_cache = ArraysCache(size=2)

    kv_cache.update_and_fetch(
        mx.ones((1, 2, len(tokens) - 1, 4)),
        mx.ones((1, 2, len(tokens) - 1, 4)) * 2,
    )
    recurrent_cache.state = [
        mx.ones((1, 3, 8)),
        mx.ones((1, 2, 4, 4)) * 3,
    ]
    mx.eval(kv_cache.keys, kv_cache.values, *recurrent_cache.state)
    cache = [kv_cache, recurrent_cache]
    rollback = _capture_prompt_cache_one_token_rollback(cache)
    assert rollback is not None

    # Simulate mlx_lm's lookahead: both cache kinds now include the final
    # emitted token, while rollback still owns the prior recurrent arrays.
    kv_cache.update_and_fetch(
        mx.ones((1, 2, 1, 4)) * 4,
        mx.ones((1, 2, 1, 4)) * 5,
    )
    recurrent_cache.state = [
        mx.ones((1, 3, 8)) * 6,
        mx.ones((1, 2, 4, 4)) * 7,
    ]
    mx.eval(kv_cache.keys, kv_cache.values, *recurrent_cache.state)

    handle = lru.bind_resume(
        key,
        tokens,
        prompt_cache=cache,
        one_token_rollback=rollback,
    )
    retained_with_rollback = lru.retained_bytes()
    assert retained_with_rollback > len(tokens) * 16

    resumed_cache, remaining, resumed, failure = lru.fetch_resume(
        handle,
        key,
        can_trim_prompt_cache=lambda _cache: False,
        trim_prompt_cache=lambda _cache, _count: None,
    )

    assert failure == ""
    assert remaining == [tokens[-1]]
    assert resumed == tokens
    assert resumed_cache is cache
    assert kv_cache.offset == len(tokens) - 1
    assert mx.array_equal(recurrent_cache.state[0], mx.ones((1, 3, 8))).item()
    assert mx.array_equal(
        recurrent_cache.state[1], mx.ones((1, 2, 4, 4)) * 3
    ).item()
    assert lru.retained_entries() == 0


def test_hybrid_one_token_rollback_replays_identical_qwen35_logits():
    import mlx.core as mx
    from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    from core.brain.llm.mlx_worker import (
        _capture_prompt_cache_one_token_rollback,
        _PromptCacheLRU,
    )

    text_config = {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "max_position_embeddings": 256,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 2,
        "full_attention_interval": 2,
        "head_dim": 16,
        "tie_word_embeddings": False,
    }
    model = Model(ModelArgs(model_type="qwen3_5", text_config=text_config))
    mx.eval(model.parameters())
    cache = model.make_cache()

    prompt = mx.array([[3, 5, 7, 9]], dtype=mx.int32)
    mx.eval(model(prompt, cache=cache))
    rollback = _capture_prompt_cache_one_token_rollback(cache)
    assert rollback is not None

    replay_token = mx.array([[11]], dtype=mx.int32)
    expected_logits = model(replay_token, cache=cache)
    mx.eval(expected_logits)

    lru = _PromptCacheLRU(max_size=2)
    key = (31, "user_surface")
    tokens = [3, 5, 7, 9, 11]
    handle = lru.bind_resume(
        key,
        tokens,
        prompt_cache=cache,
        one_token_rollback=rollback,
    )
    resumed_cache, remaining, resumed, failure = lru.fetch_resume(
        handle,
        key,
        can_trim_prompt_cache=can_trim_prompt_cache,
        trim_prompt_cache=trim_prompt_cache,
    )

    assert failure == ""
    assert remaining == [11]
    assert resumed == tokens
    assert resumed_cache is cache
    replayed_logits = model(mx.array([remaining], dtype=mx.int32), cache=resumed_cache)
    mx.eval(replayed_logits)
    assert mx.allclose(replayed_logits, expected_logits, rtol=0, atol=0).item()


def test_hybrid_resume_then_append_matches_uninterrupted_qwen35_logits():
    import mlx.core as mx
    from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    from core.brain.llm.mlx_worker import (
        _capture_prompt_cache_one_token_rollback,
        _PromptCacheLRU,
    )

    text_config = {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "max_position_embeddings": 256,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 2,
        "full_attention_interval": 2,
        "head_dim": 16,
        "tie_word_embeddings": False,
    }
    model = Model(ModelArgs(model_type="qwen3_5", text_config=text_config))
    mx.eval(model.parameters())
    prompt = mx.array([[3, 5, 7, 9]], dtype=mx.int32)
    old_tail = mx.array([[11]], dtype=mx.int32)
    append = [13, 15]

    uninterrupted_cache = model.make_cache()
    mx.eval(model(prompt, cache=uninterrupted_cache))
    expected_logits = model(
        mx.array([[11, *append]], dtype=mx.int32),
        cache=uninterrupted_cache,
    )
    mx.eval(expected_logits)

    resumable_cache = model.make_cache()
    mx.eval(model(prompt, cache=resumable_cache))
    rollback = _capture_prompt_cache_one_token_rollback(resumable_cache)
    assert rollback is not None
    mx.eval(model(old_tail, cache=resumable_cache))

    lru = _PromptCacheLRU(max_size=2)
    key = (32, "user_surface")
    handle = lru.bind_resume(
        key,
        [3, 5, 7, 9, 11],
        prompt_cache=resumable_cache,
        one_token_rollback=rollback,
        context_digest="authority-v1",
    )
    resumed_cache, remaining, resumed, failure = lru.fetch_resume(
        handle,
        key,
        can_trim_prompt_cache=can_trim_prompt_cache,
        trim_prompt_cache=trim_prompt_cache,
        append_tokens=[11, *append],
        context_digest="authority-v1",
    )

    assert failure == ""
    assert remaining == [11, 13, 15]
    assert resumed == [3, 5, 7, 9, 11, 13, 15]
    actual_logits = model(
        mx.array([remaining], dtype=mx.int32),
        cache=resumed_cache,
    )
    mx.eval(actual_logits)
    assert mx.allclose(actual_logits, expected_logits, rtol=0, atol=0).item()


def test_prompt_cache_resume_refuses_unrewindable_hybrid_state_without_mutation():
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=2)
    key = (29, "user_surface")
    tokens = [51, 52, 53]
    kv_cache = KVCache()
    kv_cache.update_and_fetch(
        mx.ones((1, 1, len(tokens), 2)),
        mx.ones((1, 1, len(tokens), 2)),
    )
    recurrent_cache = ArraysCache(size=2)
    recurrent_cache.state = [mx.ones((1, 2)), mx.ones((1, 2))]
    cache = [kv_cache, recurrent_cache]
    handle = lru.bind_resume(key, tokens, prompt_cache=cache)

    resumed_cache, _remaining, _resumed, failure = lru.fetch_resume(
        handle,
        key,
        can_trim_prompt_cache=lambda _cache: False,
        trim_prompt_cache=lambda _cache, _count: None,
    )

    assert resumed_cache is None
    assert failure == "hybrid_rollback_unavailable"
    assert kv_cache.offset == len(tokens)
    assert lru.retained_entries() == 1


def test_mlx_client_drops_an_invalid_continuation_resume_handle():
    from core.brain.llm.mlx_client import _sanitize_surface_control_receipt

    receipt = _sanitize_surface_control_receipt(
        {
            "continuation_resume_available": True,
            "continuation_resume_handle": "../../not-a-capability",
        }
    )

    assert receipt["continuation_resume_available"] is True
    assert "continuation_resume_handle" not in receipt


def test_mlx_client_drops_invalid_conversation_resume_proof_fields():
    from core.brain.llm.mlx_client import _sanitize_surface_control_receipt

    receipt = _sanitize_surface_control_receipt(
        {
            "conversation_resume_available": True,
            "conversation_resume_handle": "../../not-a-capability",
            "conversation_resume_output_sha256": "not-a-digest",
        }
    )

    assert receipt["conversation_resume_available"] is True
    assert "conversation_resume_handle" not in receipt
    assert "conversation_resume_output_sha256" not in receipt


def test_mlx_client_arms_the_oom_ladder_with_the_prompt_cache():
    """The rung the ladder was missing, reachable from the parent process.

    The boot verifier warned on EVERY boot: "no organ exposes a shed hook, so
    the OOM ladder has no rungs: the only available response to memory pressure
    is a restart" — while the prompt KV cache in the worker was the largest
    trivially droppable allocation in the tree. It was unreachable because the
    ladder runs in the parent and the cache lives in the worker; the worker had
    always accepted a `clear_cache` action and nothing ever sent one.
    """
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient("Aura-32B-crsm-closeout-jul1-20260701-215118")

    # This is exactly the contract core/runtime/foundations.py discovers.
    assert callable(client.shed_memory)
    assert callable(client.memory_footprint_bytes)
    assert client.oom_score_adj > 0, "a pure cache should volunteer, not resist"
    assert client.oom_recoverable is True
    assert client.oom_rationale

    # Footprint is read from what the worker reported, not guessed.
    assert client.memory_footprint_bytes() == 0
    client._record_surface_control_receipt_from_response(
        {"status": "ok", "prompt_cache_bytes": 1_234_567, "tokens_used": 5}
    )
    assert client.memory_footprint_bytes() == 1_234_567

    # With no live worker, shedding frees nothing and must not raise into the
    # ladder — a shed loop that explodes under pressure is worse than no rung.
    assert client.shed_memory() == 0


def test_shed_memory_never_claims_more_than_was_held(monkeypatch):
    """An unverified reclaim number is how a ladder reports progress it did not make."""
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient("Aura-32B-crsm-closeout-jul1-20260701-215118")
    client._record_surface_control_receipt_from_response(
        {"status": "ok", "prompt_cache_bytes": 1000, "tokens_used": 1}
    )

    monkeypatch.setattr(
        client,
        "_send_worker_control_action",
        lambda action, timeout_s=10.0: {
            "status": "ok",
            "prompt_cache_bytes_freed": 999_999_999,  # worker over-reports
            "prompt_cache_bytes": 0,
        },
    )
    assert client.shed_memory() == 1000, "must be clamped to what was known held"
    assert client.memory_footprint_bytes() == 0, "footprint must follow the shed"

    # A control message that fails is reported as zero reclaimed, not as success.
    def _boom(action, timeout_s=10.0):
        raise BrokenPipeError("worker gone")

    monkeypatch.setattr(client, "_send_worker_control_action", _boom)
    assert client.shed_memory() == 0


def test_prompt_cache_registers_itself_on_the_oom_ladder_at_construction():
    """Boot-time discovery cannot see a lazily-created organ.

    core/runtime/foundations.py registers services that are ALREADY
    instantiated, and this client is built when a model first loads — long
    after that sweep. So the live ladder kept reporting "0 sheddable organs"
    and the verifier kept warning that its only response to memory pressure
    was a restart, with a working shed_memory() sitting right there.
    """
    from core.brain.llm.mlx_client import MLXLocalClient
    from core.runtime.oom_policy import oom_report, reset_oom_policy_for_test

    reset_oom_policy_for_test()
    assert oom_report()["sheddable_organs"] == 0

    client = MLXLocalClient("Aura-32B-crsm-closeout-jul1-20260701-215118")
    try:
        client._record_surface_control_receipt_from_response(
            {"status": "ok", "prompt_cache_bytes": 3 * 1024**3, "tokens_used": 1}
        )
        report = oom_report()
        assert report["sheddable_organs"] >= 1, "the ladder still has no rungs"
        rows = [r for r in report["scoring_table"] if r["organ"] == "mlx_prompt_cache"]
        assert rows, "the prompt cache did not register itself"
        assert rows[0]["sheddable"] is True
        assert rows[0]["footprint_bytes"] == 3 * 1024**3
        assert rows[0]["oom_score_adj"] > 0
    finally:
        client._process = None
        reset_oom_policy_for_test()


def test_distinct_internal_prompt_families_stop_evicting_each_other():
    """One slot for the whole internal lane made its families thrash.

    The internal lane carries many DISTINCT prompt families — the reflective
    persona, the pre-linguistic decision narrator, enrichment, dreaming. With a
    50/50 split of a 2-entry budget it held exactly one, and the live log showed
    the same two families taking turns destroying each other's prefix, over and
    over: "trimmed hit — reused 3/792 tokens".
    """
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=12, max_total_tokens=100_000, kv_bytes_per_token=1)
    internal = (9, "default")

    families = {
        "persona": [1, 2, 3] + list(range(100, 400)),
        "prelinguistic": [1, 2, 4] + list(range(400, 700)),
        "enrichment": [1, 2, 5] + list(range(700, 1000)),
        "dreaming": [1, 2, 6] + list(range(1000, 1300)),
    }
    for tokens in families.values():
        lru.insert_cache(internal, tokens, ["kv"])

    # Every family must still be reusable, not just the most recent one.
    for name, tokens in families.items():
        hit, remaining = lru.fetch_nearest_cache(
            internal, tokens + [99_999],
            can_trim_prompt_cache=lambda _pc: True,
            trim_prompt_cache=lambda _pc, _n: None,
        )
        assert hit is not None, f"{name} was evicted by its siblings"
        assert len(remaining) <= 2, (
            f"{name} reused almost nothing ({len(remaining)} of "
            f"{len(tokens) + 1} to prefill)"
        )


def test_the_conversation_lane_keeps_its_own_reserved_slots():
    """Widening the internal lane must not cost the conversation its guarantee."""
    from core.brain.llm.mlx_worker import _PromptCacheLRU

    lru = _PromptCacheLRU(max_size=12, max_total_tokens=100_000, kv_bytes_per_token=1)
    surface = (9, "user_surface")
    internal = (9, "default")

    conversation = list(range(1, 601))
    lru.insert_cache(surface, conversation, ["kv-conversation"])
    for tick in range(40):
        lru.insert_cache(internal, [50_000 + tick, tick, tick + 1], ["kv-internal"])

    hit, _ = lru.fetch_nearest_cache(
        surface, conversation + [7777],
        can_trim_prompt_cache=lambda _pc: True,
        trim_prompt_cache=lambda _pc, _n: None,
    )
    assert hit == ["kv-conversation"], "internal churn took the conversation's slot"
    assert lru._lane_budget("user_surface") >= 1
    assert lru._lane_budget("default") > lru._lane_budget("user_surface"), (
        "the lane with many distinct families needs the larger share"
    )


def test_memory_pressure_sheds_caches_before_killing_workers():
    """The ladder and the watchdog were two answers that never met.

    Measured live: "swap exhaustion: managed RSS 33868MB swap 8.3GB ...
    terminated 0 heavy workers and forced gc out-of-band" — nothing to kill,
    nothing reclaimed — while the prompt KV cache sat registered as sheddable
    with a bounded ~3GB footprint. A rung nothing pulls is not a rung.

    Killing a worker costs a full model reload; dropping a cache costs a
    re-prefill. Caches must go first.
    """
    from core.resilience import memory_watchdog as mw
    from core.runtime.oom_policy import register_organ, reset_oom_policy_for_test

    reset_oom_policy_for_test()
    try:
        order: list[str] = []

        def _shed_cache():
            order.append("shed")
            return 3 * 1024**3

        register_organ(
            "kv_cache", oom_score_adj=300, footprint=lambda: 3 * 1024**3,
            shed=_shed_cache, rationale="test cache",
        )

        organs, freed = mw._shed_registered_organs()
        assert organs == 1 and freed == 3 * 1024**3
        assert order == ["shed"]

        # A raising organ must not break the reclaim under pressure.
        def _explodes():
            raise RuntimeError("shed failed")

        register_organ("broken", oom_score_adj=10, shed=_explodes, rationale="t")
        organs_again, _ = mw._shed_registered_organs()
        assert organs_again >= 1, "one broken organ must not abort the whole reclaim"
    finally:
        reset_oom_policy_for_test()


def test_the_hard_reclaim_pulls_the_ladder_before_terminating_workers():
    from core.resilience.memory_watchdog import MemoryWatchdog

    calls: list[str] = []

    watchdog = MemoryWatchdog(
        worker_terminator=lambda: (calls.append("kill"), 0)[1],
        gc_collect=lambda: (calls.append("gc"), 0)[1],
        ladder_shed=lambda: (calls.append("shed"), (1, 3 * 1024**3))[1],
    )
    sample = watchdog._sampler()
    watchdog._handle_hard(sample, 10_000.0, swap_escalation=True)

    assert calls, "the hard tier must attempt a reclaim"
    assert calls[0] == "shed", (
        f"caches must be shed before workers are killed, got {calls}"
    )
    assert "kill" in calls
