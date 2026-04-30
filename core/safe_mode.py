"""Runtime mode configuration for Aura startup.

Historically this module monkey-patched live orchestrator methods at boot.
That created drift as native runtime paths evolved, and parts of the patch
layer became stale or ineffective. The safe/full mode contract now works by
installing explicit runtime configuration that native subsystems consume.
"""
from __future__ import annotations


import logging
from copy import deepcopy
from typing import Any

logger = logging.getLogger("Aura.SafeMode")


SAFE_MODE_CONFIG = {
    # Core features — always on
    "llm_enabled": True,
    "memory_basic": True,
    "personality_engine": True,
    "voice_stt": True,
    "voice_tts": True,
    # Features disabled in safe mode
    "self_modification": False,
    "self_preservation": False,
    "persona_evolution": False,
    "dream_cycle": False,
    "device_discovery": False,
    "stealth_mode": False,
    "singularity_monitor": False,
    "vector_memory_write": True,
    "context_pruning": False,
    "memory_consolidation": False,
    # Tuning
    "autonomous_thought_interval_s": 60.0,
    "health_poll_interval_ms": 10000,
    "max_conversation_history": 50,
    "singularity_acceleration_cap": 1.0,
}

FULL_MODE_CONFIG = {
    "llm_enabled": True,
    "memory_basic": True,
    "personality_engine": True,
    "voice_stt": True,
    "voice_tts": True,
    "self_modification": True,
    "self_preservation": False,
    "persona_evolution": True,
    "dream_cycle": True,
    "device_discovery": False,
    "stealth_mode": True,
    "singularity_monitor": True,
    "vector_memory_write": True,
    "context_pruning": True,
    "memory_consolidation": True,
    "autonomous_thought_interval_s": 45.0,
    "health_poll_interval_ms": 5000,
    "max_conversation_history": 100,
    "singularity_acceleration_cap": 2.0,
}


def _build_runtime_mode_config(orchestrator: Any, *, safe_mode: bool) -> dict[str, Any]:
    runtime_config = deepcopy(SAFE_MODE_CONFIG if safe_mode else FULL_MODE_CONFIG)

    kernel = getattr(orchestrator, "kernel", None)
    volition = getattr(kernel, "volition_level", 0) if kernel else 0

    if not safe_mode:
        if volition >= 1:
            runtime_config["dream_cycle"] = True
            runtime_config["context_pruning"] = True
            runtime_config["memory_consolidation"] = True
        if volition >= 2:
            runtime_config["persona_evolution"] = True
        if volition >= 3:
            runtime_config["self_modification"] = True
            runtime_config["self_preservation"] = True
            runtime_config["singularity_monitor"] = True

    return runtime_config


def get_runtime_mode_config(orchestrator: Any | None = None) -> dict[str, Any]:
    runtime_config = getattr(orchestrator, "_runtime_mode_config", None) if orchestrator else None
    if isinstance(runtime_config, dict):
        return runtime_config
    return FULL_MODE_CONFIG


def runtime_mode_value(orchestrator: Any | None, key: str, default: Any = None) -> Any:
    value = get_runtime_mode_config(orchestrator).get(key, default)
    return default if value is None else value


def runtime_feature_enabled(orchestrator: Any | None, key: str, default: bool = True) -> bool:
    return bool(runtime_mode_value(orchestrator, key, default))


def apply_orchestrator_patches(orchestrator: Any, safe_mode: bool = False) -> None:
    """Legacy compatibility hook for installing runtime mode configuration.

    The entry point name is preserved so older boot code keeps working, but it
    now configures native execution paths instead of monkey-patching methods.
    """

    runtime_config = _build_runtime_mode_config(orchestrator, safe_mode=safe_mode)
    orchestrator._runtime_mode_config = runtime_config
    orchestrator._safe_mode_enabled = bool(safe_mode)

    kernel = getattr(orchestrator, "kernel", None)
    volition = getattr(kernel, "volition_level", 0) if kernel else 0

    _apply_runtime_caps(orchestrator, runtime_config)

    logger.info(
        "Applied orchestrator runtime configuration (safe_mode=%s, volition=%d)",
        safe_mode,
        volition,
    )


def _apply_runtime_caps(orchestrator: Any, runtime_config: dict[str, Any]) -> None:
    cap = float(runtime_config.get("singularity_acceleration_cap", 1.0))

    monitor = getattr(orchestrator, "singularity_monitor", None)
    if monitor and hasattr(monitor, "acceleration_factor"):
        original_factor = float(getattr(monitor, "acceleration_factor", 1.0))
        monitor.acceleration_factor = min(original_factor, cap)

    engine = getattr(orchestrator, "cognitive_engine", None)
    if engine and hasattr(engine, "singularity_factor"):
        original_factor = float(getattr(engine, "singularity_factor", 1.0))
        engine.singularity_factor = min(original_factor, cap)

    orchestrator._min_thought_interval = float(runtime_config.get("autonomous_thought_interval_s", 45.0))

    history = getattr(orchestrator, "conversation_history", None)
    max_history = int(runtime_config.get("max_conversation_history", 100))
    if isinstance(history, list) and len(history) > max_history:
        orchestrator.conversation_history = history[-max_history:]
