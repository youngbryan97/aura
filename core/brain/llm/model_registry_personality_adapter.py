"""Which personality adapter, if any, her MLX model loads.

Lifted whole out of `model_registry`, which imports them straight back: every caller and every patch that names them
there still finds them. What they take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _personality_lora_disabled(backend: str) -> bool:
    backend_key = "AURA_DISABLE_MLX_LORA"
    return _env_truthy("AURA_DISABLE_PERSONALITY_LORA") or _env_truthy(backend_key)


def _default_personality_lora_enabled(backend: str) -> bool:
    backend_key = "AURA_ENABLE_MLX_LORA"
    return _env_truthy("AURA_ENABLE_PERSONALITY_LORA") or _env_truthy(backend_key)


def resolve_personality_adapter(
    target_model: str | None,
    *,
    backend: str = "mlx",
) -> str | None:
    """Return a compatible Aura personality adapter for the requested model.

    MLX overrides are supported:
      - `AURA_LORA_PATH`, `AURA_LORA_TARGET_MODEL`

    The bundled personality adapter is opt-in. It has historically been useful
    for experiments, but live Cortex already runs against Aura-tuned/fused
    weights and the extra adapter can pollute user-visible prose. Explicit
    adapter paths still work; the default adapter only loads when an enable env
    flag is set.
    """
    from . import model_paths
    from .model_registry import (
        _FLAG_LORA_PATH,
        _FLAG_LORA_TARGET_MODEL,
        ACTIVE_MODEL,
        _read_adapter_target_model,
        model_identities_compatible,
    )

    normalized_backend = str(backend or "mlx").strip().lower()
    if normalized_backend != "mlx":
        return None
    target_model = str(target_model or "").strip()

    if _personality_lora_disabled(normalized_backend):
        return None

    adapter_dir = _FLAG_LORA_PATH.value().strip()
    if not adapter_dir:
        if not _default_personality_lora_enabled(normalized_backend):
            return None
        default_dir = model_paths.BASE_DIR / "training" / "adapters" / "aura-personality"
        if (default_dir / "adapters.safetensors").exists():
            adapter_dir = str(default_dir)
    if not adapter_dir or not Path(adapter_dir).is_dir():
        return None

    configured_target = (
        _FLAG_LORA_TARGET_MODEL.value().strip()
        or _read_adapter_target_model(Path(adapter_dir))
        or ACTIVE_MODEL
    )
    if target_model and configured_target and not model_identities_compatible(configured_target, target_model):
        return None
    return adapter_dir
