from __future__ import annotations
from core.runtime.errors import record_degradation

"""Shared helpers for safer macOS desktop boot behavior."""


import os
import platform
import threading
from typing import Any, Mapping


_GIB = 1024 ** 3
_INPROCESS_MLX_LOCK = threading.Lock()
_INPROCESS_MLX_STATE: dict[str, Any] = {
    "configured": False,
    "device": "unknown",
    "reason": "uninitialized",
}


def env_flag_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def desktop_safe_boot_enabled(env: Mapping[str, str] | None = None) -> bool:
    env = env or os.environ
    explicit = str(env.get("AURA_SAFE_BOOT_DESKTOP", "")).strip().lower()
    if explicit in {"1", "true", "yes", "on"}:
        return True
    if explicit in {"0", "false", "no", "off"}:
        return False
    return env_flag_enabled(env.get("AURA_LAUNCHED_FROM_APP"))


def compute_mlx_cache_limit(total_ram_bytes: int, env: Mapping[str, str] | None = None) -> int:
    env = env or os.environ
    total_ram_bytes = max(int(total_ram_bytes), 8 * _GIB)

    if desktop_safe_boot_enabled(env):
        ratio = float(env.get("AURA_SAFE_BOOT_METAL_CACHE_RATIO", "0.56"))
        hard_cap_gb = float(env.get("AURA_SAFE_BOOT_METAL_CACHE_CAP_GB", "36"))
        floor_gb = float(env.get("AURA_SAFE_BOOT_METAL_CACHE_FLOOR_GB", "16"))
        limit = int(total_ram_bytes * ratio)
        limit = min(limit, int(hard_cap_gb * _GIB))
        return max(int(floor_gb * _GIB), limit)

    ratio = float(env.get("AURA_METAL_CACHE_RATIO", "0.75"))
    limit = int(total_ram_bytes * ratio)
    hard_cap_gb = float(env.get("AURA_METAL_CACHE_CAP_GB", "0"))
    if hard_cap_gb > 0:
        limit = min(limit, int(hard_cap_gb * _GIB))
    return max(8 * _GIB, limit)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _macos_major_version(version: str | None = None) -> int:
    release = str(version or platform.mac_ver()[0] or "").strip()
    if not release:
        return 0
    head = release.split(".", 1)[0].strip()
    try:
        return int(head)
    except ValueError:
        return 0


def inprocess_mlx_metal_enabled(
    env: Mapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
    mac_version: str | None = None,
) -> tuple[bool, str]:
    env = env or os.environ
    platform_name = str(platform_name or os.sys.platform).lower()

    if _truthy(env.get("AURA_FORCE_INPROCESS_MLX_METAL")) or _truthy(
        env.get("AURA_ALLOW_UNSAFE_INPROCESS_MLX_METAL")
    ):
        return True, "forced"

    if _truthy(env.get("AURA_DISABLE_INPROCESS_MLX_METAL")):
        return False, "env_disabled"

    if desktop_safe_boot_enabled(env):
        return False, "desktop_safe_boot"

    if platform_name == "darwin" and _macos_major_version(mac_version) >= 26:
        return False, "macos26_guard"

    return True, "enabled"


def configure_inprocess_mlx_runtime(
    env: Mapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
    mac_version: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    enabled, reason = inprocess_mlx_metal_enabled(
        env,
        platform_name=platform_name,
        mac_version=mac_version,
    )

    desired_device = "metal" if enabled else "cpu"
    with _INPROCESS_MLX_LOCK:
        if (
            not force
            and _INPROCESS_MLX_STATE["configured"]
            and _INPROCESS_MLX_STATE["device"] == desired_device
            and _INPROCESS_MLX_STATE["reason"] == reason
        ):
            return dict(_INPROCESS_MLX_STATE)

        if not enabled:
            _INPROCESS_MLX_STATE.update(
                {
                    "configured": True,
                    "device": "cpu",
                    "reason": reason,
                }
            )
            return dict(_INPROCESS_MLX_STATE)

        try:
            import mlx.core as mx
        except Exception:
            _INPROCESS_MLX_STATE.update(
                {
                    "configured": True,
                    "device": "unavailable",
                    "reason": f"{reason}:mlx_unavailable",
                }
            )
            return dict(_INPROCESS_MLX_STATE)

        try:
            _INPROCESS_MLX_STATE.update(
                {
                    "configured": True,
                    "device": "metal",
                    "reason": reason,
                }
            )
        except Exception as exc:
            record_degradation('desktop_boot_safety', exc)
            _INPROCESS_MLX_STATE.update(
                {
                    "configured": True,
                    "device": "failed",
                    "reason": f"{reason}:{type(exc).__name__}",
                }
            )
        return dict(_INPROCESS_MLX_STATE)
