"""Shared helpers for full desktop boot resource protection.

The normal desktop profile is a complete Aura runtime.  Resource guards bound
MLX, process RSS, and duplicate Metal ownership without disabling cognitive
organs.  ``AURA_SAFE_BOOT_DESKTOP`` is reserved for an explicit recovery boot.
"""

from __future__ import annotations

import importlib
import os
import platform
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_GIB = 1024**3
SAFE_BOOT_MLX_MEMORY_CAP_GB = 34.0
SAFE_BOOT_PROCESS_RSS_CAP_GB = 56.0
DESKTOP_PROCESS_RSS_RATIO = 0.81
DESKTOP_HOST_RESERVE_RATIO = 0.18
DESKTOP_HOST_RESERVE_FLOOR_GB = 8.0
_INPROCESS_MLX_LOCK = threading.Lock()
_INPROCESS_MLX_STATE: dict[str, Any] = {
    "configured": False,
    "device": "unknown",
    "reason": "uninitialized",
    "verified": False,
}


def env_flag_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(env.get(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        return default


def _resource_env_float(
    env: Mapping[str, str],
    name: str,
    legacy_name: str,
    default: float,
) -> float:
    """Read a desktop guard setting with recovery-profile compatibility."""

    if name in env:
        return _env_float(env, name, default)
    return _env_float(env, legacy_name, default)


def _unsafe_memory_limits_allowed(env: Mapping[str, str]) -> bool:
    return env_flag_enabled(env.get("AURA_ALLOW_UNSAFE_MEMORY_LIMITS"))


def desktop_safe_boot_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the explicit reduced recovery profile was requested."""

    env = env or os.environ
    explicit = str(env.get("AURA_SAFE_BOOT_DESKTOP", "")).strip().lower()
    if explicit in {"1", "true", "yes", "on"}:
        return True
    if explicit in {"0", "false", "no", "off"}:
        return False
    return False


def desktop_resource_guard_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether desktop memory/process ownership guards are active.

    The guard is on for normal app launches and explicit recovery boots.  It is
    deliberately independent of subsystem admission: a protected desktop boot
    is still the full runtime.
    """

    env = env or os.environ
    explicit = str(env.get("AURA_DESKTOP_RESOURCE_GUARD", "")).strip().lower()
    if explicit in {"1", "true", "yes", "on"}:
        return True
    if explicit in {"0", "false", "no", "off"}:
        return False
    return desktop_safe_boot_enabled(env) or env_flag_enabled(env.get("AURA_LAUNCHED_FROM_APP"))


def compute_mlx_cache_limit(total_ram_bytes: int, env: Mapping[str, str] | None = None) -> int:
    env = env or os.environ
    total_ram_bytes = max(int(total_ram_bytes), 8 * _GIB)

    if desktop_resource_guard_enabled(env):
        ratio = _resource_env_float(
            env, "AURA_DESKTOP_METAL_CACHE_RATIO", "AURA_SAFE_BOOT_METAL_CACHE_RATIO", 0.16
        )
        hard_cap_gb = _resource_env_float(
            env, "AURA_DESKTOP_METAL_CACHE_CAP_GB", "AURA_SAFE_BOOT_METAL_CACHE_CAP_GB", 10.0
        )
        floor_gb = _resource_env_float(
            env, "AURA_DESKTOP_METAL_CACHE_FLOOR_GB", "AURA_SAFE_BOOT_METAL_CACHE_FLOOR_GB", 4.0
        )
        limit = int(total_ram_bytes * ratio)
        limit = min(limit, int(hard_cap_gb * _GIB))
        limit = max(int(floor_gb * _GIB), limit)
        if not _unsafe_memory_limits_allowed(env):
            limit = min(limit, 10 * _GIB)
        return limit

    ratio = _env_float(env, "AURA_METAL_CACHE_RATIO", 0.75)
    limit = int(total_ram_bytes * ratio)
    hard_cap_gb = _env_float(env, "AURA_METAL_CACHE_CAP_GB", 0.0)
    if hard_cap_gb > 0:
        limit = min(limit, int(hard_cap_gb * _GIB))
    return max(8 * _GIB, limit)


def compute_mlx_memory_limit(total_ram_bytes: int, env: Mapping[str, str] | None = None) -> int:
    """Return the active MLX memory ceiling for model/KV allocations."""

    env = env or os.environ
    total_ram_bytes = max(int(total_ram_bytes), 8 * _GIB)
    resource_guard = desktop_resource_guard_enabled(env)
    unsafe_allowed = _unsafe_memory_limits_allowed(env)
    configured = str(env.get("AURA_MLX_MEMORY_LIMIT_GB", "") or "").strip()
    if configured:
        try:
            configured_gb = float(configured)
        except (TypeError, ValueError, OverflowError):
            configured_gb = 0.0
        if configured_gb > 0.0:
            configured_limit = int(configured_gb * _GIB)
            if resource_guard and not unsafe_allowed:
                safe_cap_gb = min(
                    _resource_env_float(
                        env,
                        "AURA_DESKTOP_MLX_MEMORY_CAP_GB",
                        "AURA_SAFE_BOOT_MLX_MEMORY_CAP_GB",
                        SAFE_BOOT_MLX_MEMORY_CAP_GB,
                    ),
                    SAFE_BOOT_MLX_MEMORY_CAP_GB,
                )
                return min(configured_limit, int(safe_cap_gb * _GIB))
            return configured_limit

    if resource_guard:
        ratio = _resource_env_float(
            env, "AURA_DESKTOP_MLX_MEMORY_RATIO", "AURA_SAFE_BOOT_MLX_MEMORY_RATIO", 0.54
        )
        hard_cap_gb = _resource_env_float(
            env,
            "AURA_DESKTOP_MLX_MEMORY_CAP_GB",
            "AURA_SAFE_BOOT_MLX_MEMORY_CAP_GB",
            SAFE_BOOT_MLX_MEMORY_CAP_GB,
        )
        floor_gb = _resource_env_float(
            env, "AURA_DESKTOP_MLX_MEMORY_FLOOR_GB", "AURA_SAFE_BOOT_MLX_MEMORY_FLOOR_GB", 18.0
        )
        limit = min(int(total_ram_bytes * ratio), int(hard_cap_gb * _GIB))
        limit = max(int(floor_gb * _GIB), limit)
        if not unsafe_allowed:
            limit = min(limit, int(SAFE_BOOT_MLX_MEMORY_CAP_GB * _GIB))
        return limit

    ratio = _env_float(env, "AURA_MLX_MEMORY_RATIO", 0.72)
    limit = int(total_ram_bytes * ratio)
    hard_cap_gb = _env_float(env, "AURA_MLX_MEMORY_CAP_GB", 0.0)
    if hard_cap_gb > 0:
        limit = min(limit, int(hard_cap_gb * _GIB))
    return max(8 * _GIB, limit)


def compute_process_rss_limit(total_ram_bytes: int, env: Mapping[str, str] | None = None) -> int:
    """Return the process-tree RSS guard used by full desktop runtime.

    This is intentionally lower than the external sentinel kill ceiling. The
    in-process guard should refuse/recycle before the out-of-process sentinel
    has to SIGKILL Aura to protect the host.
    """

    env = env or os.environ
    total_ram_bytes = max(int(total_ram_bytes), 8 * _GIB)
    resource_guard = desktop_resource_guard_enabled(env)
    unsafe_allowed = _unsafe_memory_limits_allowed(env)
    ratio = _resource_env_float(
        env,
        "AURA_DESKTOP_PROCESS_RSS_RATIO",
        "AURA_SAFE_BOOT_PROCESS_RSS_RATIO",
        DESKTOP_PROCESS_RSS_RATIO,
    )
    hard_cap_gb = min(
        _resource_env_float(
            env,
            "AURA_DESKTOP_PROCESS_RSS_CAP_GB",
            "AURA_SAFE_BOOT_PROCESS_RSS_CAP_GB",
            SAFE_BOOT_PROCESS_RSS_CAP_GB,
        ),
        SAFE_BOOT_PROCESS_RSS_CAP_GB,
    )
    reserve_ratio = _resource_env_float(
        env,
        "AURA_DESKTOP_HOST_RESERVE_RATIO",
        "AURA_SAFE_BOOT_HOST_RESERVE_RATIO",
        DESKTOP_HOST_RESERVE_RATIO,
    )
    reserve_floor_gb = _resource_env_float(
        env,
        "AURA_DESKTOP_HOST_RESERVE_FLOOR_GB",
        "AURA_SAFE_BOOT_HOST_RESERVE_FLOOR_GB",
        DESKTOP_HOST_RESERVE_FLOOR_GB,
    )
    reserve_bytes = max(int(reserve_floor_gb * _GIB), int(total_ram_bytes * reserve_ratio))
    host_safe_cap = max(4 * _GIB, total_ram_bytes - reserve_bytes)
    floor_gb = _resource_env_float(
        env,
        "AURA_DESKTOP_PROCESS_RSS_FLOOR_GB",
        "AURA_SAFE_BOOT_PROCESS_RSS_FLOOR_GB",
        24.0,
    )
    if resource_guard and not unsafe_allowed:
        floor_gb = min(floor_gb, 24.0)
    canonical_limit = min(
        int(total_ram_bytes * ratio),
        int(hard_cap_gb * _GIB),
        host_safe_cap,
    )
    canonical_limit = max(min(int(floor_gb * _GIB), host_safe_cap), canonical_limit)

    configured = str(env.get("AURA_PROCESS_RSS_LIMIT_GB", "") or "").strip()
    if configured:
        try:
            configured_gb = float(configured)
        except (TypeError, ValueError, OverflowError):
            configured_gb = 0.0
        if configured_gb > 0.0:
            configured_limit = int(configured_gb * _GIB)
            if resource_guard and not unsafe_allowed:
                return min(configured_limit, canonical_limit)
            return configured_limit

    if resource_guard:
        return canonical_limit

    ratio = _env_float(env, "AURA_PROCESS_RSS_RATIO", 0.56)
    hard_cap_gb = _env_float(env, "AURA_PROCESS_RSS_CAP_GB", 38.0)
    floor_gb = _env_float(env, "AURA_PROCESS_RSS_FLOOR_GB", 30.0)
    limit = min(int(total_ram_bytes * ratio), int(hard_cap_gb * _GIB))
    return max(int(floor_gb * _GIB), limit)


@dataclass(frozen=True)
class DesktopMemoryEnvelope:
    """One ordered memory policy shared by admission, governors, and sentinels."""

    process_limit_mb: float
    governor_prune_mb: float
    governor_unload_mb: float
    governor_critical_mb: float
    watchdog_soft_mb: float
    watchdog_hard_mb: float
    watchdog_lethal_mb: float


def compute_desktop_memory_envelope(
    total_ram_bytes: int,
    env: Mapping[str, str] | None = None,
) -> DesktopMemoryEnvelope:
    """Derive an ordered envelope that admits the 32B lane and protects the host.

    The process limit is the canonical admission ceiling. Lower rungs shed
    caches and optional workers before it; the lethal rung retains an external
    margin while preserving at least an eighth of host RAM for macOS and other
    applications. Explicit lower process limits remain valid recovery policy.
    """

    env = env or os.environ
    total_ram_bytes = max(int(total_ram_bytes), 8 * _GIB)
    process_limit_mb = compute_process_rss_limit(total_ram_bytes, env) / float(1024**2)
    total_mb = total_ram_bytes / float(1024**2)
    prune_mb = process_limit_mb * 0.88
    unload_mb = process_limit_mb * 0.93
    critical_mb = process_limit_mb * 0.97
    lethal_host_cap_mb = total_mb - max(6 * 1024.0, total_mb * 0.125)
    lethal_mb = min(lethal_host_cap_mb, process_limit_mb + 4 * 1024.0)
    lethal_mb = max(process_limit_mb + 1024.0, lethal_mb)
    return DesktopMemoryEnvelope(
        process_limit_mb=process_limit_mb,
        governor_prune_mb=prune_mb,
        governor_unload_mb=unload_mb,
        governor_critical_mb=critical_mb,
        watchdog_soft_mb=prune_mb,
        watchdog_hard_mb=process_limit_mb,
        watchdog_lethal_mb=lethal_mb,
    )


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

    if desktop_resource_guard_enabled(env):
        return False, "desktop_resource_guard"

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
    return configure_mlx_process_device(
        desired_device,
        reason=reason,
        force=force,
    )


def configure_mlx_process_device(
    device: str,
    *,
    reason: str,
    force: bool = False,
) -> dict[str, Any]:
    """Set and verify this process's MLX default device.

    MLX defaults to Metal on Apple Silicon. Recording ``device='cpu'`` in
    Aura's own state without calling ``mx.set_default_device`` therefore does
    not move any work: the next parent-side array still allocates on Metal.
    Device ownership is process-local, so the desktop parent and isolated
    model worker both use this function and receive independently verified
    contracts.
    """

    desired_device = str(device or "").strip().lower()
    if desired_device == "gpu":
        desired_device = "metal"
    if desired_device not in {"cpu", "metal"}:
        raise ValueError(f"unsupported MLX process device: {device!r}")

    with _INPROCESS_MLX_LOCK:
        try:
            mx = importlib.import_module("mlx.core")
        except (ImportError, AttributeError, RuntimeError) as exc:
            _INPROCESS_MLX_STATE.update(
                {
                    "configured": True,
                    "device": "unavailable",
                    "reason": f"{reason}:mlx_unavailable:{type(exc).__name__}",
                    "verified": False,
                }
            )
            return dict(_INPROCESS_MLX_STATE)

        target = mx.cpu if desired_device == "cpu" else mx.gpu
        try:
            current = mx.default_device()
            if force or current != target:
                mx.set_default_device(target)
            actual = mx.default_device()
            if actual != target:
                raise RuntimeError(
                    f"MLX default device remained {actual!s}; expected {target!s}"
                )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _INPROCESS_MLX_STATE.update(
                {
                    "configured": True,
                    "device": "unavailable",
                    "reason": (
                        f"{reason}:device_configuration_failed:{type(exc).__name__}"
                    ),
                    "verified": False,
                }
            )
            return dict(_INPROCESS_MLX_STATE)

        _INPROCESS_MLX_STATE.update(
            {
                "configured": True,
                "device": desired_device,
                "reason": reason,
                "verified": True,
            }
        )
        return dict(_INPROCESS_MLX_STATE)


def mlx_process_runtime_status() -> dict[str, Any]:
    """Return this process's last verified MLX device contract."""

    with _INPROCESS_MLX_LOCK:
        return dict(_INPROCESS_MLX_STATE)


def mlx_process_uses_metal() -> bool:
    """Whether this process owns a verified Metal MLX default device."""

    status = mlx_process_runtime_status()
    return bool(status.get("verified") and status.get("device") == "metal")
