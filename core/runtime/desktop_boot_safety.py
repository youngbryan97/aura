"""Shared helpers for safer macOS desktop boot behavior."""

from __future__ import annotations

import os
from typing import Mapping


_GIB = 1024 ** 3


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
