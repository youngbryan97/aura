"""Host thermal state, readable everywhere Aura runs — no sudo required.

The previous "thermal guard" called psutil.sensors_temperatures(), which
does not exist on macOS: on the primary deployment platform the gate was a
silent no-op and sustained load could cook the host unchecked. This module
gives one canonical reading:

    level 0 nominal · 1 fair · 2 serious · 3 critical

Sources, in order:
  1. NSProcessInfo.thermalState (macOS canonical, matches the levels above)
  2. `pmset -g therm` CPU_Speed_Limit (macOS fallback when pyobjc is absent)
  3. psutil.sensors_temperatures (Linux), mapped by °C thresholds

Readings are cached briefly; failures degrade to level 0 with the source
recorded, so callers can distinguish "cool" from "blind".
"""
from __future__ import annotations

import logging
import re
import subprocess  # noqa: F401 — via gateway below; kept for typing clarity
import time
from dataclasses import dataclass

logger = logging.getLogger("Aura.Runtime.Thermal")

_CACHE_TTL_S = 5.0
_PMSET_TIMEOUT_S = 3.0
# Linux psutil mapping: °C at which we call it serious / critical.
_TEMP_SERIOUS_C = 78.0
_TEMP_CRITICAL_C = 90.0


@dataclass(frozen=True)
class ThermalReading:
    level: int  # 0 nominal, 1 fair, 2 serious, 3 critical
    source: str  # nsprocessinfo | pmset | psutil | blind
    detail: str = ""

    @property
    def blind(self) -> bool:
        return self.source == "blind"


_cached: ThermalReading | None = None
_cached_at: float = 0.0


def _read_nsprocessinfo() -> ThermalReading | None:
    try:
        from Foundation import NSProcessInfo  # type: ignore

        level = int(NSProcessInfo.processInfo().thermalState())
        return ThermalReading(level=max(0, min(3, level)), source="nsprocessinfo")
    # not a failure: None means this source has nothing to say, and the
    # caller tries the next one.
    except (ImportError, AttributeError, ValueError, TypeError):
        return None


def _read_pmset() -> ThermalReading | None:
    try:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        proc = get_subprocess_gateway().run(
            ["pmset", "-g", "therm"],
            timeout=_PMSET_TIMEOUT_S,
            read_only=True,
            source="runtime.thermal.pmset_probe",
            accelerator_capability="none",
        )
        out = proc.stdout or ""
    # not a failure: None means this source has nothing to say, and the
    # caller tries the next one.
    except (ImportError, OSError, RuntimeError, ValueError):
        return None
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    if not match:
        # pmset answered but reports nothing: treat as nominal evidence.
        return ThermalReading(level=0, source="pmset", detail="no warning recorded")
    limit = int(match.group(1))
    if limit >= 100:
        level = 0
    elif limit >= 80:
        level = 1
    elif limit >= 50:
        level = 2
    else:
        level = 3
    return ThermalReading(level=level, source="pmset", detail=f"cpu_speed_limit={limit}")


def _read_psutil() -> ThermalReading | None:
    try:
        import psutil

        sensors = getattr(psutil, "sensors_temperatures", None)
        if not callable(sensors):
            return None
        temps = sensors() or {}
    # not a failure: None means this source has nothing to say, and it is the
    # last one — the caller reports thermal state as unknown.
    except (ImportError, OSError, RuntimeError, AttributeError):
        return None
    hottest = 0.0
    for entries in temps.values():
        for entry in list(entries or []):
            current = getattr(entry, "current", None)
            if current is not None:
                hottest = max(hottest, float(current))
    if hottest <= 0.0:
        return None
    if hottest >= _TEMP_CRITICAL_C:
        level = 3
    elif hottest >= _TEMP_SERIOUS_C:
        level = 2
    else:
        level = 0
    return ThermalReading(level=level, source="psutil", detail=f"max_temp_c={hottest:.1f}")


def thermal_state(*, max_age_s: float = _CACHE_TTL_S) -> ThermalReading:
    """Current host thermal reading (cached ~5s). Never raises."""
    global _cached, _cached_at
    now = time.monotonic()
    if _cached is not None and (now - _cached_at) < max_age_s:
        return _cached
    reading = _read_nsprocessinfo() or _read_pmset() or _read_psutil()
    if reading is None:
        reading = ThermalReading(level=0, source="blind", detail="no thermal source available")
    if reading.level >= 2:
        logger.warning(
            "🔥 Host thermal pressure: level=%d via %s (%s)",
            reading.level,
            reading.source,
            reading.detail,
        )
    _cached = reading
    _cached_at = now
    return reading


def reset_thermal_cache() -> None:
    """Testing hook."""
    global _cached, _cached_at
    _cached = None
    _cached_at = 0.0


__all__ = ["ThermalReading", "thermal_state", "reset_thermal_cache"]
