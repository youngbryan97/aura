"""Hardware substrate telemetry adapters.

The resilience layer should not assume Darwin ``sysctl`` or mandatory
``psutil``. This module provides a small hardware-agnostic interface that can
sample process memory, CPU, and thermal pressure on macOS, Linux, and a
best-effort generic fallback.
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SubstrateMonitor")


@dataclass(frozen=True)
class SubstrateTelemetry:
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    thermal_level: int = 0
    thermal_pressure: float = 0.0
    psutil_available: bool = False
    source: str = "generic"


class SubstrateMonitor:
    """Pluggable substrate sampler used by resilience and world-state code."""

    def sample(self, *, process: Optional[Any] = None) -> SubstrateTelemetry:
        memory_mb = 0.0
        memory_percent = 0.0
        cpu_percent = 0.0
        psutil_available = False

        try:
            import psutil

            psutil_available = True
            proc = process or psutil.Process(os.getpid())
            memory_mb = proc.memory_info().rss / (1024 * 1024)
            memory_percent = float(proc.memory_percent())
            raw_cpu = float(proc.cpu_percent(interval=0.1))
            cores = max(1, int(psutil.cpu_count() or 1))
            cpu_percent = raw_cpu / cores
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("substrate_monitor", exc)
            memory_mb = _resource_rss_mb()
            cpu_percent = _load_average_percent()

        thermal_level, thermal_pressure, source = self.thermal()
        return SubstrateTelemetry(
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            cpu_percent=cpu_percent,
            thermal_level=thermal_level,
            thermal_pressure=thermal_pressure,
            psutil_available=psutil_available,
            source=source,
        )

    def thermal(self) -> tuple[int, float, str]:
        system = platform.system()
        if system == "Darwin":
            level = _darwin_thermal_level()
            return level, min(1.0, max(0.0, level / 3.0)), "darwin_sysctl"
        if system == "Linux":
            level, pressure = _linux_thermal()
            return level, pressure, "linux_thermal_zone"
        if system == "Windows":
            level, pressure = _windows_thermal()
            return level, pressure, "windows_wmi"
        return 0, 0.0, "generic"


def _darwin_thermal_level() -> int:
    for key in ("hw.thermallevel", "kern.thermal_pressure"):
        try:
            result = subprocess.run(
                ["sysctl", "-n", key],
                capture_output=True,
                text=True,
                check=False,
                timeout=1.0,
            )
            if result.returncode == 0 and result.stdout.strip():
                return max(0, min(3, int(result.stdout.strip())))
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            logger.debug("Darwin thermal probe %s failed: %s", key, exc)
    return 0


def _linux_thermal() -> tuple[int, float]:
    temps_c: list[float] = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                continue
            value = float(raw)
            if value > 1000:
                value /= 1000.0
            if 0.0 < value < 150.0:
                temps_c.append(value)
        except (OSError, ValueError):
            continue
    if not temps_c:
        return 0, 0.0
    max_temp = max(temps_c)
    pressure = min(1.0, max(0.0, (max_temp - 60.0) / 35.0))
    if max_temp >= 90.0:
        level = 3
    elif max_temp >= 80.0:
        level = 2
    elif max_temp >= 70.0:
        level = 1
    else:
        level = 0
    return level, pressure


def _windows_thermal() -> tuple[int, float]:
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance MSAcpi_ThermalZoneTemperature -Namespace root/wmi "
                "| Select-Object -First 1 -ExpandProperty CurrentTemperature)",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0, 0.0
        kelvin_tenths = float(result.stdout.strip().splitlines()[0])
        celsius = (kelvin_tenths / 10.0) - 273.15
        pressure = min(1.0, max(0.0, (celsius - 60.0) / 35.0))
        if celsius >= 90.0:
            return 3, pressure
        if celsius >= 80.0:
            return 2, pressure
        if celsius >= 70.0:
            return 1, pressure
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        logger.debug("Windows thermal probe failed: %s", exc)
    return 0, 0.0


def _resource_rss_mb() -> float:
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system() == "Darwin":
            return rss / (1024 * 1024)
        return rss / 1024.0
    except (ImportError, OSError, AttributeError, ValueError):
        return 0.0


def _load_average_percent() -> float:
    try:
        load1, _, _ = os.getloadavg()
        cores = max(1, os.cpu_count() or 1)
        return min(100.0, max(0.0, (load1 / cores) * 100.0))
    except (OSError, AttributeError):
        return 0.0


__all__ = ["SubstrateMonitor", "SubstrateTelemetry"]
