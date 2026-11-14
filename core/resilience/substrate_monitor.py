"""Hardware substrate telemetry adapters.

The resilience layer should not assume Darwin ``sysctl`` or mandatory
``psutil``. This module provides a small hardware-agnostic interface that can
sample process memory, CPU, and thermal pressure on macOS, Linux, and a
best-effort generic fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.resource_observation import ResourceObserver, get_resource_observer


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

    def __init__(self, *, observer: ResourceObserver | None = None) -> None:
        self._observer = observer

    @property
    def observer(self) -> ResourceObserver:
        return self._observer or get_resource_observer()

    def sample(self, *, process: Any | None = None) -> SubstrateTelemetry:
        observer = self.observer
        provenance = observer.provenance
        memory = observer.memory()
        compute = observer.compute()
        memory_mb = float(memory.process_rss_bytes) / float(1024**2)
        memory_percent = float(memory.percent)
        cpu_percent = float(compute.cpu_percent)
        observation_available = bool(memory.available and compute.cpu_percent_readable)
        source_prefix = provenance.source.value

        if process is not None:
            try:
                memory_mb = float(process.memory_info().rss) / float(1024**2)
                memory_percent = float(process.memory_percent())
                raw_cpu = float(process.cpu_percent(interval=0.1))
                cpu_percent = raw_cpu / max(1, int(compute.cpu_count))
                source_prefix = f"explicit_process+{source_prefix}"
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation("substrate_monitor", exc)

        thermal_level, thermal_pressure, source = self.thermal()
        return SubstrateTelemetry(
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            cpu_percent=cpu_percent,
            thermal_level=thermal_level,
            thermal_pressure=thermal_pressure,
            psutil_available=observation_available,
            source=f"{source_prefix}:{source}",
        )

    def thermal(self) -> tuple[int, float, str]:
        reading = self.observer.thermal()
        level = max(0, min(3, int(reading.level)))
        pressure = min(1.0, max(0.0, level / 3.0))
        return level, pressure, reading.provider


__all__ = ["SubstrateMonitor", "SubstrateTelemetry"]
