"""core/body/environment_snapshot.py
Sensor capturing physical and system environment snapshots.
Reads system thermals, battery level, CPU performance, and memory pressures.
"""
import time
from typing import Any, Dict

from core.body.sensor_registry import BaseSensor
from core.runtime import resource_psutil as psutil
from core.runtime.errors import record_degradation
from core.runtime.resource_observation import get_resource_observer

_ENVIRONMENT_SENSOR_ERRORS = (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError)


class EnvironmentSnapshotSensor(BaseSensor):
    """Monitors OS telemetry, storage levels, and host hardware states."""

    @property
    def name(self) -> str:
        return "environment_snapshot"

    async def read(self) -> Dict[str, Any]:
        observer = get_resource_observer()
        provenance = observer.provenance
        # CPU levels
        cpu_pct = psutil.cpu_percent(interval=None)
        
        # RAM memory pressure
        mem = psutil.virtual_memory()
        
        # Disk usage
        disk = observer.disk("/")
        disk_pct = float(disk.percent) if disk.available else 100.0

        # Thermal estimations (fallback to standard macOS levels if no platform temperature sensor)
        thermal_c = 42.0
        try:
            temps = getattr(psutil, "sensors_temperatures", None)
            if temps:
                t_sensors = temps()
                if t_sensors:
                    thermal_c = next(iter(next(iter(t_sensors.values()))), None).current
        except _ENVIRONMENT_SENSOR_ERRORS as exc:
            record_degradation("body.environment_snapshot.thermal", exc)

        # Battery tracking
        battery_pct = 100.0
        power_plugged = True
        try:
            bat = getattr(psutil, "sensors_battery", None)
            if bat:
                b_info = bat()
                if b_info:
                    battery_pct = b_info.percent
                    power_plugged = b_info.power_plugged
        except _ENVIRONMENT_SENSOR_ERRORS as exc:
            record_degradation("body.environment_snapshot.battery", exc)

        return {
            "timestamp": time.time(),
            "cpu_percent": cpu_pct,
            "memory_percent": mem.percent,
            "memory_available_gb": mem.available / (1024 ** 3),
            "disk_percent": disk_pct,
            "disk_free_gb": disk.free_bytes / (1024 ** 3),
            "temperature_c": thermal_c,
            "battery_percent": battery_pct,
            "battery_plugged": power_plugged,
            "observation_source": provenance.source.value,
            "observation_scenario_id": provenance.scenario_id,
        }
