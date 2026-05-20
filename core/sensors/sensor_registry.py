"""core/sensors/sensor_registry.py
=============================
Grounded Sensor Registry.

Manages all physical sensors mapping environmental and system variables
to live telemetry signals. Keeps rolling historical buffers of sensor readings.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.SensorRegistry")


@dataclass
class PhysicalSensor:
    """Represents a live physical sensor telemetry source."""
    sensor_id: str
    description: str
    unit: str
    current_value: float = 0.0
    reliability: float = 1.0  # Weight in free energy / prediction calculations
    history_limit: int = 100
    history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_updated: float = field(default_factory=time.time)

    def record(self, value: float) -> None:
        """Records a new sensor reading into the rolling buffer."""
        self.current_value = float(value)
        self.history.append(self.current_value)
        self.last_updated = time.time()


class SensorRegistry:
    """Manages the register of physical sensors and telemetry ingestion."""

    def __init__(self) -> None:
        self.sensors: Dict[str, PhysicalSensor] = {}
        self._initialize_default_sensors()

    def _initialize_default_sensors(self) -> None:
        """Registers canonical physical sensors matching our World Model entities."""
        self.register(PhysicalSensor("port_east_load", "East Port Cargo Queue Load", "containers"))
        self.register(PhysicalSensor("port_west_load", "West Port Cargo Queue Load", "containers"))
        self.register(PhysicalSensor("port_east_latency", "East Port Bottleneck Waiting Delay", "hours"))
        self.register(PhysicalSensor("port_west_latency", "West Port Bottleneck Waiting Delay", "hours"))
        self.register(PhysicalSensor("vessel_alpha_speed", "Vessel Alpha Current Flow Velocity", "knots"))
        self.register(PhysicalSensor("warehouse_load", "Central Warehouse Inventory Level", "units"))
        self.register(PhysicalSensor("warehouse_latency", "Central Warehouse Delivery Wait Time", "hours"))
        self.register(PhysicalSensor("system_cpu_usage", "System Core CPU Load Percentage", "percent"))

    def register(self, sensor: PhysicalSensor) -> None:
        self.sensors[sensor.sensor_id] = sensor
        logger.info("Registered sensor: %s (%s)", sensor.sensor_id, sensor.description)

    def get_sensor(self, sensor_id: str) -> Optional[PhysicalSensor]:
        return self.sensors.get(sensor_id)

    def record_reading(self, sensor_id: str, value: float) -> bool:
        """Records a new reading for a registered sensor."""
        sensor = self.get_sensor(sensor_id)
        if sensor:
            sensor.record(value)
            return True
        return False

    def read_all(self) -> Dict[str, float]:
        """Returns the current state vector of all live sensor values."""
        return {sid: s.current_value for sid, s in self.sensors.items()}

    def get_reliability_vector(self) -> Dict[str, float]:
        """Returns reliability scale for all registered sensors."""
        return {sid: s.reliability for sid, s in self.sensors.items()}

    def sync_from_world_model(self) -> None:
        """Pulls physical state parameters from the active PhysicsWorldModel."""
        try:
            from core.world.world_model import get_physics_world_model
            model = get_physics_world_model()
            snapshot = model.get_state_snapshot()
            entities = snapshot.get("entities", {})

            # Sync East Port
            pe = entities.get("Port_East", {})
            if pe:
                self.record_reading("port_east_load", pe.get("load", 0.0))
                self.record_reading("port_east_latency", pe.get("latency", 0.0))

            # Sync West Port
            pw = entities.get("Port_West", {})
            if pw:
                self.record_reading("port_west_load", pw.get("load", 0.0))
                self.record_reading("port_west_latency", pw.get("latency", 0.0))

            # Sync Vessel Alpha
            va = entities.get("Vessel_Alpha", {})
            if va:
                self.record_reading("vessel_alpha_speed", va.get("flow_rate", 0.0))

            # Sync Warehouse Central
            wh = entities.get("Warehouse_Central", {})
            if wh:
                self.record_reading("warehouse_load", wh.get("load", 0.0))
                self.record_reading("warehouse_latency", wh.get("latency", 0.0))

            # Sync dummy system cpu usage
            import psutil
            self.record_reading("system_cpu_usage", psutil.cpu_percent())

        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Failed to sync sensors from world model: %s", exc)


# Singleton Pattern
_instance: Optional[SensorRegistry] = None


def get_sensor_registry() -> SensorRegistry:
    global _instance
    if _instance is None:
        _instance = SensorRegistry()
    return _instance
