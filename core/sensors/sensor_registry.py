"""core/sensors/sensor_registry.py
=============================
Grounded Sensor Registry.

Manages all physical sensors mapping environmental and system variables
to live telemetry signals. Keeps rolling historical buffers of sensor readings.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

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
    last_updated: float = field(default_factory=lambda: time.time())

    def record(self, value: float) -> bool:
        """Records a new sensor reading into the rolling buffer."""
        try:
            reading = float(value)
        except (TypeError, ValueError):
            logger.warning("Rejected non-numeric reading for sensor %s: %r", self.sensor_id, value)
            return False
        if not math.isfinite(reading):
            logger.warning("Rejected non-finite reading for sensor %s: %r", self.sensor_id, value)
            return False
        self.current_value = reading
        self.history.append(self.current_value)
        self.last_updated = time.time()
        return True


class SensorRegistry:
    """Manages the register of physical sensors and telemetry ingestion."""

    def __init__(self) -> None:
        self.sensors: dict[str, PhysicalSensor] = {}
        self._initialize_default_sensors()

    def _initialize_default_sensors(self) -> None:
        """Registers canonical physical sensors matching our World Model entities."""
        self.register(PhysicalSensor("port_east_load", "East Port Cargo Queue Load", "containers"), log=False)
        self.register(PhysicalSensor("port_west_load", "West Port Cargo Queue Load", "containers"), log=False)
        self.register(
            PhysicalSensor("port_east_latency", "East Port Bottleneck Waiting Delay", "hours"),
            log=False,
        )
        self.register(
            PhysicalSensor("port_west_latency", "West Port Bottleneck Waiting Delay", "hours"),
            log=False,
        )
        self.register(
            PhysicalSensor("vessel_alpha_speed", "Vessel Alpha Current Flow Velocity", "knots"),
            log=False,
        )
        self.register(
            PhysicalSensor("warehouse_load", "Central Warehouse Inventory Level", "units"),
            log=False,
        )
        self.register(
            PhysicalSensor("warehouse_latency", "Central Warehouse Delivery Wait Time", "hours"),
            log=False,
        )
        self.register(
            PhysicalSensor("system_cpu_usage", "System Core CPU Load Percentage", "percent"),
            log=False,
        )
        self._initialize_runtime_sensors()
        logger.debug("Initialized %d default sensors.", len(self.sensors))

    def _initialize_runtime_sensors(self) -> None:
        """Sensors describing AURA, not a shipping network.

        CP126 fae09510: the immune system's behavioural rules are generated
        over whatever sensors exist, and the only ones that existed described
        ports, vessels and a warehouse. So every B cell Aura evolved was an
        opinion about maritime logistics, and the adaptive population began
        with no code, service, queue, model, network, memory or UI repair
        grammar — the vocabulary was the ceiling on what it could ever learn
        to repair.

        These name the subsystems that actually fail. A sensor with no
        recorded reading simply has none; declaring it is what lets a repair
        rule be ABOUT it, and what makes the absence of a reading visible
        rather than the subsystem being invisible.
        """
        runtime_sensors = (
            ("runtime_memory_pressure", "Process resident memory against its budget", "ratio"),
            ("runtime_event_loop_lag", "Event-loop scheduling delay", "seconds"),
            ("runtime_task_queue_depth", "Pending tracked runtime tasks", "tasks"),
            ("runtime_degradation_rate", "Recorded degradations per minute", "per_minute"),
            ("model_inference_latency", "Model generation latency", "seconds"),
            ("model_context_utilisation", "Prompt tokens against the context window", "ratio"),
            ("model_worker_restarts", "Inference worker restarts", "count"),
            ("memory_recall_hit_rate", "Recall attempts returning a usable result", "ratio"),
            ("memory_store_latency", "Memory write latency", "seconds"),
            ("storage_write_failures", "Failed durable writes", "count"),
            ("network_request_failure_rate", "Outbound request failures", "ratio"),
            ("service_unavailable_count", "Service-container lookups returning nothing", "count"),
            ("interface_response_latency", "User-facing response latency", "seconds"),
            ("interface_error_rate", "Errors surfaced to the interface", "ratio"),
        )
        for sensor_id, description, unit in runtime_sensors:
            self.register(PhysicalSensor(sensor_id, description, unit), log=False)

    def register(self, sensor: PhysicalSensor, *, log: bool = True) -> None:
        self.sensors[sensor.sensor_id] = sensor
        if log:
            logger.info("Registered sensor: %s (%s)", sensor.sensor_id, sensor.description)

    def get_sensor(self, sensor_id: str) -> PhysicalSensor | None:
        return self.sensors.get(sensor_id)

    def record_reading(self, sensor_id: str, value: float) -> bool:
        """Records a new reading for a registered sensor."""
        sensor = self.get_sensor(sensor_id)
        if sensor:
            return sensor.record(value)
        return False

    def read_all(self) -> dict[str, float]:
        """Returns the current state vector of all live sensor values."""
        return {sid: s.current_value for sid, s in self.sensors.items()}

    def get_reliability_vector(self) -> dict[str, float]:
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

            # Sync host CPU telemetry through psutil.
            from core.runtime import resource_psutil as psutil

            self.record_reading("system_cpu_usage", psutil.cpu_percent())

        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Failed to sync sensors from world model: %s", exc)


# Singleton Pattern
_instance: SensorRegistry | None = None


def get_sensor_registry() -> SensorRegistry:
    global _instance
    if _instance is None:
        _instance = SensorRegistry()
    return _instance
