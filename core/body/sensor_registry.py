"""core/body/sensor_registry.py
Sensor registry and registration system for Aura's perceptual body.
"""
import logging
import re
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Body.SensorRegistry")

_SENSOR_READ_ERRORS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class BaseSensor:
    """Abstract base class for all environmental and physical body sensors."""

    @property
    def name(self) -> str:
        configured_name = getattr(self, "sensor_name", None)
        if isinstance(configured_name, str) and configured_name:
            return configured_name
        class_name = self.__class__.__name__
        normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
        return normalized.removesuffix("_sensor")

    async def initialize(self) -> None:
        return None

    async def read(self) -> dict[str, Any]:
        return {}


class SensorRegistry:
    """Registry maintaining active perceptual body sensors."""

    def __init__(self):
        self._sensors: dict[str, BaseSensor] = {}

    def register(self, sensor: BaseSensor) -> None:
        """Register a sensor into the perceptual body catalog."""
        self._sensors[sensor.name] = sensor
        logger.info("Registered sensor: %s", sensor.name)

    def get_sensor(self, name: str) -> BaseSensor | None:
        return self._sensors.get(name)

    def list_sensors(self) -> list[str]:
        return list(self._sensors.keys())

    async def read_all(self) -> dict[str, Any]:
        """Poll all active sensors simultaneously."""
        results = {}
        for name, sensor in self._sensors.items():
            try:
                results[name] = await sensor.read()
            except _SENSOR_READ_ERRORS as e:
                record_degradation("body.sensor_registry.read", e)
                logger.warning("Failed to read sensor %s: %s", name, e)
                results[name] = {"error": str(e)}
        return results


# Global Registry Instance
_registry = SensorRegistry()


def get_sensor_registry() -> SensorRegistry:
    return _registry
