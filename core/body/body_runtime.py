"""core/body/body_runtime.py
Central controller for Aura's perceptual body, coordinating all sensors.
Includes homeostatic health indicators, resource scaling policies, and diagnostics.
"""
import logging
import os
from typing import Any

from core.body.app_focus_sensor import AppFocusSensor
from core.body.browser_state_sensor import BrowserStateSensor
from core.body.camera_sensor import CameraSensor
from core.body.clipboard_sensor import ClipboardSensor
from core.body.environment_snapshot import EnvironmentSnapshotSensor
from core.body.filesystem_sensor import FilesystemSensor
from core.body.keyboard_mouse_state import KeyboardMouseSensor
from core.body.microphone_sensor import MicrophoneSensor
from core.body.screen_sensor import ScreenSensor
from core.body.sensor_registry import get_sensor_registry
from core.body.ui_accessibility_sensor import UiAccessibilitySensor
from core.runtime.errors import record_degradation

logger = logging.getLogger("Body.BodyRuntime")


class BodyRuntime:
    """Manages structural sensory perception and homeostatic state transitions."""

    def __init__(self):
        self.registry = get_sensor_registry()
        self._initialized = False

    def initialize_sensors(self) -> None:
        """Register default sensor plugins."""
        if self._initialized:
            return
        
        self.registry.register(ScreenSensor())
        self.registry.register(MicrophoneSensor())
        self.registry.register(CameraSensor())
        self.registry.register(KeyboardMouseSensor())
        self.registry.register(AppFocusSensor())
        self.registry.register(ClipboardSensor())
        self.registry.register(FilesystemSensor())
        self.registry.register(BrowserStateSensor())
        self.registry.register(UiAccessibilitySensor())
        self.registry.register(EnvironmentSnapshotSensor())
        
        self._initialized = True
        logger.info("Perceptual body sensors initialized successfully.")

    async def perceive_all(self, state: Any | None = None) -> dict[str, Any]:
        """Poll all sensors and consolidate results."""
        self.initialize_sensors()
        world_model = getattr(state, "world_model", {}) if state is not None else {}
        if isinstance(world_model, dict) and world_model.get("sensor_blackout"):
            logger.warning("Sensor blackout active. Returning degraded sensor readings.")
            return {
                name: {
                    "status": "degraded",
                    "error": "Sensor blackout active due to policy or testing",
                    "has_optical_feed": False,
                    "available": False
                }
                for name in self.registry.list_sensors()
            }
        return await self.registry.read_all()

    async def get_system_status(self) -> dict[str, Any]:
        """Utility extracting vital body stats to feed LifeState directly."""
        self.initialize_sensors()
        
        env_sensor = self.registry.get_sensor("environment_snapshot")
        focus_sensor = self.registry.get_sensor("app_focus")
        clip_sensor = self.registry.get_sensor("clipboard")

        status = {}
        if env_sensor:
            env_data = await env_sensor.read()
            status["cpu"] = env_data.get("cpu_percent", 10.0)
            status["memory"] = env_data.get("memory_percent", 50.0)
            status["battery"] = env_data.get("battery_percent", 100.0)
            status["temperature"] = env_data.get("temperature_c", 42.0)
        
        if focus_sensor:
            focus_data = await focus_sensor.read()
            status["focus_app"] = focus_data.get("active_app", "Terminal")
        
        if clip_sensor:
            clip_data = await clip_sensor.read()
            status["clipboard"] = clip_data.get("content", "")

        return status

    def calculate_resource_scaling(self, status: dict[str, Any]) -> dict[str, Any]:
        """Enforces homeostatic regulation recommendations based on resource load."""
        memory_usage = self._bounded_float(status.get("memory", 50.0), default=50.0)
        cpu_usage = self._bounded_float(status.get("cpu", 10.0), default=10.0)
        temp = self._bounded_float(status.get("temperature", 42.0), default=42.0, upper=130.0)
        from core.config import get_config

        llm_config = get_config().llm
        cortex_model = os.getenv("AURA_DEFAULT_CORTEX_MODEL", llm_config.fast_model)
        pressure_model = os.getenv("AURA_PRESSURE_MODEL", llm_config.chat_model)

        # Base targets
        scaling = {
            "model_capacity": cortex_model,
            "unload_vision_worker": False,
            "defer_dream_cycles": False,
            "compress_context": False,
            "governance_integrity": 1.0
        }

        # Memory pressure triggers
        if memory_usage > 80.0:
            scaling["model_capacity"] = pressure_model
            scaling["unload_vision_worker"] = True
            scaling["compress_context"] = True
            logger.warning("High memory pressure detected (%.1f%%). Recommending downscaled capacity.", memory_usage)

        # Thermal pressure triggers
        if temp > 75.0 or cpu_usage > 90.0:
            scaling["defer_dream_cycles"] = True
            logger.warning("Elevated thermal pressure detected (%.1fC). Deferring offline training cycles.", temp)

        return scaling

    @staticmethod
    def summarize_sensor_health(observations: dict[str, Any]) -> dict[str, bool]:
        """Summarize sensor liveness from actual read outcomes."""
        health: dict[str, bool] = {}
        for name, reading in observations.items():
            if not isinstance(reading, dict):
                health[str(name)] = False
                continue
            status = str(reading.get("status", "")).lower()
            available = reading.get("available")
            has_error = bool(reading.get("error"))
            health[str(name)] = bool(
                not has_error
                and available is not False
                and status not in {"degraded", "error", "failed", "unavailable"}
            )
        return health

    @staticmethod
    def summarize_actuator_health() -> dict[str, bool]:
        """Summarize available motor channels from the canonical action body."""
        try:
            from core.body.action_body import get_action_body
            action_body = get_action_body()
            action_body.initialize_motors()
            motors = set(action_body.controller.list_motors())
        except (AttributeError, ImportError, LookupError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("body.runtime.actuator_health", exc)
            return {}
        return {name: True for name in sorted(motors)}

    @staticmethod
    def _bounded_float(
        value: Any,
        *,
        default: float,
        lower: float = 0.0,
        upper: float = 100.0,
    ) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return max(lower, min(upper, parsed))


# Singleton Access
_body_runtime: BodyRuntime | None = None


def get_body_runtime() -> BodyRuntime:
    global _body_runtime
    if _body_runtime is None:
        _body_runtime = BodyRuntime()
    return _body_runtime
