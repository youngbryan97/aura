"""core/body/motor_controller.py
Motor actuator registry and base class configurations for Action Body.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("Body.MotorController")


class BaseMotor(ABC):
    """Abstract interface for somatic actuators mapping intent to action outcomes."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def actuate(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute motor activation. Returns receipt details."""
        raise NotImplementedError


class MotorController:
    """Registry maintaining available physical or virtual motor channels."""

    def __init__(self):
        self._motors: dict[str, BaseMotor] = {}

    def register(self, motor: BaseMotor) -> None:
        self._motors[motor.name] = motor
        logger.info("Registered motor actuator: %s", motor.name)

    def get_motor(self, name: str) -> BaseMotor | None:
        return self._motors.get(name)

    def list_motors(self) -> list[str]:
        return list(self._motors.keys())


# Global singleton registry
_motor_controller = MotorController()


def get_motor_controller() -> MotorController:
    return _motor_controller
