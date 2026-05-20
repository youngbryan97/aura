"""core/actuators/actuator_registry.py
===================================
Open-Ended Actuators & Action Primitives.

Implements executable physical commands that modify the state of entities
in the PhysicsWorldModel. All operations are sandboxed and validated.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.Actuators")


@dataclass
class ActuatorResult:
    """The result of executing an action primitive."""
    success: bool
    message: str
    updates: Dict[str, Any]


class BaseActuator(ABC):
    """Abstract base class for all physical open-ended actuators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this actuator."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable explanation of what this actuator does."""
        pass

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validates that parameters satisfy all safety and physical constraints."""
        pass

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> ActuatorResult:
        """Executes the action on the PhysicsWorldModel."""
        pass


class RerouteVesselActuator(BaseActuator):
    """Actuator to adjust headings and speeds of maritime vessel edges."""

    @property
    def name(self) -> str:
        return "reroute_vessel"

    @property
    def description(self) -> str:
        return "Adjusts heading (degrees) and speed (knots) of a target maritime vessel edge."

    def validate_params(self, params: Dict[str, Any]) -> bool:
        vessel_id = params.get("vessel_id")
        heading = params.get("heading")
        speed = params.get("speed")

        if not vessel_id or heading is None or speed is None:
            return False

        try:
            heading_f = float(heading)
            speed_f = float(speed)
            if not (0.0 <= heading_f <= 360.0):
                return False
            if speed_f < 0.0 or speed_f > 40.0:  # Enforce maximum safe maritime speed
                return False
            return True
        except (ValueError, TypeError):
            return False

    def execute(self, params: Dict[str, Any]) -> ActuatorResult:
        if not self.validate_params(params):
            return ActuatorResult(False, "Parameter validation failed", {})

        try:
            from core.world.world_model import get_physics_world_model
            model = get_physics_world_model()
            vessel_id = str(params["vessel_id"])
            heading = float(params["heading"])
            speed = float(params["speed"])

            vessel = model.get_entity(vessel_id)
            if not vessel:
                return ActuatorResult(False, f"Vessel '{vessel_id}' not found", {})

            # Apply step update
            model.simulate(1.0, actions=[{
                "type": "reroute",
                "entity_id": vessel_id,
                "heading": heading,
                "speed": speed,
            }])

            logger.info("Executed Actuator: reroute_vessel %s to heading=%s, speed=%s", vessel_id, heading, speed)
            return ActuatorResult(
                success=True,
                message=f"Vessel '{vessel_id}' successfully rerouted.",
                updates={vessel_id: {"heading": heading, "speed": speed}}
            )

        except Exception as exc:
            return ActuatorResult(False, f"Actuator execution failed: {exc}", {})


class ReallocateFlowActuator(BaseActuator):
    """Actuator to transfer assets/cargo from one inventory node to another."""

    @property
    def name(self) -> str:
        return "reallocate_flow"

    @property
    def description(self) -> str:
        return "Transfers inventory quantity (units) between two nodes to relieve bottleneck pressure."

    def validate_params(self, params: Dict[str, Any]) -> bool:
        source_id = params.get("source_id")
        target_id = params.get("target_id")
        amount = params.get("amount")

        if not source_id or not target_id or amount is None:
            return False

        try:
            amount_f = float(amount)
            if amount_f <= 0.0:
                return False
            return True
        except (ValueError, TypeError):
            return False

    def execute(self, params: Dict[str, Any]) -> ActuatorResult:
        if not self.validate_params(params):
            return ActuatorResult(False, "Parameter validation failed", {})

        try:
            from core.world.world_model import get_physics_world_model
            model = get_physics_world_model()
            source_id = str(params["source_id"])
            target_id = str(params["target_id"])
            amount = float(params["amount"])

            source = model.get_entity(source_id)
            target = model.get_entity(target_id)

            if not source or not target:
                return ActuatorResult(False, "Source or target node not found", {})

            if source.load < amount:
                return ActuatorResult(False, f"Source '{source_id}' load {source.load} insufficient for transfer of {amount}", {})

            if target.load + amount > target.capacity:
                # Capacity constraint check
                transferable = target.capacity - target.load
                if transferable <= 0.0:
                    return ActuatorResult(False, f"Target '{target_id}' at maximum capacity {target.capacity}", {})
                amount = transferable  # Clip transfer

            model.simulate(1.0, actions=[{
                "type": "transfer",
                "entity_id": source_id,
                "target_id": target_id,
                "amount": amount
            }])

            logger.info("Executed Actuator: reallocate_flow transferred %s from %s to %s", amount, source_id, target_id)
            return ActuatorResult(
                success=True,
                message=f"Flow of {amount} successfully reallocated from '{source_id}' to '{target_id}'.",
                updates={
                    source_id: {"load": source.load},
                    target_id: {"load": target.load}
                }
            )

        except Exception as exc:
            return ActuatorResult(False, f"Actuator execution failed: {exc}", {})


class ActuatorRegistry:
    """Registry of executable physical open-ended actuators."""

    def __init__(self) -> None:
        self.actuators: Dict[str, BaseActuator] = {}
        self._register_default_actuators()

    def _register_default_actuators(self) -> None:
        self.register(RerouteVesselActuator())
        self.register(ReallocateFlowActuator())

    def register(self, actuator: BaseActuator) -> None:
        self.actuators[actuator.name] = actuator
        logger.info("Registered actuator: %s (%s)", actuator.name, actuator.description)

    def get_actuator(self, name: str) -> Optional[BaseActuator]:
        return self.actuators.get(name)

    def execute_action(self, name: str, params: Dict[str, Any]) -> ActuatorResult:
        """Safely retrieves and executes a physical action primitive."""
        actuator = self.get_actuator(name)
        if not actuator:
            return ActuatorResult(False, f"Actuator '{name}' not found", {})
        return actuator.execute(params)


# Singleton Pattern
_instance: Optional[ActuatorRegistry] = None


def get_actuator_registry() -> ActuatorRegistry:
    global _instance
    if _instance is None:
        _instance = ActuatorRegistry()
    return _instance
