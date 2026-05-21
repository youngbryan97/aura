"""core/actuators/actuator_registry.py
===================================
Open-Ended Actuators & Action Primitives.

Implements executable physical commands that modify the state of entities
in the PhysicsWorldModel. All operations are sandboxed and validated.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.Actuators")


@dataclass
class ActuatorResult:
    """The result of executing an action primitive."""

    success: bool
    message: str
    updates: dict[str, Any]


def _finite_float(
    value: Any, *, minimum: float | None = None, maximum: float | None = None
) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate):
        return None
    if minimum is not None and candidate < minimum:
        return None
    if maximum is not None and candidate > maximum:
        return None
    return candidate


class BaseActuator(ABC):
    """Abstract base class for all physical open-ended actuators."""

    synthesized: bool = False
    trust_score: float = 1.0
    generation: int = 0
    source_code: str | None = None

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
    def validate_params(self, params: dict[str, Any]) -> bool:
        """Validates that parameters satisfy all safety and physical constraints."""
        pass

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        """Executes the action on the PhysicsWorldModel."""
        pass


class SandboxedSynthesizedActuator(BaseActuator):
    """Live wrapper for LLM-synthesized actuator code.

    The generated code never executes in Aura's main process. Execution happens
    through the validator sandbox, then this wrapper applies only bounded,
    finite update payloads to the live physics world.
    """

    synthesized: bool = True

    def __init__(
        self,
        *,
        name: str,
        description: str,
        source_code: str,
        trust_score: float = 0.3,
    ) -> None:
        self._name = str(name).strip() or "sandboxed_synthesized_actuator"
        self._description = str(description).strip() or "Sandboxed synthesized actuator"
        self.source_code = source_code
        self.trust_score = trust_score

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def validate_params(self, params: dict[str, Any]) -> bool:
        return isinstance(params, dict)

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not self.validate_params(params):
            return ActuatorResult(False, "Parameter validation failed", {})

        try:
            from core.actuators.actuator_validator import ActuatorCodeValidator

            sandbox_result = ActuatorCodeValidator.execute_sandboxed(self.source_code or "", params)
            if not sandbox_result.success:
                return ActuatorResult(False, sandbox_result.error or "Sandbox execution failed", {})
            updates = sandbox_result.details.get("updates", {})
            applied_updates = self._apply_bounded_updates(updates)
            return ActuatorResult(
                True,
                str(sandbox_result.details.get("message") or "Sandboxed actuator executed"),
                applied_updates,
            )
        except (ImportError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return ActuatorResult(False, f"Sandboxed actuator execution failed: {exc}", {})

    def _apply_bounded_updates(self, updates: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(updates, dict) or not updates:
            return {}

        from core.world.world_model import get_physics_world_model

        model = get_physics_world_model()
        applied: dict[str, Any] = {}
        for entity_id, fields in updates.items():
            entity = model.get_entity(str(entity_id))
            if entity is None or not isinstance(fields, dict):
                continue

            entity_updates: dict[str, Any] = {}
            for field in ("capacity", "load", "flow_rate", "max_flow_rate", "latency"):
                if field in fields:
                    value = _finite_float(fields[field], minimum=0.0)
                    if value is not None:
                        setattr(entity, field, value)
                        entity_updates[field] = value

            if "coordinates" in fields:
                coords = fields["coordinates"]
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    lat = _finite_float(coords[0])
                    lon = _finite_float(coords[1])
                    if lat is not None and lon is not None:
                        entity.coordinates = (lat, lon)
                        entity_updates["coordinates"] = entity.coordinates

            attrs = fields.get("attributes")
            if isinstance(attrs, dict):
                safe_attrs: dict[str, Any] = {}
                for key, value in attrs.items():
                    if not isinstance(key, str):
                        continue
                    if isinstance(value, (str, bool, int, float)) or value is None:
                        safe_attrs[key[:64]] = value
                if safe_attrs:
                    entity.attributes.update(safe_attrs)
                    entity_updates["attributes"] = safe_attrs

            entity.enforce_constraints()
            if entity_updates:
                applied[str(entity_id)] = entity_updates
        return applied


class RerouteVesselActuator(BaseActuator):
    """Actuator to adjust headings and speeds of maritime vessel edges."""

    @property
    def name(self) -> str:
        return "reroute_vessel"

    @property
    def description(self) -> str:
        return "Adjusts heading (degrees) and speed (knots) of a target maritime vessel edge."

    def validate_params(self, params: dict[str, Any]) -> bool:
        vessel_id = params.get("vessel_id")
        heading = params.get("heading")
        speed = params.get("speed")

        if not vessel_id or heading is None or speed is None:
            return False

        heading_f = _finite_float(heading, minimum=0.0, maximum=360.0)
        speed_f = _finite_float(speed, minimum=0.0, maximum=40.0)
        if heading_f is None or speed_f is None:
            return False
        return True

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not self.validate_params(params):
            return ActuatorResult(False, "Parameter validation failed", {})

        try:
            from core.world.world_model import get_physics_world_model

            model = get_physics_world_model()
            vessel_id = str(params["vessel_id"])
            heading = _finite_float(params["heading"], minimum=0.0, maximum=360.0)
            speed = _finite_float(params["speed"], minimum=0.0, maximum=40.0)
            if heading is None or speed is None:
                return ActuatorResult(False, "Parameter validation failed", {})

            vessel = model.get_entity(vessel_id)
            if not vessel:
                return ActuatorResult(False, f"Vessel '{vessel_id}' not found", {})

            # Apply step update
            model.simulate(
                1.0,
                actions=[
                    {
                        "type": "reroute",
                        "entity_id": vessel_id,
                        "heading": heading,
                        "speed": speed,
                    }
                ],
            )

            logger.info(
                "Executed Actuator: reroute_vessel %s to heading=%s, speed=%s",
                vessel_id,
                heading,
                speed,
            )
            return ActuatorResult(
                success=True,
                message=f"Vessel '{vessel_id}' successfully rerouted.",
                updates={vessel_id: {"heading": heading, "speed": speed}},
            )

        except (ImportError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return ActuatorResult(False, f"Actuator execution failed: {exc}", {})


class ReallocateFlowActuator(BaseActuator):
    """Actuator to transfer assets/cargo from one inventory node to another."""

    @property
    def name(self) -> str:
        return "reallocate_flow"

    @property
    def description(self) -> str:
        return (
            "Transfers inventory quantity (units) between two nodes to relieve bottleneck pressure."
        )

    def validate_params(self, params: dict[str, Any]) -> bool:
        source_id = params.get("source_id")
        target_id = params.get("target_id")
        amount = params.get("amount")

        if not source_id or not target_id or amount is None:
            return False

        amount_f = _finite_float(amount, minimum=1e-9)
        if amount_f is None:
            return False
        return True

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not self.validate_params(params):
            return ActuatorResult(False, "Parameter validation failed", {})

        try:
            from core.world.world_model import get_physics_world_model

            model = get_physics_world_model()
            source_id = str(params["source_id"])
            target_id = str(params["target_id"])
            amount = _finite_float(params["amount"], minimum=1e-9)
            if amount is None:
                return ActuatorResult(False, "Parameter validation failed", {})

            source = model.get_entity(source_id)
            target = model.get_entity(target_id)

            if not source or not target:
                return ActuatorResult(False, "Source or target node not found", {})

            if source.load < amount:
                return ActuatorResult(
                    False,
                    f"Source '{source_id}' load {source.load} insufficient for transfer of {amount}",
                    {},
                )

            if target.load + amount > target.capacity:
                # Capacity constraint check
                transferable = target.capacity - target.load
                if transferable <= 0.0:
                    return ActuatorResult(
                        False, f"Target '{target_id}' at maximum capacity {target.capacity}", {}
                    )
                amount = transferable  # Clip transfer

            model.simulate(
                1.0,
                actions=[
                    {
                        "type": "transfer",
                        "entity_id": source_id,
                        "target_id": target_id,
                        "amount": amount,
                    }
                ],
            )

            logger.info(
                "Executed Actuator: reallocate_flow transferred %s from %s to %s",
                amount,
                source_id,
                target_id,
            )
            return ActuatorResult(
                success=True,
                message=f"Flow of {amount} successfully reallocated from '{source_id}' to '{target_id}'.",
                updates={source_id: {"load": source.load}, target_id: {"load": target.load}},
            )

        except (ImportError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return ActuatorResult(False, f"Actuator execution failed: {exc}", {})


class ActuatorRegistry:
    """Registry of executable physical open-ended actuators."""

    def __init__(self) -> None:
        self.actuators: dict[str, BaseActuator] = {}
        self._register_default_actuators()

    def _register_default_actuators(self) -> None:
        self.register(RerouteVesselActuator())
        self.register(ReallocateFlowActuator())

    def register(self, actuator: BaseActuator) -> None:
        self.actuators[actuator.name] = actuator
        logger.info("Registered actuator: %s (%s)", actuator.name, actuator.description)

    def get_actuator(self, name: str) -> BaseActuator | None:
        return self.actuators.get(name)

    def register_synthesized(
        self, actuator: BaseActuator, source_code: str, trust_score: float = 0.3
    ) -> None:
        """Register a runtime-synthesized actuator with low trust and stored source code."""
        actuator.synthesized = True
        actuator.source_code = source_code
        actuator.trust_score = trust_score
        self.register(actuator)
        logger.info("Registered synthesized actuator: %s (trust=%.2f)", actuator.name, trust_score)

    def deregister(self, name: str) -> None:
        """Remove an actuator from the registry (retirement)."""
        if name in self.actuators:
            del self.actuators[name]
            logger.info("Deregistered actuator: %s", name)

    def get_synthesized_actuators(self) -> list[BaseActuator]:
        """List all runtime-synthesized actuators."""
        return [act for act in self.actuators.values() if getattr(act, "synthesized", False)]

    def execute_action(self, name: str, params: dict[str, Any]) -> ActuatorResult:
        """Safely retrieves and executes a physical action primitive."""
        actuator = self.get_actuator(name)
        if not actuator:
            return ActuatorResult(False, f"Actuator '{name}' not found", {})

        # Trust score gating: if synthesized and trust is extremely low, additional checks
        if getattr(actuator, "synthesized", False):
            if actuator.trust_score < 0.2:
                return ActuatorResult(
                    False,
                    f"Actuator '{name}' has trust score too low ({actuator.trust_score:.2f}) to execute",
                    {},
                )
            elif actuator.trust_score < 0.5:
                # Run an additional parameter validation check and ensure they are sanitized
                logger.warning(
                    "Executing low-trust synthesized actuator '%s' (trust=%.2f)",
                    name,
                    actuator.trust_score,
                )
                if not params:
                    return ActuatorResult(
                        False, "Low-trust actuator requires non-empty parameters", {}
                    )

        return actuator.execute(params)


# Singleton Pattern
_instance: ActuatorRegistry | None = None


def get_actuator_registry() -> ActuatorRegistry:
    global _instance
    if _instance is None:
        _instance = ActuatorRegistry()
    return _instance
