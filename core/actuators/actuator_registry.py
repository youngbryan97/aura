"""core/actuators/actuator_registry.py
===================================
Open-Ended Actuators & Action Primitives.

Implements executable physical commands that modify the state of entities
in the PhysicsWorldModel. All operations are sandboxed and validated.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.runtime.task_ownership import (
    drain_owned_awaitable,
    runtime_shutdown_blocks_new_work,
)
from core.utils.task_tracker import get_task_tracker

# Bounds for synthesized world-model mutations. LLM-generated actuator output
# is untrusted, so both the fan-out (how many entities/fields it can touch)
# and individual values are capped to protect the live world model.
_MAX_SYNTH_ENTITIES = 256
_MAX_SYNTH_ATTRS_PER_ENTITY = 64
_MAX_SYNTH_STRING_LEN = 512
_MAX_SAFE_INT = 2**53  # exact-integer ceiling; beyond this floats lose precision
_SANDBOX_TIMEOUT_MIN_S = 0.5
_SANDBOX_TIMEOUT_MAX_S = 120.0

#: Result-contract bounds. An ActuatorResult crosses into world state, receipts
#: and logs, so its shape is bounded at construction rather than at each reader.
MAX_RESULT_MESSAGE_CHARS = 2000
MAX_RESULT_UPDATE_KEYS = 512

#: Parameter-contract bounds for synthesized (LLM-authored) actuators.
MAX_SYNTH_PARAM_KEYS = 64
MAX_SYNTH_PARAM_KEY_CHARS = 96
MAX_SYNTH_PARAM_SEQUENCE = 256

#: Wall-clock ceiling on one actuator body. A thread cannot be killed, so this
#: bounds how long a CALLER waits, not how long the work runs — see
#: ``ACTUATOR_DEADLINE_S`` handling in :meth:`ActuatorRegistry.execute_action_async`.
DEFAULT_ACTUATOR_DEADLINE_S = 300.0

#: An actuator declaring ``blocking_execution = False`` promises it will not
#: occupy the owner loop. Past this, the promise is measurably false.
NONBLOCKING_BUDGET_S = 0.25

#: Context keys a caller must never be able to assert. These are not policy
#: INPUTS a requester describes; they are policy CONCLUSIONS the gateway
#: reaches. A caller supplying one is claiming the answer to the question it is
#: asking. (``will_receipt_id`` is deliberately absent — it is a reference the
#: Will itself validates, not a self-asserted verdict.)
_FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "principal",
        "identity",
        "authenticated",
        "approved",
        "verified",
        "capability",
        "capability_token_id",
        "signed_capability",
        "_aura_authorized",
        "_capability_token_id",
        "bypass_authority",
    }
)
#: Bounds on caller context: a governance decision input must not also be a
#: memory or log bomb.
MAX_CONTEXT_KEYS = 64
MAX_CONTEXT_STRING_CHARS = 4096
#: Authorization fields the REGISTRY owns. A caller that supplies them is
#: attempting to forge authorization, so they are stripped on entry.
_REGISTRY_OWNED_PARAM_KEYS = ("_aura_authorized", "_capability_token_id", "_params_digest")

logger = logging.getLogger("Aura.Actuators")


@dataclass
class ActuatorResult:
    """The result of executing an action primitive.

    CP126 894bf628: these were three plain fields with no validation, yet the
    value flows into world state, authority receipts and logs. The shape is
    now normalized once, here, instead of being re-guessed by every reader.
    """

    success: bool
    message: str
    updates: dict[str, Any]
    updates_truncated: bool = field(default=False)

    def __post_init__(self) -> None:
        self.success = bool(self.success)
        message = self.message if isinstance(self.message, str) else str(self.message)
        if len(message) > MAX_RESULT_MESSAGE_CHARS:
            message = (
                message[:MAX_RESULT_MESSAGE_CHARS]
                + f"… [{len(message) - MAX_RESULT_MESSAGE_CHARS} more characters]"
            )
        self.message = message
        if not isinstance(self.updates, dict):
            # A non-mapping `updates` used to reach world-state merge code and
            # fail there, far from the actuator that produced it.
            logger.warning(
                "Actuator result carried %s updates, not a mapping; discarding",
                type(self.updates).__name__,
            )
            self.updates = {}
            self.updates_truncated = True
            return
        if len(self.updates) > MAX_RESULT_UPDATE_KEYS:
            kept = dict(list(self.updates.items())[:MAX_RESULT_UPDATE_KEYS])
            logger.warning(
                "Actuator result carried %d update keys; bounded to %d",
                len(self.updates),
                MAX_RESULT_UPDATE_KEYS,
            )
            self.updates = kept
            self.updates_truncated = True

    def digest(self) -> str:
        """A stable digest of what this result claims (CP126 6424c991).

        The authority receipt records this instead of the payload, so a
        closure can be checked against the effect without copying possibly
        sensitive update values into the audit trail.
        """
        try:
            body = json.dumps(
                {
                    "success": self.success,
                    "message": self.message,
                    "update_keys": sorted(str(key) for key in self.updates),
                },
                sort_keys=True,
                default=str,
            )
        except (TypeError, ValueError):
            body = f"{self.success}|{self.message}|{len(self.updates)}"
        return hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()


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
    # Fail SAFE: an actuator that forgets to declare its authority requirement
    # is governed by default. Low-risk, in-memory-only actuators opt OUT
    # explicitly with requires_authority = False.
    requires_authority: bool = True
    generation: int = 0
    source_code: str | None = None
    blocking_execution: bool = True
    #: Where this actuator came from and whether anything validated it.
    #: Empty for built-ins, populated by :meth:`ActuatorRegistry.register_synthesized`.
    provenance: dict[str, Any] | None = None
    # Evolved immune rules run repeatedly in cloned world models. They may
    # only target actuators that explicitly opt into that narrower contract.
    immune_rule_compatible: bool = False

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

    def immune_rule_seed_params(self) -> dict[str, Any] | None:
        """A bounded parameter seed for immune simulation, or None.

        None is the base default and the honest answer for most actuators:
        an actuator opts in by returning a seed, and immune simulation must
        never infer one from an actuator that never offered it. The previous
        docstring promised a seed unconditionally while the body returned
        None, which reads as an unfinished implementation rather than a
        deliberate opt-in.
        """
        return None


class SandboxedSynthesizedActuator(BaseActuator):
    """Live wrapper for LLM-synthesized actuator code.

    The generated code never executes in Aura's main process. Execution happens
    through the validator sandbox, then this wrapper applies only bounded,
    finite update payloads to the live physics world.
    """

    synthesized: bool = True
    # Generated code that mutates the live world model is always governed.
    requires_authority: bool = True

    def __init__(
        self,
        *,
        name: str,
        description: str,
        source_code: str,
        trust_score: float = 0.3,
        param_schema: dict[str, Any] | None = None,
    ) -> None:
        self._name = str(name).strip() or "sandboxed_synthesized_actuator"
        self._description = str(description).strip() or "Sandboxed synthesized actuator"
        self.source_code = source_code
        self.trust_score = trust_score
        # CP126 a1ec1e8a: a declared schema is optional, but its ABSENCE is
        # recorded rather than silently treated as "anything goes".
        self.param_schema = dict(param_schema) if isinstance(param_schema, dict) else None
        self.param_rejection: str = ""

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def schema_declared(self) -> bool:
        """Whether this actuator declares a per-action parameter schema."""
        return self.param_schema is not None

    def validate_params(self, params: dict[str, Any]) -> bool:
        """Enforce the parameter contract before generated code sees the input.

        CP126 a1ec1e8a: this was ``isinstance(params, dict)``, which is not a
        contract — a synthesized actuator received any dictionary at all,
        including one that was itself the attack. Two layers now apply:

        * a **structural floor** that holds even with no declared schema —
          string keys, bounded cardinality, bounded values, no deep nesting;
        * the **declared schema** when the synthesizer supplied one, with
          required keys and per-key types checked.
        """
        self.param_rejection = ""
        if not isinstance(params, dict):
            self.param_rejection = "parameters must be a mapping"
            return False
        # Registry-owned authorization keys never count against the budget.
        payload = {k: v for k, v in params.items() if k not in _REGISTRY_OWNED_PARAM_KEYS}
        if len(payload) > MAX_SYNTH_PARAM_KEYS:
            self.param_rejection = (
                f"{len(payload)} parameters exceeds the {MAX_SYNTH_PARAM_KEYS}-key bound"
            )
            return False
        for key, value in payload.items():
            if not isinstance(key, str) or not key:
                self.param_rejection = f"parameter key {key!r} is not a non-empty string"
                return False
            if len(key) > MAX_SYNTH_PARAM_KEY_CHARS:
                self.param_rejection = f"parameter key '{key[:32]}…' is too long"
                return False
            ok, why = self._bounded_param_value(value)
            if not ok:
                self.param_rejection = f"parameter '{key}': {why}"
                return False
        if self.param_schema is not None:
            ok, why = self._matches_schema(payload)
            if not ok:
                self.param_rejection = why
                return False
        return True

    @classmethod
    def _bounded_param_value(cls, value: Any, *, depth: int = 0) -> tuple[bool, str]:
        """One level of nesting, bounded primitives, no arbitrary objects."""
        if isinstance(value, (list, tuple)):
            if depth >= 1:
                return False, "nested sequences are not accepted"
            if len(value) > MAX_SYNTH_PARAM_SEQUENCE:
                return False, f"sequence of {len(value)} exceeds {MAX_SYNTH_PARAM_SEQUENCE}"
            for item in value:
                ok, why = cls._bounded_param_value(item, depth=depth + 1)
                if not ok:
                    return False, why
            return True, ""
        if isinstance(value, dict):
            if depth >= 1:
                return False, "nested mappings are not accepted"
            if len(value) > MAX_SYNTH_PARAM_KEYS:
                return False, f"mapping of {len(value)} keys exceeds {MAX_SYNTH_PARAM_KEYS}"
            for key, item in value.items():
                if not isinstance(key, str):
                    return False, "mapping keys must be strings"
                ok, why = cls._bounded_param_value(item, depth=depth + 1)
                if not ok:
                    return False, why
            return True, ""
        if cls._bounded_primitive_ok(value):
            return True, ""
        return False, f"value of type {type(value).__name__} is not an accepted primitive"

    @staticmethod
    def _bounded_primitive_ok(value: Any) -> bool:
        if value is None or isinstance(value, bool):
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, int):
            return abs(value) <= _MAX_SAFE_INT
        if isinstance(value, str):
            return len(value) <= _MAX_SYNTH_STRING_LEN
        return False

    _SCHEMA_TYPES: dict[str, tuple[type, ...]] = {
        "string": (str,),
        "number": (int, float),
        "integer": (int,),
        "boolean": (bool,),
        "array": (list, tuple),
        "object": (dict,),
    }

    def _matches_schema(self, payload: dict[str, Any]) -> tuple[bool, str]:
        schema = self.param_schema or {}
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        for name in schema.get("required") or ():
            if str(name) not in payload:
                return False, f"required parameter '{name}' is missing"
        if not schema.get("additional_properties", False):
            extra = sorted(set(payload) - set(properties))
            if properties and extra:
                return False, f"undeclared parameters: {', '.join(extra[:5])}"
        for name, spec in properties.items():
            if name not in payload or not isinstance(spec, dict):
                continue
            value = payload[name]
            wanted = self._SCHEMA_TYPES.get(str(spec.get("type") or ""))
            if wanted and not isinstance(value, wanted):
                return False, f"parameter '{name}' must be {spec.get('type')}"
            if isinstance(value, bool) and wanted in (
                self._SCHEMA_TYPES["number"],
                self._SCHEMA_TYPES["integer"],
            ):
                return False, f"parameter '{name}' must be {spec.get('type')}, not a boolean"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                low, high = spec.get("minimum"), spec.get("maximum")
                if low is not None and value < low:
                    return False, f"parameter '{name}' is below the minimum {low}"
                if high is not None and value > high:
                    return False, f"parameter '{name}' is above the maximum {high}"
            if isinstance(value, str):
                allowed = spec.get("enum")
                if isinstance(allowed, (list, tuple)) and value not in allowed:
                    return False, f"parameter '{name}' is not one of the declared values"
        return True, ""

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not self.validate_params(params):
            return ActuatorResult(
                False,
                f"Parameter validation failed: {self.param_rejection or 'unspecified'}",
                {},
            )

        try:
            from core.actuators.actuator_validator import ActuatorCodeValidator

            sandbox_result = ActuatorCodeValidator.execute_sandboxed(self.source_code or "", params)
            if not sandbox_result.success:
                return ActuatorResult(False, sandbox_result.error or "Sandbox execution failed", {})
            if "updates" not in sandbox_result.details:
                return ActuatorResult(
                    False,
                    "Sandboxed actuator produced no update payload; no world-model change applied",
                    {},
                )
            updates = sandbox_result.details["updates"]
            if not isinstance(updates, dict):
                return ActuatorResult(
                    False,
                    "Sandboxed actuator produced a malformed update payload; "
                    "no world-model change applied",
                    {},
                )
            if not updates:
                return ActuatorResult(
                    False,
                    "Sandboxed actuator produced no updates; no world-model change applied",
                    {},
                )
            applied_updates = self._apply_bounded_updates(updates)
            # Delivery truth: sandbox completion is not an effect. Success
            # requires at least one validated mutation to reach the live model.
            if not applied_updates:
                return ActuatorResult(
                    False,
                    "Sandboxed actuator produced no valid updates; no world-model change applied",
                    {},
                )
            return ActuatorResult(
                True,
                str(sandbox_result.details.get("message") or "Sandboxed actuator executed"),
                applied_updates,
            )
        except (ImportError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return ActuatorResult(False, f"Sandboxed actuator execution failed: {exc}", {})

    @staticmethod
    def _bounded_primitive(value: Any) -> Any:
        """Reject non-finite floats, oversized ints, and unbounded strings."""
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, int):
            return value if abs(value) <= _MAX_SAFE_INT else None
        if isinstance(value, str):
            return value[:_MAX_SYNTH_STRING_LEN]
        return None

    def _apply_bounded_updates(self, updates: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(updates, dict) or not updates:
            return {}

        from core.world.world_model import get_physics_world_model

        model = get_physics_world_model()
        applied: dict[str, Any] = {}
        entity_count = 0
        for entity_id, fields in updates.items():
            if entity_count >= _MAX_SYNTH_ENTITIES:
                logger.warning(
                    "Synthesized update fan-out exceeded %d entities; ignoring the rest.",
                    _MAX_SYNTH_ENTITIES,
                )
                break
            entity = model.get_entity(str(entity_id))
            if entity is None or not isinstance(fields, dict):
                continue
            entity_count += 1

            # Snapshot the fields this update may touch so a constraint failure
            # or an invalid later field rolls THIS entity back to its prior
            # state instead of leaving a half-applied mutation committed.
            snapshot_fields = (
                "capacity",
                "load",
                "flow_rate",
                "max_flow_rate",
                "latency",
                "coordinates",
            )
            before = {f: getattr(entity, f, None) for f in snapshot_fields}
            before_attrs = dict(getattr(entity, "attributes", {}) or {})

            try:
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
                        lat = _finite_float(coords[0], minimum=-90.0, maximum=90.0)
                        lon = _finite_float(coords[1], minimum=-180.0, maximum=180.0)
                        if lat is not None and lon is not None:
                            entity.coordinates = (lat, lon)
                            entity_updates["coordinates"] = entity.coordinates

                attrs = fields.get("attributes")
                if isinstance(attrs, dict):
                    safe_attrs: dict[str, Any] = {}
                    for key, value in attrs.items():
                        if len(safe_attrs) >= _MAX_SYNTH_ATTRS_PER_ENTITY:
                            break
                        if not isinstance(key, str):
                            continue
                        bounded = self._bounded_primitive(value)
                        if bounded is not None or value is None:
                            safe_attrs[key[:64]] = bounded
                    if safe_attrs:
                        entity.attributes.update(safe_attrs)
                        entity_updates["attributes"] = safe_attrs

                entity.enforce_constraints()
            except (AttributeError, TypeError, ValueError, KeyError) as exc:
                # Roll this entity back — partial mutations must not survive a
                # constraint or field error.
                for field, prior in before.items():
                    if prior is not None or hasattr(entity, field):
                        setattr(entity, field, prior)
                if hasattr(entity, "attributes"):
                    entity.attributes.clear()
                    entity.attributes.update(before_attrs)
                logger.warning(
                    "Synthesized update for entity %s rolled back after error: %s",
                    entity_id,
                    exc,
                )
                continue

            if entity_updates:
                applied[str(entity_id)] = entity_updates
        return applied


class RerouteVesselActuator(BaseActuator):
    """Actuator to adjust headings and speeds of maritime vessel edges."""

    # In-memory simulation only (mutates the physics world model, no external
    # effect) — explicitly opts out of the governed-by-default policy.
    requires_authority = False
    blocking_execution = False

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

    # In-memory simulation only — explicitly opts out of governed-by-default.
    requires_authority = False
    blocking_execution = False
    immune_rule_compatible = True

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

    def immune_rule_seed_params(self) -> dict[str, Any] | None:
        """Construct a valid transfer from the current cloned world."""
        try:
            from core.world.world_model import get_physics_world_model

            entities = list(get_physics_world_model().entities.values())
            nodes = [
                entity
                for entity in entities
                if str(getattr(entity, "kind", "")) == "node"
                and _finite_float(getattr(entity, "capacity", None), minimum=1e-9) is not None
                and _finite_float(getattr(entity, "load", None), minimum=0.0) is not None
            ]
            if len(nodes) < 2:
                return None
            source = max(
                nodes,
                key=lambda entity: float(entity.load) / max(float(entity.capacity), 1e-9),
            )
            target = min(
                (entity for entity in nodes if entity.entity_id != source.entity_id),
                key=lambda entity: float(entity.load) / max(float(entity.capacity), 1e-9),
            )
            transferable = min(
                float(source.load) * 0.20,
                max(0.0, float(target.capacity) - float(target.load)),
            )
            if transferable <= 1e-9:
                return None
            params = {
                "source_id": str(source.entity_id),
                "target_id": str(target.entity_id),
                "amount": float(transferable),
                "allow_partial": True,
            }
            return params if self.validate_params(params) else None
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return None

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
            requested = amount

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

            clipped = False
            if target.load + amount > target.capacity:
                # Capacity constraint check
                transferable = target.capacity - target.load
                if transferable <= 0.0:
                    return ActuatorResult(
                        False, f"Target '{target_id}' at maximum capacity {target.capacity}", {}
                    )
                # CP126 e2148790: silently substituting a smaller amount and
                # reporting success let a caller believe the transfer it asked
                # for happened. A partial effect now needs acknowledgement.
                if not params.get("allow_partial"):
                    return ActuatorResult(
                        False,
                        f"Target '{target_id}' can accept only {transferable} of the "
                        f"requested {amount}; pass allow_partial=True to accept a "
                        "partial transfer.",
                        {"_partial_available": transferable, "_requested": amount},
                    )
                amount = transferable  # Clip transfer, with the caller's consent
                clipped = True

            # Report what the world actually did, not what was asked of it.
            # The world clips a transfer against live capacity and constraints,
            # so the requested amount is a hope; the measured delta is the
            # fact. Claiming the requested figure made a no-op transfer read as
            # a success, which is how the same remedy can be re-issued forever
            # against a bottleneck it never relieved.
            source_load_before = float(source.load)
            target_load_before = float(target.load)

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

            moved = source_load_before - float(source.load)
            received = float(target.load) - target_load_before
            if moved <= 0.0:
                return ActuatorResult(
                    False,
                    f"Transfer from '{source_id}' to '{target_id}' moved nothing "
                    f"(requested {amount}); the world did not accept it.",
                    {},
                )

            logger.info(
                "Executed Actuator: reallocate_flow transferred %s from %s to %s",
                moved,
                source_id,
                target_id,
            )
            message = (
                f"Flow of {moved} successfully reallocated from '{source_id}' to '{target_id}'."
            )
            partial = moved + 1e-9 < requested
            if partial:
                message += f" (PARTIAL: clipped from the requested {requested})"
            return ActuatorResult(
                success=True,
                message=message,
                updates={
                    source_id: {"load": source.load},
                    target_id: {"load": target.load},
                    "_measured": {
                        "moved": moved,
                        "received": received,
                        "requested": requested,
                        "partial": partial,
                        "clip_acknowledged": clipped,
                    },
                },
            )

        except (ImportError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return ActuatorResult(False, f"Actuator execution failed: {exc}", {})


class SandboxActuator(BaseActuator):
    """Actuator wrapper for the SandboxOperator, enabling dynamic code synthesis."""

    requires_authority = True

    def __init__(self) -> None:
        from core.actuators.sandbox_operator import SandboxOperator

        self.operator = SandboxOperator()

    @property
    def name(self) -> str:
        return "execute_in_sandbox"

    @property
    def description(self) -> str:
        return (
            "Executes raw Python code synthesized by Aura in a sandbox subprocess. "
            "Returns a dictionary with 'success', 'stdout', 'stderr', and 'exit_code'."
        )

    def validate_params(self, params: dict[str, Any]) -> bool:
        return isinstance(params, dict) and "code" in params and isinstance(params["code"], str)

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        # CP126 4eaaca21: `params["_aura_authorized"]` is a value any direct
        # caller can set. Arbitrary code execution must be gated on the
        # registry's live authorization context and a token that validates.
        from core.actuators.authority import verify_actuator_authority

        authorized, reason = verify_actuator_authority(params, actuator=self.name)
        if not authorized:
            return ActuatorResult(False, reason, {})
        if not self.validate_params(params):
            return ActuatorResult(
                False, "Parameter validation failed: 'code' string parameter is required.", {}
            )

        code = params["code"]
        # Bound the sandbox timeout: an unbounded or non-finite value could
        # hang the sandbox subprocess indefinitely.
        timeout_s = _finite_float(
            params.get("timeout_s", 10.0),
            minimum=_SANDBOX_TIMEOUT_MIN_S,
            maximum=_SANDBOX_TIMEOUT_MAX_S,
        )
        if timeout_s is None:
            timeout_s = 10.0

        res = self.operator.execute_synthesized_tool(code, timeout_s=timeout_s)

        msg = f"Execution completed with exit code {res['exit_code']}."
        if not res["success"]:
            msg = f"Execution failed (exit code {res['exit_code']}): {res['stderr']}"

        return ActuatorResult(success=res["success"], message=msg, updates={"sandbox_result": res})


def _log_late_actuator(name: str, finished: Any) -> None:
    """Report an over-deadline actuator's real outcome when it finally lands.

    The caller already received an "outcome unknown" result. This is the only
    place that ever learns what actually happened, so it must not be silent.
    """
    try:
        drained = finished.result()
        late = drained.task.result()
        logger.warning(
            "Actuator '%s' finished AFTER its deadline: success=%s (%s)",
            name,
            getattr(late, "success", None),
            getattr(late, "message", "")[:200],
        )
    except asyncio.CancelledError:
        logger.warning("Actuator '%s' was cancelled after its deadline", name)
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        logger.warning(
            "Actuator '%s' failed after its deadline: %s: %s", name, type(exc).__name__, exc
        )


class ActuatorRegistry:
    """Registry of executable physical open-ended actuators."""

    def __init__(self) -> None:
        self.actuators: dict[str, BaseActuator] = {}
        # Guards registry mutations/lookups against concurrent registration,
        # deregistration, and execution-time lookups.
        self._lock = threading.RLock()
        # Names of required default actuators that failed to register at boot.
        self._missing_default_capabilities: list[str] = []
        self._register_default_actuators()

    def _register_default_actuators(self) -> None:
        self.register(RerouteVesselActuator())
        self.register(ReallocateFlowActuator())
        self.register(SandboxActuator())

        # Each required capability that fails to import is recorded as a
        # capability blocker so boot health can surface a degraded runtime,
        # instead of the failure only reaching the log.
        def _register_required(loader, capability: str) -> None:
            try:
                loader()
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                self._missing_default_capabilities.append(capability)
                logger.error("Failed to register required actuator '%s': %s", capability, exc)
                # CP126 0a7f6c74: a blocker nothing reads is not a blocker. The
                # degradation ledger IS read by the health surface, so a
                # required capability that failed to load reaches the runtime's
                # own account of itself rather than only the log file.
                try:
                    from core.runtime.errors import record_degradation

                    record_degradation(
                        "actuator_registry",
                        exc,
                        severity="critical",
                        action=(
                            f"required actuator capability '{capability}' did not "
                            "register; actions needing it will be refused"
                        ),
                        extra={"capability": capability},
                        enforce_failure_policy=False,
                    )
                except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
                    logger.error("Could not record the '%s' capability degradation", capability)

        def _code() -> None:
            from core.actuators.code_execution_actuator import CodeExecutionActuator

            self.register(CodeExecutionActuator())

        def _web() -> None:
            from core.actuators.web_actuators import WebFetchActuator, WebSearchActuator

            self.register(WebSearchActuator())
            self.register(WebFetchActuator())

        def _git_pkg() -> None:
            from core.actuators.git_pkg_actuators import GitActuator, PackageInstallActuator

            self.register(GitActuator())
            self.register(PackageInstallActuator())

        def _process() -> None:
            from core.actuators.process_supervisor import ProcessSupervisorActuator

            self.register(ProcessSupervisorActuator())

        def _doc() -> None:
            from core.actuators.doc_ingest import DocumentIngestActuator

            self.register(DocumentIngestActuator())

        _register_required(_code, "code_execution")
        _register_required(_web, "web")
        _register_required(_git_pkg, "git_package")
        _register_required(_process, "process_supervisor")
        _register_required(_doc, "document_ingest")

    def missing_default_capabilities(self) -> list[str]:
        """Required default actuators that failed to register at boot."""
        with self._lock:
            return list(self._missing_default_capabilities)

    def health_blocker(self) -> str | None:
        """Return a capability-health blocker string, or None when complete."""
        missing = self.missing_default_capabilities()
        if not missing:
            return None
        return "actuator_capabilities_missing:" + ",".join(sorted(missing))

    def register(self, actuator: BaseActuator, *, allow_replace: bool = False) -> None:
        with self._lock:
            existing = self.actuators.get(actuator.name)
            if existing is not None and existing is not actuator and not allow_replace:
                # Refuse to silently hijack a canonical actuator name. A
                # deliberate swap must pass allow_replace=True.
                raise ValueError(
                    f"Actuator name '{actuator.name}' is already registered by "
                    f"{type(existing).__name__}; refusing silent replacement"
                )
            self.actuators[actuator.name] = actuator
        logger.info("Registered actuator: %s (%s)", actuator.name, actuator.description)

    def get_actuator(self, name: str) -> BaseActuator | None:
        with self._lock:
            return self.actuators.get(name)

    def register_synthesized(
        self,
        actuator: BaseActuator,
        source_code: str,
        trust_score: float = 0.3,
        *,
        validation_receipt: dict[str, Any] | None = None,
        registered_by: str = "",
    ) -> None:
        """Register a runtime-synthesized actuator with provenance.

        CP126 f80cc444: this marked any object synthesized and accepted the
        caller's trust score with no digest, no signer and no evidence the code
        had passed the validator. Trust arrived as an assertion.

        Provenance is now recorded and, crucially, **caller-asserted trust
        without a validation receipt is not trust**: the score is floored to
        0.0, which the execution preflight already refuses (< 0.2). Registering
        still succeeds — the actuator is visible, inspectable and re-registrable
        once validated — but it cannot execute on an assertion alone.
        """
        code = source_code if isinstance(source_code, str) else ""
        actuator.synthesized = True
        actuator.source_code = code
        digest = hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()
        # Non-finite trust must not slip past the execution-time thresholds
        # (NaN comparisons are always false), so clamp at registration.
        bounded_trust = _finite_float(trust_score, minimum=0.0, maximum=1.0)
        bounded_trust = bounded_trust if bounded_trust is not None else 0.0

        validated, why = self._validation_receipt_matches(validation_receipt, digest)
        if not validated:
            bounded_trust = 0.0
        actuator.trust_score = bounded_trust
        actuator.provenance = {
            "source_digest": digest,
            "source_chars": len(code),
            "registered_by": str(registered_by or "unattributed"),
            "registered_at": time.time(),
            "validated": validated,
            "validation_detail": why,
            "generation": int(getattr(actuator, "generation", 0) or 0),
        }
        # Synthesized actuators may replace a prior generation of themselves.
        self.register(actuator, allow_replace=True)
        if validated:
            logger.info(
                "Registered synthesized actuator: %s (trust=%.2f, digest=%s)",
                actuator.name,
                actuator.trust_score,
                digest[:12],
            )
        else:
            logger.warning(
                "Registered UNVALIDATED synthesized actuator '%s' at trust 0.00 "
                "(%s); it will be refused at execution until validated",
                actuator.name,
                why,
            )

    @staticmethod
    def _validation_receipt_matches(
        receipt: dict[str, Any] | None, digest: str
    ) -> tuple[bool, str]:
        """Whether a validator receipt actually covers THIS source.

        A receipt for different code is worse than no receipt, because it reads
        as evidence. The digest binding is the whole check.
        """
        if not isinstance(receipt, dict):
            return False, "no validator receipt supplied"
        if not receipt.get("passed"):
            return False, "validator receipt does not record a pass"
        claimed = str(receipt.get("source_digest") or "")
        if not claimed:
            return False, "validator receipt names no source digest"
        if claimed != digest:
            return False, "validator receipt covers different source code"
        return True, f"validated by {receipt.get('validator') or 'unnamed validator'}"

    def deregister(self, name: str) -> None:
        """Remove an actuator from the registry (retirement)."""
        with self._lock:
            removed = self.actuators.pop(name, None)
        if removed is not None:
            logger.info("Deregistered actuator: %s", name)

    def get_synthesized_actuators(self) -> list[BaseActuator]:
        """List all runtime-synthesized actuators."""
        with self._lock:
            return [act for act in self.actuators.values() if getattr(act, "synthesized", False)]

    @staticmethod
    def _sanitize_context(context: Any) -> dict[str, Any]:
        """Bound caller context and strip any self-asserted verdict.

        CP126 e19cb515: free-form caller context drove priority, is_critical,
        source and arbitrary additional policy data straight into the
        AuthorityGateway with no schema and no authenticated caller identity.

        The governance lane legitimately carries rich structured context —
        objectives, expectations, selection provenance, an orchestrator handle
        — so an allowlist would break governance rather than protect it. What
        the registry removes instead is the class of key by which a caller
        could assert the CONCLUSION rather than describe the request: a
        principal, an "authenticated" flag, a capability. Those are the
        gateway's to decide. Everything surviving is bounded so a decision
        input cannot also be a memory or log bomb.

        The remaining half — proving WHICH process asked — needs an
        authenticated channel the gateway itself owns, and is not something the
        registry can synthesize from a dictionary.
        """
        if not isinstance(context, dict):
            return {}
        clean: dict[str, Any] = {}
        for key, value in context.items():
            name = str(key)
            if name.lower() in _FORBIDDEN_CONTEXT_KEYS:
                logger.warning(
                    "Dropped caller-asserted authority verdict '%s' from actuator context; "
                    "the AuthorityGateway decides this, the caller does not",
                    name,
                )
                continue
            if len(clean) >= MAX_CONTEXT_KEYS:
                logger.warning(
                    "Actuator context exceeded %d keys; the remainder was dropped",
                    MAX_CONTEXT_KEYS,
                )
                break
            if isinstance(value, str):
                clean[name] = value[:MAX_CONTEXT_STRING_CHARS]
            elif isinstance(value, bool) or value is None:
                clean[name] = value
            elif isinstance(value, (int, float)):
                # A non-finite priority or weight would propagate NaN into
                # every comparison the gateway makes.
                bounded = _finite_float(value)
                if bounded is not None:
                    clean[name] = bounded
                else:
                    logger.warning(
                        "Dropped non-finite context value '%s' from actuator context", name
                    )
            else:
                clean[name] = value
        return clean

    @staticmethod
    def _params_digest(name: str, params: dict[str, Any]) -> str:
        """The digest the AuthorityGateway binds into the signed capability."""
        try:
            from core.governance.capability_chain import compute_action_digest

            return compute_action_digest(name, params)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            return ""

    @classmethod
    async def _authorize_actuator(
        cls,
        name: str,
        params: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        from core.executive.authority_gateway import get_authority_gateway

        gateway = get_authority_gateway()
        # Bound caller-supplied priority to [0, 1] and reject non-finite
        # values — an unvalidated priority could otherwise skew the gateway's
        # admission ordering.
        priority = _finite_float(context.get("priority", 0.7), minimum=0.0, maximum=1.0)
        if priority is None:
            priority = 0.7
        return await gateway.authorize_tool_execution(
            name,
            params,
            source=str(context.get("source") or "actuator_registry"),
            priority=priority,
            is_critical=bool(context.get("is_critical", False)),
            context=dict(context),
        )

    @staticmethod
    def _verify_capability_binding(
        decision: Any, name: str, expected_digest: str
    ) -> tuple[bool, str]:
        """Check the SIGNED capability, bound to these exact parameters.

        CP126 2d127a7f: post-authorization verification called
        ``verify_tool_access(name, token_id)``, whose own docstring says it
        proves only that *some* token exists in this process naming this tool.
        It bound no parameters, so an approval for one call authorized any
        other call to the same actuator, and any code that can import the
        capability system could mint the token it checks.

        The signed capability carries an ``action_digest`` over (action,
        params) under the Will's key, so verifying it against the digest of the
        parameters we are about to execute is what actually binds approval to
        THIS call. ``consume=False`` because the downstream sink, not the
        registry, is the one-shot consumer.
        """
        capability = getattr(decision, "signed_capability", None)
        if capability is None:
            # Honest degradation: say which check was unavailable rather than
            # reporting the weaker one as if it were this one.
            return False, "authority decision carried no signed capability to bind"
        if not expected_digest:
            return False, "parameter digest could not be computed for binding"
        try:
            from core.governance.capability_chain import get_capability_verifier

            result = get_capability_verifier().verify(
                capability,
                expected_action_digest=expected_digest,
                consume=False,
            )
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            return False, f"capability verification failed: {type(exc).__name__}"
        if not getattr(result, "ok", False):
            denial = getattr(result, "denial", None)
            return False, (
                f"capability rejected for '{name}': "
                f"{getattr(denial, 'value', denial) or 'denied'} "
                f"({getattr(result, 'detail', '')})"
            )
        return True, ""

    @staticmethod
    def _execute_actuator_body(
        actuator: BaseActuator,
        params: dict[str, Any],
        authority_decision: Any,
    ) -> ActuatorResult:
        name = getattr(actuator, "name", "") or type(actuator).__name__
        if authority_decision is None:
            # CP126 7fe2e1b7: an actuator that opts out of AuthorityGateway
            # review still MUTATES something, and running it with no governance
            # token at all meant its writes were ungoverned and its effects
            # unattributed. It gets a least-privilege local decision instead —
            # not authorization (no actuator_authorization context is entered,
            # so a privileged actuator still refuses), just an attributed,
            # least-privilege scope for a declared local-effect actuator.
            from core.governance_context import local_internal_governed_scope

            with local_internal_governed_scope(
                f"actuator:{name}",
                domain="state_mutation",
                constraints={"local_effect_only": True, "authority_required": False},
            ):
                return actuator.execute(params)

        from core.actuators.authority import actuator_authorization
        from core.governance_context import governed_scope_sync

        # The authorization is put on a ContextVar for the dynamic extent of the
        # call, so a privileged actuator can PROVE it was reached through the
        # registry rather than trusting an injected `_aura_authorized` flag that
        # any direct caller could set (CP126 8900fa05 / 27651212 / 9f94bf4d /
        # 251ada47 / bdb4255d / 5ce6b589 …). ContextVars propagate into
        # asyncio.to_thread, which is where blocking actuator bodies run.
        with (
            actuator_authorization(
                name,
                capability_token_id=params.get("_capability_token_id"),
                decision_reason=str(getattr(authority_decision, "reason", "") or ""),
                principal=str(getattr(authority_decision, "principal", "") or ""),
            ),
            governed_scope_sync(authority_decision),
        ):
            return actuator.execute(params)

    @staticmethod
    def _preflight_actuator(
        actuator: BaseActuator,
        name: str,
        params: dict[str, Any],
    ) -> ActuatorResult | None:
        if not getattr(actuator, "synthesized", False):
            return None
        # Non-finite trust must fail CLOSED. NaN < 0.2 is False, so a NaN
        # trust score would otherwise bypass both thresholds and execute.
        trust = _finite_float(getattr(actuator, "trust_score", None))
        if trust is None:
            return ActuatorResult(
                False,
                f"Actuator '{name}' has a non-finite trust score; refusing execution",
                {},
            )
        if trust < 0.2:
            return ActuatorResult(
                False,
                f"Actuator '{name}' has trust score too low ({trust:.2f}) to execute",
                {},
            )
        if trust < 0.5:
            logger.warning(
                "Executing low-trust synthesized actuator '%s' (trust=%.2f)",
                name,
                trust,
            )
            if not params:
                return ActuatorResult(
                    False,
                    "Low-trust actuator requires non-empty parameters",
                    {},
                )
        return None

    @staticmethod
    def _demote_to_blocking(actuator: BaseActuator, name: str, elapsed: float) -> None:
        """An actuator that occupied the owner loop loses its exemption.

        CP126 acf1e08c: ``blocking_execution = False`` was a self-declaration
        the registry trusted forever. A physics simulation that grows expensive
        keeps running on the event loop, and nothing notices. Measuring it is
        the only enforcement available in-process, so the first breach is the
        last one that runs inline.
        """
        try:
            actuator.blocking_execution = True
        except (AttributeError, TypeError):
            logger.error(
                "Actuator '%s' overran the non-blocking budget and could not be demoted",
                name,
            )
            return
        logger.error(
            "Actuator '%s' declared non-blocking but held the owner loop for %.3fs; "
            "demoted to threaded execution for subsequent calls",
            name,
            elapsed,
        )
        try:
            from core.runtime.errors import record_degradation

            record_degradation(
                "actuator_registry",
                RuntimeError(
                    f"actuator '{name}' held the event loop for {elapsed:.3f}s "
                    f"while declaring blocking_execution=False"
                ),
                action="demoted the actuator to threaded execution",
                extra={"actuator": name, "elapsed_s": round(elapsed, 4)},
                enforce_failure_policy=False,
            )
        # Not a failure: this is the report ABOUT a slow actuator. The
        # actuator has already been demoted; a degradation sink that
        # cannot be reached must not also take down the demotion.
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            pass

    async def execute_action_async(
        self,
        name: str,
        params: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        deadline_s: float | None = None,
    ) -> ActuatorResult:
        """Execute one actuator without blocking the runtime owner loop."""
        actuator = self.get_actuator(name)
        if not actuator:
            return ActuatorResult(False, f"Actuator '{name}' not found", {})

        # CP126 a3b58be8: authorization lives in the same dictionary as the
        # business parameters, so a caller that sets these keys is attempting
        # to forge it. The registry owns them; anything inbound is discarded
        # before the gateway ever sees the request.
        exec_params = {
            key: value
            for key, value in dict(params or {}).items()
            if key not in _REGISTRY_OWNED_PARAM_KEYS
        }
        forged = sorted(set(dict(params or {})) & set(_REGISTRY_OWNED_PARAM_KEYS))
        if forged:
            logger.warning(
                "Discarded caller-supplied authorization field(s) %s for actuator '%s'",
                ", ".join(forged),
                name,
            )
        safe_context = self._sanitize_context(context)
        authority_decision = None
        capability_token_id = None
        result: ActuatorResult | None = None
        cancellation: asyncio.CancelledError | None = None
        started = time.monotonic()
        outcome_certain = True
        error_class = ""

        # Reject deterministic local preflight failures before acquiring an
        # authority lease. Once a lease is acquired, every path below closes it.
        preflight = self._preflight_actuator(actuator, name, exec_params)
        if preflight is not None:
            return preflight
        if runtime_shutdown_blocks_new_work(
            f"actuator:{name}",
            resource_kind="actuator_execution",
        ):
            return ActuatorResult(
                False,
                f"Actuator '{name}' refused during runtime shutdown",
                {},
            )

        if getattr(actuator, "requires_authority", False):
            try:
                authority_decision = await self._authorize_actuator(
                    name,
                    exec_params,
                    safe_context,
                )
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                return ActuatorResult(
                    False,
                    f"AuthorityGateway unavailable for actuator '{name}': {type(exc).__name__}: {exc}",
                    {},
                )
            if not getattr(authority_decision, "approved", False):
                return ActuatorResult(
                    False,
                    f"Actuator '{name}' refused by AuthorityGateway: {getattr(authority_decision, 'reason', '')}",
                    {},
                )
            capability_token_id = getattr(authority_decision, "capability_token_id", None)

        try:
            if authority_decision is not None:
                try:
                    from core.executive.authority_gateway import get_authority_gateway

                    if not get_authority_gateway().verify_tool_access(name, capability_token_id):
                        return ActuatorResult(
                            False, f"Capability token rejected for actuator '{name}'", {}
                        )
                except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    return ActuatorResult(
                        False,
                        f"Capability token verification failed for actuator '{name}': {type(exc).__name__}: {exc}",
                        {},
                    )
                # CP126 2d127a7f: the opaque-token check above binds nothing.
                # The signed capability binds this decision to THESE parameters
                # under the Will's key; verifying it is what makes an approval
                # non-transferable between calls.
                params_digest = self._params_digest(name, exec_params)
                bound, why = self._verify_capability_binding(
                    authority_decision, name, params_digest
                )
                if not bound:
                    # Absence is not a lesser failure than rejection.
                    #
                    # This used to refuse a capability that verified badly and
                    # PROCEED when none had been minted at all, on the reasoning
                    # that recording the weaker guarantee beats pretending the
                    # strong one held. But the two branches differ only in
                    # whether an attacker (or a mint bug) has to produce a bad
                    # capability or none — and producing none is strictly
                    # easier. The absent-capability branch was therefore the
                    # cheapest way past the binding check, on exactly the
                    # actuators that declared `requires_authority`.
                    #
                    # AuthorityGateway._mint_signed_capability already documents
                    # the intended contract — "a mint failure fails the action
                    # closed rather than degrading it into an unauthenticated
                    # execution". This is the sink that has to hold it. Every
                    # approving path in the gateway mints, so a missing
                    # capability means the mint FAILED, which is a governance
                    # fault to surface rather than a mode to run in.
                    return ActuatorResult(
                        False,
                        f"Actuator '{name}' refused: {why}",
                        {
                            "authority": "unbound",
                            "capability_present": bool(
                                getattr(authority_decision, "signed_capability", None)
                            ),
                        },
                    )
                exec_params["_aura_authorized"] = True
                exec_params["_capability_token_id"] = capability_token_id
                exec_params["_params_digest"] = params_digest

            if runtime_shutdown_blocks_new_work(
                f"actuator:{name}",
                resource_kind="actuator_execution",
            ):
                result = ActuatorResult(
                    False,
                    f"Actuator '{name}' refused because runtime shutdown began during authorization",
                    {},
                )
                return result

            if getattr(actuator, "blocking_execution", True):
                budget = _finite_float(
                    deadline_s if deadline_s is not None else DEFAULT_ACTUATOR_DEADLINE_S,
                    minimum=0.1,
                )
                if budget is None:
                    budget = DEFAULT_ACTUATOR_DEADLINE_S
                # CP126 436f7e9a: the CALLER's wait is bounded. The worker
                # thread is deliberately NOT cancelled — Python cannot stop a
                # running thread safely, and `drain_owned_awaitable` exists
                # precisely to observe non-cancellable work to completion. So
                # the deadline must expire by *not waiting any longer*, never
                # by cancelling: cancelling here would be absorbed by the drain
                # and would wait for the thread anyway.
                observer = get_task_tracker().create_task(
                    drain_owned_awaitable(
                        asyncio.to_thread(
                            self._execute_actuator_body,
                            actuator,
                            exec_params,
                            authority_decision,
                        ),
                        name=f"actuator:{name}",
                        owner="core.actuators.actuator_registry",
                        allow_during_shutdown=True,
                    )
                )
                loop = asyncio.get_running_loop()
                expires_at = loop.time() + budget
                while not observer.done():
                    remaining = expires_at - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.wait({observer}, timeout=remaining)
                    except asyncio.CancelledError as exc:
                        # The caller went away; keep observing so authority
                        # closure still describes what really happened.
                        if cancellation is None:
                            cancellation = exc
                if observer.done():
                    drained = observer.result()
                    cancellation = cancellation or drained.cancellation
                    result = drained.task.result()
                    if cancellation is not None:
                        logger.info(
                            "Actuator '%s' completed after caller cancellation; "
                            "closing authority before propagating cancellation",
                            name,
                        )
                else:
                    outcome_certain = False
                    logger.error(
                        "Actuator '%s' exceeded its %.1fs deadline; the worker thread "
                        "is STILL RUNNING and its effect is unknown",
                        name,
                        budget,
                    )
                    observer.add_done_callback(
                        lambda finished, actuator_name=name: _log_late_actuator(
                            actuator_name, finished
                        )
                    )
                    result = ActuatorResult(
                        False,
                        f"Actuator '{name}' exceeded its {budget:.1f}s deadline; the "
                        "worker is still running and the effect is UNKNOWN, not failed",
                        {"_outcome": "unknown", "_deadline_s": budget},
                    )
            else:
                # CP126 acf1e08c: `blocking_execution = False` is a PROMISE that
                # this body will not occupy the owner loop. The promise is now
                # measured, and an actuator that breaks it is demoted so the
                # next call goes to a thread instead of the loop.
                inline_started = time.monotonic()
                result = self._execute_actuator_body(
                    actuator,
                    exec_params,
                    authority_decision,
                )
                inline_elapsed = time.monotonic() - inline_started
                if inline_elapsed > NONBLOCKING_BUDGET_S:
                    self._demote_to_blocking(actuator, name, inline_elapsed)
            if not isinstance(result, ActuatorResult):
                result = ActuatorResult(
                    False,
                    f"Actuator '{name}' returned invalid result type {type(result).__name__}",
                    {},
                )
        except asyncio.CancelledError:
            raise
        except (
            ImportError,
            OSError,
            RuntimeError,
            AttributeError,
            LookupError,
            TypeError,
            ValueError,
        ) as exc:
            # Full detail goes to the log; the returned message carries only the
            # error CLASS. Raw exception text can leak internal paths, provider
            # details, source snippets, or secrets to the caller surface.
            logger.exception("Actuator '%s' raised during execution", name)
            error_class = type(exc).__name__
            result = ActuatorResult(
                False,
                f"Actuator '{name}' failed ({error_class}); see logs for detail",
                {},
            )
        finally:
            if authority_decision is not None:
                success = bool(result and result.success)
                update_keys = (
                    sorted(result.updates.keys())
                    if result and isinstance(result.updates, dict)
                    else []
                )
                # CP126 6424c991: a receipt that records only a boolean cannot
                # be reconciled against what actually happened. It now carries
                # the result digest, duration, error class, and — the field the
                # old receipt could not express — whether the outcome is even
                # KNOWN.
                receipt = {
                    "success": success,
                    "actuator": name,
                    "actuator_generation": int(getattr(actuator, "generation", 0) or 0),
                    "synthesized": bool(getattr(actuator, "synthesized", False)),
                    "provenance": dict(getattr(actuator, "provenance", None) or {}),
                    "update_key_count": len(update_keys),
                    "update_keys": update_keys[:32],
                    "updates_truncated": bool(getattr(result, "updates_truncated", False)),
                    "result_digest": result.digest() if result else "",
                    "params_digest": exec_params.get("_params_digest", ""),
                    "duration_s": round(time.monotonic() - started, 4),
                    "error_class": error_class,
                    "message": (result.message[:240] if result else ""),
                    "cancelled": cancellation is not None,
                    "outcome_certain": outcome_certain,
                }
                try:
                    from core.executive.authority_gateway import get_authority_gateway

                    get_authority_gateway().finalize_tool_execution(
                        executive_intent_id=getattr(
                            authority_decision, "executive_intent_id", None
                        ),
                        capability_token_id=capability_token_id,
                        standing_authority_token=getattr(
                            authority_decision, "standing_authority_token", None
                        ),
                        success=success,
                        result=receipt,
                    )
                except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    # A successful external effect whose authority closure could
                    # not be recorded is NOT a clean success — surface the
                    # uncertain audit state instead of returning bare success.
                    logger.error(
                        "Failed to finalize actuator authority receipt for %s: %s", name, exc
                    )
                    if result is not None and result.success:
                        result = ActuatorResult(
                            False,
                            f"Actuator '{name}' effect applied but authority closure failed; audit incomplete",
                            result.updates,
                        )

        if cancellation is not None:
            raise cancellation
        return result or ActuatorResult(False, f"Actuator '{name}' produced no result", {})

    def execute_action(
        self,
        name: str,
        params: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        deadline_s: float | None = None,
    ) -> ActuatorResult:
        """Synchronous bridge for callers that do not own an event loop.

        Async runtime code must await :meth:`execute_action_async`; blocking an
        active owner loop here would invalidate both latency and thread-bound
        standing-authority guarantees.

        CP126 1159a34f: this called ``asyncio.run`` per invocation, which
        creates AND DESTROYS a loop each time. Loop-bound singletons built
        during one call (gateway clients, transports, standing-authority
        bookkeeping) were left bound to a closed loop and failed on the next
        one. A single long-lived bridge loop is used instead, so every
        synchronous call sees the same loop for the life of the process.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return _bridge_loop().run(
                self.execute_action_async(name, params, context=context, deadline_s=deadline_s)
            )
        raise RuntimeError(
            "execute_action cannot run on an active event loop; await execute_action_async instead"
        )


class _SyncBridgeLoop:
    """One long-lived event loop for synchronous callers (CP126 1159a34f)."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._serve, name="actuator-sync-bridge", daemon=True
        )
        self._thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Any, timeout: float | None = None) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout)

    def alive(self) -> bool:
        return self._thread.is_alive() and not self._loop.is_closed()


_bridge: _SyncBridgeLoop | None = None
_bridge_lock = threading.Lock()


def _bridge_loop() -> _SyncBridgeLoop:
    global _bridge
    with _bridge_lock:
        if _bridge is None or not _bridge.alive():
            _bridge = _SyncBridgeLoop()
        return _bridge


# Singleton Pattern
_instance: ActuatorRegistry | None = None
_instance_lock = threading.Lock()


def get_actuator_registry() -> ActuatorRegistry:
    global _instance
    # Double-checked locking: concurrent first access must not construct two
    # registries with divergent synthesized actuators / execution histories.
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ActuatorRegistry()
    return _instance
