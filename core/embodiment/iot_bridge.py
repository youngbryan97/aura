"""core/embodiment/iot_bridge.py

Physical IoT Bridge (Causal Grounding)
========================================
A two-way coupling between Aura's homeostatic state and the physical
environment. Examples:

  * Aura's CPU temperature climbs past a threshold → request a thermostat
    setpoint drop on the local network.
  * Sustained prediction-error storm → tint the room red and lower
    activity-level lights (mood signal).
  * High curiosity + idle → pull research feed brightness up so the user
    sees a quiet "I'm reading" state.
  * The environment changes (door opens, light turns on) → that change
    enters Aura's prediction-error stream as a real exteroceptive signal.

The bridge is transport-agnostic; concrete transports register through
``register_transport()``. Stock support is provided for:

  * Home Assistant REST through typed per-entity Reality Reach adapters
  * sensor-only custom transports registered by the embedding runtime
  * a non-production "noop" transport for deterministic tests

Every effect goes through ``WorldBridge.call(Channel.ENVIRONMENTAL_CHANGE,
...)`` and then a durable Reality Reach transaction. Network acceptance and
fresh state readback remain separate receipts.

The reverse direction uses declared read-only adapters and the bounded Reality
Observation Router. Arbitrary transport dictionaries never enter cognition;
only typed scalar readings with metrology, freshness, and provenance do.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from core.embodiment.home_assistant_connector import HomeAssistantConnector
from core.embodiment.home_assistant_reality import (
    HomeAssistantEffect,
    HomeAssistantRealityAdapter,
    HomeAssistantTransport,
)
from core.reality_reach.actuation import TargetCommandCompiler
from core.reality_reach.body_projection import (
    PhysicalBodyProjection,
    project_adapter_to_body,
    remove_body_projection,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_async_lock
from core.utils.concurrency import cancel_and_join
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.IoTBridge")


# ─── transports ─────────────────────────────────────────────────────────────


IoTEffect = HomeAssistantEffect
HassTransport = HomeAssistantTransport


class IoTTransport(ABC):
    name: str = "abstract"

    @abstractmethod
    async def apply(self, effect: IoTEffect) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    async def observe(self) -> dict[str, Any] | None:  # pragma: no cover - interface
        raise NotImplementedError

    async def discover(self) -> list[dict[str, Any]]:
        return []


class NoopTransport(IoTTransport):
    name = "noop"

    def __init__(self) -> None:
        self.applied: list[IoTEffect] = []
        self.events: list[dict[str, Any]] = []

    async def apply(self, effect: IoTEffect) -> dict[str, Any]:
        self.applied.append(effect)
        return {
            "applied": True,
            "effect_verified": True,
            "transport": "noop",
            "target": effect.target,
            "op": effect.op,
        }

    async def observe(self) -> dict[str, Any] | None:
        if not self.events:
            return None
        return self.events.pop(0)


# ─── policy: substrate → effect ────────────────────────────────────────────


@dataclass
class PolicyRule:
    name: str
    when: Callable[[dict[str, Any]], bool]
    effect: Callable[[dict[str, Any]], IoTEffect]
    cooldown_s: float = 60.0


def _read_substrate() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from core.container import ServiceContainer
        affect = ServiceContainer.get("affect_engine", default=None)
        if affect is not None and hasattr(affect, "snapshot"):
            out["affect"] = affect.snapshot() or {}
        homeo = ServiceContainer.get("homeostasis_engine", default=None) or ServiceContainer.get("homeostatic_engine", default=None)
        if homeo is not None and hasattr(homeo, "snapshot"):
            out["homeo"] = homeo.snapshot() or {}
    except (ImportError, AttributeError, RuntimeError):
        pass  # no-op: intentional
    try:
        from core.runtime import resource_psutil as psutil
        out["cpu_pct"] = psutil.cpu_percent(interval=None)
        out["ram_pct"] = psutil.virtual_memory().percent
    except (ImportError, AttributeError, RuntimeError):
        pass  # no-op: intentional
    try:
        from core.organism.viability import get_viability
        out["viability"] = get_viability().state.value
    except (ImportError, AttributeError, RuntimeError):
        pass  # no-op: intentional
    return out


_DEFAULT_POLICY: list[PolicyRule] = [
    PolicyRule(
        name="cpu_hot_drop_thermostat",
        when=lambda s: float(s.get("cpu_pct", 0.0)) > 85.0,
        effect=lambda s: IoTEffect(
            target="climate.studio",
            op="set_temperature",
            payload={"temperature": 21.0},
            reason="cpu_hot",
        ),
        cooldown_s=180.0,
    ),
    PolicyRule(
        name="threat_red_room",
        when=lambda s: (s.get("affect", {}) or {}).get("prediction_error", 0.0) > 0.85,
        effect=lambda s: IoTEffect(
            target="light.studio",
            op="turn_on",
            payload={"rgb_color": [255, 32, 32], "brightness_pct": 25},
            reason="prediction_error_storm",
        ),
        cooldown_s=120.0,
    ),
    PolicyRule(
        name="curiosity_reading_light",
        when=lambda s: (s.get("affect", {}) or {}).get("curiosity", 0.0) > 0.75 and s.get("viability") == "healthy",
        effect=lambda s: IoTEffect(
            target="light.desk",
            op="turn_on",
            payload={"brightness_pct": 80, "color_temp_kelvin": 5200},
            reason="curiosity_high_idle",
        ),
        cooldown_s=600.0,
    ),
]


# ─── bridge ─────────────────────────────────────────────────────────────────


class IoTBridge:
    def __init__(self) -> None:
        self._transports: dict[str, IoTTransport | HomeAssistantTransport] = {}
        self._policy: list[PolicyRule] = list(_DEFAULT_POLICY)
        self._last_fired: dict[str, float] = {}
        self._last_attempted: dict[str, float] = {}
        self._reality_service: Any | None = None
        self._reality_coordinator: Any | None = None
        self._reality_adapters: dict[str, HomeAssistantRealityAdapter] = {}
        self._body_projections: dict[str, PhysicalBodyProjection] = {}
        self._observation_router: Any | None = None
        self._attachment_broker: Any | None = None
        self._home_assistant_connector: HomeAssistantConnector | None = None
        self._adapter_lock = checked_async_lock("iot_bridge")
        self._task: asyncio.Task[Any] | None = None
        self._running = False
        try:
            self.register_transport("home_assistant", HassTransport())
            logger.info("Home Assistant IoT transport configured.")
        except RuntimeError as exc:
            logger.info("Home Assistant IoT transport disabled: %s", exc)

    def bind_reality_reach(self, service: Any, coordinator: Any) -> None:
        if service is None or not callable(getattr(service, "register_adapter", None)):
            raise TypeError("reality reach service must support adapter registration")
        if coordinator is None or not callable(getattr(coordinator, "execute", None)):
            raise TypeError("reality actuation coordinator must support execution")
        if self._reality_service is not None and self._reality_service is not service:
            raise RuntimeError("IoT bridge is already bound to another reality service")
        if (
            self._reality_coordinator is not None
            and self._reality_coordinator is not coordinator
        ):
            raise RuntimeError("IoT bridge is already bound to another coordinator")
        self._reality_service = service
        self._reality_coordinator = coordinator

    def bind_sensory_fabric(self, observation_router: Any, attachment_broker: Any) -> None:
        if observation_router is None or not callable(
            getattr(observation_router, "register_sampler", None)
        ):
            raise TypeError("IoT bridge requires a Reality Reach observation router")
        if attachment_broker is None or not callable(
            getattr(attachment_broker, "register_connector", None)
        ):
            raise TypeError("IoT bridge requires a physical attachment broker")
        if self._observation_router is not None and self._observation_router is not observation_router:
            raise RuntimeError("IoT bridge is already bound to another observation router")
        if self._attachment_broker is not None and self._attachment_broker is not attachment_broker:
            raise RuntimeError("IoT bridge is already bound to another attachment broker")
        self._observation_router = observation_router
        self._attachment_broker = attachment_broker

    def _ensure_runtime_binding(self) -> tuple[Any, Any]:
        if self._reality_service is None or self._reality_coordinator is None:
            from core.container import ServiceContainer

            service = ServiceContainer.get("reality_reach", default=None)
            coordinator = ServiceContainer.get("reality_actuation", default=None)
            self.bind_reality_reach(service, coordinator)
        return self._reality_service, self._reality_coordinator

    def register_transport(
        self,
        name: str,
        transport: IoTTransport | HomeAssistantTransport,
    ) -> None:
        canonical_name = str(name or "").strip().lower()
        if not canonical_name or not canonical_name.replace("_", "").isalnum():
            raise ValueError("iot_transport_name_invalid")
        if self._running and canonical_name in self._transports:
            raise RuntimeError("cannot replace a running IoT transport")
        if not callable(getattr(transport, "observe", None)) or not callable(
            getattr(transport, "discover", None)
        ):
            raise TypeError("IoT transports must implement observe and discover")
        self._transports[canonical_name] = transport

    def replace_policy(self, rules: list[PolicyRule]) -> None:
        self._policy = list(rules)

    def append_rule(self, rule: PolicyRule) -> None:
        self._policy.append(rule)

    async def apply_authorized(
        self,
        effect: IoTEffect,
        *,
        capability_token: str,
        transport_name: str = "",
        idempotency_key: str = "",
        source: str = "iot_bridge",
    ) -> dict[str, Any]:
        """Compile one authorized effect into the canonical physical transaction."""
        if not str(capability_token or "").strip():
            raise PermissionError("iot_capability_token_required")
        if not self._transports:
            raise RuntimeError("iot_transport_unavailable")
        requested_transport = str(transport_name or "").strip().lower()
        executable = {
            name: transport
            for name, transport in self._transports.items()
            if isinstance(transport, HomeAssistantTransport)
        }
        if requested_transport:
            transport = executable.get(requested_transport)
            if transport is None:
                raise RuntimeError("iot_transport_not_reality_reach_executable")
        elif len(executable) == 1:
            requested_transport, transport = next(iter(executable.items()))
        elif not executable:
            raise RuntimeError("iot_transport_not_reality_reach_executable")
        else:
            raise RuntimeError("iot_transport_selection_ambiguous")

        service, coordinator = self._ensure_runtime_binding()
        adapter_key = f"{requested_transport}:{effect.target}"
        async with self._adapter_lock:
            adapter = self._reality_adapters.get(adapter_key)
            if adapter is None:
                adapter = await transport.create_adapter(effect.target)
                service.register_adapter(adapter)
                try:
                    if self._observation_router is not None:
                        self._observation_router.register_sampler(adapter)
                    self._body_projections[adapter.adapter_id] = project_adapter_to_body(
                        adapter,
                        device_id=f"hass.{effect.target}",
                        display_name=effect.target,
                        transport="home_assistant.rest",
                        persistent_identity=bool(
                            str(os.getenv("AURA_HASS_INSTALLATION_ID") or "").strip()
                        ),
                    )
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                    if self._observation_router is not None:
                        try:
                            self._observation_router.unregister_sampler(adapter.adapter_id)
                        except LookupError:
                            pass
                    service.unregister_adapter(adapter.adapter_id)
                    raise
                self._reality_adapters[adapter_key] = adapter
                await asyncio.to_thread(service.refresh)
        inventory_sha256 = str(service.status().get("registry_sha256") or "")
        stable_idempotency = str(idempotency_key or "").strip()
        if not stable_idempotency:
            stable_idempotency = (
                "hass.auth."
                + str(capability_token).replace("\x00", "")[-32:]
                + "."
                + effect.sha256.removeprefix("sha256:")[:24]
            )
        command = await adapter.compile_effect(
            effect,
            inventory_sha256=inventory_sha256,
            deadline_s=30.0,
            idempotency_key=stable_idempotency,
            source=source,
        )
        try:
            output = await coordinator.execute(command)
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            record_degradation("iot_bridge", exc)
            raise
        if not isinstance(output, dict):
            raise RuntimeError("reality_actuation_result_not_mapping")
        transport_succeeded = bool(
            output.get("transport_succeeded") is True
            or output.get("executed") is True
        )
        effect_verified = output.get("effect_verified") is True
        failures = [] if effect_verified else [
            str(output.get("error") or output.get("reason") or "effect_unverified")[:300]
        ]
        return {
            "effects": [
                {
                    "transport": requested_transport,
                    "target": effect.target,
                    "op": effect.op,
                    "adapter_id": adapter.adapter_id,
                    "command_sha256": command.sha256,
                    "output": output,
                }
            ],
            "transport_succeeded": transport_succeeded,
            "effect_verified": effect_verified,
            "failures": failures,
            "reality_reach_transaction": output.get("reality_reach_transaction"),
        }

    async def apply_target_authorized(
        self,
        channel_id: str,
        target_value: float,
        *,
        capability_token: str,
        timeout_s: float = 30.0,
        idempotency_key: str = "",
        source: str = "iot_bridge",
    ) -> dict[str, Any]:
        """Compile a registered scalar target only after WorldBridge authority."""
        token = str(capability_token or "").strip()
        if not token:
            raise PermissionError("iot_capability_token_required")
        canonical_channel = str(channel_id or "").strip().lower()
        if not canonical_channel:
            raise ValueError("reality_channel_id_required")
        if isinstance(target_value, bool):
            raise TypeError("reality_target_value_must_be_numeric")
        target = float(target_value)
        if isinstance(timeout_s, bool):
            raise TypeError("reality_target_timeout_must_be_numeric")
        deadline_s = float(timeout_s)
        if not math.isfinite(target):
            raise ValueError("reality_target_value_must_be_finite")
        if not math.isfinite(deadline_s) or not 0.1 <= deadline_s <= 300.0:
            raise ValueError("reality_target_timeout_out_of_bounds")

        service, coordinator = self._ensure_runtime_binding()
        adapter = service.actuator_adapter(canonical_channel)
        if adapter is None:
            raise LookupError(f"physical_channel_not_executable:{canonical_channel}")
        if not isinstance(adapter, TargetCommandCompiler):
            raise TypeError(
                "physical_channel_target_compiler_unavailable:"
                f"{canonical_channel}"
            )
        inventory_sha256 = str(service.status().get("registry_sha256") or "")
        stable_idempotency = str(idempotency_key or "").strip()
        if not stable_idempotency:
            authority_sha256 = sha256_hex(token.encode("utf-8"))
            request_sha256 = sha256_hex(
                canonical_json(
                    {
                        "authority_sha256": authority_sha256,
                        "channel_id": canonical_channel,
                        "target_value": target,
                    }
                )
            )
            stable_idempotency = (
                "reality.target.auth."
                + request_sha256.removeprefix("sha256:")[:32]
            )
        command = await adapter.compile_target(
            target,
            inventory_sha256=inventory_sha256,
            deadline_s=deadline_s,
            idempotency_key=stable_idempotency,
            source=source,
        )
        try:
            raw_output = await coordinator.execute(command)
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            record_degradation("iot_bridge", exc)
            raise
        if not isinstance(raw_output, Mapping):
            raise RuntimeError("reality_actuation_result_not_mapping")
        output = dict(raw_output)
        transport_succeeded = bool(
            output.get("transport_succeeded") is True
            or output.get("executed") is True
        )
        effect_verified = output.get("effect_verified") is True
        failures = [] if effect_verified else [
            str(output.get("error") or output.get("reason") or "effect_unverified")[:300]
        ]
        return {
            "channel_id": canonical_channel,
            "target_value": target,
            "effects": [
                {
                    "adapter_id": str(getattr(adapter, "adapter_id", "")),
                    "channel_id": canonical_channel,
                    "command_sha256": str(getattr(command, "sha256", "")),
                    "output": output,
                }
            ],
            "transport_succeeded": transport_succeeded,
            "effect_verified": effect_verified,
            "failures": failures,
            "reality_reach_transaction": output.get("reality_reach_transaction"),
        }

    async def discover_authorized(
        self,
        *,
        capability_token: str,
    ) -> list[dict[str, Any]]:
        if not str(capability_token or "").strip():
            raise PermissionError("iot_capability_token_required")
        if not self._transports:
            raise RuntimeError("iot_transport_unavailable")
        devices: list[dict[str, Any]] = []
        for transport_name, transport in self._transports.items():
            discovered = await transport.discover()
            devices.extend(
                {"transport": transport_name, **item}
                for item in discovered
                if isinstance(item, dict)
            )
        return devices[:5000]

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "configured": bool(self._transports),
            "transports": sorted(self._transports),
            "policy_rules": len(self._policy),
            "reality_reach_bound": self._reality_service is not None,
            "reality_adapter_count": len(self._reality_adapters),
            "observation_router_bound": self._observation_router is not None,
            "attachment_broker_bound": self._attachment_broker is not None,
            "home_assistant_connector": self._home_assistant_connector is not None,
        }

    def is_alive(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    def is_ready(self) -> bool:
        return self.is_alive() and (
            not self._transports
            or (
                self._reality_service is not None
                and self._reality_coordinator is not None
            )
        )

    def status(self) -> dict[str, Any]:
        return self.get_status()

    async def tick(self) -> list[dict[str, Any]]:
        snapshot = _read_substrate()
        results: list[dict[str, Any]] = []
        now = time.time()
        for rule in self._policy:
            try:
                if not rule.when(snapshot):
                    continue
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('iot_bridge', exc)
                logger.debug("iot rule predicate failed: %s", exc)
                continue
            last = max(
                self._last_fired.get(rule.name, 0.0),
                self._last_attempted.get(rule.name, 0.0),
            )
            if (now - last) < rule.cooldown_s:
                continue
            try:
                effect = rule.effect(snapshot)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('iot_bridge', exc)
                logger.debug("iot rule effect build failed: %s", exc)
                continue
            from core.embodiment.world_bridge import Channel, get_world_bridge

            self._last_attempted[rule.name] = now
            result = await get_world_bridge().call(
                Channel.ENVIRONMENTAL_CHANGE,
                action=f"iot:{effect.target}:{effect.op}",
                intent=effect.reason or rule.name,
                payload={
                    "operation": "apply",
                    "transport": "home_assistant",
                    "target": effect.target,
                    "op": effect.op,
                    "effect": dict(effect.payload),
                    "reason": effect.reason or rule.name,
                    "idempotency_key": (
                        f"iot.policy.{rule.name}.{int(now // rule.cooldown_s)}"
                    ),
                },
            )
            results.append(
                {
                    "rule": rule.name,
                    "ok": result.ok,
                    "receipt_id": result.receipt_id,
                    "data": result.data,
                    "error": result.error,
                    "status": result.status,
                    "transport_succeeded": result.transport_succeeded,
                    "effect_verified": result.effect_verified,
                    "manual_reconciliation_required": (
                        result.manual_reconciliation_required
                    ),
                }
            )
            if result.ok:
                self._last_fired[rule.name] = now
        return results

    async def start(self, *, interval: float = 5.0) -> None:
        if self._running:
            return
        if any(
            isinstance(transport, HomeAssistantTransport)
            for transport in self._transports.values()
        ):
            self._ensure_runtime_binding()
            if self._observation_router is None or self._attachment_broker is None:
                raise RuntimeError("iot_reality_sensory_fabric_unbound")
            if self._home_assistant_connector is None:
                home_assistant = next(
                    transport
                    for transport in self._transports.values()
                    if isinstance(transport, HomeAssistantTransport)
                )
                self._home_assistant_connector = HomeAssistantConnector(
                    home_assistant,
                    discover_callback=self._discover_home_assistant_governed,
                )
                self._attachment_broker.register_connector(
                    self._home_assistant_connector
                )
        self._running = True

        async def _loop() -> None:
            while self._running:
                try:
                    await self.tick()
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('iot_bridge', exc)
                    logger.debug("iot bridge tick failed: %s", exc)
                await asyncio.sleep(interval)

        self._task = get_task_tracker().create_task(_loop(), name="IoTBridge")

    async def _discover_home_assistant_governed(self) -> list[dict[str, Any]]:
        from core.embodiment.world_bridge import Channel, get_world_bridge

        result = await get_world_bridge().call(
            Channel.ENVIRONMENTAL_CHANGE,
            action="iot:discover:home_assistant",
            intent="discover bounded physical sensors and actuators I may connect to",
            payload={"operation": "discover"},
        )
        if not result.ok:
            raise RuntimeError(str(result.error or "home_assistant_discovery_refused")[:300])
        data = result.data if isinstance(result.data, dict) else {}
        devices = data.get("devices", [])
        if not isinstance(devices, list):
            raise RuntimeError("home_assistant_discovery_result_invalid")
        return [
            {key: value for key, value in item.items() if key != "transport"}
            for item in devices
            if isinstance(item, dict) and item.get("transport") == "home_assistant"
        ][:5000]

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await cancel_and_join(self._task, owner="core.embodiment.iot_bridge")
            self._task = None
        if self._reality_service is not None:
            for adapter_key, adapter in list(self._reality_adapters.items()):
                removed = False
                try:
                    await asyncio.to_thread(
                        self._reality_service.unregister_adapter,
                        adapter.adapter_id,
                    )
                    removed = True
                except LookupError:
                    removed = True
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    record_degradation("iot_bridge", exc)
                    logger.warning(
                        "Could not revoke Home Assistant adapter %s during shutdown: %s",
                        adapter.adapter_id,
                        exc,
                    )
                if removed:
                    if self._observation_router is not None:
                        try:
                            self._observation_router.unregister_sampler(adapter.adapter_id)
                        except LookupError:
                            pass
                    projection = self._body_projections.pop(adapter.adapter_id, None)
                    if projection is not None:
                        try:
                            remove_body_projection(projection)
                        except (
                            ImportError,
                            AttributeError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            record_degradation(
                                "iot_bridge.body_schema",
                                exc,
                                action="removed IoT adapter while recording stale body projection",
                            )
                    self._reality_adapters.pop(adapter_key, None)
            await asyncio.to_thread(self._reality_service.refresh)


async def _environmental_change_handler(
    payload: dict[str, Any],
    *,
    capability_token: str,
) -> dict[str, Any]:
    bridge = get_iot_bridge()
    operation = str(payload.get("operation") or "apply").strip().lower()
    if operation == "discover":
        return {
            "devices": await bridge.discover_authorized(
                capability_token=capability_token,
            ),
            "transport_succeeded": True,
            "effect_verified": True,
        }
    if operation == "hardware_apply":
        from core.actuation.robotics_actuator import RoboticsActuator

        target = str(payload.get("target") or "").strip()
        command = str(payload.get("op") or "").strip()
        parameters = payload.get("parameters")
        if not target or not command or not isinstance(parameters, dict):
            raise ValueError("hardware_target_command_parameters_required")
        outcome = await RoboticsActuator.command_device(
            target,
            command,
            dict(parameters),
            source=str(payload.get("source") or "world_bridge.environmental_change"),
            idempotency_key=str(payload.get("idempotency_key") or "") or None,
        )
        verified = outcome.get("effect_verified") is True
        return {
            **outcome,
            "transport_succeeded": bool(
                outcome.get("transport_succeeded") is True
                or outcome.get("executed") is True
                or verified
            ),
            "effect_verified": verified,
        }
    if operation == "reality_target":
        channel_id = str(payload.get("channel_id") or "").strip().lower()
        target_value = payload.get("target_value")
        timeout_s = payload.get("timeout_s", 30.0)
        if not channel_id:
            raise ValueError("reality_channel_id_required")
        if target_value is None or isinstance(target_value, bool):
            raise ValueError("reality_target_value_must_be_numeric")
        if isinstance(timeout_s, bool):
            raise ValueError("reality_target_timeout_must_be_numeric")
        return await bridge.apply_target_authorized(
            channel_id,
            float(target_value),
            capability_token=capability_token,
            timeout_s=float(timeout_s),
            idempotency_key=str(payload.get("idempotency_key") or ""),
            source=str(payload.get("source") or "world_bridge.environmental_change"),
        )
    if operation not in {"apply", "home_assistant_apply"}:
        raise ValueError(f"unknown_iot_operation:{operation}")
    target = str(payload.get("target") or "").strip().lower()
    action = str(payload.get("op") or "").strip().lower()
    effect_payload = payload.get("effect")
    if not target or not action or not isinstance(effect_payload, dict):
        raise ValueError("iot_effect_target_op_payload_required")
    effect = IoTEffect(
        target=target,
        op=action,
        payload=dict(effect_payload),
        reason=str(payload.get("reason") or "")[:240],
    )
    outcome = await bridge.apply_authorized(
        effect,
        capability_token=capability_token,
        transport_name=str(payload.get("transport") or ""),
        idempotency_key=str(payload.get("idempotency_key") or ""),
        source=str(payload.get("source") or "world_bridge.environmental_change"),
    )
    return outcome


_BRIDGE: IoTBridge | None = None


def get_iot_bridge() -> IoTBridge:
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = IoTBridge()
    return _BRIDGE


__all__ = [
    "IoTEffect",
    "IoTTransport",
    "NoopTransport",
    "HassTransport",
    "PolicyRule",
    "IoTBridge",
    "get_iot_bridge",
]
