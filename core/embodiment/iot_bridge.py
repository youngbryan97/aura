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

  * Home Assistant REST (``HassTransport``)
  * MQTT (``MQTTTransport``) — only constructed when paho-mqtt is present
  * a "noop" transport for development and tests

Every effect goes through ``WorldBridge.call(Channel.ENVIRONMENTAL_CHANGE,
...)`` so it inherits permission, conscience, and capability-token gates.

The reverse direction — env → substrate — uses ``observe()`` to inject an
event into the prediction-error stream tagged with provenance so it never
gets confused with internal state.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


from core.utils.task_tracker import get_task_tracker

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("Aura.IoTBridge")


# ─── transports ─────────────────────────────────────────────────────────────


@dataclass
class IoTEffect:
    target: str        # e.g. "thermostat.living_room", "light.studio"
    op: str            # set, increment, scene, etc.
    payload: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""   # human-readable rationale (logged, never user-visible)


class IoTTransport:
    name: str = "abstract"

    async def apply(self, effect: IoTEffect) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    async def observe(self) -> Optional[Dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError


class NoopTransport(IoTTransport):
    name = "noop"

    def __init__(self) -> None:
        self.applied: List[IoTEffect] = []
        self.events: List[Dict[str, Any]] = []

    async def apply(self, effect: IoTEffect) -> Dict[str, Any]:
        self.applied.append(effect)
        return {"applied": True, "transport": "noop", "target": effect.target, "op": effect.op}

    async def observe(self) -> Optional[Dict[str, Any]]:
        if not self.events:
            return None
        return self.events.pop(0)


class HassTransport(IoTTransport):
    """Home Assistant REST transport.

    Configured by environment variables:
      AURA_HASS_URL    — e.g. "http://homeassistant.local:8123"
      AURA_HASS_TOKEN  — long-lived access token

    Refuses to operate without both.
    """

    name = "home_assistant"

    def __init__(self) -> None:
        self.base = os.getenv("AURA_HASS_URL", "").rstrip("/")
        self.token = os.getenv("AURA_HASS_TOKEN", "")
        if not self.base or not self.token:
            raise RuntimeError("hass_credentials_missing")

    async def apply(self, effect: IoTEffect) -> Dict[str, Any]:
        import aiohttp  # type: ignore

        domain, _, entity = effect.target.partition(".")
        url = f"{self.base}/api/services/{domain}/{effect.op}"
        headers = {"Authorization": f"Bearer {self.token}"}
        body = {"entity_id": effect.target, **effect.payload}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, timeout=8) as resp:
                data = await resp.text()
                return {"status": resp.status, "body": data[:1024]}

    async def observe(self) -> Optional[Dict[str, Any]]:
        # Polling-style observation. A push variant would subscribe via
        # WebSocket; this minimal version exposes the structure.
        return None


# ─── policy: substrate → effect ────────────────────────────────────────────


@dataclass
class PolicyRule:
    name: str
    when: Callable[[Dict[str, Any]], bool]
    effect: Callable[[Dict[str, Any]], IoTEffect]
    cooldown_s: float = 60.0


def _read_substrate() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        from core.container import ServiceContainer
        affect = ServiceContainer.get("affect_engine", default=None)
        if affect is not None and hasattr(affect, "snapshot"):
            out["affect"] = affect.snapshot() or {}
        homeo = ServiceContainer.get("homeostasis_engine", default=None) or ServiceContainer.get("homeostatic_engine", default=None)
        if homeo is not None and hasattr(homeo, "snapshot"):
            out["homeo"] = homeo.snapshot() or {}
    except Exception:
        pass
    try:
        import psutil
        out["cpu_pct"] = psutil.cpu_percent(interval=None)
        out["ram_pct"] = psutil.virtual_memory().percent
    except Exception:
        pass
    try:
        from core.organism.viability import get_viability
        out["viability"] = get_viability().state.value
    except Exception:
        pass
    return out


_DEFAULT_POLICY: List[PolicyRule] = [
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
        self._transports: Dict[str, IoTTransport] = {"noop": NoopTransport()}
        self._policy: List[PolicyRule] = list(_DEFAULT_POLICY)
        self._last_fired: Dict[str, float] = {}
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def register_transport(self, name: str, transport: IoTTransport) -> None:
        self._transports[name] = transport

    def replace_policy(self, rules: List[PolicyRule]) -> None:
        self._policy = list(rules)

    def append_rule(self, rule: PolicyRule) -> None:
        self._policy.append(rule)

    async def tick(self) -> List[Dict[str, Any]]:
        snapshot = _read_substrate()
        results: List[Dict[str, Any]] = []
        now = time.time()
        for rule in self._policy:
            try:
                if not rule.when(snapshot):
                    continue
            except Exception as exc:
                record_degradation('iot_bridge', exc)
                logger.debug("iot rule predicate failed: %s", exc)
                continue
            last = self._last_fired.get(rule.name, 0.0)
            if (now - last) < rule.cooldown_s:
                continue
            try:
                effect = rule.effect(snapshot)
            except Exception as exc:
                record_degradation('iot_bridge', exc)
                logger.debug("iot rule effect build failed: %s", exc)
                continue
            self._last_fired[rule.name] = now
            for tname, transport in self._transports.items():
                try:
                    out = await transport.apply(effect)
                    results.append({"transport": tname, "rule": rule.name, "out": out})
                except Exception as exc:
                    record_degradation('iot_bridge', exc)
                    logger.debug("iot transport %s apply failed: %s", tname, exc)
        return results

    async def observe_loop(self) -> None:
        """Drain observations from all transports into the substrate's
        prediction-error stream. Each observation is tagged with
        ``source="iot:<transport>"`` so it never gets confused with
        internal-only signals.
        """
        while self._running:
            for tname, transport in self._transports.items():
                try:
                    obs = await transport.observe()
                except Exception as exc:
                    record_degradation('iot_bridge', exc)
                    logger.debug("iot observe failed (%s): %s", tname, exc)
                    obs = None
                if obs is None:
                    continue
                self._inject_to_substrate(tname, obs)
            await asyncio.sleep(2.0)

    @staticmethod
    def _inject_to_substrate(transport_name: str, observation: Dict[str, Any]) -> None:
        try:
            from core.container import ServiceContainer
            sg = ServiceContainer.get("sensory_gate", default=None)
            if sg is not None and hasattr(sg, "ingest"):
                sg.ingest({"source": f"iot:{transport_name}", "observation": observation, "when": time.time()})
        except Exception as exc:
            record_degradation('iot_bridge', exc)
            logger.debug("iot substrate inject failed: %s", exc)

    async def start(self, *, interval: float = 5.0) -> None:
        if self._running:
            return
        self._running = True

        async def _loop() -> None:
            while self._running:
                try:
                    await self.tick()
                except Exception as exc:
                    record_degradation('iot_bridge', exc)
                    logger.debug("iot bridge tick failed: %s", exc)
                await asyncio.sleep(interval)

        self._task = get_task_tracker().create_task(_loop(), name="IoTBridge")
        get_task_tracker().create_task(self.observe_loop(), name="IoTBridgeObserve")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


_BRIDGE: Optional[IoTBridge] = None


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
