"""Live channel inventory and provenance-aware observation service."""

from __future__ import annotations

import inspect
import math
import re
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol, cast, runtime_checkable

from core.reality_reach.actuation import ActuatorCapability, RealityAdapter
from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    NumericDomain,
    ReachabilityCertificate,
    RealityIR,
    RealityLayer,
)
from core.reality_reach.reachability import ChannelRegistry, ReachabilityEngine
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lockdep import checked_lock
from core.runtime.resource_observation import ObservationSource, get_resource_observer
from core.runtime.service_registry import register_runtime_service

_SQLITE_INT_MAX = (1 << 63) - 1
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class ReadingStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    PERMISSION_DENIED = "permission_denied"
    DEGRADED = "degraded"
    SIMULATED = "simulated"
    UNCALIBRATED = "uncalibrated"


@dataclass(frozen=True, slots=True)
class ChannelReading:
    channel_id: str
    value: float | None
    unit: str
    captured_at_ns: int
    status: ReadingStatus
    source: str
    scenario_id: str = ""
    uncertainty: float | None = None
    error: str = ""
    ingested_at_ns: int = 0
    ingested_monotonic_ns: int = 0
    session_id: str = ""
    sequence: int = 0
    wall_clock_source: str = "system.time_ns"
    source_epoch: str = ""
    source_sequence: int = 0
    source_event_id: str = ""
    source_quality: str = ""
    adapter_identity_sha256: str = ""
    adapter_registration_generation: int = 0
    adapter_identity_stable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.channel_id, str) or not self.channel_id:
            raise ValueError("channel_id must be non-empty")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("unit must be non-empty")
        if (
            isinstance(self.captured_at_ns, bool)
            or not isinstance(self.captured_at_ns, int)
            or not 0 < self.captured_at_ns <= _SQLITE_INT_MAX
        ):
            raise ValueError("captured_at_ns must be a positive signed 64-bit integer")
        if not isinstance(self.status, ReadingStatus):
            raise TypeError("status must be a ReadingStatus")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be non-empty")
        for name, value in (
            ("ingested_at_ns", self.ingested_at_ns),
            ("ingested_monotonic_ns", self.ingested_monotonic_ns),
            ("sequence", self.sequence),
            ("source_sequence", self.source_sequence),
            ("adapter_registration_generation", self.adapter_registration_generation),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= _SQLITE_INT_MAX
            ):
                raise ValueError(f"{name} must be a non-negative signed 64-bit integer")
        if self.adapter_identity_sha256:
            if not _DIGEST.fullmatch(self.adapter_identity_sha256):
                raise ValueError("adapter_identity_sha256 must be a sha256 digest")
            if self.adapter_registration_generation <= 0:
                raise ValueError(
                    "identified readings require a positive adapter registration generation"
                )
        elif self.adapter_registration_generation != 0:
            raise ValueError("adapter registration generation requires adapter identity")
        elif self.adapter_identity_stable:
            raise ValueError("stable adapter identity requires adapter identity evidence")
        if not isinstance(self.adapter_identity_stable, bool):
            raise TypeError("adapter_identity_stable must be a bool")
        if not isinstance(self.session_id, str):
            raise TypeError("session_id must be a string")
        if not isinstance(self.wall_clock_source, str) or not self.wall_clock_source:
            raise ValueError("wall_clock_source must be non-empty")
        for text_name, text_value in (
            ("source_epoch", self.source_epoch),
            ("source_event_id", self.source_event_id),
            ("source_quality", self.source_quality),
        ):
            if not isinstance(text_value, str):
                raise TypeError(f"{text_name} must be a string")
            if len(text_value) > 256:
                raise ValueError(f"{text_name} exceeds its bounded contract")
        if self.value is not None:
            if isinstance(self.value, bool) or not math.isfinite(float(self.value)):
                raise ValueError("value must be finite when present")
            object.__setattr__(self, "value", float(self.value))
        if self.uncertainty is not None:
            uncertainty = float(self.uncertainty)
            if not math.isfinite(uncertainty) or uncertainty < 0.0:
                raise ValueError("uncertainty must be finite and non-negative")
            object.__setattr__(self, "uncertainty", uncertainty)
        if self.status in {ReadingStatus.AVAILABLE, ReadingStatus.SIMULATED}:
            if self.value is None:
                raise ValueError("available readings require a value")

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "value": self.value,
            "unit": self.unit,
            "captured_at_ns": self.captured_at_ns,
            "status": self.status.value,
            "source": self.source,
            "scenario_id": self.scenario_id,
            "uncertainty": self.uncertainty,
            "error": self.error,
            "ingested_at_ns": self.ingested_at_ns,
            "ingested_monotonic_ns": self.ingested_monotonic_ns,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "wall_clock_source": self.wall_clock_source,
            "source_epoch": self.source_epoch,
            "source_sequence": self.source_sequence,
            "source_event_id": self.source_event_id,
            "source_quality": self.source_quality,
            "adapter_identity_sha256": self.adapter_identity_sha256,
            "adapter_registration_generation": self.adapter_registration_generation,
            "adapter_identity_stable": self.adapter_identity_stable,
        }

    @property
    def sha256(self) -> str:
        return str(sha256_hex(canonical_json(self.to_dict())))

    @property
    def event_sha256(self) -> str:
        """Digest source evidence without Aura-owned ingestion lineage."""

        return str(
            sha256_hex(
                canonical_json(
                    {
                        "channel_id": self.channel_id,
                        "value": self.value,
                        "unit": self.unit,
                        "captured_at_ns": self.captured_at_ns,
                        "status": self.status.value,
                        "source": self.source,
                        "scenario_id": self.scenario_id,
                        "uncertainty": self.uncertainty,
                        "error": self.error,
                        "wall_clock_source": self.wall_clock_source,
                        "source_epoch": self.source_epoch,
                        "source_sequence": self.source_sequence,
                        "source_event_id": self.source_event_id,
                        "source_quality": self.source_quality,
                        "adapter_identity_sha256": self.adapter_identity_sha256,
                        "adapter_registration_generation": (
                            self.adapter_registration_generation
                        ),
                        "adapter_identity_stable": self.adapter_identity_stable,
                    }
                )
            )
        )


@runtime_checkable
class LiveChannelAdapter(Protocol):
    @property
    def adapter_id(self) -> str: ...

    def declarations(self) -> tuple[ChannelDeclaration, ...]: ...

    def read(self) -> tuple[ChannelReading, ...]: ...


@dataclass(frozen=True, slots=True)
class AdapterInventoryEntry:
    """Authoritative live ownership and physical-identity fence."""

    adapter_id: str
    channel_ids: tuple[str, ...]
    identity_sha256: str
    registration_generation: int
    stable_identity: bool

    def __post_init__(self) -> None:
        if not self.adapter_id:
            raise ValueError("adapter_id must be non-empty")
        if not self.channel_ids or len(self.channel_ids) != len(set(self.channel_ids)):
            raise ValueError("adapter inventory channels must be non-empty and unique")
        if not _DIGEST.fullmatch(self.identity_sha256):
            raise ValueError("adapter inventory identity must be a sha256 digest")
        if not 0 < self.registration_generation <= _SQLITE_INT_MAX:
            raise ValueError("adapter registration generation must be positive")


class HostResourceAdapter:
    """Adapts Aura's attributable resource observer into measurable channels."""

    adapter_id = "host.resource_observer"

    def __init__(self, observer: Any | None = None) -> None:
        self._observer = observer or get_resource_observer()
        self._declarations = self._build_declarations()

    @staticmethod
    def _build_declarations() -> tuple[ChannelDeclaration, ...]:
        common = {
            "kind": ChannelKind.SENSOR,
            "coupling": CouplingClass.SOFTWARE,
            "reality_layers": (RealityLayer.INTERNAL, RealityLayer.EFFECTIVE),
            "evidence_level": EvidenceLevel.P1,
            "owner": "core.runtime.resource_observation",
            "sample_rate_hz": 1.0,
            "max_latency_s": 2.0,
            "stale_after_s": 10.0,
            "coupling_validated": True,
        }
        return (
            ChannelDeclaration(
                channel_id="host.compute.cpu_percent",
                observable="cpu_usage_percent",
                unit="percent",
                domain=NumericDomain(0.0, 100.0),
                resolution=0.1,
                reference_id="host.kernel.cpu",
                **common,
            ),
            ChannelDeclaration(
                channel_id="host.memory.used_percent",
                observable="memory_usage_percent",
                unit="percent",
                domain=NumericDomain(0.0, 100.0),
                resolution=0.1,
                reference_id="host.kernel.memory",
                **common,
            ),
            ChannelDeclaration(
                channel_id="host.disk.root_used_percent",
                observable="disk_usage_percent",
                unit="percent",
                domain=NumericDomain(0.0, 100.0),
                resolution=0.1,
                reference_id="host.kernel.filesystem",
                **common,
            ),
            ChannelDeclaration(
                channel_id="host.thermal.pressure_level",
                observable="thermal_pressure_level",
                unit="level",
                domain=NumericDomain(0.0, 3.0),
                resolution=1.0,
                reference_id="host.kernel.thermal",
                **common,
            ),
            ChannelDeclaration(
                channel_id="host.power.battery_percent",
                observable="battery_charge_percent",
                unit="percent",
                domain=NumericDomain(0.0, 100.0),
                resolution=1.0,
                reference_id="host.power.battery",
                **common,
            ),
        )

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        return self._declarations

    def read(self) -> tuple[ChannelReading, ...]:
        snapshot = self._observer.snapshot(path="/", include_processes=False)
        return (
            self._from_observation(
                self._declarations[0],
                snapshot.compute,
                snapshot.compute.cpu_percent,
            ),
            self._from_observation(
                self._declarations[1],
                snapshot.memory,
                snapshot.memory.percent,
            ),
            self._from_observation(
                self._declarations[2],
                snapshot.disk,
                snapshot.disk.percent,
            ),
            self._from_observation(
                self._declarations[3],
                snapshot.thermal,
                snapshot.thermal.level,
            ),
            self._from_observation(
                self._declarations[4],
                snapshot.power,
                snapshot.power.battery_percent,
            ),
        )

    @staticmethod
    def _from_observation(
        declaration: ChannelDeclaration,
        observation: Any,
        value: float,
    ) -> ChannelReading:
        provenance = observation.provenance
        available = bool(getattr(observation, "available", True))
        if not available:
            status = ReadingStatus.UNAVAILABLE
            reading_value = None
        elif provenance.source == ObservationSource.SIMULATED:
            status = ReadingStatus.SIMULATED
            reading_value = value
        elif provenance.source in {ObservationSource.HOST, ObservationSource.LIVE_PRESSURE}:
            status = ReadingStatus.AVAILABLE
            reading_value = value
        else:
            status = ReadingStatus.UNAVAILABLE
            reading_value = None
        return ChannelReading(
            channel_id=declaration.channel_id,
            value=reading_value,
            unit=declaration.unit,
            captured_at_ns=max(1, int(float(provenance.captured_at) * 1_000_000_000)),
            status=status,
            source=provenance.source.value,
            scenario_id=str(provenance.scenario_id or ""),
            uncertainty=declaration.resolution,
            error=str(getattr(observation, "error", "") or ""),
        )


class RealityReachService:
    """Owns live declarations, readings, and feasibility certificates."""

    def __init__(
        self,
        adapters: Iterable[LiveChannelAdapter] = (),
        *,
        clock_ns: Any = time.time_ns,
        monotonic_clock_ns: Any = time.monotonic_ns,
        session_id: str | None = None,
    ) -> None:
        self._lock = checked_lock("live.instance", reentrant=True)
        self._refresh_lock = checked_lock("live.refresh")
        self._clock_ns = clock_ns
        self._monotonic_clock_ns = monotonic_clock_ns
        self._session_id = session_id or str(uuid.uuid4())
        if not self._session_id:
            raise ValueError("session_id must be non-empty")
        self._registry = ChannelRegistry()
        self._adapters: dict[str, LiveChannelAdapter] = {}
        self._adapter_channels: dict[str, tuple[str, ...]] = {}
        self._adapter_inventory: dict[str, AdapterInventoryEntry] = {}
        self._adapter_registration_generation = 0
        self._adapter_capabilities: dict[str, tuple[ActuatorCapability, ...]] = {}
        self._actuator_adapters: dict[str, RealityAdapter] = {}
        self._readings: dict[str, ChannelReading] = {}
        self._refresh_generation = 0
        self._ingest_generation = 0
        self._last_refresh_ns = 0
        self._last_refresh_monotonic_ns = 0
        for adapter in adapters:
            self.register_adapter(adapter)

    @property
    def session_id(self) -> str:
        """Opaque process-session identity for attachment fencing."""

        return self._session_id

    def register_adapter(self, adapter: LiveChannelAdapter) -> None:
        if not isinstance(adapter, LiveChannelAdapter):
            raise TypeError("adapter must satisfy LiveChannelAdapter")
        adapter_id = str(adapter.adapter_id or "")
        if not adapter_id:
            raise ValueError("adapter_id must be non-empty")
        declarations = tuple(adapter.declarations())
        if not declarations:
            raise ValueError("adapter must declare at least one channel")
        actuator_declarations = {
            declaration.channel_id: declaration
            for declaration in declarations
            if declaration.kind == ChannelKind.ACTUATOR
        }
        capabilities: tuple[ActuatorCapability, ...] = ()
        if actuator_declarations:
            if not isinstance(adapter, RealityAdapter):
                raise TypeError(
                    "actuator adapters must implement the complete RealityAdapter protocol"
                )
            for method_name in (
                "prepare",
                "actuate",
                "verify_effect",
                "cancel",
                "safe_state",
                "rollback",
            ):
                if not inspect.iscoroutinefunction(getattr(adapter, method_name, None)):
                    raise TypeError(f"actuator adapter method {method_name} must be asynchronous")
            capabilities = tuple(adapter.actuator_capabilities())
            capability_by_channel: dict[str, ActuatorCapability] = {}
            for capability in capabilities:
                if not isinstance(capability, ActuatorCapability):
                    raise TypeError("actuator_capabilities returned a non-capability")
                if capability.adapter_id != adapter_id:
                    raise ValueError("actuator capability adapter identity differs")
                if capability.channel_id in capability_by_channel:
                    raise ValueError("duplicate actuator capability channel")
                declaration = actuator_declarations.get(capability.channel_id)
                if declaration is None:
                    raise ValueError("actuator capability has no matching declaration")
                if (
                    capability.magnitude_domain.minimum < declaration.domain.minimum
                    or capability.magnitude_domain.maximum > declaration.domain.maximum
                ):
                    raise ValueError("actuator capability exceeds its declared domain")
                capability_by_channel[capability.channel_id] = capability
            if set(capability_by_channel) != set(actuator_declarations):
                raise ValueError("every actuator declaration requires one capability")
        with self._lock:
            if adapter_id in self._adapters:
                raise ValueError(f"adapter already registered: {adapter_id}")
            registered: list[str] = []
            try:
                for declaration in declarations:
                    self._registry.register(declaration)
                    registered.append(declaration.channel_id)
            except Exception:
                for channel_id in registered:
                    self._registry.unregister(channel_id)
                raise
            self._adapters[adapter_id] = adapter
            self._adapter_channels[adapter_id] = tuple(registered)
            self._adapter_registration_generation += 1
            registration_generation = self._adapter_registration_generation
            claimed_identity = str(
                getattr(adapter, "physical_identity_sha256", "") or ""
            ).strip()
            if claimed_identity and not _DIGEST.fullmatch(claimed_identity):
                for channel_id in registered:
                    self._registry.unregister(channel_id)
                self._adapters.pop(adapter_id, None)
                self._adapter_channels.pop(adapter_id, None)
                raise ValueError("adapter physical identity must be a sha256 digest")
            identity_sha256 = claimed_identity or str(
                sha256_hex(
                    canonical_json(
                        {
                            "adapter_id": adapter_id,
                            "adapter_type": (
                                f"{type(adapter).__module__}.{type(adapter).__qualname__}"
                            ),
                            "channel_declarations": [item.sha256 for item in declarations],
                            "registration_generation": registration_generation,
                            "service_session_id": self._session_id,
                        }
                    )
                )
            )
            self._adapter_inventory[adapter_id] = AdapterInventoryEntry(
                adapter_id=adapter_id,
                channel_ids=tuple(registered),
                identity_sha256=identity_sha256,
                registration_generation=registration_generation,
                stable_identity=bool(claimed_identity),
            )
            self._adapter_capabilities[adapter_id] = capabilities
            if isinstance(adapter, RealityAdapter):
                for capability in capabilities:
                    self._actuator_adapters[capability.channel_id] = adapter

    def unregister_adapter(self, adapter_id: str) -> None:
        """Atomically remove one adapter and every channel it owns."""

        if not isinstance(adapter_id, str) or not adapter_id:
            raise ValueError("adapter_id must be non-empty")
        with self._lock:
            adapter = self._adapters.get(adapter_id)
            if adapter is None:
                raise LookupError(f"adapter is not registered: {adapter_id}")
            channels = self._adapter_channels.get(adapter_id, ())
            capabilities = self._adapter_capabilities.get(adapter_id, ())
            for capability in capabilities:
                self._actuator_adapters.pop(capability.channel_id, None)
            for channel_id in channels:
                self._readings.pop(channel_id, None)
                self._registry.unregister(channel_id)
            self._adapter_capabilities.pop(adapter_id, None)
            self._adapter_inventory.pop(adapter_id, None)
            self._adapter_channels.pop(adapter_id, None)
            self._adapters.pop(adapter_id, None)

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        """Return the immutable public channel inventory.

        Consumers such as the sensory router must not reach into the registry
        or adapter maps.  The registry snapshot is already deterministic and
        immutable, so exposing it here preserves the ownership boundary while
        letting cognition reason over the same declarations used by the
        reachability and actuation engines.
        """

        with self._lock:
            return cast(tuple[ChannelDeclaration, ...], self._registry.snapshot())

    def adapter_channels(self) -> dict[str, tuple[str, ...]]:
        """Return a copy of adapter-to-channel ownership for provenance."""

        with self._lock:
            return dict(self._adapter_channels)

    def adapter_inventory(self) -> dict[str, AdapterInventoryEntry]:
        """Return immutable live ownership fenced by device or registration identity."""

        with self._lock:
            return dict(self._adapter_inventory)

    def adapter_identity_for_channel(self, channel_id: str) -> AdapterInventoryEntry | None:
        """Resolve the authoritative current identity that owns one channel."""

        with self._lock:
            for entry in self._adapter_inventory.values():
                if channel_id in entry.channel_ids:
                    return entry
        return None

    def adapter_id_for_channel(self, channel_id: str) -> str | None:
        """Resolve channel ownership without exposing executable adapters."""

        with self._lock:
            for adapter_id, channel_ids in self._adapter_channels.items():
                if channel_id in channel_ids:
                    return adapter_id
        return None

    def actuator_adapter(self, channel_id: str) -> RealityAdapter | None:
        """Return an executable adapter only when its observation route is live."""

        if channel_id not in self.executable_actuator_channels():
            return None
        with self._lock:
            return self._actuator_adapters.get(channel_id)

    def scalar_acceptance_adapter(self, adapter_id: str) -> Any | None:
        """Resolve one exact scalar adapter for the governed acceptance owner."""

        if not isinstance(adapter_id, str) or not adapter_id:
            raise ValueError("adapter_id must be non-empty")
        from core.reality_reach.scalar_adapter import ScalarRealityAdapter

        with self._lock:
            adapter = self._adapters.get(adapter_id)
            return adapter if isinstance(adapter, ScalarRealityAdapter) else None

    def actuator_capability(self, channel_id: str) -> ActuatorCapability | None:
        if channel_id not in self.executable_actuator_channels():
            return None
        with self._lock:
            for capabilities in self._adapter_capabilities.values():
                for capability in capabilities:
                    if capability.channel_id == channel_id:
                        return capability
        return None

    def executable_actuator_channels(self) -> tuple[str, ...]:
        readings = self.readings()
        with self._lock:
            capabilities = tuple(
                capability
                for values in self._adapter_capabilities.values()
                for capability in values
            )
        executable: list[str] = []
        for capability in capabilities:
            observation_route_ok = True
            for channel_id in capability.observation_channels:
                declaration = self._registry.get(channel_id)
                reading = readings.get(channel_id)
                if (
                    declaration is None
                    or declaration.kind != ChannelKind.SENSOR
                    or reading is None
                    or reading.status != ReadingStatus.AVAILABLE
                ):
                    observation_route_ok = False
                    break
            if observation_route_ok:
                executable.append(capability.channel_id)
        return tuple(sorted(executable))

    def refresh(self) -> dict[str, ChannelReading]:
        with self._refresh_lock:
            with self._lock:
                adapters = tuple(self._adapters.items())
                refresh_generation = self._refresh_generation + 1
            last_wall_ns = int(self._clock_ns())
            last_monotonic_ns = int(self._monotonic_clock_ns())
            for adapter_id, adapter in adapters:
                expected = self._adapter_channels[adapter_id]
                identity = self._adapter_inventory[adapter_id]
                with self._lock:
                    self._ingest_generation += 1
                    sequence = self._ingest_generation
                try:
                    returned = tuple(adapter.read())
                    last_wall_ns = int(self._clock_ns())
                    last_monotonic_ns = int(self._monotonic_clock_ns())
                    by_channel = self._validate_adapter_readings(
                        adapter_id,
                        expected,
                        returned,
                        now_ns=last_wall_ns,
                        monotonic_now_ns=last_monotonic_ns,
                        sequence=sequence,
                        adapter_identity=identity,
                    )
                except (
                    AttributeError,
                    LookupError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    last_wall_ns = int(self._clock_ns())
                    last_monotonic_ns = int(self._monotonic_clock_ns())
                    by_channel = {
                        channel_id: self._unavailable_reading(
                            channel_id,
                            captured_at_ns=last_wall_ns,
                            ingested_monotonic_ns=last_monotonic_ns,
                            sequence=sequence,
                            adapter_identity=identity,
                            error=f"{type(exc).__name__}:{exc}",
                        )
                        for channel_id in expected
                    }
                with self._lock:
                    self._readings.update(by_channel)
            with self._lock:
                self._refresh_generation = refresh_generation
                self._last_refresh_ns = last_wall_ns
                self._last_refresh_monotonic_ns = last_monotonic_ns
            return self.readings(
                now_ns=last_wall_ns,
                monotonic_now_ns=last_monotonic_ns,
            )

    def ingest_sensor_readings(
        self,
        adapter_id: str,
        readings: Iterable[ChannelReading],
    ) -> dict[str, ChannelReading]:
        """Normalize one async sensor batch through the canonical live boundary."""

        if not isinstance(adapter_id, str) or not adapter_id:
            raise ValueError("adapter_id must be non-empty")
        returned = tuple(readings)
        with self._refresh_lock:
            with self._lock:
                if adapter_id not in self._adapters:
                    raise LookupError(f"adapter is not registered: {adapter_id}")
                expected = tuple(
                    channel_id
                    for channel_id in self._adapter_channels[adapter_id]
                    if (
                        (declaration := self._registry.get(channel_id)) is not None
                        and declaration.kind == ChannelKind.SENSOR
                    )
                )
                identity = self._adapter_inventory[adapter_id]
                self._ingest_generation += 1
                sequence = self._ingest_generation
            if not expected:
                raise ValueError(f"adapter has no sensor channels: {adapter_id}")
            now_ns = int(self._clock_ns())
            monotonic_now_ns = int(self._monotonic_clock_ns())
            normalized = self._validate_adapter_readings(
                adapter_id,
                expected,
                returned,
                now_ns=now_ns,
                monotonic_now_ns=monotonic_now_ns,
                sequence=sequence,
                adapter_identity=identity,
            )
            with self._lock:
                self._readings.update(normalized)
                self._last_refresh_ns = now_ns
                self._last_refresh_monotonic_ns = monotonic_now_ns
            return dict(normalized)

    def readings(
        self,
        *,
        now_ns: int | None = None,
        monotonic_now_ns: int | None = None,
    ) -> dict[str, ChannelReading]:
        current_ns = int(self._clock_ns() if now_ns is None else now_ns)
        current_monotonic_ns = int(
            self._monotonic_clock_ns() if monotonic_now_ns is None else monotonic_now_ns
        )
        with self._lock:
            items = tuple(self._readings.items())
        return {
            channel_id: self._with_freshness(
                reading,
                now_ns=current_ns,
                monotonic_now_ns=current_monotonic_ns,
            )
            for channel_id, reading in items
        }

    def reading(
        self,
        channel_id: str,
        *,
        now_ns: int | None = None,
        monotonic_now_ns: int | None = None,
    ) -> ChannelReading | None:
        with self._lock:
            reading = self._readings.get(channel_id)
        if reading is None:
            return None
        current_ns = int(self._clock_ns() if now_ns is None else now_ns)
        current_monotonic_ns = int(
            self._monotonic_clock_ns() if monotonic_now_ns is None else monotonic_now_ns
        )
        return self._with_freshness(
            reading,
            now_ns=current_ns,
            monotonic_now_ns=current_monotonic_ns,
        )

    def analyze(self, contract: RealityIR, *, refresh: bool = True) -> ReachabilityCertificate:
        if refresh:
            self.refresh()
        readings = self.readings()
        effective = ChannelRegistry()
        for declaration in self._registry.snapshot():
            reading = readings.get(declaration.channel_id)
            enabled = bool(
                reading is not None
                and (
                    reading.status == ReadingStatus.AVAILABLE
                    or (
                        reading.status == ReadingStatus.SIMULATED
                        and contract.reality_layer == RealityLayer.INTERNAL
                    )
                )
            )
            effective.register(replace(declaration, enabled=enabled))
        return ReachabilityEngine().analyze(contract, effective)

    def status(self) -> dict[str, Any]:
        readings = self.readings()
        counts: dict[str, int] = {}
        for reading in readings.values():
            counts[reading.status.value] = counts.get(reading.status.value, 0) + 1
        with self._lock:
            return {
                "ready": self.is_ready(),
                "alive": True,
                "adapter_count": len(self._adapters),
                "channel_count": len(self._registry.snapshot()),
                "declared_actuator_count": len(self._actuator_adapters),
                "executable_actuator_count": len(self.executable_actuator_channels()),
                "registry_sha256": self._registry.sha256,
                "refresh_generation": self._refresh_generation,
                "ingest_generation": self._ingest_generation,
                "last_refresh_ns": self._last_refresh_ns,
                "last_refresh_monotonic_ns": self._last_refresh_monotonic_ns,
                "session_id": self._session_id,
                "reading_status_counts": counts,
            }

    def is_alive(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return any(
            reading.status == ReadingStatus.AVAILABLE for reading in self.readings().values()
        )

    def _validate_adapter_readings(
        self,
        adapter_id: str,
        expected: tuple[str, ...],
        returned: tuple[ChannelReading, ...],
        *,
        now_ns: int,
        monotonic_now_ns: int,
        sequence: int,
        adapter_identity: AdapterInventoryEntry,
    ) -> dict[str, ChannelReading]:
        by_channel: dict[str, ChannelReading] = {}
        expected_set = set(expected)
        for reading in returned:
            if not isinstance(reading, ChannelReading):
                raise TypeError(f"adapter {adapter_id} returned a non-reading")
            if reading.channel_id not in expected_set:
                raise ValueError(
                    f"adapter {adapter_id} returned undeclared channel {reading.channel_id}"
                )
            if reading.channel_id in by_channel:
                raise ValueError(
                    f"adapter {adapter_id} returned duplicate channel {reading.channel_id}"
                )
            declaration = self._registry.get(reading.channel_id)
            if declaration is None or declaration.unit != reading.unit:
                raise ValueError(f"adapter {adapter_id} returned a unit-mismatched reading")
            normalized = replace(
                reading,
                ingested_at_ns=now_ns,
                ingested_monotonic_ns=monotonic_now_ns,
                session_id=self._session_id,
                sequence=sequence,
                adapter_identity_sha256=adapter_identity.identity_sha256,
                adapter_registration_generation=(
                    adapter_identity.registration_generation
                ),
                adapter_identity_stable=adapter_identity.stable_identity,
            )
            if reading.session_id and reading.session_id != self._session_id:
                normalized = replace(
                    normalized,
                    status=ReadingStatus.STALE,
                    error="reading_session_mismatch",
                )
            elif reading.captured_at_ns > now_ns:
                normalized = replace(
                    normalized,
                    status=ReadingStatus.DEGRADED,
                    error="reading_wall_clock_regressed_or_future_capture",
                )
            elif now_ns - reading.captured_at_ns > int(declaration.stale_after_s * 1e9):
                normalized = replace(
                    normalized,
                    status=ReadingStatus.STALE,
                    error=f"reading_stale_at_ingest:age_ns={now_ns - reading.captured_at_ns}",
                )
            elif reading.value is not None and not declaration.domain.contains(reading.value):
                normalized = replace(
                    normalized,
                    status=ReadingStatus.DEGRADED,
                    error="reading_outside_declared_domain",
                )
            elif (
                declaration.calibration_valid_until_ns is not None
                and now_ns > declaration.calibration_valid_until_ns
            ):
                normalized = replace(
                    normalized,
                    status=ReadingStatus.UNCALIBRATED,
                    error="channel_calibration_expired",
                )
            by_channel[reading.channel_id] = normalized
        for channel_id in expected:
            if channel_id not in by_channel:
                by_channel[channel_id] = self._unavailable_reading(
                    channel_id,
                    captured_at_ns=now_ns,
                    ingested_monotonic_ns=monotonic_now_ns,
                    sequence=sequence,
                    adapter_identity=adapter_identity,
                    error=f"adapter_missing_reading:{adapter_id}",
                )
        return by_channel

    def _unavailable_reading(
        self,
        channel_id: str,
        *,
        captured_at_ns: int,
        ingested_monotonic_ns: int,
        sequence: int,
        error: str,
        adapter_identity: AdapterInventoryEntry,
    ) -> ChannelReading:
        declaration = self._registry.get(channel_id)
        if declaration is None:
            raise KeyError(channel_id)
        return ChannelReading(
            channel_id=channel_id,
            value=None,
            unit=declaration.unit,
            captured_at_ns=max(1, captured_at_ns),
            status=ReadingStatus.UNAVAILABLE,
            source=ObservationSource.UNAVAILABLE.value,
            error=error,
            ingested_at_ns=max(1, captured_at_ns),
            ingested_monotonic_ns=max(1, ingested_monotonic_ns),
            session_id=self._session_id,
            sequence=sequence,
            adapter_identity_sha256=adapter_identity.identity_sha256,
            adapter_registration_generation=adapter_identity.registration_generation,
            adapter_identity_stable=adapter_identity.stable_identity,
        )

    def _with_freshness(
        self,
        reading: ChannelReading,
        *,
        now_ns: int,
        monotonic_now_ns: int,
    ) -> ChannelReading:
        declaration = self._registry.get(reading.channel_id)
        if declaration is None or reading.status not in {
            ReadingStatus.AVAILABLE,
            ReadingStatus.SIMULATED,
        }:
            return reading
        if reading.session_id != self._session_id:
            return replace(
                reading,
                status=ReadingStatus.STALE,
                error="reading_session_mismatch",
            )
        if reading.ingested_monotonic_ns <= 0 or reading.sequence <= 0:
            return replace(
                reading,
                status=ReadingStatus.DEGRADED,
                error="reading_monotonic_lineage_missing",
            )
        if monotonic_now_ns < reading.ingested_monotonic_ns:
            return replace(
                reading,
                status=ReadingStatus.DEGRADED,
                error="reading_monotonic_clock_regressed",
            )
        age_ns = monotonic_now_ns - reading.ingested_monotonic_ns
        if age_ns <= int(declaration.stale_after_s * 1e9):
            return reading
        return replace(
            reading,
            status=ReadingStatus.STALE,
            error=f"reading_stale:age_ns={age_ns}",
        )


_SERVICE: RealityReachService | None = None
_SERVICE_LOCK = checked_lock("live.service")


def get_reality_reach_service() -> RealityReachService:
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = RealityReachService((HostResourceAdapter(),))
    return _SERVICE


def register_reality_reach_service() -> RealityReachService:
    service = get_reality_reach_service()
    register_runtime_service(
        "reality_reach",
        service,
        required=False,
        owner="core/reality_reach/live.py",
        registered_by="register_reality_reach_service",
        required_for="physical reachability and experiment evidence",
        failure_policy="degrade_with_receipt",
    )
    return service


__all__ = [
    "AdapterInventoryEntry",
    "ChannelReading",
    "HostResourceAdapter",
    "LiveChannelAdapter",
    "ReadingStatus",
    "RealityReachService",
    "get_reality_reach_service",
    "register_reality_reach_service",
]
