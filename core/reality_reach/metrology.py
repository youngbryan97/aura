"""Calibrated acquisition and hardware-in-loop evidence for Reality Reach.

This module owns the distinction between a scalar that happened to arrive and
a measurement that can support a physical claim.  It never changes adapters or
creates actuation authority.  It validates calibration identity, takes bounded
synchronized samples through the canonical live service, propagates uncertainty,
and fences live, simulated, and HIL evidence from one another.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import statistics
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

from core.governance_context import local_internal_governed_scope
from core.reality_reach.contracts import ChannelDeclaration, ChannelKind
from core.reality_reach.live import ChannelReading, ReadingStatus, RealityReachService
from core.reality_reach.middleware_contracts import canonical_identifier, sha256_digest
from core.runtime.atomic_writer import read_json_envelope
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.lockdep import checked_async_lock
from core.runtime.state_ownership import state_root
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.RealityReach.Metrology")

_STATE_SCHEMA = "aura.reality_reach.metrology"
_STATE_VERSION = 1
_MAX_CERTIFICATES = 4096
_MAX_RECEIPTS = 64
_MAX_CHANNELS = 32
_MAX_SAMPLES = 1024
_MAX_MEASUREMENTS_PER_RECEIPT = 1024
_T = TypeVar("_T")


class MetrologyError(RuntimeError):
    """A calibration, acquisition, or evidence-separation contract failed."""


class AcquisitionMode(StrEnum):
    LIVE = "live"
    SIMULATION = "simulation"
    HARDWARE_IN_LOOP = "hardware_in_loop"


class EvidenceSource(StrEnum):
    LIVE = "live"
    SIMULATED = "simulated"


@dataclass(frozen=True, slots=True)
class CalibrationCertificate:
    """Traceable affine calibration for one declared sensor channel."""

    calibration_id: str
    channel_id: str
    reference_standard_id: str
    traceability_sha256: str
    issued_at_ns: int
    valid_until_ns: int
    scale: float = 1.0
    offset: float = 0.0
    standard_uncertainty: float = 0.0
    issuer: str = ""

    def __post_init__(self) -> None:
        for name in ("calibration_id", "channel_id", "reference_standard_id"):
            object.__setattr__(
                self,
                name,
                canonical_identifier(str(getattr(self, name)), name=name),
            )
        object.__setattr__(
            self,
            "traceability_sha256",
            sha256_digest(self.traceability_sha256, name="traceability_sha256"),
        )
        for name in ("issued_at_ns", "valid_until_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.valid_until_ns <= self.issued_at_ns:
            raise ValueError("calibration validity must end after issuance")
        for name in ("scale", "offset", "standard_uncertainty"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "standard_uncertainty" and value < 0.0:
                raise ValueError("standard_uncertainty must be non-negative")
            object.__setattr__(self, name, value)
        if self.scale == 0.0:
            raise ValueError("calibration scale must be non-zero")
        issuer = str(self.issuer or "").strip()
        if not issuer or len(issuer) > 256:
            raise ValueError("issuer must be non-empty and bounded")
        object.__setattr__(self, "issuer", issuer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "channel_id": self.channel_id,
            "reference_standard_id": self.reference_standard_id,
            "traceability_sha256": self.traceability_sha256,
            "issued_at_ns": self.issued_at_ns,
            "valid_until_ns": self.valid_until_ns,
            "scale": self.scale,
            "offset": self.offset,
            "standard_uncertainty": self.standard_uncertainty,
            "issuer": self.issuer,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CalibrationCertificate:
        return cls(
            calibration_id=str(value["calibration_id"]),
            channel_id=str(value["channel_id"]),
            reference_standard_id=str(value["reference_standard_id"]),
            traceability_sha256=str(value["traceability_sha256"]),
            issued_at_ns=int(value["issued_at_ns"]),
            valid_until_ns=int(value["valid_until_ns"]),
            scale=float(value.get("scale", 1.0)),
            offset=float(value.get("offset", 0.0)),
            standard_uncertainty=float(value.get("standard_uncertainty", 0.0)),
            issuer=str(value.get("issuer") or ""),
        )

    @property
    def sha256(self) -> str:
        return str(sha256_hex(canonical_json(self.to_dict())))


@dataclass(frozen=True, slots=True)
class AcquisitionChannel:
    channel_id: str
    expected_source: EvidenceSource = EvidenceSource.LIVE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "channel_id",
            canonical_identifier(self.channel_id, name="channel_id"),
        )
        if not isinstance(self.expected_source, EvidenceSource):
            raise TypeError("expected_source must be an EvidenceSource")

    def to_dict(self) -> dict[str, str]:
        return {
            "channel_id": self.channel_id,
            "expected_source": self.expected_source.value,
        }


@dataclass(frozen=True, slots=True)
class AcquisitionTask:
    task_id: str
    channels: tuple[AcquisitionChannel, ...]
    mode: AcquisitionMode = AcquisitionMode.LIVE
    sample_count: int = 1
    sample_interval_s: float = 0.0
    timeout_s: float = 30.0
    max_capture_skew_ns: int = 100_000_000
    require_calibration: bool = False
    scenario_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", canonical_identifier(self.task_id, name="task_id"))
        channels = tuple(self.channels)
        if not channels or len(channels) > _MAX_CHANNELS:
            raise ValueError(f"acquisition requires between 1 and {_MAX_CHANNELS} channels")
        if len({item.channel_id for item in channels}) != len(channels):
            raise ValueError("acquisition channels must be unique")
        if not isinstance(self.mode, AcquisitionMode):
            raise TypeError("mode must be an AcquisitionMode")
        if isinstance(self.sample_count, bool) or not 1 <= self.sample_count <= _MAX_SAMPLES:
            raise ValueError(f"sample_count must lie inside [1, {_MAX_SAMPLES}]")
        if len(channels) * self.sample_count > _MAX_MEASUREMENTS_PER_RECEIPT:
            raise ValueError(
                "channel_count * sample_count exceeds the retained evidence bound "
                f"({_MAX_MEASUREMENTS_PER_RECEIPT})"
            )
        for name, minimum, maximum in (
            ("sample_interval_s", 0.0, 60.0),
            ("timeout_s", 0.05, 86_400.0),
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must lie inside [{minimum}, {maximum}]")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.max_capture_skew_ns, bool)
            or not 0 <= self.max_capture_skew_ns <= 10_000_000_000
        ):
            raise ValueError("max_capture_skew_ns must lie inside [0, 10000000000]")
        scenario = str(self.scenario_id or "").strip()
        if scenario:
            scenario = canonical_identifier(scenario, name="scenario_id")
        object.__setattr__(self, "scenario_id", scenario)
        expected = {item.expected_source for item in channels}
        if self.mode is AcquisitionMode.LIVE and expected != {EvidenceSource.LIVE}:
            raise ValueError("live acquisition accepts only live channels")
        if self.mode is AcquisitionMode.SIMULATION and expected != {EvidenceSource.SIMULATED}:
            raise ValueError("simulation acquisition accepts only simulated channels")
        if self.mode is AcquisitionMode.HARDWARE_IN_LOOP and expected != {
            EvidenceSource.LIVE,
            EvidenceSource.SIMULATED,
        }:
            raise ValueError("HIL acquisition requires explicit live and simulated channels")
        if self.mode is not AcquisitionMode.LIVE and not scenario:
            raise ValueError("simulation and HIL acquisition require a scenario_id")
        object.__setattr__(self, "channels", channels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "channels": [item.to_dict() for item in self.channels],
            "mode": self.mode.value,
            "sample_count": self.sample_count,
            "sample_interval_s": self.sample_interval_s,
            "timeout_s": self.timeout_s,
            "max_capture_skew_ns": self.max_capture_skew_ns,
            "require_calibration": self.require_calibration,
            "scenario_id": self.scenario_id,
        }

    @property
    def sha256(self) -> str:
        return str(sha256_hex(canonical_json(self.to_dict())))


@dataclass(frozen=True, slots=True)
class Measurement:
    channel_id: str
    value: float
    unit: str
    captured_at_ns: int
    source: EvidenceSource
    scenario_id: str
    wall_clock_source: str
    random_uncertainty: float
    resolution_uncertainty: float
    systematic_uncertainty: float
    calibration_sha256: str
    reading_sha256: str

    @property
    def standard_uncertainty(self) -> float:
        return math.sqrt(
            self.random_uncertainty**2
            + self.resolution_uncertainty**2
            + self.systematic_uncertainty**2
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "value": self.value,
            "unit": self.unit,
            "captured_at_ns": self.captured_at_ns,
            "source": self.source.value,
            "scenario_id": self.scenario_id,
            "wall_clock_source": self.wall_clock_source,
            "random_uncertainty": self.random_uncertainty,
            "resolution_uncertainty": self.resolution_uncertainty,
            "systematic_uncertainty": self.systematic_uncertainty,
            "standard_uncertainty": self.standard_uncertainty,
            "calibration_sha256": self.calibration_sha256,
            "reading_sha256": self.reading_sha256,
        }


@dataclass(frozen=True, slots=True)
class MeasurementSummary:
    channel_id: str
    unit: str
    sample_count: int
    mean: float
    minimum: float
    maximum: float
    standard_uncertainty: float
    coverage_factor: float
    expanded_uncertainty_k2: float
    source: EvidenceSource
    wall_clock_source: str
    calibration_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "unit": self.unit,
            "sample_count": self.sample_count,
            "mean": self.mean,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "standard_uncertainty": self.standard_uncertainty,
            "coverage_factor": self.coverage_factor,
            "expanded_uncertainty_k2": self.expanded_uncertainty_k2,
            "source": self.source.value,
            "wall_clock_source": self.wall_clock_source,
            "calibration_sha256": self.calibration_sha256,
        }


@dataclass(frozen=True, slots=True)
class AcquisitionReceipt:
    run_id: str
    task_sha256: str
    mode: AcquisitionMode
    mode_generation: int
    started_at_ns: int
    completed_at_ns: int
    sample_sets: int
    maximum_observed_skew_ns: int
    scenario_id: str
    measurements: tuple[Measurement, ...]
    summaries: tuple[MeasurementSummary, ...]
    evidence_sha256: str
    restored_mode: AcquisitionMode = AcquisitionMode.LIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_sha256": self.task_sha256,
            "mode": self.mode.value,
            "mode_generation": self.mode_generation,
            "started_at_ns": self.started_at_ns,
            "completed_at_ns": self.completed_at_ns,
            "sample_sets": self.sample_sets,
            "maximum_observed_skew_ns": self.maximum_observed_skew_ns,
            "scenario_id": self.scenario_id,
            "measurements": [item.to_dict() for item in self.measurements],
            "summaries": [item.to_dict() for item in self.summaries],
            "evidence_sha256": self.evidence_sha256,
            "restored_mode": self.restored_mode.value,
        }

    def evidence_document(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_sha256": self.task_sha256,
            "mode": self.mode.value,
            "mode_generation": self.mode_generation,
            "started_at_ns": self.started_at_ns,
            "completed_at_ns": self.completed_at_ns,
            "sample_sets": self.sample_sets,
            "maximum_observed_skew_ns": self.maximum_observed_skew_ns,
            "scenario_id": self.scenario_id,
            "measurements": [item.to_dict() for item in self.measurements],
            "summaries": [item.to_dict() for item in self.summaries],
        }

    def verify_evidence(self) -> bool:
        return self.evidence_sha256 == str(
            sha256_hex(canonical_json(self.evidence_document()))
        )


class RealityMetrologyService:
    """Bounded acquisition owner over the canonical Reality Reach inventory."""

    def __init__(
        self,
        reality: RealityReachService,
        *,
        state_path: Path | None = None,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(reality, RealityReachService):
            raise TypeError("reality must be a RealityReachService")
        self._reality = reality
        self._state_path = Path(state_path or (state_root() / "reality_metrology.json"))
        self._wall_clock_ns = wall_clock_ns
        self._monotonic_clock = monotonic_clock
        self._lock = checked_async_lock("reality_metrology.state")
        self._run_lock = checked_async_lock("reality_metrology.acquisition")
        self._persist_lock = checked_async_lock("reality_metrology.persist")
        self._certificates: dict[str, CalibrationCertificate] = {}
        self._receipts: list[dict[str, Any]] = []
        self._failures: list[dict[str, Any]] = []
        self._mode = AcquisitionMode.LIVE
        self._mode_generation = 0
        self._active_run: dict[str, Any] | None = None
        self._recovered_interrupted = 0
        self._closing = False
        self._stop = asyncio.Event()
        self._inflight_refresh: asyncio.Task[dict[str, ChannelReading]] | None = None
        self._load_state()

    def _load_state(self) -> None:
        if not os.path.lexists(self._state_path):
            return
        if self._state_path.is_symlink() or not self._state_path.is_file():
            raise MetrologyError("metrology state path must be a regular file")
        envelope = read_json_envelope(self._state_path)
        if envelope.get("schema_name") != _STATE_SCHEMA:
            raise MetrologyError("metrology state schema differs")
        if int(envelope.get("schema_version", 0)) != _STATE_VERSION:
            raise MetrologyError("metrology state version differs")
        payload = dict(envelope.get("payload") or {})
        recorded = str(payload.pop("state_sha256", ""))
        if recorded != str(sha256_hex(canonical_json(payload))):
            raise MetrologyError("metrology state integrity check failed")
        certificates = tuple(
            CalibrationCertificate.from_dict(item)
            for item in payload.get("certificates") or ()
        )
        if len(certificates) > _MAX_CERTIFICATES:
            raise MetrologyError("metrology calibration registry exceeds its bound")
        if len({item.channel_id for item in certificates}) != len(certificates):
            raise MetrologyError("metrology state contains duplicate channel calibrations")
        receipts = list(payload.get("receipts") or ())
        if len(receipts) > _MAX_RECEIPTS:
            raise MetrologyError("metrology receipt journal exceeds its bound")
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                raise MetrologyError("metrology receipt journal contains a non-record")
            measurements = receipt.get("measurements")
            if (
                not isinstance(measurements, list)
                or len(measurements) > _MAX_MEASUREMENTS_PER_RECEIPT
            ):
                raise MetrologyError("metrology receipt evidence is absent or unbounded")
            evidence = {
                key: receipt.get(key)
                for key in (
                    "run_id",
                    "task_sha256",
                    "mode",
                    "mode_generation",
                    "started_at_ns",
                    "completed_at_ns",
                    "sample_sets",
                    "maximum_observed_skew_ns",
                    "scenario_id",
                    "measurements",
                    "summaries",
                )
            }
            if receipt.get("evidence_sha256") != str(
                sha256_hex(canonical_json(evidence))
            ):
                raise MetrologyError("metrology receipt evidence digest differs")
        failures = list(payload.get("failures") or ())
        if len(failures) > _MAX_RECEIPTS:
            raise MetrologyError("metrology failure journal exceeds its bound")
        self._certificates = {item.channel_id: item for item in certificates}
        self._receipts = [dict(item) for item in receipts]
        self._failures = [dict(item) for item in failures]
        self._mode_generation = max(0, int(payload.get("mode_generation", 0)))
        if payload.get("active_run"):
            self._recovered_interrupted = max(
                1,
                int(payload.get("recovered_interrupted", 0)) + 1,
            )
            self._mode_generation += 1
        else:
            self._recovered_interrupted = max(
                0,
                int(payload.get("recovered_interrupted", 0)),
            )
        self._mode = AcquisitionMode.LIVE
        self._active_run = None

    async def _persist(self) -> None:
        async with self._persist_lock:
            async with self._lock:
                payload: dict[str, Any] = {
                    "saved_at_ns": int(self._wall_clock_ns()),
                    "mode": self._mode.value,
                    "mode_generation": self._mode_generation,
                    "active_run": dict(self._active_run) if self._active_run else None,
                    "recovered_interrupted": self._recovered_interrupted,
                    "certificates": [
                        self._certificates[key].to_dict()
                        for key in sorted(self._certificates)
                    ],
                    "receipts": list(self._receipts[-_MAX_RECEIPTS:]),
                    "failures": list(self._failures[-_MAX_RECEIPTS:]),
                }
                payload["state_sha256"] = str(sha256_hex(canonical_json(payload)))
            with local_internal_governed_scope(
                "reality_reach.metrology.persist",
                domain="state_mutation",
            ):
                gateway = get_file_write_gateway()
                await gateway.ensure_directory_async(
                    self._state_path.parent,
                    source="reality_reach.metrology.persist",
                )
                await gateway.write_json_async(
                    self._state_path,
                    payload,
                    schema_version=_STATE_VERSION,
                    schema_name=_STATE_SCHEMA,
                    source="reality_reach.metrology.persist",
                )

    async def _persist_shielded(self) -> None:
        persistence = get_task_tracker().create_task(
            self._persist(),
            name="RealityMetrologyPersist",
        )
        try:
            await asyncio.shield(persistence)
        except asyncio.CancelledError:
            await persistence
            raise

    async def start(self) -> None:
        self._closing = False
        self._stop.clear()
        await self._persist()

    async def shutdown(self) -> None:
        self._closing = True
        self._stop.set()
        async with self._run_lock:
            async with self._lock:
                self._mode = AcquisitionMode.LIVE
                self._active_run = None
                self._mode_generation += 1
            await self._persist()

    async def register_calibration(self, certificate: CalibrationCertificate) -> str:
        if not isinstance(certificate, CalibrationCertificate):
            raise TypeError("certificate must be a CalibrationCertificate")
        declarations = {item.channel_id: item for item in self._reality.declarations()}
        declaration = declarations.get(certificate.channel_id)
        if declaration is None or declaration.kind is not ChannelKind.SENSOR:
            raise MetrologyError("calibration channel is not a declared sensor")
        if declaration.calibration_id != certificate.calibration_id:
            raise MetrologyError("calibration identity differs from channel declaration")
        if (
            declaration.calibration_valid_until_ns is not None
            and declaration.calibration_valid_until_ns != certificate.valid_until_ns
        ):
            raise MetrologyError("calibration expiry differs from channel declaration")
        async with self._run_lock:
            async with self._lock:
                current = self._certificates.get(certificate.channel_id)
                if current is not None and current.sha256 == certificate.sha256:
                    return certificate.sha256
                if current is not None and certificate.issued_at_ns <= current.issued_at_ns:
                    raise MetrologyError("calibration replacement must advance issuance time")
                if len(self._certificates) >= _MAX_CERTIFICATES and current is None:
                    raise MetrologyError("calibration registry capacity exhausted")
                self._certificates[certificate.channel_id] = certificate
            try:
                await self._persist()
            except BaseException:
                async with self._lock:
                    if current is None:
                        self._certificates.pop(certificate.channel_id, None)
                    else:
                        self._certificates[certificate.channel_id] = current
                raise
        return certificate.sha256

    def calibrations(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                **item.to_dict(),
                "certificate_sha256": item.sha256,
                "valid": item.issued_at_ns <= self._wall_clock_ns() <= item.valid_until_ns,
            }
            for item in sorted(self._certificates.values(), key=lambda row: row.channel_id)
        )

    async def restore_live(self, *, reason: str = "operator_or_aura_request") -> dict[str, Any]:
        reason_text = str(reason or "").strip()[:256]
        async with self._run_lock:
            async with self._lock:
                prior = self._mode
                self._mode = AcquisitionMode.LIVE
                self._active_run = None
                self._mode_generation += 1
                receipt = {
                    "prior_mode": prior.value,
                    "restored_mode": self._mode.value,
                    "mode_generation": self._mode_generation,
                    "reason": reason_text,
                    "restored_at_ns": int(self._wall_clock_ns()),
                }
            await self._persist()
            return receipt

    async def acquire(self, task: AcquisitionTask) -> AcquisitionReceipt:
        if not isinstance(task, AcquisitionTask):
            raise TypeError("task must be an AcquisitionTask")
        if self._closing:
            raise MetrologyError("metrology service is shutting down")
        async with self._run_lock:
            if self._inflight_refresh is not None:
                if not self._inflight_refresh.done():
                    raise MetrologyError(
                        "a timed-out adapter refresh is still reconciling; new acquisition refused"
                    )
                try:
                    self._inflight_refresh.result()
                # not a failure: the wait ran out or the work was cancelled, which are the two
                # ways this bounded join ends without a result.
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # noqa: BLE001 - logged at warning before acquisition continues
                    logger.warning("Prior Reality Reach refresh failed before acquisition: %s", exc)
                self._inflight_refresh = None
            started_ns = int(self._wall_clock_ns())
            run_id = str(uuid.uuid4())
            async with self._lock:
                if self._active_run is not None:
                    raise MetrologyError("another acquisition is already active")
                self._mode = task.mode
                self._mode_generation += 1
                generation = self._mode_generation
                self._active_run = {
                    "run_id": run_id,
                    "task_sha256": task.sha256,
                    "mode": task.mode.value,
                    "mode_generation": generation,
                    "started_at_ns": started_ns,
                }
            await self._persist()
            measurements: list[Measurement] = []
            max_skew = 0
            identity_fence: dict[str, tuple[str, int]] = {}
            try:
                deadline = self._monotonic_clock() + task.timeout_s
                for sample_index in range(task.sample_count):
                    remaining = deadline - self._monotonic_clock()
                    if remaining <= 0.0:
                        raise MetrologyError("acquisition deadline expired")
                    refresh_task = get_task_tracker().create_task(
                        asyncio.to_thread(self._reality.refresh),
                        name=f"RealityMetrologyRefresh:{run_id}:{sample_index}",
                    )
                    try:
                        readings = await asyncio.wait_for(
                            asyncio.shield(refresh_task),
                            timeout=remaining,
                        )
                    except TimeoutError as exc:
                        self._inflight_refresh = refresh_task
                        raise MetrologyError(
                            "adapter refresh exceeded the acquisition deadline and is reconciling"
                        ) from exc
                    sample = self._validate_sample(task, readings, identity_fence)
                    captures = [item.captured_at_ns for item in sample]
                    skew = max(captures) - min(captures)
                    if skew > task.max_capture_skew_ns:
                        raise MetrologyError(
                            f"capture skew {skew}ns exceeds {task.max_capture_skew_ns}ns"
                        )
                    max_skew = max(max_skew, skew)
                    measurements.extend(sample)
                    if sample_index + 1 < task.sample_count and task.sample_interval_s > 0.0:
                        remaining = deadline - self._monotonic_clock()
                        if remaining <= task.sample_interval_s:
                            raise MetrologyError("acquisition deadline cannot fit next interval")
                        try:
                            await asyncio.wait_for(
                                self._stop.wait(),
                                timeout=task.sample_interval_s,
                            )
                        # not a failure: the wait ran out, which is what the timeout was set to decide.
                        except TimeoutError:
                            pass
                        if self._stop.is_set():
                            raise MetrologyError("acquisition cancelled by shutdown")
                summaries = self._summarize(task, measurements)
                completed_ns = int(self._wall_clock_ns())
                evidence = {
                    "run_id": run_id,
                    "task_sha256": task.sha256,
                    "mode": task.mode.value,
                    "mode_generation": generation,
                    "started_at_ns": started_ns,
                    "completed_at_ns": completed_ns,
                    "sample_sets": task.sample_count,
                    "maximum_observed_skew_ns": max_skew,
                    "scenario_id": task.scenario_id,
                    "measurements": [item.to_dict() for item in measurements],
                    "summaries": [item.to_dict() for item in summaries],
                }
                receipt = AcquisitionReceipt(
                    run_id=run_id,
                    task_sha256=task.sha256,
                    mode=task.mode,
                    mode_generation=generation,
                    started_at_ns=started_ns,
                    completed_at_ns=completed_ns,
                    sample_sets=task.sample_count,
                    maximum_observed_skew_ns=max_skew,
                    scenario_id=task.scenario_id,
                    measurements=tuple(measurements),
                    summaries=summaries,
                    evidence_sha256=str(sha256_hex(canonical_json(evidence))),
                )
                if not receipt.verify_evidence():
                    raise MetrologyError("acquisition evidence failed its self-check")
                async with self._lock:
                    self._receipts.append(receipt.to_dict())
                    self._receipts = self._receipts[-_MAX_RECEIPTS:]
                return receipt
            except BaseException as exc:
                async with self._lock:
                    failure = {
                        "run_id": run_id,
                        "task_sha256": task.sha256,
                        "mode": task.mode.value,
                        "mode_generation": generation,
                        "failed_at_ns": int(self._wall_clock_ns()),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1024],
                    }
                    failure["failure_sha256"] = str(
                        sha256_hex(canonical_json(failure))
                    )
                    self._failures.append(failure)
                    self._failures = self._failures[-_MAX_RECEIPTS:]
                raise
            finally:
                async with self._lock:
                    self._mode = AcquisitionMode.LIVE
                    self._mode_generation += 1
                    self._active_run = None
                await self._persist_shielded()

    async def acquire_around(
        self,
        task: AcquisitionTask,
        operation: Callable[[], Awaitable[_T]],
    ) -> tuple[_T, AcquisitionReceipt]:
        """Measure before and after one operation under the same acquisition."""

        if not isinstance(task, AcquisitionTask):
            raise TypeError("task must be an AcquisitionTask")
        if not callable(operation):
            raise TypeError("operation must be callable")
        if task.sample_count < 2 or task.sample_interval_s <= 0.0:
            raise MetrologyError(
                "enclosing acquisition requires at least two temporally separated samples"
            )
        acquisition = get_task_tracker().create_task(
            self.acquire(task),
            name=f"RealityMetrologyEnclosure:{task.task_id}",
        )
        startup_deadline = self._monotonic_clock() + min(5.0, task.timeout_s)
        try:
            while True:
                if acquisition.done():
                    await acquisition
                    raise MetrologyError("enclosing acquisition ended before operation")
                active = self.status().get("active_run")
                if (
                    isinstance(active, Mapping)
                    and active.get("task_sha256") == task.sha256
                ):
                    break
                if self._monotonic_clock() >= startup_deadline:
                    raise MetrologyError("enclosing acquisition did not become active")
                await asyncio.sleep(0.01)
            operation_started_ns = int(self._wall_clock_ns())
            result = await operation()
            operation_completed_ns = int(self._wall_clock_ns())
            receipt = await acquisition
        except BaseException:
            if not acquisition.done():
                acquisition.cancel()
                with suppress(asyncio.CancelledError):
                    await acquisition
            raise
        if (
            receipt.started_at_ns > operation_started_ns
            or receipt.completed_at_ns < operation_completed_ns
        ):
            raise MetrologyError("acquisition did not enclose the complete operation")
        return result, receipt

    def _validate_sample(
        self,
        task: AcquisitionTask,
        readings: Mapping[str, ChannelReading],
        identity_fence: dict[str, tuple[str, int]],
    ) -> tuple[Measurement, ...]:
        declarations = {item.channel_id: item for item in self._reality.declarations()}
        sample: list[Measurement] = []
        sessions: set[str] = set()
        clock_sources: set[str] = set()
        for requested in task.channels:
            declaration = declarations.get(requested.channel_id)
            reading = readings.get(requested.channel_id)
            if declaration is None or declaration.kind is not ChannelKind.SENSOR:
                raise MetrologyError(f"undeclared sensor channel: {requested.channel_id}")
            if reading is None:
                raise MetrologyError(f"missing reading: {requested.channel_id}")
            expected_status = (
                ReadingStatus.AVAILABLE
                if requested.expected_source is EvidenceSource.LIVE
                else ReadingStatus.SIMULATED
            )
            if reading.status is not expected_status:
                raise MetrologyError(
                    f"{requested.channel_id} source status {reading.status.value} "
                    f"does not satisfy {requested.expected_source.value}"
                )
            if reading.value is None:
                raise MetrologyError(f"{requested.channel_id} has no measurable value")
            if requested.expected_source is EvidenceSource.SIMULATED:
                if reading.scenario_id != task.scenario_id:
                    raise MetrologyError("simulated reading scenario differs from acquisition")
            elif reading.scenario_id:
                raise MetrologyError("live reading carries a simulation scenario")
            if not reading.session_id or not reading.adapter_identity_sha256:
                raise MetrologyError("measurement lacks runtime and adapter identity evidence")
            if task.require_calibration and not reading.adapter_identity_stable:
                raise MetrologyError(
                    "calibrated evidence requires a stable adapter identity"
                )
            sessions.add(reading.session_id)
            clock_sources.add(reading.wall_clock_source)
            identity = (
                reading.adapter_identity_sha256,
                reading.adapter_registration_generation,
            )
            prior_identity = identity_fence.setdefault(requested.channel_id, identity)
            if prior_identity != identity:
                raise MetrologyError("adapter identity changed during acquisition")
            sample.append(self._calibrate(declaration, reading, requested.expected_source, task))
        if len(sessions) != 1 or next(iter(sessions)) != self._reality.session_id:
            raise MetrologyError("acquisition readings do not share the current runtime session")
        if len(clock_sources) != 1:
            raise MetrologyError(
                "capture skew is not comparable across different wall-clock sources"
            )
        return tuple(sample)

    def _calibrate(
        self,
        declaration: ChannelDeclaration,
        reading: ChannelReading,
        source: EvidenceSource,
        task: AcquisitionTask,
    ) -> Measurement:
        certificate = self._certificates.get(declaration.channel_id)
        now_ns = int(self._wall_clock_ns())
        if declaration.calibration_id:
            if certificate is None:
                raise MetrologyError("declared calibration has no registered certificate")
            if certificate.calibration_id != declaration.calibration_id:
                raise MetrologyError("registered calibration identity differs from declaration")
        if task.require_calibration and certificate is None:
            raise MetrologyError("acquisition requires a calibration certificate")
        if certificate is not None and not (
            certificate.issued_at_ns <= now_ns <= certificate.valid_until_ns
        ):
            raise MetrologyError("calibration certificate is not currently valid")
        scale = certificate.scale if certificate else 1.0
        offset = certificate.offset if certificate else 0.0
        systematic = certificate.standard_uncertainty if certificate else 0.0
        value = float(reading.value) * scale + offset
        random = abs(scale) * float(reading.uncertainty or 0.0)
        resolution = abs(scale) * declaration.resolution / math.sqrt(12.0)
        return Measurement(
            channel_id=declaration.channel_id,
            value=value,
            unit=declaration.unit,
            captured_at_ns=reading.captured_at_ns,
            source=source,
            scenario_id=reading.scenario_id,
            wall_clock_source=reading.wall_clock_source,
            random_uncertainty=random,
            resolution_uncertainty=resolution,
            systematic_uncertainty=systematic,
            calibration_sha256=certificate.sha256 if certificate else "",
            reading_sha256=reading.sha256,
        )

    @staticmethod
    def _summarize(
        task: AcquisitionTask,
        measurements: Sequence[Measurement],
    ) -> tuple[MeasurementSummary, ...]:
        summaries: list[MeasurementSummary] = []
        for channel in task.channels:
            values = [item for item in measurements if item.channel_id == channel.channel_id]
            raw = [item.value for item in values]
            n = len(values)
            random_mean = math.sqrt(
                sum(
                    item.random_uncertainty**2 + item.resolution_uncertainty**2
                    for item in values
                )
            ) / n
            repeatability = statistics.stdev(raw) / math.sqrt(n) if n > 1 else 0.0
            systematic = max(item.systematic_uncertainty for item in values)
            combined = math.sqrt(random_mean**2 + repeatability**2 + systematic**2)
            clock_sources = {item.wall_clock_source for item in values}
            if len(clock_sources) != 1:
                raise MetrologyError("summary measurements do not share one clock source")
            summaries.append(
                MeasurementSummary(
                    channel_id=channel.channel_id,
                    unit=values[0].unit,
                    sample_count=n,
                    mean=statistics.fmean(raw),
                    minimum=min(raw),
                    maximum=max(raw),
                    standard_uncertainty=combined,
                    coverage_factor=2.0,
                    expanded_uncertainty_k2=2.0 * combined,
                    source=channel.expected_source,
                    wall_clock_source=next(iter(clock_sources)),
                    calibration_sha256=values[0].calibration_sha256,
                )
            )
        return tuple(summaries)

    def status(self) -> dict[str, Any]:
        refresh_recovery_required = bool(
            self._inflight_refresh is not None and not self._inflight_refresh.done()
        )
        return {
            "alive": not self._closing,
            "ready": self.is_ready(),
            "mode": self._mode.value,
            "mode_generation": self._mode_generation,
            "active_run": dict(self._active_run) if self._active_run else None,
            "calibration_count": len(self._certificates),
            "valid_calibration_count": sum(
                item.issued_at_ns <= self._wall_clock_ns() <= item.valid_until_ns
                for item in self._certificates.values()
            ),
            "receipt_count": len(self._receipts),
            "failure_count": len(self._failures),
            "last_failure": dict(self._failures[-1]) if self._failures else None,
            "recovered_interrupted_count": self._recovered_interrupted,
            "refresh_reconciliation_required": refresh_recovery_required,
            "live_restoration_required": self._mode is not AcquisitionMode.LIVE
            and self._active_run is None,
        }

    def is_alive(self) -> bool:
        return not self._closing

    def is_ready(self) -> bool:
        return bool(
            not self._closing
            and not (
                self._inflight_refresh is not None
                and not self._inflight_refresh.done()
            )
            and not (self._mode is not AcquisitionMode.LIVE and self._active_run is None)
        )


__all__ = [
    "AcquisitionChannel",
    "AcquisitionMode",
    "AcquisitionReceipt",
    "AcquisitionTask",
    "CalibrationCertificate",
    "EvidenceSource",
    "Measurement",
    "MeasurementSummary",
    "MetrologyError",
    "RealityMetrologyService",
]
