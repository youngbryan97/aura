"""Deterministic reachability analysis over declared physical channels."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable

from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    FailureCode,
    ObjectiveKind,
    ReachabilityCertificate,
    ReachabilityFailure,
    ReachabilityStatus,
    RealityIR,
    RealityLayer,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lockdep import checked_lock


class ChannelRegistry:
    """Thread-safe inventory of immutable, explicitly declared channels."""

    def __init__(self, channels: Iterable[ChannelDeclaration] = ()) -> None:
        self._lock = checked_lock("reachability", reentrant=True)
        self._channels: dict[str, ChannelDeclaration] = {}
        for channel in channels:
            self.register(channel)

    def register(self, channel: ChannelDeclaration, *, replace: bool = False) -> None:
        if not isinstance(channel, ChannelDeclaration):
            raise TypeError("channel must be a ChannelDeclaration")
        with self._lock:
            if channel.channel_id in self._channels and not replace:
                raise ValueError(f"channel already registered: {channel.channel_id}")
            self._channels[channel.channel_id] = channel

    def unregister(self, channel_id: str) -> None:
        with self._lock:
            self._channels.pop(channel_id, None)

    def get(self, channel_id: str) -> ChannelDeclaration | None:
        with self._lock:
            return self._channels.get(channel_id)

    def snapshot(self) -> tuple[ChannelDeclaration, ...]:
        with self._lock:
            return tuple(self._channels[key] for key in sorted(self._channels))

    @property
    def sha256(self) -> str:
        payload = [channel.to_dict() for channel in self.snapshot()]
        return sha256_hex(canonical_json(payload))


class ReachabilityEngine:
    """Produces a bounded realization or a tamper-evident no-go certificate."""

    _HARD_FAILURES = {
        FailureCode.NO_CHANNEL,
        FailureCode.TARGET_OUT_OF_RANGE,
        FailureCode.CONSTRAINT_UNSATISFIED,
        FailureCode.AMBIENT_IDENTITY_UNRESOLVED,
        FailureCode.NOT_CONTROLLABLE,
    }

    def analyze(
        self,
        contract: RealityIR,
        registry: ChannelRegistry,
    ) -> ReachabilityCertificate:
        if not isinstance(contract, RealityIR):
            raise TypeError("contract must be a RealityIR")
        if not isinstance(registry, ChannelRegistry):
            raise TypeError("registry must be a ChannelRegistry")

        failures: list[ReachabilityFailure] = []
        actuators = self._resolve(
            contract.allowed_actuators,
            expected_kind=ChannelKind.ACTUATOR,
            contract=contract,
            registry=registry,
            failures=failures,
        )
        sensors = self._resolve(
            contract.allowed_sensors,
            expected_kind=ChannelKind.SENSOR,
            contract=contract,
            registry=registry,
            failures=failures,
        )

        if contract.objective_kind != ObjectiveKind.OBSERVE and not actuators:
            self._append_once(
                failures,
                ReachabilityFailure.from_mapping(
                    FailureCode.NO_CHANNEL,
                    "no enabled actuator can realize the requested observable and layer",
                    channels=contract.allowed_actuators,
                ),
            )
        if not sensors:
            self._append_once(
                failures,
                ReachabilityFailure.from_mapping(
                    FailureCode.NO_CHANNEL,
                    "no enabled sensor can observe the requested observable and layer",
                    channels=contract.allowed_sensors,
                ),
            )

        self._check_constraints(contract, actuators, sensors, failures)
        self._check_metrology(contract, actuators, sensors, failures)
        evidence_ceiling = self._evidence_ceiling(actuators, sensors)
        if evidence_ceiling.rank < contract.required_proof.minimum_evidence.rank:
            self._append_once(
                failures,
                ReachabilityFailure.from_mapping(
                    FailureCode.INSUFFICIENT_EVIDENCE,
                    "declared channels cannot attain the required evidence level",
                    channels=tuple(channel.channel_id for channel in sensors),
                    details={
                        "evidence_ceiling": evidence_ceiling.value,
                        "required": contract.required_proof.minimum_evidence.value,
                    },
                ),
            )

        hard_failure = any(failure.code in self._HARD_FAILURES for failure in failures)
        if hard_failure:
            status = ReachabilityStatus.UNREACHABLE
        elif failures:
            status = ReachabilityStatus.PARTIAL
        else:
            status = ReachabilityStatus.REACHABLE
        return ReachabilityCertificate(
            contract_sha256=contract.sha256,
            registry_sha256=registry.sha256,
            status=status,
            selected_actuators=tuple(channel.channel_id for channel in actuators),
            selected_sensors=tuple(channel.channel_id for channel in sensors),
            failures=tuple(failures),
            evidence_ceiling=evidence_ceiling,
            claim_boundary=contract.reality_layer,
            issued_at_ns=time.time_ns(),
        )

    def _resolve(
        self,
        channel_ids: tuple[str, ...],
        *,
        expected_kind: ChannelKind,
        contract: RealityIR,
        registry: ChannelRegistry,
        failures: list[ReachabilityFailure],
    ) -> list[ChannelDeclaration]:
        resolved: list[ChannelDeclaration] = []
        for channel_id in channel_ids:
            channel = registry.get(channel_id)
            if channel is None or channel.kind != expected_kind or not channel.supports(contract):
                self._append_once(
                    failures,
                    ReachabilityFailure.from_mapping(
                        FailureCode.NO_CHANNEL,
                        "allowed channel is absent, disabled, type-mismatched, or incompatible",
                        channels=(channel_id,),
                        details={"expected_kind": expected_kind.value},
                    ),
                )
                continue
            if not channel.domain.contains(contract.target, tolerance=contract.tolerance):
                self._append_once(
                    failures,
                    ReachabilityFailure.from_mapping(
                        FailureCode.TARGET_OUT_OF_RANGE,
                        "requested target and tolerance exceed the declared channel domain",
                        channels=(channel_id,),
                        details={
                            "minimum": channel.domain.minimum,
                            "maximum": channel.domain.maximum,
                            "target": contract.target,
                            "tolerance": contract.tolerance,
                        },
                    ),
                )
                continue
            resolved.append(channel)
        return resolved

    def _check_constraints(
        self,
        contract: RealityIR,
        actuators: list[ChannelDeclaration],
        sensors: list[ChannelDeclaration],
        failures: list[ReachabilityFailure],
    ) -> None:
        selected = {channel.channel_id: channel for channel in (*actuators, *sensors)}
        for constraint in contract.constraints:
            if not constraint.required:
                continue
            candidates = (
                tuple(selected.get(channel_id) for channel_id in constraint.applies_to_channels)
                if constraint.applies_to_channels
                else tuple(actuators or sensors)
            )
            missing = tuple(
                channel.channel_id
                for channel in candidates
                if channel is not None
                and constraint.constraint_id not in channel.compliance_tags
            )
            absent = tuple(
                channel_id
                for channel_id in constraint.applies_to_channels
                if channel_id not in selected
            )
            affected = tuple(sorted((*missing, *absent)))
            if affected:
                self._append_once(
                    failures,
                    ReachabilityFailure.from_mapping(
                        FailureCode.CONSTRAINT_UNSATISFIED,
                        "a required channel constraint has no declared compliance evidence",
                        channels=affected,
                        details={"constraint_id": constraint.constraint_id},
                    ),
                )

    def _check_metrology(
        self,
        contract: RealityIR,
        actuators: list[ChannelDeclaration],
        sensors: list[ChannelDeclaration],
        failures: list[ReachabilityFailure],
    ) -> None:
        inadequate = tuple(
            channel.channel_id
            for channel in sensors
            if channel.resolution > contract.tolerance
        )
        if inadequate:
            self._append_once(
                failures,
                ReachabilityFailure.from_mapping(
                    FailureCode.BELOW_SENSOR_FLOOR,
                    "requested tolerance is below one or more sensor resolution floors",
                    channels=inadequate,
                    details={"requested_tolerance": contract.tolerance},
                ),
            )

        independent_references = {channel.reference_id for channel in sensors}
        required_references = contract.required_proof.minimum_independent_sensors
        if len(independent_references) < required_references:
            self._append_once(
                failures,
                ReachabilityFailure.from_mapping(
                    FailureCode.SHARED_REFERENCE,
                    "independent sensor references are insufficient for the requested proof",
                    channels=tuple(channel.channel_id for channel in sensors),
                    details={
                        "independent_references": len(independent_references),
                        "required": required_references,
                    },
                ),
            )

        if contract.reality_layer in {RealityLayer.DIRECT, RealityLayer.AMBIENT}:
            actuator_references = {
                channel.reference_id for channel in actuators if channel.reference_id
            }
            if sensors and all(
                channel.reference_id in actuator_references for channel in sensors
            ):
                self._append_once(
                    failures,
                    ReachabilityFailure.from_mapping(
                        FailureCode.SHARED_REFERENCE,
                        "all observations share the actuator reference and cannot prove an external effect",
                        channels=tuple(channel.channel_id for channel in sensors),
                    ),
                )

        needs_external = (
            contract.required_proof.external_metrology
            or contract.reality_layer == RealityLayer.AMBIENT
        )
        if needs_external and not any(channel.external_metrology for channel in sensors):
            self._append_once(
                failures,
                ReachabilityFailure.from_mapping(
                    FailureCode.AMBIENT_IDENTITY_UNRESOLVED,
                    "the requested claim requires an independently calibrated external instrument",
                    channels=tuple(channel.channel_id for channel in sensors),
                ),
            )

        if contract.reality_layer == RealityLayer.AMBIENT and actuators:
            if not any(
                channel.coupling != CouplingClass.UNKNOWN and channel.coupling_validated
                for channel in actuators
            ):
                self._append_once(
                    failures,
                    ReachabilityFailure.from_mapping(
                        FailureCode.AMBIENT_IDENTITY_UNRESOLVED,
                        "no independently validated physical coupling supports the ambient claim",
                        channels=tuple(channel.channel_id for channel in actuators),
                    ),
                )

    @staticmethod
    def _evidence_ceiling(
        actuators: list[ChannelDeclaration],
        sensors: list[ChannelDeclaration],
    ) -> EvidenceLevel:
        channels = [*actuators, *sensors]
        if not channels:
            return EvidenceLevel.P0
        return min(channels, key=lambda channel: channel.evidence_level.rank).evidence_level

    @staticmethod
    def _append_once(
        failures: list[ReachabilityFailure],
        failure: ReachabilityFailure,
    ) -> None:
        identity = (failure.code, failure.channels, failure.details)
        if any((item.code, item.channels, item.details) == identity for item in failures):
            return
        failures.append(failure)


__all__ = ["ChannelRegistry", "ReachabilityEngine"]
