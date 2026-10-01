"""Cross-protocol Reality Reach acceptance runner and fault-injection facade.

The fault transport is explicit test/HIL infrastructure. It never upgrades
simulation to live evidence, and a post-dispatch fault remains indeterminate
because the wrapped transport may already have changed external state.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

from core.reality_reach.acceptance_contracts import (
    _IDENTIFIER,
    _MAX_FAULT_RECEIPTS,
    ACCEPTANCE_GOVERNANCE_SCHEMA,
    REQUIRED_SCALAR_ACCEPTANCE_CASES,
    AcceptanceCaseResult,
    AcceptanceError,
    AcceptanceEvidenceClass,
    AcceptanceExecutor,
    AcceptanceMetrologyAcquirer,
    AcceptanceVerdict,
    ConnectorAcceptanceCertificate,
    ScalarAcceptancePlan,
    ScalarFault,
    _canonical_json_bytes,
    _digest,
    _identifier,
    _sha256,
    _strict_json_loads,
    acceptance_governance_accepted,
    acceptance_governance_document,
)
from core.reality_reach.acceptance_store import AcceptanceCertificateStore
from core.reality_reach.actuation import ActuationLease, ActuationState
from core.reality_reach.live import ReadingStatus, RealityReachService
from core.reality_reach.metrology import AcquisitionMode, AcquisitionReceipt, EvidenceSource
from core.reality_reach.scalar_adapter import (
    ScalarProtocolTransport,
    ScalarRealityAdapter,
    ScalarSample,
    ScalarTransportClass,
    ScalarWriteResult,
)
from core.runtime.lockdep import checked_async_lock


class ScalarAcceptanceRunner:
    """Exercise one scalar adapter under external governance for physical runs."""

    _CASE_IDS = REQUIRED_SCALAR_ACCEPTANCE_CASES

    def __init__(
        self,
        adapter: ScalarRealityAdapter,
        service: RealityReachService,
        plan: ScalarAcceptancePlan,
        *,
        governed_executor: AcceptanceExecutor | None = None,
        metrology_acquirer: AcceptanceMetrologyAcquirer | None = None,
    ) -> None:
        if not isinstance(adapter, ScalarRealityAdapter):
            raise TypeError("adapter must be a ScalarRealityAdapter")
        if not isinstance(service, RealityReachService):
            raise TypeError("service must be a RealityReachService")
        if not isinstance(plan, ScalarAcceptancePlan):
            raise TypeError("plan must be a ScalarAcceptancePlan")
        capabilities = adapter.actuator_capabilities()
        if not any(
            service.adapter_id_for_channel(item.channel_id) == adapter.adapter_id
            for item in capabilities
        ):
            raise AcceptanceError("acceptance_adapter_not_registered")
        if not capabilities:
            raise AcceptanceError("acceptance_adapter_is_read_only")
        self._adapter = adapter
        self._service = service
        self._plan = plan
        self._governed_executor = governed_executor
        self._metrology_acquirer = metrology_acquirer
        self._metrology_receipt = plan.metrology_receipt
        self._observation_channels = tuple(
            dict.fromkeys(
                channel_id
                for capability in capabilities
                for channel_id in capability.observation_channels
            )
        )
        self._case_evidence: dict[str, Any] = {}
        self._governance_evidence: dict[str, Any] = {}

    @property
    def case_evidence(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            _strict_json_loads(_canonical_json_bytes(self._case_evidence)),
        )

    @property
    def metrology_receipt(self) -> AcquisitionReceipt | None:
        return self._metrology_receipt

    @property
    def governance_evidence(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            _strict_json_loads(_canonical_json_bytes(self._governance_evidence)),
        )

    def _lease(
        self,
        command_sha256: str,
        *,
        suffix: str,
        authority_receipt_id: str,
    ) -> ActuationLease:
        now_wall = time.time_ns()
        now_mono = time.monotonic_ns()
        duration_ns = int(self._plan.deadline_s * 1_000_000_000)
        return ActuationLease(
            lease_id=f"lease.{self._plan.campaign_id}.{suffix}"[:128],
            command_sha256=command_sha256,
            adapter_id=self._adapter.adapter_id,
            session_id=self._service.session_id,
            authority_receipt_id=authority_receipt_id,
            issued_at_ns=now_wall,
            expires_at_ns=now_wall + duration_ns,
            issued_monotonic_ns=now_mono,
            expires_monotonic_ns=now_mono + duration_ns,
        )

    def _result(
        self,
        case_id: str,
        verdict: AcceptanceVerdict,
        *,
        started_ns: int,
        evidence: Any,
        detail: str = "",
    ) -> AcceptanceCaseResult:
        canonical_evidence = _strict_json_loads(_canonical_json_bytes(evidence))
        self._case_evidence[case_id] = canonical_evidence
        return AcceptanceCaseResult(
            case_id=case_id,
            verdict=verdict,
            evidence_class=self._plan.evidence_class,
            required=True,
            evidence_sha256=_digest(canonical_evidence),
            duration_ms=max(0.0, (time.monotonic_ns() - started_ns) / 1_000_000),
            detail=detail,
        )

    def _failure(
        self,
        case_id: str,
        *,
        started_ns: int,
        error: BaseException,
    ) -> AcceptanceCaseResult:
        error_type = type(error).__name__
        return self._result(
            case_id,
            AcceptanceVerdict.FAIL,
            started_ns=started_ns,
            evidence={
                "error_type": error_type,
                "error_sha256": _digest(str(error)),
            },
            detail=error_type,
        )

    async def _run_cases(self, *, authority_receipt_id: str) -> ConnectorAcceptanceCertificate:
        self._case_evidence = {}
        started_at_ns = max(1, time.time_ns())
        results: dict[str, AcceptanceCaseResult] = {}
        command = None
        actuation = None

        case_started = time.monotonic_ns()
        try:
            reading = await self._adapter.refresh_readback()
            observation_passed = bool(
                reading.status is ReadingStatus.AVAILABLE
                and reading.value is not None
                and reading.source_event_id
            )
            results["observation.fresh"] = self._result(
                "observation.fresh",
                AcceptanceVerdict.PASS if observation_passed else AcceptanceVerdict.FAIL,
                started_ns=case_started,
                evidence=reading.to_dict(),
                detail="fresh identified readback"
                if observation_passed
                else "readback unavailable",
            )
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            results["observation.fresh"] = self._failure(
                "observation.fresh",
                started_ns=case_started,
                error=exc,
            )

        if results["observation.fresh"].verdict is AcceptanceVerdict.PASS:
            case_started = time.monotonic_ns()
            try:
                cancel_command = await self._adapter.compile_target(
                    self._plan.target,
                    inventory_sha256=self._service.status()["registry_sha256"],
                    deadline_s=self._plan.deadline_s,
                    idempotency_key=f"{self._plan.campaign_id}.cancel",
                    source="reality_reach.acceptance",
                )
                cancel_lease = self._lease(
                    cancel_command.sha256,
                    suffix="cancel",
                    authority_receipt_id=authority_receipt_id,
                )
                cancel_prepared = await self._adapter.prepare(
                    cancel_command,
                    cancel_lease,
                )
                cancellation = await self._adapter.cancel(
                    cancel_command,
                    cancel_prepared,
                )
                passed = bool(
                    cancellation.state is ActuationState.CANCELLED
                    and cancellation.executed is False
                    and cancellation.transport_completed is False
                )
                results["cancellation.pre_dispatch"] = self._result(
                    "cancellation.pre_dispatch",
                    AcceptanceVerdict.PASS if passed else AcceptanceVerdict.FAIL,
                    started_ns=case_started,
                    evidence=cancellation.to_dict(),
                    detail="cancelled before transport"
                    if passed
                    else "cancellation contract failed",
                )
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                results["cancellation.pre_dispatch"] = self._failure(
                    "cancellation.pre_dispatch",
                    started_ns=case_started,
                    error=exc,
                )

        if all(
            results.get(case_id) is not None and results[case_id].verdict is AcceptanceVerdict.PASS
            for case_id in ("observation.fresh", "cancellation.pre_dispatch")
        ):
            case_started = time.monotonic_ns()
            try:
                command = await self._adapter.compile_target(
                    self._plan.target,
                    inventory_sha256=self._service.status()["registry_sha256"],
                    deadline_s=self._plan.deadline_s,
                    idempotency_key=f"{self._plan.campaign_id}.actuate",
                    source="reality_reach.acceptance",
                )
                lease = self._lease(
                    command.sha256,
                    suffix="actuate",
                    authority_receipt_id=authority_receipt_id,
                )
                prepared = await self._adapter.prepare(command, lease)
                results["actuation.prepare"] = self._result(
                    "actuation.prepare",
                    AcceptanceVerdict.PASS,
                    started_ns=case_started,
                    evidence=prepared.to_dict(),
                    detail="preconditions fenced",
                )
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                results["actuation.prepare"] = self._failure(
                    "actuation.prepare",
                    started_ns=case_started,
                    error=exc,
                )
            else:
                case_started = time.monotonic_ns()
                try:
                    actuation = await self._adapter.actuate(command, lease, prepared)
                    passed = actuation.state is ActuationState.EXECUTED
                    results["actuation.dispatch"] = self._result(
                        "actuation.dispatch",
                        AcceptanceVerdict.PASS if passed else AcceptanceVerdict.FAIL,
                        started_ns=case_started,
                        evidence=actuation.to_dict(),
                        detail="transport completed" if passed else "transport not completed",
                    )
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                    results["actuation.dispatch"] = self._failure(
                        "actuation.dispatch",
                        started_ns=case_started,
                        error=exc,
                    )

        if (
            command is not None
            and actuation is not None
            and results.get("actuation.dispatch") is not None
            and results["actuation.dispatch"].verdict is AcceptanceVerdict.PASS
        ):
            case_started = time.monotonic_ns()
            try:
                effect = await self._adapter.verify_effect(command, actuation)
                passed = bool(
                    effect.state is ActuationState.EFFECT_VERIFIED and effect.independently_observed
                )
                results["effect.independent_readback"] = self._result(
                    "effect.independent_readback",
                    AcceptanceVerdict.PASS if passed else AcceptanceVerdict.FAIL,
                    started_ns=case_started,
                    evidence=effect.to_dict(),
                    detail="independent fresh effect"
                    if passed
                    else "effect not independently verified",
                )
                if (
                    passed
                    and self._plan.evidence_class
                    is not AcceptanceEvidenceClass.SIMULATION
                ):
                    await asyncio.sleep(self._plan.metrology_effect_hold_s)
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                results["effect.independent_readback"] = self._failure(
                    "effect.independent_readback",
                    started_ns=case_started,
                    error=exc,
                )

        if command is not None:
            case_started = time.monotonic_ns()
            try:
                rollback = (
                    await self._adapter.rollback(command, actuation)
                    if actuation is not None
                    else await self._adapter.safe_state(command, None)
                )
                passed = rollback.state in {
                    ActuationState.ROLLED_BACK,
                    ActuationState.SAFE_STATE,
                }
                results["restoration.rollback"] = self._result(
                    "restoration.rollback",
                    AcceptanceVerdict.PASS if passed else AcceptanceVerdict.FAIL,
                    started_ns=case_started,
                    evidence=rollback.to_dict(),
                    detail=(
                        "initial or safe state restored"
                        if passed
                        else "restoration not independently verified"
                    ),
                )
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                results["restoration.rollback"] = self._failure(
                    "restoration.rollback",
                    started_ns=case_started,
                    error=exc,
                )

        for case_id in self._CASE_IDS:
            if case_id not in results:
                results[case_id] = self._result(
                    case_id,
                    AcceptanceVerdict.UNMEASURED,
                    started_ns=time.monotonic_ns(),
                    evidence={"blocked_by": [item.to_dict() for item in results.values()]},
                    detail="blocked by an earlier required acceptance failure",
                )
        completed_at_ns = max(started_at_ns, time.time_ns())
        return ConnectorAcceptanceCertificate(
            campaign_id=self._plan.campaign_id,
            connector_id=self._plan.connector_id,
            adapter_id=self._adapter.adapter_id,
            physical_identity_sha256=self._adapter.physical_identity_sha256,
            source_commit_sha256=self._plan.source_commit_sha256,
            target=self._plan.target,
            target_tolerance=self._adapter.effect_tolerance,
            started_at_ns=started_at_ns,
            completed_at_ns=completed_at_ns,
            cases=tuple(results[case_id] for case_id in self._CASE_IDS),
            scenario_id=self._plan.scenario_id,
        )

    def _bind_metrology(
        self,
        certificate: ConnectorAcceptanceCertificate,
        receipt: AcquisitionReceipt,
    ) -> ConnectorAcceptanceCertificate:
        if not isinstance(receipt, AcquisitionReceipt) or not receipt.verify_evidence():
            raise AcceptanceError("acceptance_metrology_evidence_invalid")
        expected_mode = (
            AcquisitionMode.HARDWARE_IN_LOOP
            if self._plan.evidence_class is AcceptanceEvidenceClass.HARDWARE_IN_LOOP
            else AcquisitionMode.LIVE
        )
        if receipt.mode is not expected_mode or receipt.restored_mode is not AcquisitionMode.LIVE:
            raise AcceptanceError("acceptance_metrology_mode_mismatch")
        if not (
            receipt.started_at_ns <= certificate.started_at_ns
            and receipt.completed_at_ns >= certificate.completed_at_ns
        ):
            raise AcceptanceError("acceptance_metrology_does_not_enclose_operation")
        sources = {item.source for item in receipt.measurements}
        expected_sources = (
            {EvidenceSource.LIVE, EvidenceSource.SIMULATED}
            if expected_mode is AcquisitionMode.HARDWARE_IN_LOOP
            else {EvidenceSource.LIVE}
        )
        if sources != expected_sources:
            raise AcceptanceError("acceptance_metrology_source_class_mismatch")
        if (
            expected_mode is AcquisitionMode.HARDWARE_IN_LOOP
            and receipt.scenario_id != self._plan.scenario_id
        ):
            raise AcceptanceError("acceptance_metrology_scenario_mismatch")
        measured_live = {
            item.channel_id
            for item in receipt.measurements
            if item.source is EvidenceSource.LIVE
        }
        if not set(self._observation_channels).issubset(measured_live):
            raise AcceptanceError("acceptance_metrology_readback_channel_missing")
        target_observed = any(
            item.source is EvidenceSource.LIVE
            and item.channel_id in self._observation_channels
            and abs(float(item.value) - certificate.target)
            <= certificate.target_tolerance
            for item in receipt.measurements
        )
        if not target_observed:
            raise AcceptanceError("acceptance_metrology_target_not_observed")
        self._metrology_receipt = receipt
        return replace(
            certificate,
            metrology_evidence_sha256=receipt.evidence_sha256,
        )

    @staticmethod
    def _governance_document(result: Mapping[str, Any]) -> dict[str, Any]:
        return acceptance_governance_document(result)

    @staticmethod
    def _governance_accepted(evidence: Mapping[str, Any]) -> bool:
        return acceptance_governance_accepted(evidence)

    async def _run_governed(self) -> ConnectorAcceptanceCertificate:
        from core.governance.action_domain import ActionDomain
        from core.runtime.action_executor import ActionExecutor
        from core.runtime.skill_contract import ActionExpectation

        executor = self._governed_executor or ActionExecutor.execute
        completed: dict[str, ConnectorAcceptanceCertificate] = {}

        async def effect_handler(context: Mapping[str, Any]) -> Mapping[str, Any]:
            authority_receipt_id = str(context.get("will_receipt_id") or "")
            if not _IDENTIFIER.fullmatch(authority_receipt_id):
                raise AcceptanceError("acceptance_governance_receipt_invalid")
            acquirer = self._metrology_acquirer
            if acquirer is None:
                raise AcceptanceError("acceptance_metrology_acquirer_missing")
            certificate, receipt = await acquirer(
                lambda: self._run_cases(authority_receipt_id=authority_receipt_id)
            )
            certificate = self._bind_metrology(certificate, receipt)
            completed["certificate"] = certificate
            dispatch = next(
                item for item in certificate.cases if item.case_id == "actuation.dispatch"
            )
            return {
                "ok": certificate.deterministic_passed,
                "transport_succeeded": dispatch.verdict is AcceptanceVerdict.PASS,
                "acceptance_certificate_sha256": certificate.sha256,
                "metrology_evidence_sha256": certificate.metrology_evidence_sha256,
            }

        async def effect_verifier(_context: Mapping[str, Any]) -> Mapping[str, Any]:
            certificate = completed.get("certificate")
            return {
                "effect_verified": bool(
                    certificate is not None and certificate.physical_evidence_passed
                ),
                "acceptance_certificate_sha256": (
                    certificate.sha256 if certificate is not None else ""
                ),
            }

        raw_result = await executor(
            domain=ActionDomain.ENVIRONMENT_ACTION,
            action_name=f"reality_reach.acceptance.{self._plan.connector_id}",
            params={
                "campaign_id": self._plan.campaign_id,
                "adapter_id": self._adapter.adapter_id,
                "physical_identity_sha256": self._adapter.physical_identity_sha256,
                "evidence_class": self._plan.evidence_class.value,
                "target": self._plan.target,
            },
            source="reality_reach.acceptance",
            rollback_target="adapter.rollback_or_safe_state",
            expectation=ActionExpectation(
                objective="exercise and restore one declared physical effect",
                required_evidence=[
                    "acceptance_certificate_sha256",
                    "metrology_evidence_sha256",
                ],
                rollback_hint="restore the pre-dispatch value or declared safe state",
                allow_partial=False,
            ),
            effect_handler=effect_handler,
            effect_verifier=effect_verifier,
            execution_timeout_s=self._plan.deadline_s,
            verification_timeout_s=self._plan.deadline_s,
            action_id=f"acceptance.{self._plan.campaign_id}"[:128],
        )
        if not isinstance(raw_result, Mapping):
            raise AcceptanceError("acceptance_governance_result_invalid")
        certificate = completed.get("certificate")
        if certificate is None:
            raise AcceptanceError("acceptance_governance_refused_before_dispatch")
        governance = self._governance_document(raw_result)
        self._governance_evidence = governance
        accepted = self._governance_accepted(governance)
        return replace(
            certificate,
            governance_evidence_sha256=_digest(governance),
            governance_accepted=accepted,
        )

    async def run(self) -> ConnectorAcceptanceCertificate:
        self._governance_evidence = {}
        self._metrology_receipt = self._plan.metrology_receipt
        if self._plan.evidence_class is AcceptanceEvidenceClass.SIMULATION:
            if self._adapter.transport_class is not ScalarTransportClass.SIMULATED:
                raise AcceptanceError(
                    "simulation_acceptance_requires_simulated_adapter"
                )
            return await self._run_cases(
                authority_receipt_id=self._plan.authority_receipt_id,
            )
        if self._adapter.transport_class is not ScalarTransportClass.PHYSICAL:
            raise AcceptanceError("physical_acceptance_requires_physical_adapter")
        if self._metrology_acquirer is None:
            raise AcceptanceError("physical_acceptance_requires_metrology_acquirer")
        return await self._run_governed()

    async def run_and_persist(
        self,
        store: AcceptanceCertificateStore | None = None,
    ) -> ConnectorAcceptanceCertificate:
        """Run once and create-once publish both verdict and replay evidence."""

        target_store = store or AcceptanceCertificateStore()
        if not isinstance(target_store, AcceptanceCertificateStore):
            raise TypeError("store must be an AcceptanceCertificateStore")
        certificate = await self.run()
        target_store.persist(certificate)
        target_store.persist_evidence(
            certificate,
            self.case_evidence,
            metrology_receipt=self.metrology_receipt,
            governance_evidence=(
                self.governance_evidence if certificate.governance_evidence_sha256 else None
            ),
        )
        return certificate


@dataclass(frozen=True, slots=True)
class FaultInjectionReceipt:
    sequence: int
    fault: ScalarFault
    operation: str
    resource_sha256: str
    injected_at_ns: int
    delegate_called: bool
    outcome_indeterminate: bool
    evidence_class: AcceptanceEvidenceClass
    scenario_id: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or self.sequence <= 0:
            raise ValueError("fault receipt sequence must be positive")
        if not isinstance(self.fault, ScalarFault):
            raise TypeError("fault must be a ScalarFault")
        object.__setattr__(self, "operation", _identifier(self.operation, name="operation"))
        object.__setattr__(
            self,
            "resource_sha256",
            _sha256(self.resource_sha256, name="resource_sha256"),
        )
        if isinstance(self.injected_at_ns, bool) or self.injected_at_ns <= 0:
            raise ValueError("injected_at_ns must be positive")
        if not isinstance(self.delegate_called, bool) or not isinstance(
            self.outcome_indeterminate,
            bool,
        ):
            raise TypeError("fault receipt booleans must be explicit")
        if self.outcome_indeterminate and not self.delegate_called:
            raise ValueError("indeterminate effect requires delegate dispatch")
        if self.evidence_class not in {
            AcceptanceEvidenceClass.SIMULATION,
            AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
        }:
            raise ValueError("fault injection evidence must be simulation or HIL")
        scenario = _identifier(self.scenario_id, name="scenario_id")
        object.__setattr__(self, "scenario_id", scenario)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "fault": self.fault.value,
            "operation": self.operation,
            "resource_sha256": self.resource_sha256,
            "injected_at_ns": self.injected_at_ns,
            "delegate_called": self.delegate_called,
            "outcome_indeterminate": self.outcome_indeterminate,
            "evidence_class": self.evidence_class.value,
            "scenario_id": self.scenario_id,
        }


class FaultInjectedReadError(ConnectionError):
    """A deterministic read partition was injected before transport."""


class FaultInjectedWriteError(ConnectionError):
    """A deterministic write partition was injected before transport."""


class FaultInjectedOutcomeUnknownError(TimeoutError):
    """The wrapped write completed but its caller lost the acknowledgement."""


class FaultInjectingScalarTransport:
    """One-shot deterministic faults around a real or simulated scalar transport."""

    def __init__(
        self,
        delegate: ScalarProtocolTransport,
        *,
        evidence_class: AcceptanceEvidenceClass,
        scenario_id: str,
        stale_age_s: float = 3600.0,
    ) -> None:
        if not isinstance(delegate, ScalarProtocolTransport):
            raise TypeError("delegate must satisfy ScalarProtocolTransport")
        if evidence_class not in {
            AcceptanceEvidenceClass.SIMULATION,
            AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
        }:
            raise ValueError("fault transport requires simulation or HIL evidence")
        self._delegate = delegate
        self._evidence_class = evidence_class
        self._scenario_id = _identifier(scenario_id, name="scenario_id")
        stale_age = float(stale_age_s)
        if not math.isfinite(stale_age) or not 1.0 <= stale_age <= 604_800.0:
            raise ValueError("stale_age_s must lie inside [1, 604800]")
        self._stale_age_ns = int(stale_age * 1_000_000_000)
        self._armed: deque[ScalarFault] = deque()
        self._samples: deque[ScalarSample] = deque(maxlen=2)
        self._receipts: deque[FaultInjectionReceipt] = deque(maxlen=_MAX_FAULT_RECEIPTS)
        self._sequence = 0
        self._lock = checked_async_lock("reality_reach.fault_injection")

    @property
    def transport_id(self) -> str:
        return f"acceptance.{self._delegate.transport_id}"

    @property
    def receipts(self) -> tuple[FaultInjectionReceipt, ...]:
        return tuple(self._receipts)

    def arm(self, *faults: ScalarFault) -> None:
        if not faults:
            raise ValueError("at least one fault is required")
        for fault in faults:
            if not isinstance(fault, ScalarFault):
                raise TypeError("faults must be ScalarFault values")
            self._armed.append(fault)

    def clear(self) -> None:
        self._armed.clear()

    def _take(self, operation: str) -> ScalarFault | None:
        if not self._armed:
            return None
        fault = self._armed[0]
        read_faults = {
            ScalarFault.READ_PARTITION,
            ScalarFault.STALE_READBACK,
            ScalarFault.DUPLICATE_READBACK,
            ScalarFault.REORDERED_READBACK,
        }
        write_faults = {
            ScalarFault.WRITE_PARTITION,
            ScalarFault.WRITE_OUTCOME_UNKNOWN,
        }
        allowed = read_faults if operation == "read" else write_faults
        if fault not in allowed:
            return None
        return self._armed.popleft()

    def _record(
        self,
        fault: ScalarFault,
        *,
        operation: str,
        resource_id: str,
        delegate_called: bool,
        outcome_indeterminate: bool = False,
    ) -> None:
        self._sequence += 1
        self._receipts.append(
            FaultInjectionReceipt(
                sequence=self._sequence,
                fault=fault,
                operation=operation,
                resource_sha256=_digest(resource_id),
                injected_at_ns=max(1, time.time_ns()),
                delegate_called=delegate_called,
                outcome_indeterminate=outcome_indeterminate,
                evidence_class=self._evidence_class,
                scenario_id=self._scenario_id,
            )
        )

    async def read_scalar(self, resource_id: str) -> ScalarSample:
        async with self._lock:
            fault = self._take("read")
            if fault is ScalarFault.READ_PARTITION:
                self._record(
                    fault,
                    operation="read",
                    resource_id=resource_id,
                    delegate_called=False,
                )
                raise FaultInjectedReadError("fault_injected_read_partition")
            if fault is ScalarFault.DUPLICATE_READBACK and self._samples:
                self._record(
                    fault,
                    operation="read",
                    resource_id=resource_id,
                    delegate_called=False,
                )
                return self._samples[-1]
            sample = await self._delegate.read_scalar(resource_id)
            if fault is ScalarFault.STALE_READBACK:
                sample = replace(
                    sample,
                    captured_at_ns=max(1, time.time_ns() - self._stale_age_ns),
                    source_event_id=_digest(
                        {
                            "fault": fault.value,
                            "source_event_id": sample.source_event_id,
                        }
                    ),
                    quality="fault_injected_stale",
                )
            elif fault is ScalarFault.REORDERED_READBACK and self._samples:
                current = sample
                sample = self._samples[0]
                self._samples.append(current)
            else:
                self._samples.append(sample)
            if fault is not None:
                self._record(
                    fault,
                    operation="read",
                    resource_id=resource_id,
                    delegate_called=True,
                )
            return sample

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult:
        async with self._lock:
            fault = self._take("write")
            if fault is ScalarFault.WRITE_PARTITION:
                self._record(
                    fault,
                    operation="write",
                    resource_id=resource_id,
                    delegate_called=False,
                )
                raise FaultInjectedWriteError("fault_injected_write_partition")
            result = await self._delegate.write_scalar(
                resource_id,
                value,
                idempotency_key=idempotency_key,
                recovery=recovery,
            )
            if fault is ScalarFault.WRITE_OUTCOME_UNKNOWN:
                self._record(
                    fault,
                    operation="write",
                    resource_id=resource_id,
                    delegate_called=True,
                    outcome_indeterminate=True,
                )
                raise FaultInjectedOutcomeUnknownError("fault_injected_write_acknowledgement_loss")
            return result


__all__ = [
    "ACCEPTANCE_GOVERNANCE_SCHEMA",
    "AcceptanceCertificateStore",
    "AcceptanceCaseResult",
    "AcceptanceError",
    "AcceptanceEvidenceClass",
    "AcceptanceVerdict",
    "ConnectorAcceptanceCertificate",
    "FaultInjectedOutcomeUnknownError",
    "FaultInjectedReadError",
    "FaultInjectedWriteError",
    "FaultInjectingScalarTransport",
    "FaultInjectionReceipt",
    "ScalarAcceptancePlan",
    "ScalarAcceptanceRunner",
    "ScalarFault",
    "REQUIRED_SCALAR_ACCEPTANCE_CASES",
    "acceptance_governance_accepted",
    "acceptance_governance_document",
]
