"""Operational owner for governed Reality Reach acceptance campaigns."""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from core.reality_reach.acceptance import (
    REQUIRED_SCALAR_ACCEPTANCE_CASES,
    AcceptanceCertificateStore,
    AcceptanceError,
    AcceptanceEvidenceClass,
    AcceptanceExecutor,
    ConnectorAcceptanceCertificate,
    ScalarAcceptancePlan,
    ScalarAcceptanceRunner,
    acceptance_governance_document,
)
from core.reality_reach.acceptance_mandate import (
    AcceptanceMandateError,
    AcceptanceMandateStore,
    AcceptanceVerificationMandate,
)
from core.reality_reach.acceptance_requests import (
    AcousticA1MandateRequest,
    AcousticA1Request,
    ScalarAcceptanceMandateRequest,
    ScalarAcceptanceRequest,
)
from core.reality_reach.acoustic_acceptance import (
    ACOUSTIC_A1_CONNECTOR_ID,
    ACOUSTIC_A1_REQUIRED_CASES,
    AcousticA1AcceptanceReceipt,
    AcousticA1CampaignRecord,
    AcousticA1CampaignStore,
    AcousticAcceptanceConfig,
    AcousticAcceptanceError,
    AcousticTrialDriver,
    run_acoustic_a1_acceptance,
)
from core.reality_reach.live import RealityReachService
from core.reality_reach.metrology import (
    AcquisitionChannel,
    AcquisitionMode,
    AcquisitionTask,
    EvidenceSource,
    RealityMetrologyService,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lockdep import checked_async_lock

_GIT_OID = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
SourceIdentityProvider = Callable[[], Mapping[str, Any]]
AcousticAdapterFactory = Callable[[], Awaitable[Any]]


def capture_runtime_source_identity() -> Mapping[str, Any]:
    from core.config import config
    from core.runtime.launch_provenance import (
        collect_runtime_launch_provenance,
        collect_source_identity,
    )

    root = config.paths.project_root
    provenance = collect_runtime_launch_provenance(root)
    required = provenance.get("required") is True
    source = provenance.get("actual") if required else collect_source_identity(root)
    if not isinstance(source, Mapping):
        source = {}
    return {
        "identity_bound": bool(
            (provenance.get("verified") is True if required else True)
            and provenance.get("source_verified", True) is True
        ),
        "source_commit": str(source.get("commit_sha") or "").strip().lower(),
        "source_dirty": source.get("source_dirty") is True,
        "workspace_state_sha256": str(
            source.get("workspace_state_sha256") or ""
        ).strip().lower(),
    }


def _normalize_source_identity(raw: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(raw, Mapping) or raw.get("identity_bound") is not True:
        raise AcceptanceError("acceptance_runtime_source_identity_unbound")
    commit = str(raw.get("source_commit") or "").strip().lower()
    if not _GIT_OID.fullmatch(commit):
        raise AcceptanceError("acceptance_runtime_source_commit_invalid")
    if raw.get("source_dirty") is True:
        raise AcceptanceError("acceptance_runtime_source_dirty")
    workspace = str(raw.get("workspace_state_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", workspace):
        raise AcceptanceError("acceptance_runtime_workspace_identity_invalid")
    return {
        "source_commit_sha256": str(sha256_hex(canonical_json(commit))),
        "workspace_state_sha256": f"sha256:{workspace}",
    }


class RealityAcceptanceService:
    """Serialize, execute, persist, and summarize physical acceptance runs."""

    def __init__(
        self,
        reality: RealityReachService,
        metrology: RealityMetrologyService,
        *,
        store: AcceptanceCertificateStore | None = None,
        mandate_store: AcceptanceMandateStore | None = None,
        governed_executor: AcceptanceExecutor | None = None,
        acoustic_adapter_factory: AcousticAdapterFactory | None = None,
        acoustic_campaign_store: AcousticA1CampaignStore | None = None,
        pinned_source_identity: Mapping[str, Any],
        source_identity_provider: SourceIdentityProvider | None = None,
    ) -> None:
        if not isinstance(reality, RealityReachService):
            raise TypeError("reality must be a RealityReachService")
        if not isinstance(metrology, RealityMetrologyService):
            raise TypeError("metrology must be a RealityMetrologyService")
        self._reality = reality
        self._metrology = metrology
        self._store = store or AcceptanceCertificateStore()
        self._mandate_store = mandate_store
        self._mandate_status: dict[str, Any] = {
            "healthy": mandate_store is not None,
            "state": "available" if mandate_store is not None else "unavailable",
            "error_type": "" if mandate_store is not None else "mandate_store_unavailable",
        }
        self._governed_executor = governed_executor
        self._acoustic_adapter_factory = acoustic_adapter_factory
        self._acoustic_campaign_store = (
            acoustic_campaign_store or AcousticA1CampaignStore()
        )
        if not isinstance(pinned_source_identity, Mapping):
            raise TypeError("pinned_source_identity must be a mapping")
        self._pinned_source_identity = dict(pinned_source_identity)
        self._source_identity_provider = (
            source_identity_provider or capture_runtime_source_identity
        )
        self._lock = checked_async_lock("reality_reach.acceptance_service")
        self._generation = 0
        self._active_campaign_id = ""
        self._last_result: dict[str, Any] | None = None
        self._last_failure: dict[str, Any] | None = None

    @staticmethod
    def _observation_channels(adapter: Any) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                channel_id
                for capability in tuple(adapter.actuator_capabilities())
                for channel_id in capability.observation_channels
            )
        )

    async def _source_binding(self, expected_sha256: str | None) -> dict[str, str]:
        raw = await asyncio.to_thread(self._source_identity_provider)
        pinned = _normalize_source_identity(self._pinned_source_identity)
        current = _normalize_source_identity(raw)
        if current != pinned:
            raise AcceptanceError("acceptance_runtime_source_drifted_since_boot")
        if (
            expected_sha256 is not None
            and current["source_commit_sha256"] != expected_sha256
        ):
            raise AcceptanceError("acceptance_runtime_source_commit_mismatch")
        return current

    async def preflight(self, adapter_id: str) -> dict[str, Any]:
        adapter = self._reality.scalar_acceptance_adapter(adapter_id)
        if adapter is None:
            raise AcceptanceError("acceptance_scalar_adapter_not_registered")
        source = await self._source_binding(None)
        capabilities = tuple(adapter.actuator_capabilities())
        observation_channels = self._observation_channels(adapter)
        evidence_classes = (
            (AcceptanceEvidenceClass.SIMULATION.value,)
            if adapter.transport_class.value == "simulated"
            else (
                AcceptanceEvidenceClass.HARDWARE_IN_LOOP.value,
                AcceptanceEvidenceClass.LIVE.value,
            )
        )
        return {
            "ready": bool(
                capabilities
                and observation_channels
                and self._metrology.is_ready()
                and self._mandate_status.get("healthy") is True
            ),
            "adapter_id": adapter.adapter_id,
            "physical_identity_sha256": adapter.physical_identity_sha256,
            "transport_class": adapter.transport_class.value,
            "target_tolerance": adapter.effect_tolerance,
            "required_cases": list(REQUIRED_SCALAR_ACCEPTANCE_CASES),
            "observation_channels": list(observation_channels),
            "supported_evidence_classes": list(evidence_classes),
            **source,
            "trust_boundary": "producer_observation_not_independent_acceptance",
        }

    async def provision_macos_acoustic_adapter(self) -> dict[str, Any]:
        """Attach the built-in reversible acoustic loop only on explicit demand."""

        from core.embodiment.macos_acoustic_reality import (
            ACOUSTIC_ADAPTER_ID,
            MacOSAcousticRealityError,
            build_macos_acoustic_reality_adapter,
        )
        from core.reality_reach.scalar_adapter import (
            ScalarRealityAdapter,
            ScalarTransportClass,
        )

        provisioned = False
        async with self._lock:
            if self._active_campaign_id:
                raise AcceptanceError("acceptance_campaign_already_active")
            adapter = self._reality.scalar_acceptance_adapter(ACOUSTIC_ADAPTER_ID)
            if adapter is None:
                factory = (
                    self._acoustic_adapter_factory
                    or build_macos_acoustic_reality_adapter
                )
                try:
                    adapter = await factory()
                except MacOSAcousticRealityError as exc:
                    raise AcceptanceError(
                        f"acceptance_acoustic_hardware_unavailable:{exc}"
                    ) from exc
                except (ImportError, OSError, RuntimeError, TimeoutError) as exc:
                    raise AcceptanceError(
                        "acceptance_acoustic_hardware_unavailable"
                    ) from exc
                if not isinstance(adapter, ScalarRealityAdapter):
                    raise AcceptanceError("acceptance_acoustic_adapter_invalid")
                if adapter.transport_class is not ScalarTransportClass.PHYSICAL:
                    raise AcceptanceError("acceptance_acoustic_adapter_not_physical")
                if adapter.adapter_id != ACOUSTIC_ADAPTER_ID:
                    raise AcceptanceError("acceptance_acoustic_adapter_identity_invalid")
                self._reality.register_adapter(adapter)
                provisioned = True
        result = await self.preflight(ACOUSTIC_ADAPTER_ID)
        readings = tuple(adapter.read())
        baseline_reading = next(
            (
                reading
                for reading in readings
                if isinstance(reading.value, (int, float))
                and not isinstance(reading.value, bool)
            ),
            None,
        )
        baseline = (
            float(baseline_reading.value)
            if baseline_reading is not None
            else math.nan
        )
        recommended_target = (
            min(-18.0, baseline + 12.0) if math.isfinite(baseline) else math.nan
        )
        signal_margin = recommended_target - baseline
        campaign_blockers: list[str] = []
        if not math.isfinite(baseline):
            campaign_blockers.append("acoustic_baseline_unavailable")
        elif signal_margin < 8.0:
            campaign_blockers.append("acoustic_signal_margin_insufficient")
        return {
            **result,
            "provisioned": provisioned,
            "campaign_ready": bool(result["ready"] and not campaign_blockers),
            "campaign_blockers": campaign_blockers,
            "stimulus": {
                "frequency_hz": 997.0,
                "baseline_dbfs": baseline if math.isfinite(baseline) else None,
                "recommended_target_dbfs": (
                    recommended_target if math.isfinite(recommended_target) else None
                ),
                "required_signal_margin_db": 8.0,
                "raw_audio_retained": False,
                "maximum_output_amplitude": 0.08,
                "restoration": "silence_then_independent_microphone_readback",
            },
        }

    def _acoustic_trial_adapter(self) -> tuple[Any, AcousticTrialDriver]:
        from core.embodiment.macos_acoustic_reality import ACOUSTIC_ADAPTER_ID

        adapter = self._reality.scalar_acceptance_adapter(ACOUSTIC_ADAPTER_ID)
        if adapter is None:
            raise AcceptanceError("acceptance_acoustic_adapter_not_registered")
        if adapter.transport_class.value != "physical":
            raise AcceptanceError("acceptance_acoustic_adapter_not_physical")
        if not isinstance(adapter, AcousticTrialDriver):
            raise AcceptanceError("acceptance_acoustic_trial_driver_unavailable")
        return adapter, adapter

    async def precommit_acoustic_a1(
        self,
        request: AcousticA1MandateRequest,
    ) -> dict[str, Any]:
        """Precommit the immutable A1 protocol before any physical stimulus."""

        from core.embodiment.macos_acoustic_reality import ACOUSTIC_ADAPTER_ID

        if not isinstance(request, AcousticA1MandateRequest):
            raise TypeError("request must be an AcousticA1MandateRequest")
        if self._mandate_store is None:
            raise AcceptanceError("acceptance_mandate_custody_unavailable")
        adapter, _driver = self._acoustic_trial_adapter()
        source = await self._source_binding(request.expected_source_commit_sha256)
        config = AcousticAcceptanceConfig(campaign_id=request.campaign_id)
        live_channels = self._observation_channels(adapter)
        try:
            provision = await asyncio.to_thread(
                self._mandate_store.provision,
                campaign_id=request.campaign_id,
                connector_id=ACOUSTIC_A1_CONNECTOR_ID,
                adapter_id=ACOUSTIC_ADAPTER_ID,
                expected_source_commit_sha256=source["source_commit_sha256"],
                expected_physical_identity_sha256=adapter.physical_identity_sha256,
                expected_evidence_class=AcceptanceEvidenceClass.LIVE,
                target=config.required_error_reduction,
                target_tolerance=0.0,
                scenario_id="",
                expected_live_channel_ids=live_channels,
                expected_simulated_channel_ids=(),
                required_cases=ACOUSTIC_A1_REQUIRED_CASES,
            )
            mandate = await asyncio.to_thread(
                self._mandate_store.get,
                request.campaign_id,
            )
            self._mandate_status = dict(
                await asyncio.to_thread(self._mandate_store.status)
            )
        except AcceptanceMandateError as exc:
            self._mandate_status = {
                "healthy": False,
                "state": "degraded",
                "error_type": type(exc).__name__,
            }
            raise AcceptanceError(str(exc)) from exc
        return {
            "mandate": mandate.to_dict(),
            "provision_receipt": provision.to_dict(),
            "config_sha256": config.sha256,
            "required_error_reduction": config.required_error_reduction,
            "required_cases": list(ACOUSTIC_A1_REQUIRED_CASES),
            "source_commit_sha256": source["source_commit_sha256"],
            "workspace_state_sha256": source["workspace_state_sha256"],
            "trust_boundary": "machine_local_precommit_not_external_witness",
        }

    async def _require_acoustic_a1_mandate(
        self,
        request: AcousticA1Request,
        adapter: Any,
        source: Mapping[str, str],
    ) -> AcceptanceVerificationMandate:
        if self._mandate_store is None:
            raise AcceptanceError("acceptance_mandate_custody_unavailable")
        config = AcousticAcceptanceConfig(campaign_id=request.campaign_id)
        try:
            mandate = await asyncio.to_thread(
                self._mandate_store.get,
                request.campaign_id,
            )
        except AcceptanceMandateError as exc:
            raise AcceptanceError(str(exc)) from exc
        expected = {
            "campaign_id": request.campaign_id,
            "connector_id": ACOUSTIC_A1_CONNECTOR_ID,
            "adapter_id": adapter.adapter_id,
            "expected_source_commit_sha256": source["source_commit_sha256"],
            "expected_physical_identity_sha256": adapter.physical_identity_sha256,
            "expected_evidence_class": AcceptanceEvidenceClass.LIVE,
            "target": config.required_error_reduction,
            "target_tolerance": 0.0,
            "scenario_id": "",
            "expected_live_channel_ids": self._observation_channels(adapter),
            "expected_simulated_channel_ids": (),
            "required_cases": ACOUSTIC_A1_REQUIRED_CASES,
        }
        if mandate.sha256 != request.mandate_sha256:
            raise AcceptanceError("acceptance_mandate_digest_mismatch")
        for field, value in expected.items():
            if getattr(mandate, field) != value:
                raise AcceptanceError(f"acceptance_mandate_{field}_mismatch")
        return mandate

    async def run_acoustic_a1(self, request: AcousticA1Request) -> dict[str, Any]:
        """Run the exact precommitted A1 campaign through Will/ActionExecutor."""

        from core.governance.action_domain import ActionDomain
        from core.runtime.action_executor import ActionExecutor
        from core.runtime.skill_contract import ActionExpectation

        if not isinstance(request, AcousticA1Request):
            raise TypeError("request must be an AcousticA1Request")
        adapter, driver = self._acoustic_trial_adapter()
        async with self._lock:
            self._generation += 1
            self._active_campaign_id = request.campaign_id
            started_at_ns = time.time_ns()
            source_before: dict[str, str] | None = None
            mandate: AcceptanceVerificationMandate | None = None
            completed: dict[str, AcousticA1AcceptanceReceipt] = {}
            persist_attempted = False

            async def persist_interrupted_evidence(error: BaseException) -> None:
                receipt = completed.get("receipt")
                if (
                    receipt is None
                    or source_before is None
                    or mandate is None
                    or persist_attempted
                ):
                    return
                governance = acceptance_governance_document(
                    {
                        "action_id": f"acceptance.{request.campaign_id}"[:128],
                        "status": "interrupted_after_effect",
                        "transport_succeeded": True,
                        "effect_verified": False,
                        "receipt_persisted": False,
                        "welfare_transaction_completed": False,
                    }
                )
                record = AcousticA1CampaignRecord(
                    campaign_id=request.campaign_id,
                    adapter_id=adapter.adapter_id,
                    source_commit_sha256=source_before["source_commit_sha256"],
                    workspace_state_sha256=source_before["workspace_state_sha256"],
                    physical_identity_sha256=adapter.physical_identity_sha256,
                    mandate_sha256=mandate.sha256,
                    receipt=receipt,
                    governance_evidence=governance,
                    started_at_ns=started_at_ns,
                    completed_at_ns=max(time.time_ns(), receipt.completed_at_ns),
                )
                try:
                    await asyncio.shield(
                        asyncio.to_thread(
                            self._acoustic_campaign_store.persist,
                            record,
                        )
                    )
                except Exception as persist_exc:  # noqa: BLE001 - preserve root cause
                    error.add_note(
                        "completed acoustic evidence could not be persisted: "
                        f"{type(persist_exc).__name__}: {persist_exc}"
                    )

            try:
                source_before = await self._source_binding(
                    request.expected_source_commit_sha256
                )
                mandate = await self._require_acoustic_a1_mandate(
                    request,
                    adapter,
                    source_before,
                )
                config = AcousticAcceptanceConfig(campaign_id=request.campaign_id)
                try:
                    existing = await asyncio.to_thread(
                        self._acoustic_campaign_store.load,
                        request.campaign_id,
                    )
                except AcousticAcceptanceError as exc:
                    if str(exc) != "acoustic_a1_campaign_unavailable":
                        raise AcceptanceError(str(exc)) from exc
                else:
                    if (
                        existing.source_commit_sha256
                        != source_before["source_commit_sha256"]
                        or existing.workspace_state_sha256
                        != source_before["workspace_state_sha256"]
                        or existing.physical_identity_sha256
                        != adapter.physical_identity_sha256
                        or existing.mandate_sha256 != mandate.sha256
                        or existing.receipt.config_sha256 != config.sha256
                    ):
                        raise AcceptanceError("acoustic_a1_campaign_replay_mismatch")
                    result = {
                        **existing.to_dict(),
                        "published": False,
                        "replayed": True,
                        "trust_boundary": (
                            "producer_governed_result_pending_independent_witness"
                        ),
                    }
                    self._last_result = result
                    self._last_failure = None
                    return dict(result)
                async def effect_handler(
                    context: Mapping[str, Any],
                ) -> Mapping[str, Any]:
                    authority_receipt_id = str(context.get("will_receipt_id") or "")
                    if not _IDENTIFIER.fullmatch(authority_receipt_id):
                        raise AcceptanceError("acceptance_governance_receipt_invalid")
                    receipt = await run_acoustic_a1_acceptance(driver, config)
                    completed["receipt"] = receipt
                    return {
                        "ok": True,
                        "transport_succeeded": True,
                        "acoustic_a1_receipt_sha256": receipt.sha256,
                        "error_reduction": receipt.error_reduction,
                    }

                async def effect_verifier(
                    _context: Mapping[str, Any],
                ) -> Mapping[str, Any]:
                    receipt = completed.get("receipt")
                    return {
                        "effect_verified": bool(receipt and receipt.accepted),
                        "acoustic_a1_receipt_sha256": (
                            receipt.sha256 if receipt is not None else ""
                        ),
                    }

                executor = self._governed_executor or ActionExecutor.execute
                raw_result = await executor(
                    domain=ActionDomain.ENVIRONMENT_ACTION,
                    action_name="reality_reach.acceptance.macos_acoustic_a1",
                    params={
                        "campaign_id": request.campaign_id,
                        "adapter_id": adapter.adapter_id,
                        "physical_identity_sha256": adapter.physical_identity_sha256,
                        "config_sha256": config.sha256,
                        "required_error_reduction": config.required_error_reduction,
                    },
                    source="reality_reach.acceptance.acoustic_a1",
                    rollback_target="macos_acoustic.restore_silence",
                    expectation=ActionExpectation(
                        objective=(
                            "calibrate speaker-to-microphone control and beat an "
                            "equal-work open-loop arm on held-out targets"
                        ),
                        required_evidence=[
                            "acoustic_a1_receipt_sha256",
                            "error_reduction",
                        ],
                        rollback_hint="emit silence and remeasure the baseline",
                        allow_partial=False,
                    ),
                    effect_handler=effect_handler,
                    effect_verifier=effect_verifier,
                    execution_timeout_s=60.0,
                    verification_timeout_s=5.0,
                    action_id=f"acceptance.{request.campaign_id}"[:128],
                )
                if not isinstance(raw_result, Mapping):
                    raise AcceptanceError("acceptance_governance_result_invalid")
                receipt = completed.get("receipt")
                if receipt is None:
                    raise AcceptanceError("acceptance_governance_refused_before_dispatch")
                source_after = await self._source_binding(
                    request.expected_source_commit_sha256
                )
                if source_after != source_before:
                    raise AcceptanceError("acceptance_runtime_source_changed_during_run")
                governance = acceptance_governance_document(raw_result)
                completed_at_ns = max(time.time_ns(), receipt.completed_at_ns)
                record = AcousticA1CampaignRecord(
                    campaign_id=request.campaign_id,
                    adapter_id=adapter.adapter_id,
                    source_commit_sha256=source_after["source_commit_sha256"],
                    workspace_state_sha256=source_after["workspace_state_sha256"],
                    physical_identity_sha256=adapter.physical_identity_sha256,
                    mandate_sha256=mandate.sha256,
                    receipt=receipt,
                    governance_evidence=governance,
                    started_at_ns=started_at_ns,
                    completed_at_ns=completed_at_ns,
                )
                persist_attempted = True
                try:
                    published = await asyncio.to_thread(
                        self._acoustic_campaign_store.persist,
                        record,
                    )
                except AcousticAcceptanceError as exc:
                    raise AcceptanceError(str(exc)) from exc
                result = {
                    **record.to_dict(),
                    "published": published,
                    "replayed": False,
                    "trust_boundary": (
                        "producer_governed_result_pending_independent_witness"
                    ),
                }
                self._last_result = result
                self._last_failure = None
                return dict(result)
            except asyncio.CancelledError as exc:
                await persist_interrupted_evidence(exc)
                self._record_failure(request, exc, started_at_ns=started_at_ns)
                raise
            except Exception as exc:
                await persist_interrupted_evidence(exc)
                self._record_failure(request, exc, started_at_ns=started_at_ns)
                raise
            finally:
                self._active_campaign_id = ""

    async def precommit(
        self,
        request: ScalarAcceptanceMandateRequest,
    ) -> dict[str, Any]:
        if not isinstance(request, ScalarAcceptanceMandateRequest):
            raise TypeError("request must be a ScalarAcceptanceMandateRequest")
        if self._mandate_store is None:
            raise AcceptanceError("acceptance_mandate_custody_unavailable")
        adapter = self._reality.scalar_acceptance_adapter(request.adapter_id)
        if adapter is None:
            raise AcceptanceError("acceptance_scalar_adapter_not_registered")
        transport_class = adapter.transport_class.value
        if request.evidence_class is AcceptanceEvidenceClass.SIMULATION:
            if transport_class != "simulated":
                raise AcceptanceError("acceptance_simulation_transport_mismatch")
            live_channels: tuple[str, ...] = ()
        else:
            if transport_class != "physical":
                raise AcceptanceError("acceptance_physical_transport_required")
            live_channels = self._observation_channels(adapter)
            if not live_channels:
                raise AcceptanceError("acceptance_adapter_has_no_observation_channels")
        source = await self._source_binding(request.expected_source_commit_sha256)
        try:
            receipt = await asyncio.to_thread(
                self._mandate_store.provision,
                campaign_id=request.campaign_id,
                connector_id=request.connector_id,
                adapter_id=request.adapter_id,
                expected_source_commit_sha256=source["source_commit_sha256"],
                expected_physical_identity_sha256=(
                    adapter.physical_identity_sha256
                ),
                expected_evidence_class=request.evidence_class,
                target=request.target,
                target_tolerance=adapter.effect_tolerance,
                scenario_id=request.scenario_id,
                expected_live_channel_ids=live_channels,
                expected_simulated_channel_ids=request.simulated_channel_ids,
                required_cases=REQUIRED_SCALAR_ACCEPTANCE_CASES,
            )
            mandate = await asyncio.to_thread(
                self._mandate_store.get,
                request.campaign_id,
            )
            self._mandate_status = dict(
                await asyncio.to_thread(self._mandate_store.status)
            )
        except AcceptanceMandateError as exc:
            self._mandate_status = {
                "healthy": False,
                "state": "degraded",
                "error_type": type(exc).__name__,
            }
            raise AcceptanceError(str(exc)) from exc
        return {
            "mandate": mandate.to_dict(),
            "provision_receipt": receipt.to_dict(),
            "source_commit_sha256": source["source_commit_sha256"],
            "workspace_state_sha256": source["workspace_state_sha256"],
            "trust_boundary": "machine_local_precommit_not_external_witness",
        }

    async def _require_physical_mandate(
        self,
        request: ScalarAcceptanceRequest,
        adapter: Any,
        source: Mapping[str, str],
    ) -> AcceptanceVerificationMandate:
        if self._mandate_store is None:
            raise AcceptanceError("acceptance_mandate_custody_unavailable")
        if not request.mandate_sha256:
            raise AcceptanceError("acceptance_mandate_sha256_missing")
        try:
            mandate = await asyncio.to_thread(
                self._mandate_store.get,
                request.campaign_id,
            )
        except AcceptanceMandateError as exc:
            self._mandate_status = {
                "healthy": False,
                "state": "degraded",
                "error_type": type(exc).__name__,
            }
            raise AcceptanceError(str(exc)) from exc
        expected = {
            "campaign_id": request.campaign_id,
            "connector_id": request.connector_id,
            "adapter_id": request.adapter_id,
            "expected_source_commit_sha256": source["source_commit_sha256"],
            "expected_physical_identity_sha256": adapter.physical_identity_sha256,
            "expected_evidence_class": request.evidence_class,
            "target": request.target,
            "target_tolerance": adapter.effect_tolerance,
            "scenario_id": request.scenario_id,
            "expected_live_channel_ids": self._observation_channels(adapter),
            "expected_simulated_channel_ids": request.simulated_channel_ids,
            "required_cases": REQUIRED_SCALAR_ACCEPTANCE_CASES,
        }
        if mandate.sha256 != request.mandate_sha256:
            raise AcceptanceError("acceptance_mandate_digest_mismatch")
        for field, value in expected.items():
            if getattr(mandate, field) != value:
                raise AcceptanceError(f"acceptance_mandate_{field}_mismatch")
        return mandate

    def _record_failure(
        self,
        request: ScalarAcceptanceRequest | AcousticA1Request,
        error: BaseException,
        *,
        started_at_ns: int,
    ) -> None:
        error_type = type(error).__name__
        self._last_failure = {
            "campaign_id_sha256": str(
                sha256_hex(canonical_json(str(request.campaign_id)))
            ),
            "error_type": error_type,
            "error_sha256": str(
                sha256_hex(
                    canonical_json(
                        {
                            "error_type": error_type,
                            "message": str(error),
                        }
                    )
                )
            ),
            "started_at_ns": started_at_ns,
            "failed_at_ns": time.time_ns(),
        }

    def _task(self, request: ScalarAcceptanceRequest, adapter: Any) -> AcquisitionTask:
        capabilities = tuple(adapter.actuator_capabilities())
        live_channels = tuple(
            dict.fromkeys(
                channel_id
                for capability in capabilities
                for channel_id in capability.observation_channels
            )
        )
        if not live_channels:
            raise AcceptanceError("acceptance_adapter_has_no_observation_channels")
        mode = (
            AcquisitionMode.HARDWARE_IN_LOOP
            if request.evidence_class is AcceptanceEvidenceClass.HARDWARE_IN_LOOP
            else AcquisitionMode.LIVE
        )
        channels = tuple(AcquisitionChannel(item) for item in live_channels)
        channels += tuple(
            AcquisitionChannel(item, EvidenceSource.SIMULATED)
            for item in request.simulated_channel_ids
        )
        duration_s = request.deadline_s + request.effect_hold_s + 1.0
        sample_count = int(math.ceil(duration_s / request.sample_interval_s)) + 1
        if sample_count > 1024:
            raise AcceptanceError("acceptance_metrology_sample_budget_exceeded")
        return AcquisitionTask(
            task_id=f"acceptance.{request.campaign_id}"[:128],
            channels=channels,
            mode=mode,
            sample_count=sample_count,
            sample_interval_s=request.sample_interval_s,
            timeout_s=duration_s + 5.0,
            scenario_id=request.scenario_id,
        )

    async def run(self, request: ScalarAcceptanceRequest) -> dict[str, Any]:
        if not isinstance(request, ScalarAcceptanceRequest):
            raise TypeError("request must be a ScalarAcceptanceRequest")
        adapter = self._reality.scalar_acceptance_adapter(request.adapter_id)
        if adapter is None:
            raise AcceptanceError("acceptance_scalar_adapter_not_registered")
        async with self._lock:
            self._generation += 1
            self._active_campaign_id = request.campaign_id
            started_at_ns = time.time_ns()
            try:
                source_before = await self._source_binding(
                    request.expected_source_commit_sha256
                )
                mandate = None
                if request.evidence_class in {
                    AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
                    AcceptanceEvidenceClass.LIVE,
                }:
                    mandate = await self._require_physical_mandate(
                        request,
                        adapter,
                        source_before,
                    )
                plan = ScalarAcceptancePlan(
                    campaign_id=request.campaign_id,
                    connector_id=request.connector_id,
                    target=request.target,
                    source_commit_sha256=source_before["source_commit_sha256"],
                    authority_receipt_id="authority.runtime.will",
                    evidence_class=request.evidence_class,
                    scenario_id=request.scenario_id,
                    deadline_s=request.deadline_s,
                    metrology_effect_hold_s=request.effect_hold_s,
                )
                if request.evidence_class is AcceptanceEvidenceClass.SIMULATION:
                    runner = ScalarAcceptanceRunner(adapter, self._reality, plan)
                else:
                    task = self._task(request, adapter)
                    runner = ScalarAcceptanceRunner(
                        adapter,
                        self._reality,
                        plan,
                        governed_executor=self._governed_executor,
                        metrology_acquirer=lambda operation: self._metrology.acquire_around(
                            task,
                            operation,
                        ),
                    )
                certificate = await runner.run()
                source_after = await self._source_binding(
                    request.expected_source_commit_sha256
                )
                if source_after != source_before:
                    raise AcceptanceError("acceptance_runtime_source_changed_during_run")
                self._store.persist(certificate)
                self._store.persist_evidence(
                    certificate,
                    runner.case_evidence,
                    metrology_receipt=runner.metrology_receipt,
                    governance_evidence=(
                        runner.governance_evidence
                        if certificate.governance_evidence_sha256
                        else None
                    ),
                )
                result = self._result(
                    certificate,
                    started_at_ns=started_at_ns,
                    source_identity=source_after,
                    mandate=mandate,
                )
                self._last_result = result
                self._last_failure = None
                return dict(result)
            except asyncio.CancelledError as exc:
                self._record_failure(request, exc, started_at_ns=started_at_ns)
                raise
            except Exception as exc:
                self._record_failure(request, exc, started_at_ns=started_at_ns)
                raise
            finally:
                self._active_campaign_id = ""

    def _result(
        self,
        certificate: ConnectorAcceptanceCertificate,
        *,
        started_at_ns: int,
        source_identity: Mapping[str, str],
        mandate: AcceptanceVerificationMandate | None,
    ) -> dict[str, Any]:
        return {
            "campaign_id": certificate.campaign_id,
            "certificate_sha256": certificate.sha256,
            "evidence_class": next(iter(certificate.cases)).evidence_class.value,
            "deterministic_passed": certificate.deterministic_passed,
            "live_acceptance_passed": certificate.live_acceptance_passed,
            "metrology_evidence_sha256": certificate.metrology_evidence_sha256,
            "governance_evidence_sha256": certificate.governance_evidence_sha256,
            "source_commit_sha256": source_identity["source_commit_sha256"],
            "workspace_state_sha256": source_identity["workspace_state_sha256"],
            "mandate_sha256": mandate.sha256 if mandate is not None else "",
            "started_at_ns": started_at_ns,
            "completed_at_ns": time.time_ns(),
        }

    def status(self) -> dict[str, Any]:
        source_blocker = ""
        try:
            _normalize_source_identity(self._pinned_source_identity)
        except AcceptanceError as exc:
            source_blocker = str(exc)
        mandate_status = dict(self._mandate_status)
        return {
            "alive": True,
            "ready": bool(
                not self._active_campaign_id
                and not source_blocker
                and self._metrology.is_ready()
                and mandate_status.get("healthy") is True
            ),
            "source_identity_blocker": source_blocker,
            "generation": self._generation,
            "active_campaign_id": self._active_campaign_id,
            "last_result": dict(self._last_result) if self._last_result else None,
            "last_failure": dict(self._last_failure) if self._last_failure else None,
            "store_root": str(self._store.root),
            "acoustic_campaign_store_root": str(self._acoustic_campaign_store.root),
            "mandate_custody": mandate_status,
        }

    def is_alive(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return bool(self.status()["ready"])

    def close(self) -> None:
        if self._mandate_store is not None:
            self._mandate_store.close()


__all__ = [
    "AcousticA1MandateRequest",
    "AcousticA1Request",
    "RealityAcceptanceService",
    "ScalarAcceptanceMandateRequest",
    "ScalarAcceptanceRequest",
    "SourceIdentityProvider",
    "capture_runtime_source_identity",
]
