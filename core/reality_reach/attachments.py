"""Portable discovery, trust, and attachment for physical Reality Reach adapters.

The broker separates four events that older Aura paths conflated:

* a connector notices a candidate;
* Aura proposes a relationship with that candidate;
* a stable device identity receives bounded trust;
* an adapter is attached to the live Reality Reach inventory.

Discovery never grants actuation.  Trusted devices may reattach after a host
migration without code changes, but only through the access classes and
manifest digest named by their durable trust grant.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable

from core.reality_reach.attachment_authority import (
    AttachmentAuthorityError,
    AttachmentCapabilityAuthorityVerifier,
    ManifestMigrationAuthorityVerifier,
    PhysicalAuthorityVerifier,
    build_attachment_authority_intent,
)
from core.reality_reach.body_projection import (
    PhysicalBodyProjection,
    project_adapter_to_body,
    remove_body_projection,
)
from core.reality_reach.digital_twin import RealityDigitalTwinGraph
from core.reality_reach.live import LiveChannelAdapter, RealityReachService
from core.reality_reach.middleware_contracts import ManagedRealityAdapter
from core.reality_reach.observation_router import RealityObservationRouter
from core.reality_reach.trust_custody import (
    AttachmentTrustStore,
    AttachmentTrustStoreError,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_async_lock, checked_lock, checked_semaphore
from core.utils.concurrency import cancel_and_join
from core.utils.task_tracker import get_task_tracker

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_METADATA_BYTES = 16 * 1024
_SESSION_GRANT_MAX_S = 8 * 60 * 60
_PERSISTENT_OBSERVE_MAX_S = 90 * 24 * 60 * 60
_PERSISTENT_CONTROL_MAX_S = 30 * 24 * 60 * 60
_DEFAULT_PERSISTENT_OBSERVE_S = 30 * 24 * 60 * 60
_DEFAULT_PERSISTENT_CONTROL_S = 7 * 24 * 60 * 60
_DEFAULT_DISAPPEARANCE_QUORUM = 3
_MAX_DISAPPEARANCE_QUORUM = 16
_TWIN_RECONCILIATION_SCHEMA = "aura.reality_reach.twin_reconciliation.v1"
_MAX_TWIN_RECONCILIATION_INTENTS = 4096
_GRANT_LIFECYCLE_ACTIVE = "active"
_GRANT_LIFECYCLE_PENDING = "pending_activation"


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _reconciliation_reason(value: Any, *, fallback: str) -> str:
    normalized = " ".join(str(value or fallback).split())
    return (normalized or fallback)[:160]


def _identifier(value: str, *, name: str) -> str:
    canonical = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(canonical):
        raise ValueError(f"{name} must be a canonical identifier")
    return canonical


def _frozen_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("device metadata must contain canonical JSON") from exc
    if len(encoded) > _MAX_METADATA_BYTES:
        raise ValueError("device metadata exceeds the bounded envelope")
    decoded = json.loads(encoded.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("device metadata must be a mapping")
    return MappingProxyType(decoded)


def _bounded_grant_ttl_s(
    access: tuple[AttachmentAccess, ...],
    *,
    persistent: bool,
    requested_ttl_s: int | None,
) -> int:
    includes_control = AttachmentAccess.CONTROL in access
    if not persistent:
        maximum = _SESSION_GRANT_MAX_S
        default = _SESSION_GRANT_MAX_S
    elif includes_control:
        maximum = _PERSISTENT_CONTROL_MAX_S
        default = _DEFAULT_PERSISTENT_CONTROL_S
    else:
        maximum = _PERSISTENT_OBSERVE_MAX_S
        default = _DEFAULT_PERSISTENT_OBSERVE_S
    if requested_ttl_s is None:
        return default
    if isinstance(requested_ttl_s, bool) or not isinstance(requested_ttl_s, int):
        raise TypeError("grant_ttl_s must be an integer number of seconds")
    if requested_ttl_s <= 0 or requested_ttl_s > maximum:
        raise ValueError(f"grant_ttl_s must lie inside [1, {maximum}]")
    return requested_ttl_s


class AttachmentAccess(StrEnum):
    OBSERVE = "observe"
    CONTROL = "control"


class ConnectionState(StrEnum):
    DISCOVERED = "discovered"
    PENDING_TRUST = "pending_trust"
    ATTACHING = "attaching"
    ATTACHED = "attached"
    REFUSED = "refused"
    LOST = "lost"
    ERROR = "error"
    REVOKED = "revoked"


@dataclass(frozen=True, slots=True)
class DeviceCandidate:
    candidate_id: str
    connector_id: str
    device_id: str
    display_name: str
    transport: str
    identity_fingerprint: str
    manifest_sha256: str
    access: tuple[AttachmentAccess, ...]
    discovered_at_ns: int
    expires_at_ns: int
    persistent_identity: bool = False
    privacy_sensitive: bool = False
    proposal_salience: float = 0.4
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        for name in ("candidate_id", "connector_id", "device_id", "transport"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))
        if not str(self.display_name or "").strip() or len(self.display_name) > 160:
            raise ValueError("display_name must be present and bounded")
        for name in ("identity_fingerprint", "manifest_sha256"):
            if not _DIGEST.fullmatch(str(getattr(self, name))):
                raise ValueError(f"{name} must be a sha256 digest")
        if not self.access or len(self.access) != len(set(self.access)):
            raise ValueError("candidate access must be non-empty and unique")
        if any(not isinstance(item, AttachmentAccess) for item in self.access):
            raise TypeError("candidate access must contain AttachmentAccess values")
        if self.discovered_at_ns <= 0 or self.expires_at_ns <= self.discovered_at_ns:
            raise ValueError("candidate discovery lifetime is invalid")
        salience = float(self.proposal_salience)
        if not math.isfinite(salience) or not 0.0 <= salience <= 1.0:
            raise ValueError("proposal_salience must lie inside [0, 1]")
        object.__setattr__(self, "proposal_salience", salience)
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "connector_id": self.connector_id,
            "device_id": self.device_id,
            "display_name": self.display_name,
            "transport": self.transport,
            "identity_fingerprint": self.identity_fingerprint,
            "manifest_sha256": self.manifest_sha256,
            "access": [item.value for item in self.access],
            "discovered_at_ns": self.discovered_at_ns,
            "expires_at_ns": self.expires_at_ns,
            "persistent_identity": self.persistent_identity,
            "privacy_sensitive": self.privacy_sensitive,
            "proposal_salience": self.proposal_salience,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ConnectionRequest:
    request_id: str
    candidate_id: str
    candidate_sha256: str
    requested_access: tuple[AttachmentAccess, ...]
    initiated_by: str
    reason: str
    state: ConnectionState
    created_at_ns: int
    updated_at_ns: int
    authority_receipt_id: str = ""
    adapter_id: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        for name in ("request_id", "candidate_id", "initiated_by"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))
        if not _DIGEST.fullmatch(self.candidate_sha256):
            raise ValueError("candidate_sha256 must be a sha256 digest")
        if not self.requested_access or len(self.requested_access) != len(
            set(self.requested_access)
        ):
            raise ValueError("requested access must be non-empty and unique")
        if any(not isinstance(item, AttachmentAccess) for item in self.requested_access):
            raise TypeError("requested access must contain AttachmentAccess values")
        if not isinstance(self.state, ConnectionState):
            raise TypeError("state must be a ConnectionState")
        if self.created_at_ns <= 0 or self.updated_at_ns < self.created_at_ns:
            raise ValueError("connection request timestamps are invalid")
        if len(self.reason) > 320 or len(self.error) > 320:
            raise ValueError("connection request text exceeds its bound")
        if self.adapter_id:
            object.__setattr__(self, "adapter_id", _identifier(self.adapter_id, name="adapter_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "candidate_sha256": self.candidate_sha256,
            "requested_access": [item.value for item in self.requested_access],
            "initiated_by": self.initiated_by,
            "reason": self.reason,
            "state": self.state.value,
            "created_at_ns": self.created_at_ns,
            "updated_at_ns": self.updated_at_ns,
            "authority_receipt_id": self.authority_receipt_id,
            "adapter_id": self.adapter_id,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class TrustGrant:
    identity_fingerprint: str
    connector_id: str
    manifest_sha256: str
    allowed_access: tuple[AttachmentAccess, ...]
    authority_receipt_id: str
    authority_intent: Mapping[str, Any]
    authority_evidence: Mapping[str, Any]
    private_device_metadata: Mapping[str, Any]
    issued_at_ns: int
    expires_at_ns: int
    persistent: bool

    def __post_init__(self) -> None:
        if not _DIGEST.fullmatch(self.identity_fingerprint):
            raise ValueError("identity_fingerprint must be a sha256 digest")
        if not _DIGEST.fullmatch(self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a sha256 digest")
        object.__setattr__(
            self, "connector_id", _identifier(self.connector_id, name="connector_id")
        )
        if not self.allowed_access or len(self.allowed_access) != len(set(self.allowed_access)):
            raise ValueError("allowed_access must be non-empty and unique")
        if not str(self.authority_receipt_id or "").strip():
            raise ValueError("authority_receipt_id must be present")
        if self.issued_at_ns <= 0 or self.expires_at_ns <= self.issued_at_ns:
            raise ValueError("trust grant lifetime is invalid")
        object.__setattr__(
            self,
            "authority_intent",
            _frozen_metadata(self.authority_intent),
        )
        object.__setattr__(
            self,
            "authority_evidence",
            _frozen_metadata(self.authority_evidence),
        )
        object.__setattr__(
            self,
            "private_device_metadata",
            _frozen_metadata(self.private_device_metadata),
        )

    def is_valid_at(self, now_ns: int) -> bool:
        return self.issued_at_ns <= now_ns < self.expires_at_ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_fingerprint": self.identity_fingerprint,
            "connector_id": self.connector_id,
            "manifest_sha256": self.manifest_sha256,
            "allowed_access": [item.value for item in self.allowed_access],
            "authority_receipt_id": self.authority_receipt_id,
            "authority_intent": dict(self.authority_intent),
            "authority_evidence": dict(self.authority_evidence),
            "private_device_metadata": dict(self.private_device_metadata),
            "issued_at_ns": self.issued_at_ns,
            "expires_at_ns": self.expires_at_ns,
            "persistent": self.persistent,
        }


@runtime_checkable
class DeviceConnector(Protocol):
    @property
    def connector_id(self) -> str: ...

    async def discover(self) -> tuple[DeviceCandidate, ...]: ...

    async def attach(
        self,
        candidate: DeviceCandidate,
        access: tuple[AttachmentAccess, ...],
    ) -> LiveChannelAdapter: ...

    async def detach(self, adapter: LiveChannelAdapter) -> None: ...


class DeviceAttachmentBroker:
    """Lifecycle owner for portable physical relationships."""

    def __init__(
        self,
        service: RealityReachService,
        observation_router: RealityObservationRouter,
        *,
        digital_twin: RealityDigitalTwinGraph | None = None,
        middleware: Any | None = None,
        state_path: Path | None = None,
        trust_store: AttachmentTrustStore | None = None,
        trust_store_error: str = "",
        authority_verifier: PhysicalAuthorityVerifier | None = None,
        attachment_observer: Callable[[str], None] | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        discovery_interval_s: float = 60.0,
        connector_timeout_s: float = 12.0,
        max_candidates: int = 2048,
        max_pending_proposals: int = 8,
        disappearance_quorum: int = _DEFAULT_DISAPPEARANCE_QUORUM,
    ) -> None:
        if not isinstance(service, RealityReachService):
            raise TypeError("service must be a RealityReachService")
        if not isinstance(observation_router, RealityObservationRouter):
            raise TypeError("observation_router must be a RealityObservationRouter")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        if attachment_observer is not None and not callable(attachment_observer):
            raise TypeError("attachment_observer must be callable")
        self._service = service
        self._router = observation_router
        if digital_twin is not None and not isinstance(digital_twin, RealityDigitalTwinGraph):
            raise TypeError("digital_twin must be a RealityDigitalTwinGraph")
        self._digital_twin = digital_twin
        if middleware is not None and (
            not callable(getattr(middleware, "register_adapter", None))
            or not callable(getattr(middleware, "unregister_adapter", None))
        ):
            raise TypeError("middleware must expose managed adapter lifecycle methods")
        self._middleware = middleware
        self._trust_store: AttachmentTrustStore | None = trust_store
        self._custody_error = str(trust_store_error or "")[:320]
        if self._trust_store is None and not self._custody_error:
            self._custody_error = "attachment_trust_custody_not_provisioned"
        if self._trust_store is not None and not isinstance(
            self._trust_store,
            AttachmentTrustStore,
        ):
            raise TypeError("trust_store must satisfy AttachmentTrustStore")
        self._authority_verifier = authority_verifier or AttachmentCapabilityAuthorityVerifier()
        self._attachment_observer = attachment_observer
        if not isinstance(self._authority_verifier, PhysicalAuthorityVerifier):
            raise TypeError("authority_verifier must satisfy PhysicalAuthorityVerifier")
        if self._digital_twin is not None and isinstance(
            self._authority_verifier,
            ManifestMigrationAuthorityVerifier,
        ):
            self._digital_twin.bind_migration_authority_verifier(self._authority_verifier)
        self._clock_ns = clock_ns
        self._discovery_interval_s = max(5.0, min(float(discovery_interval_s), 3600.0))
        self._connector_timeout_s = max(1.0, min(float(connector_timeout_s), 60.0))
        self._max_candidates = max(1, min(int(max_candidates), 10_000))
        self._max_pending_proposals = max(0, min(int(max_pending_proposals), 64))
        if isinstance(disappearance_quorum, bool) or not isinstance(
            disappearance_quorum,
            int,
        ):
            raise TypeError("disappearance_quorum must be an integer")
        if not 2 <= disappearance_quorum <= _MAX_DISAPPEARANCE_QUORUM:
            raise ValueError(
                f"disappearance_quorum must lie inside [2, {_MAX_DISAPPEARANCE_QUORUM}]"
            )
        self._disappearance_quorum = disappearance_quorum
        self._connectors: dict[str, DeviceConnector] = {}
        self._candidates: dict[str, DeviceCandidate] = {}
        self._candidate_absence_streaks: dict[str, int] = {}
        self._requests: dict[str, ConnectionRequest] = {}
        self._grants: dict[str, TrustGrant] = {}
        self._pending_grant_activations: set[str] = set()
        self._attached: dict[str, tuple[str, LiveChannelAdapter]] = {}
        self._attached_candidates: dict[str, DeviceCandidate] = {}
        self._body_projections: dict[str, PhysicalBodyProjection] = {}
        self._managed_nodes: dict[str, str] = {}
        # A pending entry retains broker ownership until local and remote
        # fencing both complete. The bool records whether remote detach already
        # succeeded, avoiding a duplicate physical command on retry.
        self._teardown_pending: dict[str, bool] = {}
        self._pending_twin_detaches: dict[str, tuple[str, bool]] = {}
        self._pending_manifest_migrations: dict[str, Mapping[str, Any]] = {}
        self._pending_twin_revocations: dict[str, str] = {}
        self._lock = checked_lock("reality_attachment_broker.state", reentrant=True)
        self._state_load_lock = checked_async_lock("reality_attachment_broker.state_load")
        self._lifecycle_lock = checked_async_lock("reality_attachment_broker.lifecycle")
        self._persistence_lock = checked_async_lock("reality_attachment_broker.persistence")
        self._state_loaded = False
        self._wake = asyncio.Event()
        self._task: asyncio.Task[Any] | None = None
        self._running = False
        self._discoveries = 0
        self._attachment_failures = 0
        self._expired_grants = 0
        self._connector_discovery_failures: dict[str, int] = {}
        self._state_needs_compaction = False

    def register_connector(self, connector: DeviceConnector) -> None:
        if not isinstance(connector, DeviceConnector):
            raise TypeError("connector must satisfy DeviceConnector")
        connector_id = _identifier(connector.connector_id, name="connector_id")
        with self._lock:
            existing = self._connectors.get(connector_id)
            if existing is not None and existing is not connector:
                raise ValueError(f"connector already registered: {connector_id}")
            self._connectors[connector_id] = connector
        if self._running:
            self._wake.set()

    def unregister_connector(self, connector_id: str) -> None:
        """Remove an idle connector; active connectors require async retirement."""

        canonical = _identifier(connector_id, name="connector_id")
        if self._lifecycle_lock.locked():
            raise RuntimeError("connector lifecycle is busy; use and await retire_connector()")
        with self._lock:
            if canonical not in self._connectors:
                raise LookupError(f"connector is not registered: {canonical}")
            active_request_ids = self._connector_active_request_ids_locked(canonical)
            if active_request_ids:
                raise RuntimeError(
                    "connector has active attachments; use and await retire_connector()"
                )
            self._remove_idle_connector_locked(
                canonical,
                reason="connector_unregistered",
            )
        if self._running:
            self._wake.set()

    async def retire_connector(
        self,
        connector_id: str,
        *,
        reason: str = "connector_retired",
    ) -> None:
        """Fence every relationship before removing a connector.

        Retirement is cancellation-safe: once admitted, the bounded teardown
        transaction finishes under a shield before cancellation is propagated.
        """

        await self._ensure_state_loaded()
        canonical = _identifier(connector_id, name="connector_id")
        reason_text = str(reason or "connector_retired")[:320]
        retirement = get_task_tracker().create_task(
            self._retire_connector_transaction(canonical, reason=reason_text),
            name=f"RealityConnectorRetirement:{canonical}",
        )
        try:
            await asyncio.shield(retirement)
        except asyncio.CancelledError:
            await self._await_task_completion(retirement)
            raise
        if self._running:
            self._wake.set()

    async def discover(self) -> tuple[DeviceCandidate, ...]:
        await self._ensure_state_loaded()
        await self._reconcile_pending_teardowns()
        await self._reconcile_digital_twin()
        with self._lock:
            connectors = tuple(self._connectors.values())
        semaphore = checked_semaphore("attachments", 8)

        async def _one(
            connector: DeviceConnector,
        ) -> tuple[str, bool, tuple[DeviceCandidate, ...]]:
            async with semaphore:
                try:
                    found = await asyncio.wait_for(
                        connector.discover(),
                        timeout=self._connector_timeout_s,
                    )
                    if any(item.connector_id != connector.connector_id for item in found):
                        raise ValueError("connector returned a foreign candidate identity")
                    return connector.connector_id, True, found
                except asyncio.CancelledError:
                    raise
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                    record_degradation(
                        "reality_attachment.discovery",
                        exc,
                        action=f"retained other connectors after {connector.connector_id} failed",
                    )
                    return connector.connector_id, False, ()

        results = await asyncio.gather(*(_one(item) for item in connectors))
        successful_connectors = {
            connector_id for connector_id, succeeded, _found in results if succeeded
        }
        failed_connectors = {
            connector_id for connector_id, succeeded, _found in results if not succeeded
        }
        now_ns = self._clock_ns()
        observed = sorted(
            (
                candidate
                for _connector_id, succeeded, batch in results
                if succeeded
                for candidate in batch
                if candidate.expires_at_ns > now_ns
            ),
            key=lambda item: (-item.proposal_salience, item.candidate_id),
        )
        observed_by_id = {candidate.candidate_id: candidate for candidate in observed}
        observed_ids_by_connector: dict[str, set[str]] = {
            connector_id: set() for connector_id in successful_connectors
        }
        for candidate in observed:
            observed_ids_by_connector[candidate.connector_id].add(candidate.candidate_id)
        fresh = observed[: self._max_candidates]
        newly_discovered: list[DeviceCandidate] = []
        lost_requests: list[tuple[str, str]] = []
        with self._lock:
            previous_candidates = dict(self._candidates)
            previous_ids = set(self._candidates)
            retained: dict[str, DeviceCandidate] = {}
            for candidate_id, candidate in previous_candidates.items():
                if candidate_id in observed_by_id:
                    self._candidate_absence_streaks.pop(candidate_id, None)
                    continue
                if candidate.connector_id not in successful_connectors:
                    if candidate.expires_at_ns > now_ns:
                        retained[candidate_id] = candidate
                    continue
                streak = self._candidate_absence_streaks.get(candidate_id, 0) + 1
                self._candidate_absence_streaks[candidate_id] = streak
                if candidate.expires_at_ns > now_ns and streak < self._disappearance_quorum:
                    retained[candidate_id] = candidate
            retained.update(
                {item.candidate_id: item for item in fresh if item.expires_at_ns > now_ns}
            )
            self._candidates = dict(
                sorted(
                    retained.items(),
                    key=lambda item: (-item[1].proposal_salience, item[0]),
                )[: self._max_candidates]
            )
            for request_id, attached_candidate in tuple(self._attached_candidates.items()):
                refreshed = observed_by_id.get(attached_candidate.candidate_id)
                if refreshed is not None:
                    self._attached_candidates[request_id] = refreshed
                    self._candidate_absence_streaks.pop(
                        attached_candidate.candidate_id,
                        None,
                    )
                    continue
                expired = attached_candidate.expires_at_ns <= now_ns
                connector_succeeded = attached_candidate.connector_id in successful_connectors
                missing_from_full_scan = (
                    attached_candidate.candidate_id
                    not in observed_ids_by_connector.get(
                        attached_candidate.connector_id,
                        set(),
                    )
                )
                if (
                    connector_succeeded
                    and missing_from_full_scan
                    and attached_candidate.candidate_id not in previous_candidates
                ):
                    self._candidate_absence_streaks[attached_candidate.candidate_id] = (
                        self._candidate_absence_streaks.get(
                            attached_candidate.candidate_id,
                            0,
                        )
                        + 1
                    )
                streak = self._candidate_absence_streaks.get(
                    attached_candidate.candidate_id,
                    0,
                )
                if expired:
                    lost_requests.append((request_id, "candidate_discovery_lease_expired"))
                elif (
                    connector_succeeded
                    and missing_from_full_scan
                    and streak >= self._disappearance_quorum
                ):
                    lost_requests.append((request_id, "candidate_disappearance_quorum_reached"))
            newly_discovered = [item for item in fresh if item.candidate_id not in previous_ids]
            self._discoveries += len(newly_discovered)
            for connector_id in successful_connectors:
                self._connector_discovery_failures.pop(connector_id, None)
            for connector_id in failed_connectors:
                self._connector_discovery_failures[connector_id] = (
                    self._connector_discovery_failures.get(connector_id, 0) + 1
                )
            live_candidate_ids = set(self._candidates)
            live_candidate_ids.update(
                candidate.candidate_id for candidate in self._attached_candidates.values()
            )
            self._candidate_absence_streaks = {
                candidate_id: streak
                for candidate_id, streak in self._candidate_absence_streaks.items()
                if candidate_id in live_candidate_ids
            }
            current_candidates = tuple(self._candidates.values())
        if self._digital_twin is not None:
            for candidate in fresh:
                await asyncio.to_thread(self._digital_twin.observe_candidate, candidate)
        for request_id, loss_reason in lost_requests:
            await self._detach_request(
                request_id,
                state=ConnectionState.LOST,
                error=loss_reason,
                lost=True,
            )
        await self._restore_trusted_connections()
        await self._propose_new_connections(newly_discovered)
        return current_candidates

    async def request_connection(
        self,
        candidate_id: str,
        *,
        requested_access: tuple[AttachmentAccess, ...] = (AttachmentAccess.OBSERVE,),
        initiated_by: str = "aura",
        reason: str = "new physical capability discovered",
    ) -> ConnectionRequest:
        await self._ensure_state_loaded()
        canonical = _identifier(candidate_id, name="candidate_id")
        with self._lock:
            candidate = self._candidates.get(canonical)
        if candidate is None or candidate.expires_at_ns <= self._clock_ns():
            raise LookupError(f"candidate is not currently discoverable: {canonical}")
        requested = tuple(dict.fromkeys(requested_access))
        if not requested or not set(requested).issubset(set(candidate.access)):
            raise PermissionError("requested attachment access is not declared by candidate")
        candidate_sha256 = _digest(candidate.to_dict())
        request_id = (
            "reality.connect."
            + _digest(
                {
                    "candidate": candidate_sha256,
                    "access": [item.value for item in requested],
                }
            ).removeprefix("sha256:")[:32]
        )
        with self._lock:
            existing = self._requests.get(request_id)
            if existing is not None and existing.state not in {
                ConnectionState.ERROR,
                ConnectionState.LOST,
                ConnectionState.REVOKED,
            }:
                return existing
            grant = self._grants.get(candidate.identity_fingerprint)
            current_time_ns = self._clock_ns()
            if grant is not None and not grant.is_valid_at(current_time_ns):
                self._grants.pop(candidate.identity_fingerprint, None)
                self._expired_grants += 1
                grant = None
        trusted = bool(
            grant is not None
            and grant.is_valid_at(current_time_ns)
            and grant.connector_id == candidate.connector_id
            and grant.manifest_sha256 == candidate.manifest_sha256
            and set(requested).issubset(set(grant.allowed_access))
        )
        now_ns = max(1, current_time_ns)
        request = ConnectionRequest(
            request_id=request_id,
            candidate_id=candidate.candidate_id,
            candidate_sha256=candidate_sha256,
            requested_access=requested,
            initiated_by=initiated_by,
            reason=str(reason or "")[:320],
            state=ConnectionState.ATTACHING if trusted else ConnectionState.PENDING_TRUST,
            created_at_ns=now_ns,
            updated_at_ns=now_ns,
            authority_receipt_id=grant.authority_receipt_id if grant is not None else "",
        )
        with self._lock:
            self._requests[request_id] = request
        if trusted:
            return await self._attach(request)
        await self._announce_request(candidate, request)
        return request

    def authority_intent(
        self,
        request_id: str,
        *,
        persistent: bool,
        grant_ttl_s: int | None = None,
    ) -> dict[str, Any]:
        """Return the deterministic payload the Will must authorize."""

        canonical = _identifier(request_id, name="request_id")
        with self._lock:
            request = self._requests.get(canonical)
            candidate = self._candidates.get(request.candidate_id) if request else None
        if request is None or candidate is None:
            raise LookupError("connection request or candidate is unavailable")
        if request.state != ConnectionState.PENDING_TRUST:
            raise RuntimeError(f"connection request is not pending trust: {request.state.value}")
        if persistent and not candidate.persistent_identity:
            raise PermissionError("candidate identity is not stable enough for persistent trust")
        ttl_s = _bounded_grant_ttl_s(
            request.requested_access,
            persistent=persistent,
            requested_ttl_s=grant_ttl_s,
        )
        intent = build_attachment_authority_intent(
            request_id=request.request_id,
            candidate_sha256=request.candidate_sha256,
            identity_fingerprint=candidate.identity_fingerprint,
            connector_id=candidate.connector_id,
            manifest_sha256=candidate.manifest_sha256,
            requested_access=tuple(item.value for item in request.requested_access),
            persistent=persistent,
            grant_ttl_s=ttl_s,
        )
        if not isinstance(intent, dict):
            raise TypeError("attachment authority intent is not an object")
        return intent

    def manifest_migration_intent(self, request_id: str) -> dict[str, Any] | None:
        canonical = _identifier(request_id, name="request_id")
        with self._lock:
            request = self._requests.get(canonical)
            candidate = self._candidates.get(request.candidate_id) if request else None
        if request is None or candidate is None:
            raise LookupError("connection request or candidate is unavailable")
        if request.state != ConnectionState.PENDING_TRUST:
            raise RuntimeError(f"connection request is not pending trust: {request.state.value}")
        if self._digital_twin is None:
            return None
        intent = self._digital_twin.manifest_migration_intent(
            candidate,
            request_id=request.request_id,
        )
        if intent is not None and not isinstance(intent, dict):
            raise TypeError("digital twin migration intent is not an object")
        return intent

    async def authorize_and_attach(
        self,
        request_id: str,
        *,
        authority_capability: Mapping[str, Any],
        manifest_migration_capability: Mapping[str, Any] | None = None,
        persistent: bool = True,
        grant_ttl_s: int | None = None,
    ) -> ConnectionRequest:
        await self._ensure_state_loaded()
        canonical = _identifier(request_id, name="request_id")
        with self._lock:
            request = self._requests.get(canonical)
            candidate = self._candidates.get(request.candidate_id) if request else None
        if request is None or candidate is None:
            raise LookupError("connection request or candidate is unavailable")
        if request.state != ConnectionState.PENDING_TRUST:
            raise RuntimeError(f"connection request is not pending trust: {request.state.value}")
        if persistent and not candidate.persistent_identity:
            raise PermissionError("candidate identity is not stable enough for persistent trust")
        if persistent and (self._trust_store is None or self._custody_error):
            raise AttachmentTrustStoreError(
                "attachment_trust_custody_unhealthy_for_persistent_grant"
            )
        migration_intent = self.manifest_migration_intent(request.request_id)
        if migration_intent is not None:
            if not isinstance(
                self._authority_verifier,
                ManifestMigrationAuthorityVerifier,
            ):
                raise AttachmentAuthorityError("manifest_migration_authority_verifier_unavailable")
            if manifest_migration_capability is None:
                raise AttachmentAuthorityError("manifest_migration_authority_capability_missing")
        intent = self.authority_intent(
            request.request_id,
            persistent=persistent,
            grant_ttl_s=grant_ttl_s,
        )
        authority_evidence = self._authority_verifier.verify(
            authority_capability,
            intent=intent,
            persistent=persistent,
        )
        capability = authority_evidence.get("capability")
        authority_receipt_id = (
            str(capability.get("receipt_id") or "") if isinstance(capability, Mapping) else ""
        )
        if not authority_receipt_id:
            raise AttachmentAuthorityError("attachment_authority_receipt_missing")
        migration_evidence: Mapping[str, Any] | None = None
        if migration_intent is not None:
            assert manifest_migration_capability is not None
            migration_verifier = cast(
                ManifestMigrationAuthorityVerifier,
                self._authority_verifier,
            )
            migration_evidence = migration_verifier.verify_manifest_migration(
                manifest_migration_capability,
                intent=migration_intent,
                persistent=bool(migration_intent["persistent"]),
            )
        issued_at_ns = max(1, self._clock_ns())
        ttl_s = int(intent["grant_ttl_s"])
        grant = TrustGrant(
            identity_fingerprint=candidate.identity_fingerprint,
            connector_id=candidate.connector_id,
            manifest_sha256=candidate.manifest_sha256,
            allowed_access=request.requested_access,
            authority_receipt_id=authority_receipt_id[:192],
            authority_intent=intent,
            authority_evidence=authority_evidence,
            private_device_metadata={
                "device_id": candidate.device_id,
                "display_name": candidate.display_name,
                "transport": candidate.transport,
                "privacy_sensitive": candidate.privacy_sensitive,
                "metadata": dict(candidate.metadata),
            },
            issued_at_ns=issued_at_ns,
            expires_at_ns=issued_at_ns + ttl_s * 1_000_000_000,
            persistent=bool(persistent),
        )
        with self._lock:
            previous_grant = self._grants.get(grant.identity_fingerprint)
            previous_request = self._requests[request.request_id]
            self._grants[grant.identity_fingerprint] = grant
            if persistent:
                # The first durable write is intentionally non-replayable. A
                # crash or cancellation can never turn an incomplete physical
                # transaction into authority after restart.
                self._pending_grant_activations.add(grant.identity_fingerprint)
            self._requests[request.request_id] = replace(
                request,
                state=ConnectionState.ATTACHING,
                authority_receipt_id=grant.authority_receipt_id,
                updated_at_ns=max(request.created_at_ns, self._clock_ns()),
            )
            attaching = self._requests[request.request_id]
            if migration_evidence is not None:
                self._pending_manifest_migrations[request.request_id] = migration_evidence
        try:
            if persistent:
                persistence = get_task_tracker().create_task(
                    self._persist_state(),
                    name=f"RealityAttachmentAuthorityPersist:{request.request_id}",
                )
                try:
                    await asyncio.shield(persistence)
                except asyncio.CancelledError:
                    try:
                        await self._await_task_completion(persistence)
                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                        record_degradation(
                            "reality_attachment.cancelled_authority_persist",
                            exc,
                            action=(
                                "completed the interrupted persistence attempt before "
                                "rolling authority back"
                            ),
                        )
                    rollback = get_task_tracker().create_task(
                        self._rollback_cancelled_authorization(
                            request=previous_request,
                            grant=grant,
                            previous_grant=previous_grant,
                        ),
                        name=(f"RealityAttachmentAuthorityPersistRollback:{request.request_id}"),
                    )
                    await self._await_task_completion(rollback)
                    with self._lock:
                        self._pending_manifest_migrations.pop(request.request_id, None)
                    raise
        except (OSError, RuntimeError, TypeError, ValueError):
            with self._lock:
                self._pending_grant_activations.discard(grant.identity_fingerprint)
                if previous_grant is None:
                    self._grants.pop(grant.identity_fingerprint, None)
                else:
                    self._grants[grant.identity_fingerprint] = previous_grant
                self._requests[request.request_id] = previous_request
                self._pending_manifest_migrations.pop(request.request_id, None)
            raise
        try:
            result = await self._attach(attaching)
            if persistent and result.state == ConnectionState.ATTACHED:
                with self._lock:
                    self._pending_grant_activations.discard(grant.identity_fingerprint)
                activation = get_task_tracker().create_task(
                    self._persist_state(),
                    name=f"RealityAttachmentAuthorityActivate:{request.request_id}",
                )
                try:
                    await asyncio.shield(activation)
                except asyncio.CancelledError:
                    # Durable activation is the transaction's linearization
                    # point. Once committed, a late caller cancellation must
                    # not manufacture a rollback that can resurrect authority.
                    await self._await_task_completion(activation)
                    return result
                except (OSError, RuntimeError, TypeError, ValueError):
                    with self._lock:
                        self._pending_grant_activations.add(grant.identity_fingerprint)
                    await self._detach_request(
                        request.request_id,
                        state=ConnectionState.ERROR,
                        error="persistent_authority_activation_failed",
                        degradation_phase=("reality_attachment.authority_activation"),
                    )
                    await self._rollback_cancelled_authorization(
                        request=previous_request,
                        grant=grant,
                        previous_grant=previous_grant,
                        error="persistent_authority_activation_failed",
                    )
                    raise
            elif persistent:
                await self._rollback_cancelled_authorization(
                    request=previous_request,
                    grant=grant,
                    previous_grant=previous_grant,
                    error="attachment_failed",
                )
            return result
        except asyncio.CancelledError:
            rollback = get_task_tracker().create_task(
                self._rollback_cancelled_authorization(
                    request=previous_request,
                    grant=grant,
                    previous_grant=previous_grant,
                ),
                name=f"RealityAttachmentAuthorityRollback:{request.request_id}",
            )
            await self._await_task_completion(rollback)
            raise
        finally:
            with self._lock:
                self._pending_manifest_migrations.pop(request.request_id, None)

    async def revoke(self, identity_fingerprint: str, *, reason: str) -> None:
        await self._ensure_state_loaded()
        if not _DIGEST.fullmatch(identity_fingerprint):
            raise ValueError("identity_fingerprint must be a sha256 digest")
        reason_text = _reconciliation_reason(reason, fallback="revoked")
        with self._lock:
            previous_grant = self._grants.pop(identity_fingerprint, None)
            previous_reconciliation = self._pending_twin_revocations.get(identity_fingerprint)
            if self._digital_twin is not None:
                self._pending_twin_revocations[identity_fingerprint] = reason_text
            attached = [
                (request_id, adapter_id, adapter, candidate)
                for request_id, (adapter_id, adapter) in self._attached.items()
                if (candidate := self._attached_candidates.get(request_id)) is not None
                and candidate.identity_fingerprint == identity_fingerprint
            ]
        try:
            if (
                previous_grant is not None and previous_grant.persistent
            ) or self._digital_twin is not None:
                await self._persist_state()
        except (OSError, RuntimeError, TypeError, ValueError):
            with self._lock:
                if previous_grant is not None:
                    self._grants[identity_fingerprint] = previous_grant
                if previous_reconciliation is None:
                    self._pending_twin_revocations.pop(identity_fingerprint, None)
                else:
                    self._pending_twin_revocations[identity_fingerprint] = previous_reconciliation
            raise
        for request_id, _adapter_id, _adapter, _candidate in attached:
            await self._detach_request(
                request_id,
                state=ConnectionState.REVOKED,
                error=str(reason or "revoked")[:320],
                degradation_phase="reality_attachment.revoke_detach",
            )
        if self._digital_twin is not None:
            await self._reconcile_digital_twin()
        await asyncio.to_thread(self._service.refresh)

    async def start(self) -> None:
        if self._running:
            return
        await self._ensure_state_loaded()
        await self._reconcile_pending_teardowns()
        await self._reconcile_digital_twin()
        self._running = True
        self._task = get_task_tracker().create_task(
            self._discovery_loop(),
            name="RealityAttachmentBroker",
        )

    async def stop(self) -> None:
        self._running = False
        self._wake.set()
        if self._task is not None:
            await cancel_and_join(self._task, owner="core.reality_reach.attachments")
            self._task = None
        with self._lock:
            attached_request_ids = list(self._attached)
        for request_id in attached_request_ids:
            await self._detach_request(
                request_id,
                state=ConnectionState.LOST,
                error="runtime_stopped",
                lost=True,
            )
        with self._lock:
            residual = tuple(sorted(self._attached))
        if residual:
            record_degradation(
                "reality_attachment.stop",
                RuntimeError("attachment_teardown_incomplete"),
                action="retained broker ownership for explicit teardown recovery",
            )
        await asyncio.to_thread(self._service.refresh)

    async def _discovery_loop(self) -> None:
        while self._running:
            try:
                await self.discover()
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                record_degradation(
                    "reality_attachment.loop",
                    exc,
                    action="continued physical discovery after one bounded scan failed",
                )
            try:
                await asyncio.wait_for(
                    self._wake.wait(),
                    timeout=self._discovery_interval_s,
                )
                self._wake.clear()
            # not a failure: the wait ran out, which is what the timeout was set to decide.
            except TimeoutError:
                pass

    async def _restore_trusted_connections(self) -> None:
        now_ns = self._clock_ns()
        with self._lock:
            expired = [
                identity
                for identity, grant in self._grants.items()
                if not grant.is_valid_at(now_ns)
            ]
            for identity in expired:
                self._grants.pop(identity, None)
            self._expired_grants += len(expired)
            candidates = tuple(self._candidates.values())
            active_candidates = {
                self._requests[request_id].candidate_id
                for request_id in self._attached
                if request_id in self._requests
            }
        for candidate in candidates:
            grant = self._grants.get(candidate.identity_fingerprint)
            if grant is None or candidate.candidate_id in active_candidates:
                continue
            with self._lock:
                activation_pending = (
                    candidate.identity_fingerprint in self._pending_grant_activations
                )
            if activation_pending:
                continue
            if (
                not grant.persistent
                or not grant.is_valid_at(now_ns)
                or grant.connector_id != candidate.connector_id
                or grant.manifest_sha256 != candidate.manifest_sha256
            ):
                continue
            await self.request_connection(
                candidate.candidate_id,
                requested_access=grant.allowed_access,
                initiated_by="aura.migration",
                reason="reattach trusted physical capability after runtime discovery",
            )
        if expired or self._state_needs_compaction:
            try:
                await self._persist_state()
            except (OSError, RuntimeError, TypeError, ValueError):
                # The expired grant has already been removed from the live
                # authority set.  Persistence stays visibly unhealthy, but a
                # Keychain outage must not blind bounded physical sensing.
                pass

    async def _propose_new_connections(
        self,
        candidates: list[DeviceCandidate],
    ) -> None:
        with self._lock:
            pending_count = sum(
                item.state == ConnectionState.PENDING_TRUST for item in self._requests.values()
            )
        remaining = max(0, self._max_pending_proposals - pending_count)
        for candidate in candidates[:remaining]:
            try:
                await self.request_connection(
                    candidate.candidate_id,
                    requested_access=(AttachmentAccess.OBSERVE,),
                    initiated_by="aura",
                    reason="I discovered a physical capability and would like to connect to its bounded sensory surface.",
                )
            except (LookupError, PermissionError, RuntimeError, TypeError, ValueError):
                continue

    async def _attach(self, request: ConnectionRequest) -> ConnectionRequest:
        transaction = get_task_tracker().create_task(
            self._attach_transaction(request),
            name=f"RealityAttachmentTransaction:{request.request_id}",
        )
        try:
            return await asyncio.shield(transaction)
        except asyncio.CancelledError:
            result = await self._await_task_completion(transaction)
            if isinstance(result, ConnectionRequest) and result.state == ConnectionState.ATTACHED:
                rollback = get_task_tracker().create_task(
                    self._detach_request(
                        request.request_id,
                        state=ConnectionState.ERROR,
                        error="attachment_cancelled",
                        degradation_phase="reality_attachment.cancelled_attach",
                    ),
                    name=f"RealityAttachmentCancellationRollback:{request.request_id}",
                )
                await self._await_task_completion(rollback)
            elif (
                isinstance(result, ConnectionRequest) and result.state == ConnectionState.ATTACHING
            ):
                await self._fail_request(result, "attachment_cancelled")
            raise

    async def _attach_transaction(self, request: ConnectionRequest) -> ConnectionRequest:
        async with self._lifecycle_lock:
            return await self._attach_transaction_locked(request)

    async def _attach_transaction_locked(
        self,
        request: ConnectionRequest,
    ) -> ConnectionRequest:
        with self._lock:
            candidate = self._candidates.get(request.candidate_id)
            connector = self._connectors.get(candidate.connector_id) if candidate else None
            migration_evidence = self._pending_manifest_migrations.get(request.request_id)
        if candidate is None or connector is None:
            return await self._fail_request(request, "candidate_or_connector_unavailable")
        adapter: LiveChannelAdapter | None = None
        service_registered = False
        managed_registered = False
        sampler_registered = False
        body_projected = False
        twin_attached = False
        try:
            adapter = await asyncio.wait_for(
                connector.attach(candidate, request.requested_access),
                timeout=self._connector_timeout_s,
            )
            if adapter is None:
                raise TypeError("connector returned no Reality Reach adapter")
            adapter_id = str(adapter.adapter_id)
            await self._reconcile_digital_twin_unlocked(adapter_id=adapter_id)
            with self._lock:
                twin_detach_pending = adapter_id in self._pending_twin_detaches
            if twin_detach_pending:
                raise RuntimeError("digital_twin_detach_reconciliation_pending_for_adapter")
            self._service.register_adapter(adapter)
            service_registered = True
            if isinstance(adapter, ManagedRealityAdapter):
                if self._middleware is None:
                    raise RuntimeError("managed_physical_runtime_unavailable")
                declaration = adapter.lifecycle_declaration()
                await self._middleware.register_adapter(adapter)
                with self._lock:
                    self._managed_nodes[adapter_id] = declaration.node_id
                managed_registered = True
            try:
                self._router.register_sampler(adapter)
                sampler_registered = True
            except (TypeError, ValueError):
                # Push-only and synchronous adapters remain valid Reality Reach
                # channels; they simply do not use the async sampling hook.
                pass
            projection = project_adapter_to_body(
                adapter,
                device_id=candidate.device_id,
                display_name=candidate.display_name,
                transport=candidate.transport,
                privacy_sensitive=candidate.privacy_sensitive,
                persistent_identity=candidate.persistent_identity,
                manifest_sha256=candidate.manifest_sha256,
            )
            self._body_projections[adapter_id] = projection
            body_projected = True
            if self._digital_twin is not None:
                twin_receipt = await asyncio.to_thread(
                    self._digital_twin.attach_adapter,
                    candidate,
                    adapter,
                    body_projection=projection,
                    migration_request_id=request.request_id,
                    migration_authority_evidence=migration_evidence,
                )
                if not twin_receipt.accepted:
                    raise RuntimeError(
                        f"digital_twin_attach_rejected:{twin_receipt.disposition.value}"
                    )
                twin_attached = True
            await asyncio.to_thread(self._service.refresh)
            now_ns = max(request.created_at_ns, self._clock_ns())
            attached = replace(
                request,
                state=ConnectionState.ATTACHED,
                adapter_id=str(adapter.adapter_id),
                updated_at_ns=now_ns,
                error="",
            )
            with self._lock:
                self._requests[request.request_id] = attached
                self._attached[request.request_id] = (adapter_id, adapter)
                self._attached_candidates[request.request_id] = candidate
            if self._attachment_observer is not None:
                try:
                    self._attachment_observer(adapter_id)
                except Exception as exc:  # noqa: BLE001 - observer cannot own attachment fate
                    record_degradation(
                        "reality_attachment.recovery_notification",
                        exc,
                        action=(
                            "retained the committed physical attachment and left pending "
                            "effects closed for the recovery supervisor's bounded retry"
                        ),
                    )
            return attached
        except Exception as exc:  # noqa: BLE001 - transaction rollback precedes classification
            if adapter is not None:
                await self._rollback_partial_attachment(
                    request=request,
                    candidate=candidate,
                    connector=connector,
                    adapter=adapter,
                    service_registered=service_registered,
                    managed_registered=managed_registered,
                    sampler_registered=sampler_registered,
                    body_projected=body_projected,
                    twin_attached=twin_attached,
                    reason="attachment_transaction_rollback",
                )
            self._attachment_failures += 1
            return await self._fail_request(
                request,
                f"{type(exc).__name__}:{exc}"[:320],
            )

    async def _detach_request(
        self,
        request_id: str,
        *,
        state: ConnectionState,
        error: str,
        lost: bool = False,
        degradation_phase: str = "reality_attachment.detach",
    ) -> bool:
        if state not in {
            ConnectionState.LOST,
            ConnectionState.REVOKED,
            ConnectionState.ERROR,
        }:
            raise ValueError("detach state is not terminal")
        async with self._lifecycle_lock:
            return await self._detach_request_unlocked(
                request_id,
                state=state,
                error=error,
                lost=lost,
                degradation_phase=degradation_phase,
            )

    async def _detach_request_unlocked(
        self,
        request_id: str,
        *,
        state: ConnectionState,
        error: str,
        lost: bool = False,
        degradation_phase: str = "reality_attachment.detach",
        require_remote_detach: bool = False,
    ) -> bool:
        with self._lock:
            attached = self._attached.get(request_id)
            request = self._requests.get(request_id)
            candidate = self._attached_candidates.get(request_id)
        if attached is None or request is None:
            return True
        adapter_id, adapter = attached
        connector = self._connectors.get(candidate.connector_id) if candidate else None
        with self._lock:
            remote_already_detached = self._teardown_pending.get(request_id, False)
        local_fenced, remote_detached = await self._fence_adapter_locally(
            connector=None if remote_already_detached else connector,
            adapter=adapter,
            adapter_id=adapter_id,
            service_registered=True,
            sampler_registered=True,
            body_projected=True,
            degradation_phase=degradation_phase,
        )
        remote_detached = remote_already_detached or remote_detached
        if not local_fenced or (require_remote_detach and not remote_detached):
            with self._lock:
                self._teardown_pending[request_id] = remote_detached
                current = self._requests.get(request_id)
                if current is not None:
                    self._requests[request_id] = replace(
                        current,
                        state=ConnectionState.ERROR,
                        updated_at_ns=max(current.created_at_ns, self._clock_ns()),
                        error=(
                            "local_adapter_fencing_pending"
                            if not local_fenced
                            else "remote_connector_teardown_pending"
                        ),
                    )
            record_degradation(
                degradation_phase,
                RuntimeError(
                    "adapter_fencing_incomplete"
                    if not local_fenced
                    else "remote_connector_teardown_incomplete"
                ),
                action=(
                    "retained connector ownership and attachment bookkeeping for "
                    "a bounded teardown retry"
                ),
            )
            return False
        with self._lock:
            current = self._requests.get(request_id)
            if current is not None:
                self._requests[request_id] = replace(
                    current,
                    state=state,
                    updated_at_ns=max(current.created_at_ns, self._clock_ns()),
                    error=str(error or state.value)[:320],
                )
            self._attached.pop(request_id, None)
            self._attached_candidates.pop(request_id, None)
            self._teardown_pending.pop(request_id, None)
            if candidate is not None:
                self._candidate_absence_streaks.pop(candidate.candidate_id, None)
        await self._detach_twin_or_queue(
            adapter_id,
            reason=str(error or state.value)[:160],
            lost=lost,
            degradation_phase=degradation_phase,
        )
        return True

    async def _rollback_partial_attachment(
        self,
        *,
        request: ConnectionRequest,
        candidate: DeviceCandidate,
        connector: DeviceConnector,
        adapter: LiveChannelAdapter,
        service_registered: bool,
        managed_registered: bool,
        sampler_registered: bool,
        body_projected: bool,
        twin_attached: bool,
        reason: str,
    ) -> None:
        adapter_id = str(getattr(adapter, "adapter_id", "") or "")
        local_fenced, remote_detached = await self._fence_adapter_locally(
            connector=connector,
            adapter=adapter,
            adapter_id=adapter_id,
            service_registered=service_registered,
            managed_registered=managed_registered,
            sampler_registered=sampler_registered,
            body_projected=body_projected,
            degradation_phase="reality_attachment.rollback",
        )
        if not local_fenced or not remote_detached:
            with self._lock:
                self._attached[request.request_id] = (adapter_id, adapter)
                self._attached_candidates[request.request_id] = candidate
                self._teardown_pending[request.request_id] = remote_detached
            record_degradation(
                "reality_attachment.rollback",
                RuntimeError("partial_attachment_fencing_incomplete"),
                action=("retained adapter ownership and queued bounded teardown retry"),
            )
        if twin_attached and local_fenced:
            try:
                await self._detach_twin_or_queue(
                    adapter_id,
                    reason=reason,
                    lost=False,
                    degradation_phase="reality_attachment.rollback",
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "reality_attachment.rollback.digital_twin_persistence",
                    exc,
                    action=(
                        "retained the rollback obligation and completed request failure accounting"
                    ),
                )

    async def _reconcile_pending_teardowns(self) -> None:
        async with self._lifecycle_lock:
            with self._lock:
                pending = tuple(sorted(self._teardown_pending))
            for request_id in pending:
                with self._lock:
                    request = self._requests.get(request_id)
                if request is None:
                    continue
                await self._detach_request_unlocked(
                    request_id,
                    state=ConnectionState.ERROR,
                    error=request.error or "attachment_rollback_retry",
                    degradation_phase="reality_attachment.rollback_retry",
                    require_remote_detach=True,
                )

    async def _fence_adapter_locally(
        self,
        *,
        connector: DeviceConnector | None,
        adapter: LiveChannelAdapter,
        adapter_id: str,
        service_registered: bool,
        managed_registered: bool | None = None,
        sampler_registered: bool,
        body_projected: bool,
        degradation_phase: str,
    ) -> tuple[bool, bool]:
        with self._lock:
            managed_node_id = self._managed_nodes.get(adapter_id)
        should_fence_managed = bool(
            managed_registered if managed_registered is not None else managed_node_id
        )
        managed_fenced = not should_fence_managed
        if should_fence_managed and self._middleware is not None:
            try:
                if managed_node_id is None and isinstance(adapter, ManagedRealityAdapter):
                    managed_node_id = adapter.lifecycle_declaration().node_id
                if managed_node_id is None:
                    raise RuntimeError("managed_node_identity_missing")
                await self._middleware.unregister_adapter(
                    managed_node_id,
                    forget_desired=True,
                )
                managed_fenced = True
                with self._lock:
                    self._managed_nodes.pop(adapter_id, None)
            except LookupError:
                managed_fenced = True
                with self._lock:
                    self._managed_nodes.pop(adapter_id, None)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    degradation_phase,
                    exc,
                    action="continued local attachment fencing after managed node removal failed",
                )
        sampler_fenced = not sampler_registered
        if sampler_registered:
            try:
                self._router.unregister_sampler(adapter_id)
                sampler_fenced = True
            except LookupError:
                sampler_fenced = True
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    degradation_phase,
                    exc,
                    action="continued remaining local attachment fencing",
                )
        service_fenced = not service_registered
        if service_registered:
            try:
                await asyncio.to_thread(self._service.unregister_adapter, adapter_id)
                service_fenced = True
            except LookupError:
                service_fenced = True
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    degradation_phase,
                    exc,
                    action="continued body and connector fencing after service removal failed",
                )
        body_fenced = not body_projected
        if body_projected:
            body_fenced = self._remove_body_projection(adapter_id)
        remote_detached = connector is None
        if connector is not None:
            try:
                await asyncio.wait_for(
                    connector.detach(adapter),
                    timeout=self._connector_timeout_s,
                )
                remote_detached = True
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                record_degradation(
                    degradation_phase,
                    exc,
                    action="retained local effect fencing after remote connector teardown failed",
                )
        service_fenced = service_fenced and (adapter_id not in self._service.adapter_channels())
        local_fenced = managed_fenced and sampler_fenced and service_fenced and body_fenced
        if not local_fenced:
            record_degradation(
                degradation_phase,
                RuntimeError(
                    "local_attachment_effects_remain_after_teardown:"
                    f"managed={not managed_fenced}:"
                    f"sampler={not sampler_fenced}:"
                    f"service={not service_fenced}:"
                    f"body={not body_fenced}"
                ),
                action="retained attachment ownership for a bounded local fencing retry",
            )
        return local_fenced, remote_detached

    async def _detach_twin_or_queue(
        self,
        adapter_id: str,
        *,
        reason: str,
        lost: bool,
        degradation_phase: str,
    ) -> None:
        if self._digital_twin is None:
            return
        canonical_reason = _reconciliation_reason(
            reason,
            fallback="attachment_detached",
        )
        intent = (canonical_reason, lost)
        with self._lock:
            self._pending_twin_detaches[adapter_id] = intent
        durable_intent = self._trust_store is not None
        if durable_intent:
            try:
                # Write-ahead: a crash after the physical fence but before the twin
                # operation must leave an authenticated replay obligation.
                await self._persist_state()
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    f"{degradation_phase}.digital_twin_intent",
                    exc,
                    action=(
                        "retained the in-memory twin detach obligation and refused "
                        "to claim durable reconciliation"
                    ),
                )
                raise
        try:
            await asyncio.to_thread(
                self._digital_twin.detach_adapter,
                adapter_id,
                reason=canonical_reason,
                lost=lost,
            )
            with self._lock:
                if self._pending_twin_detaches.get(adapter_id) == intent:
                    self._pending_twin_detaches.pop(adapter_id, None)
            if durable_intent:
                try:
                    await self._persist_state()
                except (OSError, RuntimeError, TypeError, ValueError):
                    # The write-ahead head remains durable. Mirror that state in
                    # memory so readiness and retry behavior agree with restart.
                    with self._lock:
                        self._pending_twin_detaches[adapter_id] = intent
                    raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                f"{degradation_phase}.digital_twin",
                exc,
                action=(
                    "kept connector, sampler, service, and body effects fenced; "
                    "queued canonical-twin detach reconciliation"
                ),
            )

    async def _reconcile_digital_twin(self) -> None:
        if self._digital_twin is None:
            return
        async with self._lifecycle_lock:
            await self._reconcile_digital_twin_unlocked()

    async def _reconcile_digital_twin_unlocked(
        self,
        *,
        adapter_id: str | None = None,
    ) -> None:
        if self._digital_twin is None:
            return
        with self._lock:
            pending_detaches = tuple(
                (pending_adapter_id, reason, lost)
                for pending_adapter_id, (reason, lost) in sorted(
                    self._pending_twin_detaches.items()
                )
                if adapter_id is None or pending_adapter_id == adapter_id
            )
            pending_revocations = (
                tuple(sorted(self._pending_twin_revocations.items())) if adapter_id is None else ()
            )
        for pending_adapter_id, reason, lost in pending_detaches:
            intent = (reason, lost)
            try:
                await asyncio.to_thread(
                    self._digital_twin.detach_adapter,
                    pending_adapter_id,
                    reason=reason,
                    lost=lost,
                )
                with self._lock:
                    if self._pending_twin_detaches.get(pending_adapter_id) == intent:
                        self._pending_twin_detaches.pop(pending_adapter_id, None)
                try:
                    await self._persist_state()
                except (OSError, RuntimeError, TypeError, ValueError):
                    with self._lock:
                        self._pending_twin_detaches[pending_adapter_id] = intent
                    raise
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "reality_attachment.digital_twin_reconcile",
                    exc,
                    action="retained canonical-twin detach for the next bounded scan",
                )
        for identity_fingerprint, reason in pending_revocations:
            try:
                await asyncio.to_thread(
                    self._digital_twin.revoke_identity,
                    identity_fingerprint,
                    reason=reason,
                )
                with self._lock:
                    if self._pending_twin_revocations.get(identity_fingerprint) == reason:
                        self._pending_twin_revocations.pop(identity_fingerprint, None)
                try:
                    await self._persist_state()
                except (OSError, RuntimeError, TypeError, ValueError):
                    with self._lock:
                        self._pending_twin_revocations[identity_fingerprint] = reason
                    raise
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "reality_attachment.digital_twin_reconcile",
                    exc,
                    action="retained canonical-twin revocation for the next bounded scan",
                )

    async def _retire_connector_transaction(
        self,
        connector_id: str,
        *,
        reason: str,
    ) -> None:
        async with self._lifecycle_lock:
            with self._lock:
                if connector_id not in self._connectors:
                    raise LookupError(f"connector is not registered: {connector_id}")
                request_ids = self._connector_active_request_ids_locked(connector_id)
            incomplete: list[str] = []
            for request_id in request_ids:
                detached = await self._detach_request_unlocked(
                    request_id,
                    state=ConnectionState.LOST,
                    error=reason,
                    lost=True,
                    degradation_phase="reality_attachment.connector_retirement",
                    require_remote_detach=True,
                )
                if not detached:
                    incomplete.append(request_id)
            if incomplete:
                raise RuntimeError(
                    "connector retirement could not fence every attachment: "
                    + ",".join(sorted(incomplete))
                )
            with self._lock:
                residual = self._connector_active_request_ids_locked(connector_id)
                if residual:
                    raise RuntimeError(
                        "connector retirement observed concurrent attachments: "
                        + ",".join(residual)
                    )
                self._remove_idle_connector_locked(connector_id, reason=reason)

    def _connector_active_request_ids_locked(self, connector_id: str) -> tuple[str, ...]:
        active = {
            request_id
            for request_id, candidate in self._attached_candidates.items()
            if candidate.connector_id == connector_id
        }
        candidate_ids = {
            candidate.candidate_id
            for candidate in self._candidates.values()
            if candidate.connector_id == connector_id
        }
        active.update(
            request_id
            for request_id, request in self._requests.items()
            if request.candidate_id in candidate_ids
            and request.state in {ConnectionState.ATTACHING, ConnectionState.ATTACHED}
        )
        return tuple(sorted(active))

    def _remove_idle_connector_locked(self, connector_id: str, *, reason: str) -> None:
        self._connectors.pop(connector_id, None)
        candidate_ids = {
            candidate_id
            for candidate_id, candidate in self._candidates.items()
            if candidate.connector_id == connector_id
        }
        for candidate_id in candidate_ids:
            self._candidates.pop(candidate_id, None)
            self._candidate_absence_streaks.pop(candidate_id, None)
        now_ns = self._clock_ns()
        for request_id, request in tuple(self._requests.items()):
            if request.candidate_id not in candidate_ids or request.state in {
                ConnectionState.ERROR,
                ConnectionState.LOST,
                ConnectionState.REFUSED,
                ConnectionState.REVOKED,
            }:
                continue
            self._requests[request_id] = replace(
                request,
                state=ConnectionState.LOST,
                updated_at_ns=max(request.created_at_ns, now_ns),
                error=str(reason or "connector_removed")[:320],
            )
        self._connector_discovery_failures.pop(connector_id, None)

    async def _rollback_cancelled_authorization(
        self,
        *,
        request: ConnectionRequest,
        grant: TrustGrant,
        previous_grant: TrustGrant | None,
        error: str = "attachment_cancelled",
    ) -> None:
        with self._lock:
            self._pending_grant_activations.discard(grant.identity_fingerprint)
            current_grant = self._grants.get(grant.identity_fingerprint)
            if current_grant == grant:
                if previous_grant is None:
                    self._grants.pop(grant.identity_fingerprint, None)
                else:
                    self._grants[grant.identity_fingerprint] = previous_grant
            self._requests[request.request_id] = replace(
                request,
                state=ConnectionState.ERROR,
                updated_at_ns=max(request.created_at_ns, self._clock_ns()),
                error=error,
            )
        if grant.persistent:
            try:
                await self._persist_state()
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "reality_attachment.cancelled_authority_rollback",
                    exc,
                    action=(
                        "removed cancelled attachment authority from live state; "
                        "left durable custody visibly degraded"
                    ),
                )

    @staticmethod
    async def _await_task_completion(task: asyncio.Task[Any]) -> Any:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        return task.result()

    def _remove_body_projection(self, adapter_id: str) -> bool:
        canonical = str(adapter_id or "")
        projection = self._body_projections.get(canonical)
        if projection is None:
            return True
        try:
            remove_body_projection(projection)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "reality_attachment.body_schema",
                exc,
                action="retained attachment teardown after body-schema projection removal failed",
            )
            return False
        with self._lock:
            if self._body_projections.get(canonical) == projection:
                self._body_projections.pop(canonical, None)
        return True

    async def _fail_request(
        self,
        request: ConnectionRequest,
        error: str,
    ) -> ConnectionRequest:
        failed = replace(
            request,
            state=ConnectionState.ERROR,
            updated_at_ns=max(request.created_at_ns, self._clock_ns()),
            error=error[:320],
        )
        with self._lock:
            self._requests[request.request_id] = failed
        return failed

    async def _announce_request(
        self,
        candidate: DeviceCandidate,
        request: ConnectionRequest,
    ) -> None:
        try:
            from core.observability.neural_feed import get_feed

            get_feed().push(
                content=(
                    f"I discovered {candidate.display_name} via {candidate.transport} and "
                    f"would like to attach its {', '.join(item.value for item in request.requested_access)} "
                    f"surface. Connection request: {request.request_id}."
                ),
                title="PHYSICAL_CONNECTION_PROPOSED",
                category="SOMATIC",
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("reality_attachment.announce", exc)
        try:
            from core.advanced_cognition.integration import (
                get_advanced_cognition_runtime,
            )

            runtime = get_advanced_cognition_runtime()
            await asyncio.to_thread(
                runtime.observe_state,
                "physical_attachment",
                {
                    "candidate_id": candidate.candidate_id,
                    "device_id": candidate.device_id,
                    "transport": candidate.transport,
                    "requested_access": [item.value for item in request.requested_access],
                    "privacy_sensitive": candidate.privacy_sensitive,
                    "connection_state": request.state.value,
                    "request_id": request.request_id,
                },
                source=f"reality_discovery:{candidate.connector_id}",
                confidence=0.8 if candidate.persistent_identity else 0.55,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("reality_attachment.cognition", exc)

    def candidates(self) -> tuple[DeviceCandidate, ...]:
        with self._lock:
            return tuple(sorted(self._candidates.values(), key=lambda item: item.candidate_id))

    def requests(self) -> tuple[ConnectionRequest, ...]:
        with self._lock:
            return tuple(sorted(self._requests.values(), key=lambda item: item.request_id))

    def status(self) -> dict[str, Any]:
        with self._lock:
            state_counts: dict[str, int] = {}
            for request in self._requests.values():
                state_counts[request.state.value] = state_counts.get(request.state.value, 0) + 1
            custody = (
                dict(self._trust_store.status())
                if self._trust_store is not None
                else {
                    "healthy": False,
                    "error": "attachment_trust_custody_unavailable",
                }
            )
            if self._custody_error:
                custody["healthy"] = False
                custody["error"] = self._custody_error
            try:
                twin_status = (
                    self._digital_twin.health_snapshot() if self._digital_twin is not None else None
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                twin_status = {
                    "ready": False,
                    "healthy": False,
                    "error": f"{type(exc).__name__}:{exc}"[:320],
                }
            pending_twin_detaches = len(self._pending_twin_detaches)
            pending_twin_revocations = len(self._pending_twin_revocations)
            pending_authority_activations = len(self._pending_grant_activations)
            return {
                "alive": self.is_alive(),
                "ready": self.is_ready(),
                "connectors": len(self._connectors),
                "degraded_connectors": len(self._connector_discovery_failures),
                "connector_failure_streaks": dict(
                    sorted(self._connector_discovery_failures.items())
                ),
                "disappearance_quorum": self._disappearance_quorum,
                "candidate_absence_streaks": dict(sorted(self._candidate_absence_streaks.items())),
                "candidates": len(self._candidates),
                "trust_grants": len(self._grants),
                "pending_authority_activations": pending_authority_activations,
                "attached": len(self._attached),
                "teardown_pending": len(self._teardown_pending),
                "discoveries": self._discoveries,
                "attachment_failures": self._attachment_failures,
                "expired_grants": self._expired_grants,
                "request_states": state_counts,
                "persistent_trust_ready": self._state_loaded
                and self._trust_store is not None
                and not self._custody_error,
                "trust_state_loaded": self._state_loaded,
                "trust_custody": custody,
                "digital_twin": twin_status,
                "twin_reconciliation": {
                    "healthy": not (pending_twin_detaches or pending_twin_revocations),
                    "pending": pending_twin_detaches + pending_twin_revocations,
                    "pending_detaches": pending_twin_detaches,
                    "pending_revocations": pending_twin_revocations,
                },
            }

    def get_status(self) -> dict[str, Any]:
        return self.status()

    def is_alive(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    def is_ready(self) -> bool:
        with self._lock:
            reconciliation_pending = bool(
                self._teardown_pending
                or self._pending_twin_detaches
                or self._pending_twin_revocations
                or self._pending_grant_activations
                or self._custody_error
            )
        if not self.is_alive() or reconciliation_pending:
            return False
        if self._digital_twin is None:
            return True
        try:
            return bool(self._digital_twin.is_ready())
        except (OSError, RuntimeError, TypeError, ValueError):
            return False

    @staticmethod
    def _parse_twin_reconciliation(
        raw: Any,
    ) -> tuple[dict[str, tuple[str, bool]], dict[str, str]]:
        if raw is None:
            return {}, {}
        if not isinstance(raw, dict) or set(raw) != {
            "schema",
            "detaches",
            "revocations",
        }:
            raise ValueError("twin reconciliation state shape is invalid")
        if raw.get("schema") != _TWIN_RECONCILIATION_SCHEMA:
            raise ValueError("twin reconciliation schema is unsupported")
        detach_rows = raw.get("detaches")
        revocation_rows = raw.get("revocations")
        if not isinstance(detach_rows, list) or not isinstance(revocation_rows, list):
            raise ValueError("twin reconciliation intents must be lists")
        if len(detach_rows) + len(revocation_rows) > _MAX_TWIN_RECONCILIATION_INTENTS:
            raise ValueError("twin reconciliation intent limit exceeded")

        detaches: dict[str, tuple[str, bool]] = {}
        for row in detach_rows:
            if not isinstance(row, dict) or set(row) != {
                "adapter_id",
                "reason",
                "lost",
            }:
                raise ValueError("twin detach reconciliation intent is invalid")
            adapter_id_value = row.get("adapter_id")
            if not isinstance(adapter_id_value, str):
                raise ValueError("twin detach reconciliation intent is invalid")
            adapter_id = _identifier(adapter_id_value, name="adapter_id")
            reason = row.get("reason")
            lost = row.get("lost")
            if (
                not isinstance(reason, str)
                or not reason
                or reason != reason.strip()
                or len(reason) > 160
                or any(ord(character) < 32 for character in reason)
                or not isinstance(lost, bool)
                or adapter_id in detaches
            ):
                raise ValueError("twin detach reconciliation intent is invalid")
            detaches[adapter_id] = (reason, lost)

        revocations: dict[str, str] = {}
        for row in revocation_rows:
            if not isinstance(row, dict) or set(row) != {
                "identity_fingerprint",
                "reason",
            }:
                raise ValueError("twin revocation reconciliation intent is invalid")
            identity_fingerprint = row.get("identity_fingerprint")
            reason = row.get("reason")
            if (
                not isinstance(identity_fingerprint, str)
                or not _DIGEST.fullmatch(identity_fingerprint)
                or not isinstance(reason, str)
                or not reason
                or reason != reason.strip()
                or len(reason) > 160
                or any(ord(character) < 32 for character in reason)
                or identity_fingerprint in revocations
            ):
                raise ValueError("twin revocation reconciliation intent is invalid")
            revocations[identity_fingerprint] = reason
        return detaches, revocations

    def _load_state(self) -> None:
        if self._trust_store is None:
            return
        try:
            payload = self._trust_store.load()
            if payload is None:
                return
            grants = payload.get("grants", [])
            if not isinstance(grants, list):
                raise ValueError("attachment grants must be a list")
            pending_detaches, pending_revocations = self._parse_twin_reconciliation(
                payload.get("twin_reconciliation")
            )
            # Safety reconciliation is independent of grant validity. A corrupt
            # authority row must not erase a valid physical detach or revocation
            # obligation recovered from the same authenticated envelope.
            self._pending_twin_detaches = pending_detaches
            self._pending_twin_revocations = pending_revocations
            loaded: dict[str, TrustGrant] = {}
            incomplete_lifecycle = False
            now_ns = self._clock_ns()
            for raw in grants:
                if not isinstance(raw, dict):
                    raise ValueError("attachment grant must be a mapping")
                authority_intent = raw.get("authority_intent")
                authority_evidence = raw.get("authority_evidence")
                private_device_metadata = raw.get("private_device_metadata")
                if (
                    not isinstance(authority_intent, dict)
                    or not isinstance(authority_evidence, dict)
                    or not isinstance(private_device_metadata, dict)
                ):
                    raise ValueError("attachment grant private evidence is invalid")
                grant = TrustGrant(
                    identity_fingerprint=str(raw.get("identity_fingerprint") or ""),
                    connector_id=str(raw.get("connector_id") or ""),
                    manifest_sha256=str(raw.get("manifest_sha256") or ""),
                    allowed_access=tuple(
                        AttachmentAccess(str(item)) for item in raw.get("allowed_access", [])
                    ),
                    authority_receipt_id=str(raw.get("authority_receipt_id") or ""),
                    authority_intent=authority_intent,
                    authority_evidence=authority_evidence,
                    private_device_metadata=private_device_metadata,
                    issued_at_ns=int(raw.get("issued_at_ns") or 0),
                    expires_at_ns=int(raw.get("expires_at_ns") or 0),
                    persistent=bool(raw.get("persistent")),
                )
                if not grant.persistent:
                    raise ValueError("session trust must not be present in durable state")
                self._validate_loaded_grant(grant)
                lifecycle = raw.get("lifecycle", _GRANT_LIFECYCLE_ACTIVE)
                if lifecycle not in {
                    _GRANT_LIFECYCLE_ACTIVE,
                    _GRANT_LIFECYCLE_PENDING,
                }:
                    raise ValueError("attachment grant lifecycle is invalid")
                if lifecycle == _GRANT_LIFECYCLE_PENDING:
                    incomplete_lifecycle = True
                    self._state_needs_compaction = True
                    continue
                if grant.is_valid_at(now_ns):
                    loaded[grant.identity_fingerprint] = grant
                else:
                    self._expired_grants += 1
                    self._state_needs_compaction = True
            self._grants = loaded
            self._custody_error = (
                "incomplete_authority_lifecycle_requires_compaction" if incomplete_lifecycle else ""
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._custody_error = f"{type(exc).__name__}:{exc}"[:320]
            record_degradation(
                "reality_attachment.load",
                exc,
                action=(
                    "refused invalid physical trust state while retaining discovery "
                    "without attachment authority"
                ),
            )
            self._grants.clear()

    async def _ensure_state_loaded(self) -> None:
        if self._state_loaded:
            return
        async with self._state_load_lock:
            if self._state_loaded:
                return
            await asyncio.to_thread(self._load_state)
            self._state_loaded = True

    def _validate_loaded_grant(self, grant: TrustGrant) -> None:
        intent = dict(grant.authority_intent)
        expected_access = sorted(item.value for item in grant.allowed_access)
        lifetime_ns = grant.expires_at_ns - grant.issued_at_ns
        if (
            intent.get("identity_fingerprint") != grant.identity_fingerprint
            or intent.get("connector_id") != grant.connector_id
            or intent.get("manifest_sha256") != grant.manifest_sha256
            or intent.get("requested_access") != expected_access
            or intent.get("persistent") is not True
            or not isinstance(intent.get("grant_ttl_s"), int)
            or isinstance(intent.get("grant_ttl_s"), bool)
            or lifetime_ns != int(intent["grant_ttl_s"]) * 1_000_000_000
        ):
            raise AttachmentAuthorityError("attachment_authority_grant_binding_invalid")
        evidence = self._authority_verifier.validate_persisted(
            grant.authority_evidence,
            intent=intent,
            persistent=True,
        )
        capability = evidence.get("capability")
        if (
            not isinstance(capability, Mapping)
            or capability.get("receipt_id") != grant.authority_receipt_id
            or int(evidence.get("verified_at_ns") or 0) > grant.issued_at_ns + 5_000_000_000
        ):
            raise AttachmentAuthorityError("attachment_authority_grant_evidence_invalid")

    def _persistent_body(self) -> dict[str, Any]:
        now_ns = self._clock_ns()
        with self._lock:
            expired = [
                identity
                for identity, grant in self._grants.items()
                if grant.persistent and not grant.is_valid_at(now_ns)
            ]
            for identity in expired:
                self._grants.pop(identity, None)
            self._expired_grants += len(expired)
            return {
                "grants": [
                    {
                        **item.to_dict(),
                        "lifecycle": (
                            _GRANT_LIFECYCLE_PENDING
                            if item.identity_fingerprint in self._pending_grant_activations
                            else _GRANT_LIFECYCLE_ACTIVE
                        ),
                    }
                    for item in sorted(
                        self._grants.values(),
                        key=lambda item: item.identity_fingerprint,
                    )
                    if item.persistent
                ],
                "twin_reconciliation": {
                    "schema": _TWIN_RECONCILIATION_SCHEMA,
                    "detaches": [
                        {
                            "adapter_id": adapter_id,
                            "reason": reason,
                            "lost": lost,
                        }
                        for adapter_id, (reason, lost) in sorted(
                            self._pending_twin_detaches.items()
                        )
                    ],
                    "revocations": [
                        {
                            "identity_fingerprint": identity_fingerprint,
                            "reason": reason,
                        }
                        for identity_fingerprint, reason in sorted(
                            self._pending_twin_revocations.items()
                        )
                    ],
                },
            }

    async def _persist_state(self) -> None:
        async with self._persistence_lock:
            body = self._persistent_body()
            reconciliation = body["twin_reconciliation"]
            has_reconciliation = bool(reconciliation["detaches"] or reconciliation["revocations"])
            if self._trust_store is None:
                if body["grants"] or has_reconciliation:
                    raise AttachmentTrustStoreError(
                        "attachment_trust_custody_unavailable_for_persistence"
                    )
                return
            try:
                await asyncio.to_thread(self._trust_store.save, body)
                self._custody_error = ""
                self._state_needs_compaction = False
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                self._custody_error = f"{type(exc).__name__}:{exc}"[:320]
                record_degradation(
                    "reality_attachment.persist",
                    exc,
                    action="refused to acknowledge uncommitted physical state",
                )
                raise

    async def rotate_trust_custody(self) -> Mapping[str, Any]:
        """Rotate the Keychain root and atomically re-encrypt the trust head."""

        await self._ensure_state_loaded()
        if self._trust_store is None:
            raise AttachmentTrustStoreError("attachment_trust_custody_unavailable")
        body = self._persistent_body()
        try:
            receipt = await asyncio.to_thread(
                self._trust_store.rotate_and_save,
                body,
            )
            if not isinstance(receipt, Mapping):
                raise AttachmentTrustStoreError("attachment custody receipt is not an object")
            self._custody_error = ""
            return receipt
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._custody_error = f"{type(exc).__name__}:{exc}"[:320]
            record_degradation(
                "reality_attachment.rotate_custody",
                exc,
                action="retained the prior authenticated trust head",
            )
            raise


__all__ = [
    "AttachmentAccess",
    "ConnectionRequest",
    "ConnectionState",
    "DeviceAttachmentBroker",
    "DeviceCandidate",
    "DeviceConnector",
    "TrustGrant",
]
