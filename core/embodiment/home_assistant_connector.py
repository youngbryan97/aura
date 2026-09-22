"""Home Assistant device discovery for the portable attachment broker."""

from __future__ import annotations

import os
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from core.embodiment.home_assistant_reality import (
    HomeAssistantRealityAdapter,
    HomeAssistantRealityError,
    HomeAssistantSensorAdapter,
    HomeAssistantTransport,
)
from core.reality_reach.attachments import (
    AttachmentAccess,
    DeviceCandidate,
)
from core.reality_reach.live import LiveChannelAdapter
from core.runtime.audit_chain import canonical_json, sha256_hex

_PRIVACY_CLASSES = frozenset(
    {
        "audio",
        "door",
        "garage_door",
        "motion",
        "occupancy",
        "opening",
        "presence",
        "sound",
        "window",
    }
)
_SAFETY_CLASSES = frozenset(
    {
        "carbon_dioxide",
        "carbon_monoxide",
        "gas",
        "moisture",
        "nitrogen_dioxide",
        "nitrogen_monoxide",
        "nitrous_oxide",
        "ozone",
        "pm1",
        "pm10",
        "pm25",
        "smoke",
        "sulphur_dioxide",
        "volatile_organic_compounds",
    }
)


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


class HomeAssistantConnector:
    """Translate Home Assistant entities into stable attachment candidates.

    Persistent identity requires ``AURA_HASS_INSTALLATION_ID``.  An endpoint
    URL alone is sufficient for a session proposal but deliberately not for a
    migration-surviving trust grant because DNS names and addresses are not
    device identity.
    """

    connector_id = "home_assistant.local"

    def __init__(
        self,
        transport: HomeAssistantTransport,
        *,
        discover_callback: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
        candidate_ttl_s: float = 180.0,
    ) -> None:
        if not isinstance(transport, HomeAssistantTransport):
            raise TypeError("transport must be a HomeAssistantTransport")
        self._transport = transport
        self._discover_callback = discover_callback
        self._candidate_ttl_s = max(30.0, min(float(candidate_ttl_s), 3600.0))
        self._installation_id = str(os.getenv("AURA_HASS_INSTALLATION_ID") or "").strip()

    async def discover(self) -> tuple[DeviceCandidate, ...]:
        states = (
            await self._discover_callback()
            if self._discover_callback is not None
            else await self._transport.discover()
        )
        candidates: list[DeviceCandidate] = []
        for state in states[:5000]:
            if not isinstance(state, Mapping):
                continue
            try:
                candidates.append(self._candidate_for_state(state))
            except (
                HomeAssistantRealityError,
                TypeError,
                ValueError,
            ):
                continue
        return tuple(sorted(candidates, key=lambda item: item.candidate_id))

    async def attach(
        self,
        candidate: DeviceCandidate,
        access: tuple[AttachmentAccess, ...],
    ) -> LiveChannelAdapter:
        if candidate.connector_id != self.connector_id:
            raise ValueError("home_assistant_candidate_connector_mismatch")
        requested = set(access)
        if not requested or not requested.issubset(set(candidate.access)):
            raise PermissionError("home_assistant_attachment_access_not_declared")
        if AttachmentAccess.CONTROL in requested and AttachmentAccess.OBSERVE not in requested:
            raise PermissionError("home_assistant_control_requires_observation")
        state = await self._transport.read_state(candidate.metadata["entity_id"])
        current = self._candidate_for_state(state)
        if (
            current.identity_fingerprint != candidate.identity_fingerprint
            or current.manifest_sha256 != candidate.manifest_sha256
        ):
            raise RuntimeError("home_assistant_candidate_changed_before_attachment")
        if AttachmentAccess.CONTROL in requested:
            return HomeAssistantRealityAdapter(
                self._transport,
                current.metadata["entity_id"],
                initial_state=state,
            )
        return HomeAssistantSensorAdapter(
            self._transport,
            current.metadata["entity_id"],
            initial_state=state,
        )

    async def detach(self, _adapter: LiveChannelAdapter) -> None:
        # Home Assistant REST is request-scoped; unregistering the adapter is
        # the complete attachment teardown and leaves no open socket behind.
        return None

    def _candidate_for_state(self, state: Mapping[str, Any]) -> DeviceCandidate:
        entity_id = str(state.get("entity_id") or "").strip().lower()
        sensor = HomeAssistantSensorAdapter(
            self._transport,
            entity_id,
            initial_state=state,
        )
        access = [AttachmentAccess.OBSERVE]
        control_manifest: dict[str, Any] | None = None
        try:
            control = HomeAssistantRealityAdapter(
                self._transport,
                entity_id,
                initial_state=state,
            )
            access.append(AttachmentAccess.CONTROL)
            control_manifest = {
                "adapter_id": control.adapter_id,
                "declarations": [item.to_dict() for item in control.declarations()],
                "capabilities": [
                    item.to_dict() for item in control.actuator_capabilities()
                ],
            }
        # not a failure: a control that will not describe itself has no manifest, and the
        # caller checks for None.
        except (HomeAssistantRealityError, TypeError, ValueError):
            control_manifest = None
        attributes = state.get("attributes")
        if not isinstance(attributes, Mapping):
            attributes = {}
        device_class = str(attributes.get("device_class") or "").strip().lower()
        persistent = bool(self._installation_id)
        installation_key = self._installation_id or f"session-endpoint:{self._transport.base}"
        identity_fingerprint = _digest(
            {
                "connector": self.connector_id,
                "installation": installation_key,
                "entity_id": entity_id,
            }
        )
        manifest_sha256 = _digest(
            {
                "sensor_manifest_sha256": sensor.manifest_sha256,
                "control": control_manifest,
            }
        )
        candidate_id = (
            "hass.candidate."
            + _digest(
                {
                    "identity": identity_fingerprint,
                    "manifest": manifest_sha256,
                }
            ).removeprefix("sha256:")[:32]
        )
        now_ns = max(1, time.time_ns())
        friendly_name = str(attributes.get("friendly_name") or entity_id).strip()
        return DeviceCandidate(
            candidate_id=candidate_id,
            connector_id=self.connector_id,
            device_id=f"hass.{entity_id}",
            display_name=friendly_name[:160] or entity_id,
            transport="home_assistant.rest",
            identity_fingerprint=identity_fingerprint,
            manifest_sha256=manifest_sha256,
            access=tuple(access),
            discovered_at_ns=now_ns,
            expires_at_ns=now_ns + int(self._candidate_ttl_s * 1_000_000_000),
            persistent_identity=persistent,
            privacy_sensitive=device_class in _PRIVACY_CLASSES,
            proposal_salience=0.8 if device_class in _SAFETY_CLASSES else 0.4,
            metadata={
                "entity_id": entity_id,
                "device_class": device_class,
                "identity_strength": (
                    "installation_scoped" if persistent else "session_endpoint_scoped"
                ),
                "sensor_manifest_sha256": sensor.manifest_sha256,
                "sensor_domain_source": sensor.domain_source,
                "control_available": AttachmentAccess.CONTROL in access,
            },
        )


__all__ = ["HomeAssistantConnector"]
