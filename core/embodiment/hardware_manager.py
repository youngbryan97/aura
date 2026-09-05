from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

from core.reality_reach.body_projection import (
    PhysicalBodyProjection,
    project_adapter_to_body,
    remove_body_projection,
)
from core.runtime.base_module import AuraBaseModule
from core.runtime.errors import FallbackClassification, Severity, record_degradation

from .base_device import BaseHardwareDevice
from .reality_adapter import HardwareRealityAdapter, HardwareRealityManifest
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Embodiment.Manager")

_HARDWARE_MANAGER_ERRORS = (
    AttributeError,
    ConnectionError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_hardware_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "hardware_manager",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation(
                "hardware_manager",
                error,
                severity=severity,
                action=action or "hardware manager degraded",
            )
        except TypeError:
            logger.warning(
                "HardwareManager degradation could not be recorded: %s",
                signature_exc,
            )


def _safe_device_id(value: object) -> str:
    try:
        text = str(value or "").replace("\x00", "").strip()
    except (RuntimeError, TypeError, ValueError):
        text = ""
    return text[:128]

class HardwareManager(AuraBaseModule):  # type: ignore[misc]  # skipped import is untyped
    """
    Manages the lifecycle, discovery, and coordination of all connected physical components.
    Acts as the bridge between Aura's software brain and her physical body.
    """
    def __init__(self) -> None:
        super().__init__("HardwareManager")
        self.devices: dict[str, BaseHardwareDevice] = {}
        self.connection_failures: dict[str, str] = {}
        self.reality_adapters: dict[str, HardwareRealityAdapter] = {}
        self._reality_service: Any | None = None
        self._observation_router: Any | None = None
        self._body_projections: dict[str, PhysicalBodyProjection] = {}
        self._started = False

    def bind_reality_reach(self, service: Any) -> None:
        """Bind the canonical live inventory before activating physical devices."""

        if service is None or not callable(getattr(service, "register_adapter", None)):
            raise TypeError("reality reach service must support adapter registration")
        if self._reality_service is not None and self._reality_service is not service:
            raise RuntimeError("hardware manager is already bound to another reality service")
        self._reality_service = service

    def bind_observation_router(self, router: Any) -> None:
        """Bind the bounded sensory route used by connected readback adapters."""

        if router is None or not callable(getattr(router, "register_sampler", None)):
            raise TypeError("observation router must support sampler registration")
        if self._observation_router is not None and self._observation_router is not router:
            raise RuntimeError("hardware manager is already bound to another observation router")
        self._observation_router = router

    def register_configured_devices(self) -> tuple[str, ...]:
        """Materialize only explicitly configured production hardware."""

        if not str(os.environ.get("AURA_IOT_ENDPOINT") or "").strip():
            return ()
        from .mock_iot_plug import RestSmartPlug

        device_id = str(os.environ.get("AURA_IOT_DEVICE_ID") or "generic_relay_01")
        device_name = str(os.environ.get("AURA_IOT_DEVICE_NAME") or "REST API Relay")
        device = RestSmartPlug(device_id=device_id, name=device_name)
        if self.get_device(device.device_id) is None:
            self.register_device(device)
        return (device.device_id,)
        
    async def start(self) -> None:
        """Initialize and auto-connect to registered hardware."""
        if self._started:
            return
        self.logger.info("Initializing Embodiment Hardware Manager...")
        self._started = True

        # During startup, attempt to connect all registered devices.
        for device_id, device in list(self.devices.items()):
            try:
                success = await device.connect()
                if success:
                    self.connection_failures.pop(device_id, None)
                    device.is_connected = True
                    self.logger.info("✓ Connected to hardware: %s (%s)", device.device_name, device_id)
                    await self._activate_reality_adapter(device)
                else:
                    self.connection_failures[device_id] = "connect returned false"
                    device.is_connected = False
                    self.logger.warning("Failed to connect to hardware: %s", device.device_name)
            except _HARDWARE_MANAGER_ERRORS as e:
                device_name = getattr(device, "device_name", device_id)
                self.connection_failures[device_id] = f"{type(e).__name__}: {str(e)[:200]}"
                try:
                    device.is_connected = False
                except _HARDWARE_MANAGER_ERRORS as state_exc:
                    self.logger.debug(
                        "Could not mark hardware %s disconnected: %s",
                        device_id,
                        state_exc,
                    )
                _record_hardware_degradation(
                    e,
                    action="kept hardware manager online while marking device unavailable",
                    severity="degraded",
                    extra={"device_id": device_id, "device_name": str(device_name)[:128]},
                )
                self.logger.error("Exception connecting %s: %s", device_name, e)

        if self._observation_router is not None:
            for adapter in self.reality_adapters.values():
                try:
                    self._observation_router.register_sampler(adapter)
                except (TypeError, ValueError) as exc:
                    _record_hardware_degradation(
                        exc,
                        action="kept device attached while marking its sensory sampler unavailable",
                        severity="warning",
                        extra={"adapter_id": str(adapter.adapter_id)},
                    )

    async def stop(self) -> None:
        """Gracefully disconnect all hardware during shutdown."""
        self.logger.info("Safely decoupling from physical hardware...")
        for device_id, device in list(self.devices.items()):
            try:
                if device.is_connected:
                    await device.disconnect()
                device.is_connected = False
            except _HARDWARE_MANAGER_ERRORS as e:
                device_name = getattr(device, "device_name", device_id)
                self.connection_failures[device_id] = f"disconnect failed: {type(e).__name__}"
                _record_hardware_degradation(
                    e,
                    action="continued hardware manager shutdown after device disconnect failed",
                    severity="warning",
                    extra={"device_id": device_id, "device_name": str(device_name)[:128]},
                )
        if self._observation_router is not None:
            for adapter in self.reality_adapters.values():
                try:
                    self._observation_router.unregister_sampler(adapter.adapter_id)
                except LookupError:
                    pass
        for adapter_id, projection in list(self._body_projections.items()):
            try:
                remove_body_projection(projection)
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _record_hardware_degradation(
                    exc,
                    action="completed hardware shutdown while recording stale body projection",
                    severity="warning",
                    extra={"adapter_id": adapter_id},
                )
            finally:
                self._body_projections.pop(adapter_id, None)
        if self._reality_service is not None:
            await asyncio.to_thread(self._reality_service.refresh)
        self._started = False

    async def activate_device(self, device_id: str) -> HardwareRealityAdapter | None:
        """Connect and register a device added after manager startup."""

        device = self.get_device(device_id)
        if device is None:
            raise LookupError(f"hardware device is not registered: {device_id}")
        if not self._started:
            raise RuntimeError("hardware manager must be started before device activation")
        if not device.is_connected:
            connected = await device.connect()
            if not connected:
                device.is_connected = False
                self.connection_failures[device.device_id] = "connect returned false"
                raise RuntimeError(f"hardware device failed to connect: {device.device_id}")
            device.is_connected = True
        return await self._activate_reality_adapter(device)

    async def _activate_reality_adapter(
        self,
        device: BaseHardwareDevice,
    ) -> HardwareRealityAdapter | None:
        manifest = device.reality_manifest()
        if manifest is None:
            return None
        if not isinstance(manifest, HardwareRealityManifest):
            raise TypeError("device reality_manifest returned an invalid contract")
        existing = self.reality_adapters.get(device.device_id)
        if existing is not None:
            await existing.refresh_readback()
            if self._observation_router is not None:
                self._observation_router.register_sampler(existing)
            self._project_device_adapter(device, existing)
            if self._reality_service is not None:
                await asyncio.to_thread(self._reality_service.refresh)
            return existing
        if self._reality_service is None:
            raise RuntimeError("reality reach service is not bound")
        adapter = HardwareRealityAdapter(device, manifest)
        reading = await adapter.refresh_readback()
        if reading.value is None:
            raise RuntimeError("device readback is unavailable; adapter not registered")
        self._reality_service.register_adapter(adapter)
        try:
            if self._observation_router is not None:
                self._observation_router.register_sampler(adapter)
            self._project_device_adapter(device, adapter)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            if self._observation_router is not None:
                try:
                    self._observation_router.unregister_sampler(adapter.adapter_id)
                except LookupError:
                    pass
            self._reality_service.unregister_adapter(adapter.adapter_id)
            raise
        self.reality_adapters[device.device_id] = adapter
        await asyncio.to_thread(self._reality_service.refresh)
        return adapter

    def _project_device_adapter(
        self,
        device: BaseHardwareDevice,
        adapter: HardwareRealityAdapter,
    ) -> None:
        existing = self._body_projections.get(adapter.adapter_id)
        if existing is not None:
            return
        self._body_projections[adapter.adapter_id] = project_adapter_to_body(
            adapter,
            device_id=str(device.device_id),
            display_name=str(device.device_name),
            transport=f"hardware.{getattr(device, 'device_type', 'device')}",
            persistent_identity=True,
        )

    def is_alive(self) -> bool:
        """The manager is alive once its lifecycle has started."""

        return self._started

    def is_ready(self) -> bool:
        """Readiness means registry operations are live, not that hardware exists."""

        return self._started

    def status(self) -> dict[str, Any]:
        return dict(self.get_health())

    def register_device(self, device: BaseHardwareDevice) -> None:
        """Add a new hardware device to the registry."""
        device_id = _safe_device_id(getattr(device, "device_id", ""))
        if not device_id:
            raise ValueError("hardware device must expose a non-empty device_id")
        if device_id in self.devices:
            self.logger.warning("Overwriting existing device registration for ID: %s", device_id)
        self.devices[device_id] = device
        self.connection_failures.pop(device_id, None)
        self.logger.info(
            "Registered physical device: %s [%s]",
            getattr(device, "device_name", device_id),
            getattr(device, "device_type", "unknown"),
        )

    def unregister_device(self, device_id: str) -> None:
        """Remove a device from the physical registry."""
        device_id = _safe_device_id(device_id)
        adapter = self.reality_adapters.get(device_id)
        device = self.devices.get(device_id)
        if adapter is not None:
            if self._started or bool(getattr(device, "is_connected", False)):
                raise RuntimeError(
                    "cannot unregister an active physical adapter; stop the manager first"
                )
            if self._reality_service is None or not callable(
                getattr(self._reality_service, "unregister_adapter", None)
            ):
                raise RuntimeError("reality service cannot remove the device adapter")
            self._reality_service.unregister_adapter(adapter.adapter_id)
            if self._observation_router is not None:
                try:
                    self._observation_router.unregister_sampler(adapter.adapter_id)
                except LookupError:
                    pass
            self.reality_adapters.pop(device_id, None)
            projection = self._body_projections.pop(adapter.adapter_id, None)
            if projection is not None:
                try:
                    remove_body_projection(projection)
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    _record_hardware_degradation(
                        exc,
                        action="removed hardware registry entry while recording stale body projection",
                        severity="warning",
                        extra={"adapter_id": adapter.adapter_id},
                    )
        if device_id in self.devices:
            del self.devices[device_id]
        self.connection_failures.pop(device_id, None)

    def get_device(self, device_id: str) -> BaseHardwareDevice | None:
        """Fetch a specific device by ID."""
        return self.devices.get(_safe_device_id(device_id))

    def get_reality_adapter(self, device_id: str) -> HardwareRealityAdapter | None:
        """Return only a fully registered, explicit physical capability adapter."""

        return self.reality_adapters.get(_safe_device_id(device_id))

    @property
    def reality_service(self) -> Any | None:
        return self._reality_service

    def list_devices(self) -> list[dict[str, Any]]:
        """Return a serialized list of all devices and their status."""
        serialized = []
        for device_id, device in self.devices.items():
            try:
                serialized.append(device.to_dict())
            except _HARDWARE_MANAGER_ERRORS as e:
                _record_hardware_degradation(
                    e,
                    action="returned fallback metadata for unserializable hardware device",
                    severity="warning",
                    extra={"device_id": device_id},
                )
                serialized.append(
                    {
                        "device_id": device_id,
                        "device_name": getattr(device, "device_name", device_id),
                        "device_type": getattr(device, "device_type", "unknown"),
                        "is_connected": bool(getattr(device, "is_connected", False)),
                        "serialization_error": type(e).__name__,
                    }
                )
        return serialized

    def get_health(self) -> dict[str, Any]:
        base: dict[str, Any] = dict(super().get_health())
        connected = sum(1 for device in self.devices.values() if getattr(device, "is_connected", False))
        base.update(
            {
                "started": self._started,
                "registered_devices": len(self.devices),
                "connected_devices": connected,
                "reality_adapter_count": len(self.reality_adapters),
                "reality_reach_bound": self._reality_service is not None,
                "observation_router_bound": self._observation_router is not None,
                "body_projection_count": len(self._body_projections),
                "connection_failures": dict(self.connection_failures),
                "status": "degraded" if self.connection_failures else base["status"],
            }
        )
        return base


_HARDWARE_MANAGER: HardwareManager | None = None
_HARDWARE_MANAGER_LOCK = checked_lock("hardware_manager")


def get_hardware_manager() -> HardwareManager:
    global _HARDWARE_MANAGER
    if _HARDWARE_MANAGER is None:
        with _HARDWARE_MANAGER_LOCK:
            if _HARDWARE_MANAGER is None:
                _HARDWARE_MANAGER = HardwareManager()
    return _HARDWARE_MANAGER


__all__ = ["HardwareManager", "get_hardware_manager"]
