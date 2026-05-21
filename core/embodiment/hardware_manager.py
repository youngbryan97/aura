from __future__ import annotations

import logging
from typing import Any

from core.base_module import AuraBaseModule
from core.runtime.errors import FallbackClassification, Severity, record_degradation

from .base_device import BaseHardwareDevice

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

class HardwareManager(AuraBaseModule):
    """
    Manages the lifecycle, discovery, and coordination of all connected physical components.
    Acts as the bridge between Aura's software brain and her physical body.
    """
    def __init__(self):
        super().__init__("HardwareManager")
        self.devices: dict[str, BaseHardwareDevice] = {}
        self.connection_failures: dict[str, str] = {}
        self._started = False
        
    async def start(self) -> None:
        """Initialize and auto-connect to registered hardware."""
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
        self.devices.clear()
        self._started = False

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
        if device_id in self.devices:
            del self.devices[device_id]
        self.connection_failures.pop(device_id, None)

    def get_device(self, device_id: str) -> BaseHardwareDevice | None:
        """Fetch a specific device by ID."""
        return self.devices.get(_safe_device_id(device_id))

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
        base = super().get_health()
        connected = sum(1 for device in self.devices.values() if getattr(device, "is_connected", False))
        base.update(
            {
                "started": self._started,
                "registered_devices": len(self.devices),
                "connected_devices": connected,
                "connection_failures": dict(self.connection_failures),
                "status": "degraded" if self.connection_failures else base["status"],
            }
        )
        return base
