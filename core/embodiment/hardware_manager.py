import asyncio
import logging
from typing import Dict, List, Optional, Any

from core.base_module import AuraBaseModule
from .base_device import BaseHardwareDevice

logger = logging.getLogger("Embodiment.Manager")

class HardwareManager(AuraBaseModule):
    """
    Manages the lifecycle, discovery, and coordination of all connected physical components.
    Acts as the bridge between Aura's software brain and her physical body.
    """
    def __init__(self):
        super().__init__("HardwareManager")
        self.devices: Dict[str, BaseHardwareDevice] = {}
        
    async def start(self) -> None:
        """Initialize and auto-connect to registered hardware."""
        self.logger.info("Initializing Embodiment Hardware Manager...")
        
        # During startup, attempt to connect all registered devices.
        for device_id, device in self.devices.items():
            try:
                success = await device.connect()
                if success:
                    self.logger.info("✓ Connected to hardware: %s (%s)", device.device_name, device_id)
                else:
                    self.logger.warning("Failed to connect to hardware: %s", device.device_name)
            except Exception as e:
                self.logger.error("Exception connecting %s: %s", device.device_name, e)

    async def stop(self) -> None:
        """Gracefully disconnect all hardware during shutdown."""
        self.logger.info("Safely decoupling from physical hardware...")
        for device in self.devices.values():
            if device.is_connected:
                await device.disconnect()
        self.devices.clear()

    def register_device(self, device: BaseHardwareDevice) -> None:
        """Add a new hardware device to the registry."""
        if device.device_id in self.devices:
            self.logger.warning("Overwriting existing device registration for ID: %s", device.device_id)
        self.devices[device.device_id] = device
        self.logger.info("Registered physical device: %s [%s]", device.device_name, device.device_type)

    def unregister_device(self, device_id: str) -> None:
        """Remove a device from the physical registry."""
        if device_id in self.devices:
            del self.devices[device_id]

    def get_device(self, device_id: str) -> Optional[BaseHardwareDevice]:
        """Fetch a specific device by ID."""
        return self.devices.get(device_id)

    def list_devices(self) -> List[Dict[str, Any]]:
        """Return a serialized list of all devices and their status."""
        return [dev.to_dict() for dev in self.devices.values()]