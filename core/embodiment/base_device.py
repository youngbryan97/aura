from core.runtime.errors import record_degradation
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger("Embodiment.BaseDevice")

class BaseHardwareDevice(ABC):
    """
    Abstract Base Class for all physical hardware components that Aura can embody.
    Provides a standardized interface and threading locks to prevent concurrent hardware collisions.
    """
    def __init__(self, device_id: str, device_name: str, device_type: str):
        self.device_id = device_id
        self.device_name = device_name
        self.device_type = device_type
        self.is_connected = False
        
        # A lock to ensure hardware commands are serialized
        self.hardware_lock = asyncio.Lock()

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the hardware device."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Gracefully disconnect from the hardware device."""
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Query the device for its current state and telemetry."""
        pass

    @abstractmethod
    async def execute_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a hardware-specific command.
        Should ideally be wrapped by `safe_execute` for thread safety.
        """
        pass

    async def safe_execute(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Thread-safe execution wrapper. Most hardware interfaces (like Serial or distinct IoT sockets)
        can crash or corrupt if hit with concurrent commands.
        """
        if not self.is_connected:
            return {"ok": False, "error": f"Device {self.device_name} is not connected."}
            
        async with self.hardware_lock:
            try:
                # Add a timeout to prevent deadlocks on hardware hangs
                return await asyncio.wait_for(
                    self.execute_command(command, **kwargs),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.error("Timeout executing command '%s' on device %s", command, self.device_id)
                # Attempt to recover connection state on timeout
                self.is_connected = False
                return {"ok": False, "error": "Hardware command timed out. Connection forced closed."}
            except Exception as e:
                record_degradation('base_device', e)
                logger.error("Execution failed on device %s: %s", self.device_id, e, exc_info=True)
                return {"ok": False, "error": str(e)}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata for the cognitive orchestrator."""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "is_connected": self.is_connected
        }