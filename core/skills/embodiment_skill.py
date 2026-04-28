from core.runtime.errors import record_degradation
from typing import Any, Dict
import logging

from core.skills.base_skill import BaseSkill
from core.container import ServiceContainer
from core.embodiment.hardware_manager import HardwareManager

logger = logging.getLogger("Skills.Embodiment")

class EmbodimentSkill(BaseSkill):
    """
    Hardware Interface Skill. Gives Aura the ability to physically interact
    with connected devices (IoT, Serial, Robotics).
    """
    name = "embodiment"
    description = "Control or query physical hardware devices in the real world."
    inputs = {
        "action": "list_devices | query_device | command_device",
        "device_id": "(Optional) Required for query or command",
        "command": "(Optional) The hardware-specific command to send (e.g. 'turn_on', 'turn_off', 'toggle')"
    }
    
    def __init__(self):
        super().__init__()

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hardware commands securely."""
        # Retrieve the manager from the DI container dynamically at runtime.
        # It may not exist if the node is running in cloud/software-only mode.
        try:
            manager: HardwareManager = ServiceContainer.get("hardware_manager", default=None)
        except Exception as e:
            record_degradation('embodiment_skill', e)
            logger.warning("Hardware manager not available: %s", e)
            manager = None
            
        if not manager:
            return {"ok": False, "error": "HardwareManager is offline. This instance has no physical body."}
            
        params = goal.get("params", {}) if "params" in goal else goal
        action = params.get("action", "list_devices")
        device_id = params.get("device_id")
        
        if action == "list_devices":
            devices = manager.list_devices()
            return {
                "ok": True,
                "devices": devices,
                "summary": f"Found {len(devices)} physical devices connected to my central nervous system."
            }
            
        elif action == "query_device":
            if not device_id:
                return {"ok": False, "error": "device_id is required."}
            
            device = manager.get_device(device_id)
            if not device:
                return {"ok": False, "error": f"Device {device_id} not found."}
                
            status_response = await device.get_status()
            return status_response
            
        elif action == "command_device":
            if not device_id:
                return {"ok": False, "error": "device_id is required."}
                
            command = params.get("command")
            if not command:
                return {"ok": False, "error": "command string is required."}
                
            device = manager.get_device(device_id)
            if not device:
                return {"ok": False, "error": f"Device {device_id} not found in my sensor array."}
                
            # Use safe execution to prevent hardware lockups
            logger.info("Sending physical pulse to '%s'. Command: %s", device_id, command)
            result = await device.safe_execute(command)
            return result
            
        else:
            return {"ok": False, "error": f"Unknown hardware action: {action}"}