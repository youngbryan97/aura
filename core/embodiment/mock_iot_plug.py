import aiohttp
import asyncio
import logging
import os
from typing import Any, Dict

from .base_device import BaseHardwareDevice

logger = logging.getLogger("Embodiment.RestSmartPlug")

class RestSmartPlug(BaseHardwareDevice):
    """
    A live REST-based IoT Smart Plug adapter.
    Translates hardware commands into HTTP POST requests to a configurable endpoint 
    (e.g., HomeAssistant webhooks or a custom local IoT relay).
    """
    def __init__(self, device_id: str = "generic_relay_01", name: str = "REST API Relay"):
        super().__init__(device_id, name, "IoT.Relay")
        self.power_state = False
        self.current_draw_watts = 0.0
        
        # Environment configuration for dynamic endpoint binding
        self.endpoint_url = os.environ.get("AURA_IOT_ENDPOINT", "http://localhost:8123/api/webhook/aura_relay")
        self.api_key = os.environ.get("AURA_IOT_KEY", "")
        
    async def connect(self) -> bool:
        """Establish network connectivity/auth to the IoT hub."""
        logger.info("Initializing connection pool to %s...", self.endpoint_url)
        # We don't maintain a persistent websocket here, just test HTTP
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                # Fire a quick ping or handshake payload
                async with session.post(self.endpoint_url, json={"action": "ping"}, headers=headers, timeout=5) as resp:
                    if resp.status in (200, 201, 202, 204):
                        self.is_connected = True
                        logger.info("✅ Connected to real IoT generic relay: %s", self.device_name)
                        return True
                    else:
                        logger.warning("IoT handshake returned %d. Marking connected but degraded.", resp.status)
                        self.is_connected = True
                        return True
        except Exception as e:
            logger.error("❌ Failed to connect to IoT endpoint %s: %s", self.endpoint_url, e)
            return False

    async def disconnect(self) -> bool:
        """Teardown network."""
        logger.info("Disconnecting from %s...", self.device_id)
        self.is_connected = False
        return True

    async def get_status(self) -> Dict[str, Any]:
        """Query the remote endpoint for real hardware state."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                async with session.post(self.endpoint_url, json={"action": "status"}, headers=headers, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.power_state = data.get("state") == "on"
                        self.current_draw_watts = float(data.get("power_draw_watts", 0.0))
        except Exception as e:
            logger.debug("IoT Status fetch failed, using cached state: %s", e)
            
        return {
            "ok": True,
            "status": "on" if self.power_state else "off",
            "power_draw_watts": self.current_draw_watts,
            "connected": self.is_connected
        }

    async def execute_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """Dispatch live actuator command to the REST endpoint."""
        command = command.lower()
        target_state = None
        
        if command == "turn_on": target_state = "on"
        elif command == "turn_off": target_state = "off"
        elif command == "toggle": target_state = "toggle"
        else:
            return {"ok": False, "error": f"Unknown hardware command: {command}"}

        payload = {"action": target_state, "device_id": self.device_id}
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                async with session.post(self.endpoint_url, json=payload, headers=headers, timeout=5) as resp:
                    if resp.status in (200, 204):
                        logger.info("✅ [%s] Dispatched hardware command HTTP %d: %s", self.device_name, resp.status, target_state)
                        
                        # Optimistic update
                        if target_state == "on": self.power_state = True
                        elif target_state == "off": self.power_state = False
                        else: self.power_state = not self.power_state
                        
                        return {"ok": True, "message": f"Successfully signaled {self.device_name} via REST.", "new_state": "on" if self.power_state else "off"}
                    else:
                        logger.error("❌ [%s] Hardware dispatch failed with HTTP %d", self.device_name, resp.status)
                        return {"ok": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            logger.error("❌ [%s] Network failure instructing hardware: %s", self.device_name, e)
            return {"ok": False, "error": str(e)}