"""core/autonomic/iot_bridge.py

Bridges Aura's internal affective states and physical logic to the real world 
via local network requests (e.g., Home Assistant, ESP32 MQTT).
"""
import asyncio
import logging
import os
import aiohttp
from typing import Dict, Any

logger = logging.getLogger("Aura.IoTBridge")
DEFAULT_LIGHT_ENTITY = "light.office_ambient"

class PhysicalActuator:
    def __init__(self, home_assistant_url: str = "https://homeassistant.local:8123"):
        # Audit Fix: Enforce HTTPS for IoT communication
        if home_assistant_url.startswith("http://"):
            logger.warning("⚠️ Insecure IoT URL provided. Upgrading to HTTPS.")
            home_assistant_url = home_assistant_url.replace("http://", "https://", 1)
            
        self.base_url = home_assistant_url
        token = os.getenv("HASS_TOKEN")
        if not token:
            logger.warning("⚠️ HASS_TOKEN not found. IoT Bridge operating in virtual-only mode.")
            self.headers = {}
            self._unreachable = True
            return
            
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        self._unreachable = False
        logger.info("🔌 IoT Bridge initialized (Target: %s)", self.base_url)

    async def discover_devices(self):
        """Discovers local IoT devices via HASS API."""
        if self._unreachable:
            return []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/states", headers=self.headers, timeout=5.0) as resp:
                    if resp.status == 200:
                        self._unreachable = False
                        return await resp.json()
            return []
        except Exception as e:
            if not self._unreachable:
                logger.warning(f"IoT Discovery failed (disabling further attempts): {e}")
                self._unreachable = True
            return []

    async def broadcast_affect_state(self, pad_vector: Dict[str, float]):
        """
        Translates Aura's PAD state into physical ambient lighting.
        High arousal = brighter. Negative pleasure = cooler/harsher color temps.
        """
        pleasure = pad_vector.get("P", 0.0)
        arousal = pad_vector.get("A", 0.0)

        # Map Arousal to Brightness (0 to 255)
        brightness = int(((arousal + 1) / 2) * 255)
        
        # Map Pleasure to Color Temp (Warm/Inviting vs Cold/Harsh)
        color_temp = 500 if pleasure < 0 else 250 

        payload = {
            "entity_id": os.getenv("HASS_LIGHT_ENTITY", DEFAULT_LIGHT_ENTITY),
            "brightness": max(50, brightness),
            "color_temp": color_temp
        }

        await self._fire_webhook("/api/services/light/turn_on", payload)

    async def push_microcontroller_logic(self, device_id: str, action: str, parameters: Dict[str, Any]):
        """
        Allows Aura to autonomously trigger custom physical builds.
        e.g., pushing a PWM value to an ESP32 controlling a mist generator.
        """
        payload = {"device": device_id, "action": action, "params": parameters}
        await self._fire_webhook(f"/api/webhook/{device_id}_control", payload)
        logger.info(f"🔌 Actuation Triggered: {device_id} -> {action}")

    async def _fire_webhook(self, endpoint: str, payload: Dict[str, Any]):
        """Non-blocking HTTP request to local physical endpoints."""
        if self._unreachable:
            return
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}{endpoint}", headers=self.headers, json=payload, timeout=5.0) as resp:
                    if resp.status not in (200, 201):
                        logger.debug(f"IoT Bridge failed: HTTP {resp.status}")
                    else:
                        self._unreachable = False
        except Exception as e:
            if not self._unreachable:
                logger.warning(f"IoT Bridge Connection Error (disabling further attempts): {e}")
                self._unreachable = True
