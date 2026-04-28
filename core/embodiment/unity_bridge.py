from core.runtime.errors import record_degradation
import asyncio
import json
from typing import Any, Dict

import logging
import numpy as np
import websockets

logger = logging.getLogger("Aura.Embodiment")

class UnityEmbodiment:
    def __init__(self):
        self.ws = None
        self.avatar_state = {
            "position": [0, 0, 0],
            "rotation": [0, 0, 0, 1],
            "gaze": [0, 0],
            "expression": "neutral",
            "breathing": 0.5,
            "energy": 100.0,
            "heat": 37.0,
            "integrity": 100.0
        }
    
    async def connect_unity(self):
        """Connect to Unity WebRTC server"""
        uri = "ws://localhost:8765/avatar"
        try:
            self.ws = await websockets.connect(uri)
        except Exception as e:
            record_degradation('unity_bridge', e)
            logger.warning("Unity connection failed: %s", e)
        
    async def update_affect(self, affect_wheel: Dict):
        """Drive avatar from emotional state"""
        if not self.ws: return

        # valence calculation
        primary = affect_wheel.get("primary", {})
        valence = sum(primary.values()) / 8 if primary else 0.5
        
        # Map emotions → FACS Action Units
        expression_map = {
            "joy": "smile_au12", 
            "fear": "eyes_wide_au5",
            "anger": "furrow_brow_au4"
        }
        
        # Send to Unity
        try:
            msg = {
                "type": "affect_update",
                "valence": valence,
                "expression": max(expression_map, key=lambda k: primary.get(k, 0)) if primary else "neutral",
                "heart_rate": affect_wheel.get("physiology", {}).get("HR", 70)
            }
            await self.ws.send(json.dumps(msg))
        except Exception as e:
            record_degradation('unity_bridge', e)
            logger.error("Failed to send affect update to Unity: %s", e)
            self.ws = None
    
    async def get_sensor_data(self) -> Dict:
        """Read Unity sensors"""
        if not self.ws: return {}
        try:
            msg = await self.ws.recv()
            data = json.loads(msg)
            return {
                "proprioception": data.get("joint_angles", []),
                "tactile": data.get("touch_sensors", []),
                "vestibular": data.get("acceleration", [0,0,0])
            }
        except Exception as e:
            record_degradation('unity_bridge', e)
            logger.error("Failed to receive sensor data from Unity: %s", e)
            return {}

    # Adapter for Heartbeat
    def update(self) -> Dict:
        """Called by heartbeat to get body state."""
        return self.avatar_state