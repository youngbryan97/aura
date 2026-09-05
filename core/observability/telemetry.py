import logging
import time
from typing import Any

from core.runtime import resource_psutil as psutil

logger = logging.getLogger("Aura.Telemetry")

class TelemetryEngine:
    """
    Central 2026 telemetry system.
    Aggregates cognitive, hardware, and emotional state into unified pulses.
    """
    def __init__(self) -> None:
        self.last_pulse = time.time()
        
    def get_pulse(self) -> dict[str, Any]:
        """Collects a complete snapshot of system state."""
        try:
            from core.runtime import CoreRuntime
            rt = CoreRuntime.get_sync()
        except (ImportError, AttributeError, RuntimeError):
            return {"type": "telemetry", "cpu_usage": 0, "memory_usage": 0}

        # Hardware
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        battery = psutil.sensors_battery()
        
        # Cognitive (from Substrate)
        liquid_state = {}
        ls = rt.container.get("liquid_state")
        if ls:
            liquid_state = {
                "valence": getattr(ls, 'valence', 0.0),
                "arousal": getattr(ls, 'arousal', 0.0),
                "dominance": getattr(ls, 'dominance', 0.0)
            }
            
        # Agency
        orch = rt.container.get("orchestrator")
        cycle_count = getattr(orch, 'cycle_count', 0) if orch else 0
        
        pulse = {
            "type": "telemetry",
            "timestamp": time.time(),
            "cpu_usage": cpu,
            "memory_usage": mem,
            "battery_level": battery.percent if battery else 100,
            "liquid_state": liquid_state,
            "cycle_count": cycle_count,
            "version": "2026.3.8-TRANSVERSE"
        }
        
        return pulse

    async def broadcast_pulse(self) -> None:
        """Dispatches telemetry pulse to the EventBus/HUD."""
        from core.event_bus import get_event_bus
        pulse = self.get_pulse()
        await get_event_bus().publish("telemetry", pulse)
