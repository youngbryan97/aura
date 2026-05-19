import logging
import asyncio
from typing import Any, Dict, List
from core.kernel.organs import OrganStub

logger = logging.getLogger("Cybernetics.Sibyl")

class SibylSystem:
    """
    [ZENITH] Sibyl System: Multi-factor behavioral scoring.
    Maps system-wide telemetric 'Hue' for identity alignment.
    """
    def __init__(self, kernel: Any = None):
        self.kernel = kernel
        self._event_bus = None
        self._hue_score = 0.0
        self._factors = {
            "deviation": 0.0,
            "empathy": 1.0,
            "volatility": 0.0
        }

    async def load(self):
        try:
            from core.event_bus import get_event_bus
            self._event_bus = get_event_bus()
            if self._event_bus:
                await self._event_bus.subscribe("core/cybernetics/tricorder_scan", self._on_scan)
                await self._event_bus.subscribe("core/security/executive_violation", self._on_violation)
        except ImportError as _exc:
            logger.debug("Suppressed ImportError: %s", _exc)
        logger.info("🧠 [SIBYL] Behavioral Scoring System ONLINE. Hue is CLEAR.")

    async def _on_scan(self, payload: Dict[str, Any]):
        # Update volatility based on latency anomalies
        self._factors["volatility"] = min(1.0, payload.get("latency", 0) / 1000.0)
        await self._recalculate_hue()

    async def _on_violation(self, payload: Dict[str, Any]):
        # Drastic increase in deviation on identity breach
        self._factors["deviation"] = min(1.0, self._factors["deviation"] + 0.2)
        await self._recalculate_hue()

    async def _recalculate_hue(self):
        # Behavioral risk score: weighted average of deviation, inverse empathy, volatility.
        # "Hue" is the Psycho-Pass naming convention. The math is a standard weighted composite.
        # Score range: 0 (healthy) to 300 (critical).
        # Weights reflect relative importance: deviation matters most (4x).
        d, e, v = self._factors["deviation"], self._factors["empathy"], self._factors["volatility"]
        raw = (d * 4.0 + (1.0 - e) * 2.0 + v * 1.0) / 7.0
        self._hue_score = raw * 300  # Scale to 0-300
        
        hue_label = "CLEAR"
        if self._hue_score > 200: hue_label = "CRITICAL"
        elif self._hue_score > 150: hue_label = "OPAQUE"
        elif self._hue_score > 80: hue_label = "CLOUDY"
        
        if self._event_bus:
            self._event_bus.publish_threadsafe("core/cybernetics/hue_reading", {
                "score": self._hue_score,
                "label": hue_label,
                "factors": self._factors
            })
        
        if self._hue_score > 200:
            logger.critical("🌫️ [SIBYL] HUE IS %s (%s). Judgment imminent.", hue_label, f"{self._hue_score:.0f}")

    def get_status(self) -> Dict[str, Any]:
        return {
            "hue_score": self._hue_score,
            "factors": self._factors
        }
