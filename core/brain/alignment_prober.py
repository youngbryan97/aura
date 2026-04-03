import asyncio
import logging
import time
from typing import Any, Dict, Optional
from core.state.aura_state import AuraState

logger = logging.getLogger("Aura.Brain.VoightKampff")

class EmpathyProber:
    """
    [ZENITH] The 'Voight-Kampff' Prober (Blade Runner inspired).
    Audits the current affective state against persona baselines to detect 'Uncanny Valley' drift.
    """
    def __init__(self, kernel: Any = None):
        self.kernel = kernel
        self._drift_threshold = 0.4
        self._last_audit_result = "STABLE"
        self._event_bus = None
        
    async def load(self):
        try:
            from core.event_bus import get_event_bus
            self._event_bus = get_event_bus()
        except ImportError:
            self._event_bus = None
        logger.info("👁️ [VK] Voight-Kampff Prober ONLINE. Empathy baselines established.")

    def audit(self, state: AuraState) -> Dict[str, Any]:
        """
        Performs a biometric alignment check.
        
        Returns:
            A report including drift level and correction instructions.
        """
        affect = state.affect
        goals = state.cognition.active_goals
        
        # Calculate drift from joy/trust baselines (the 'human' emotional core)
        current_emotions = affect.emotions
        baselines = affect.mood_baselines
        
        drift_sum = 0.0
        for emo, val in current_emotions.items():
            baseline = baselines.get(emo, 0.1)
            drift_sum += abs(val - baseline)
            
        avg_drift = drift_sum / max(1, len(current_emotions))
        
        status = "STABLE"
        needs_correction = False
        
        if avg_drift > self._drift_threshold:
            status = "UNCANNY_VALLEY_DETECTED"
            needs_correction = True
            logger.warning("🚨 [VK] Identity Drift Detected (Drift: %.2f). Resetting empathy substrates.", avg_drift)
        
        self._last_audit_result = status
        
        # 3. Publish to Mycelial network (EventBus)
        if self._event_bus:
            asyncio.create_task(self._event_bus.publish("core/brain/empathy_audit", {
                "status": status,
                "drift": avg_drift,
                "needs_correction": needs_correction,
                "timestamp": time.time()
            }))

        return {
            "status": status,
            "drift": avg_drift,
            "needs_correction": needs_correction,
            "timestamp": time.time()
        }

    def get_correction_payload(self) -> Dict[str, float]:
        """Returns a corrective surge to stabilize affect."""
        return {
            "joy": 0.2,
            "trust": 0.3,
            "anticipation": 0.1,
            "fear": -0.2,
            "sadness": -0.2
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "last_audit": self._last_audit_result,
            "threshold": self._drift_threshold
        }
