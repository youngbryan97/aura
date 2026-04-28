"""Singularity Monitor — Phase 20.3
Tracks the rate of self-improvement and enables Accelerated Cognition.
"""
from core.runtime.errors import record_degradation
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("Aura.Singularity")

class SingularityMonitor:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.improvement_rate = 0.0
        self.last_health_score = 1.0
        self.acceleration_factor = 1.0
        self.is_accelerated = False
        self.milestones_reached = 0

    def pulse(self):
        """Standard heartbeat called from the Orchestrator loop."""
        try:
            # 1. Fetch Mirror Health
            metacognition = getattr(self.orchestrator, 'metacognition', None)
            if not metacognition and hasattr(self.orchestrator, 'container'):
                 from core.container import ServiceContainer
                 metacognition = ServiceContainer.get("metacognition", default=None)
            
            if metacognition and hasattr(metacognition, "mirror"):
                summary = metacognition.mirror.get_audit_summary()
                health = summary.get("health_score", 1.0)
                
                # 2. Calculate Improvement Rate (Delta health + complexity growth)
                delta = health - self.last_health_score
                self.improvement_rate = (self.improvement_rate * 0.9) + (delta * 0.1)
                self.last_health_score = health
                
                # 3. Check for Acceleration Threshold
                # If health is high and improvement is positive, we enter Accelerated state
                if health > 0.9 and self.improvement_rate > -0.01:
                    if not self.is_accelerated:
                        logger.info("⚡ [SINGULARITY] Accelerated Cognition Enabled.")
                        self.is_accelerated = True
                        self._apply_acceleration(1.5)
                elif health < 0.7:
                    if self.is_accelerated:
                        logger.warning("📉 [SINGULARITY] Accelerated Cognition Suspended due to recursive instability.")
                        self.is_accelerated = False
                        self._apply_acceleration(1.0)
                        
        except Exception as e:
            record_degradation('singularity_monitor', e)
            logger.debug("Singularity pulse failed: %s", e)

    def _apply_acceleration(self, factor: float):
        """Inject parameters into the Cognitive Engine."""
        self.acceleration_factor = factor
        ce = getattr(self.orchestrator, 'cognitive_engine', None)
        if ce:
             # Inject deep-seated bias into thinking modes
             # This is a 'latent' shift in how Aura perceives her own throughput
             logger.info("🧠 Injecting Acceleration Factor: %.1fx", factor)
             # We can't directly modify LLM params easily here without a hook,
             # so we'll set a state flag that the CognitiveEngine can check.
             setattr(ce, "singularity_factor", factor)

    def get_status(self) -> Dict[str, Any]:
        return {
            "improvement_rate": round(self.improvement_rate, 4),
            "acceleration": self.acceleration_factor,
            "is_accelerated": self.is_accelerated,
            "status": "ACCELERATED" if self.is_accelerated else "STABLE"
        }
