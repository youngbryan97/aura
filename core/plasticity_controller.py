"""core/plasticity_controller.py — Learning Rate Management for Aura Zenith.

Coordinates high-plasticity learning phases with low-plasticity consolidation phases.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("Aura.Plasticity")

class PlasticityMode(Enum):
    LEARNING = "learning"      # High delta, high intake
    CONSOLIDATING = "consolidating" # Transitioning, dream state trigger
    STABLE = "stable"          # Low delta, high integration

@dataclass
class PlasticityState:
    score: float # 0.0 to 1.0
    mode: PlasticityMode
    last_belief_count: int
    timestamp: float = field(default_factory=time.time)

class PlasticityController:
    """Governing Aura's learning vs consolidation rhythms."""
    
    def __init__(self):
        self.state = PlasticityMode.STABLE
        self.plasticity_score = 0.5
        self._history: List[Dict[str, Any]] = []
        self._last_check = 0.0
        logger.info("PlasticityController initialized.")

    async def update_plasticity(self) -> float:
        """Compute current plasticity based on belief deltas and insight flow."""
        from core.container import ServiceContainer
        belief_system = ServiceContainer.get("belief_graph", default=None)
        insight_journal = ServiceContainer.get("insight_journal", default=None)
        
        if not (belief_system and insight_journal):
            return 0.5

        # 1. Delta check
        current_beliefs = len(belief_system.get_all_beliefs())
        # We'd compare vs history.
        # Logic: If many new beliefs/insights in last hour -> high plasticity.
        
        # 2. Heuristic for now
        recent_insights = len(insight_journal.get_highest_confidence_insights(limit=10))
        
        score = min(1.0, recent_insights * 0.1) # Simplified score
        self.plasticity_score = score
        
        # 3. Transition modes
        if score > 0.7:
            change = (self.state != PlasticityMode.LEARNING)
            self.state = PlasticityMode.LEARNING
            if change: logger.info("🧠 Plasticity: Entering HIGH LEARNING mode. Throttling challenges.")
        elif score < 0.3:
            change = (self.state != PlasticityMode.STABLE)
            self.state = PlasticityMode.STABLE
            if change: logger.info("🧠 Plasticity: Stabilized. Initiating consolidation bias.")
        else:
            self.state = PlasticityMode.CONSOLIDATING
            
        return score

    def get_rate_multiplier(self, system_name: str) -> float:
        """Return a multiplier (0.0 to 1.0) for a learning system's cycle rate."""
        if self.state == PlasticityMode.LEARNING:
            if system_name == "belief_challenger": return 0.2 # Don't challenge while building
            return 1.2 # Speed up inquiry
        elif self.state == PlasticityMode.STABLE:
            if system_name == "belief_challenger": return 1.5 # Challenge more when stable
            return 0.5 # Slow down inquiry
        return 1.0

# Service Registration
def register_plasticity_controller():
    """Register the plasticity controller."""
    from core.container import ServiceContainer, ServiceLifetime
    ServiceContainer.register(
        "plasticity_controller",
        factory=lambda: PlasticityController(),
        lifetime=ServiceLifetime.SINGLETON
    )
