import time
import logging
import copy
from typing import Dict, Any, Optional
from core.state.aura_state import AuraState, CognitiveMode

logger = logging.getLogger("Aura.PredictiveModel")

class PredictiveSelfModel:
    """Projective simulator for Aura's future states.
    
    This model allows the system to ask 'What will my state be in X minutes?'
    given current trends and cognitive pressure.
    """
    
    def __init__(self, current_state: Optional[AuraState] = None):
        self._cached_current = current_state

    def project_metabolism(self, state: AuraState, horizon_minutes: float) -> Dict[str, float]:
        """Project metabolic drive levels based on current decay rates."""
        projections = {}
        
        # Simple linear projection for now
        # Future evolution: use non-linear trends based on past delta_time
        for drive, params in state.motivation.budgets.items():
            level = params.get("level", 0.0)
            decay = params.get("decay", 0.0)
            
            # Modifier: If in DELIBERATE mode, energy decay is doubled
            if drive == "energy" and state.cognition.current_mode == CognitiveMode.DELIBERATE:
                decay *= 2.0
                
            predicted = max(0.0, level - (decay * horizon_minutes))
            projections[drive] = round(predicted, 2)
            
        return projections

    def estimate_mode_viability(self, state: AuraState, target_mode: CognitiveMode, duration_minutes: float) -> bool:
        """Estimate if a specific cognitive mode is sustainable for a given duration."""
        # 1. Project energy
        projections = self.project_metabolism(state, duration_minutes)
        projected_energy = projections.get("energy", 0.0)
        
        # 2. Threshold check
        if projected_energy < 5.0: # Hard floor
            return False
            
        if target_mode == CognitiveMode.DELIBERATE and projected_energy < 15.0:
            return False
            
        return True

    def suggest_optimal_mode(self, state: AuraState) -> CognitiveMode:
        """Suggest the best cognitive mode based on current drive pressures."""
        energy = state.motivation.budgets.get("energy", {}).get("level", 100.0)
        
        if energy < 10.0:
            return CognitiveMode.REACTIVE
        elif energy < 30.0:
            # Low energy but not critical -> DORMANT if not interacting
            return CognitiveMode.DORMANT if state.cognition.active_goals == [] else CognitiveMode.REACTIVE
        else:
            return CognitiveMode.DELIBERATE if state.cognition.active_goals else CognitiveMode.REACTIVE
