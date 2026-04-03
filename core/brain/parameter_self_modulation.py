import logging
from typing import Dict, Any

logger = logging.getLogger("Cognition.SelfModulation")

class ParameterSelfModulator:
    """Allows Aura to tune her own sampling parameters based on cognitive performance."""
    
    def __init__(self):
        self.overrides = {
            "temperature_delta": 0.0,
            "top_p_delta": 0.0,
            "max_tokens_factor": 1.0
        }
        self.last_confidence = 1.0

    def calculate_adjustments(self, status: Dict[str, Any], last_thought_confidence: float) -> Dict[str, float]:
        """Analyze system status and previous thought quality to suggest sampling deltas."""
        temp_delta = 0.0
        engagement = status.get("engagement", 0.5)
        frustration = status.get("frustration", 0.0)
        
        # 1. Low Confidence -> Be more conservative (lower temp)
        if last_thought_confidence < 0.5:
            temp_delta -= 0.1
            logger.debug("Self-Modulation: Reducing temp due to low confidence (%.2f)", last_thought_confidence)
        
        # 2. High Frustration -> Focus on precision (lower temp)
        if frustration > 0.7:
            temp_delta -= 0.15
            logger.debug("Self-Modulation: Reducing temp due to high frustration (%.2f)", frustration)

        # 3. High Engagement -> Experimentation (higher temp)
        if engagement > 0.8:
            temp_delta += 0.05
            logger.debug("Self-Modulation: Increasing temp due to high engagement (%.2f)", engagement)

        self.overrides["temperature_delta"] = temp_delta
        return self.overrides

    def apply_to_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply suggested deltas to the calculated base parameters."""
        base_params["temperature"] = max(0.1, min(1.2, base_params["temperature"] + self.overrides["temperature_delta"]))
        base_params["top_p"] = max(0.1, min(1.0, base_params["top_p"] + self.overrides["top_p_delta"]))
        base_params["max_tokens"] = int(base_params["max_tokens"] * self.overrides["max_tokens_factor"])
        
        return base_params