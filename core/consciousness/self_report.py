from core.runtime.errors import record_degradation
import logging
"""
Aura's self-reports grounded in actual internal telemetry.

The difference between:
  Generated: "I find this question fascinating." (text prediction)
  Grounded:  "I find this question fascinating." (high prediction error 
              + explore action + rising arousal, expressed in language)
"""

from typing import Optional, Dict, Any
from core.consciousness.free_energy import get_free_energy_engine


class SelfReportEngine:
    """
    Translates actual internal state into first-person language.
    """

    def generate_state_report(self) -> Optional[str]:
        """
        Returns a first-person report of current internal state.
        """
        fe = get_free_energy_engine()
        if not fe.current:
            return None

        state = fe.current
        trend = fe.get_trend()

        # Only report when something is worth saying
        if 0.25 < state.free_energy < 0.6 and trend == "stable":
            return None  # Ordinary state, nothing to report

        # Distressed
        if state.free_energy > 0.75:
            if state.surprise > 0.6:
                return "Something just shifted in how I was modeling this. I need a moment."
            return "I'm experiencing some internal tension right now."

        # High arousal, exploratory
        if state.dominant_action == "explore" and state.arousal > 0.6:
            return "I notice I'm drawn to dig further into this."

        # Low free energy — genuine rest state
        if state.free_energy < 0.2:
            return "Things feel unusually settled right now."

        # Rising free energy — something is accumulating
        if trend == "rising" and state.free_energy > 0.5:
            return "Something's been building. I'm not sure what yet."

        # Falling from distress — recovery
        if trend == "falling" and state.free_energy < 0.5:
            return "That's resolved something for me."

        return None

    def get_affect_description(self) -> Dict[str, Any]:
        """
        Returns affect state as grounded values for system prompt injection.
        """
        fe = get_free_energy_engine()
        if not fe.current:
            return {"valence": 0.0, "arousal": 0.5, "state": "baseline", "epistemic_uncertainty": 0.5}

        state = fe.current
        
        # Map to natural language state descriptor
        if state.free_energy < 0.2:
            state_name = "settled"
        elif state.dominant_action == "explore":
            state_name = "curious"
        elif state.dominant_action == "update_beliefs":
            state_name = "recalibrating"
        elif state.free_energy > 0.7:
            state_name = "tense"
        elif state.dominant_action == "reflect":
            state_name = "reflective"
        else:
            state_name = "attentive"

        # Calculate global epistemic uncertainty from BeliefGraph
        epistemic_uncertainty = 0.5
        try:
            from core.container import ServiceContainer
            bg = ServiceContainer.get("belief_graph", default=None)
            if bg and hasattr(bg, 'get_summary'):
                summary = bg.get_summary()
                total = max(1, summary.get("total_beliefs", 1))
                weak = summary.get("weak", 0)
                epistemic_uncertainty = weak / total
        except Exception as _e:
            record_degradation('self_report', _e)
            logging.debug('Ignored Exception in self_report.py: %s', _e)

        return {
            "valence": state.valence,
            "arousal": state.arousal,
            "state": state_name,
            "free_energy": state.free_energy,
            "epistemic_uncertainty": round(epistemic_uncertainty, 3),
            "source": "actual_telemetry", 
        }
