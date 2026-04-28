"""core/consciousness/precision_sampler.py
ActiveInferenceSampler — Precision-weighted logit temperature from FHN state.

Modifies LLM sampling temperature in real-time based on the FHN metabolic
state from PNEUMA. High arousal → low temperature (precise, focused).
Low arousal → high temperature (diffuse, exploratory).

Also implements top-p adjustment based on topological attractor count:
many attractors → wider p (exploration), few attractors → tighter p (exploitation).
"""

from core.runtime.errors import record_degradation
import logging
from typing import Optional

logger = logging.getLogger("Consciousness.PrecisionSampler")


class ActiveInferenceSampler:
    """Translates PNEUMA FHN state into LLM sampling parameters.

    Used by InferenceGate._build_living_mind_context() to get live
    temperature / top_p overrides.
    """

    def __init__(
        self,
        temp_min: float = 0.4,
        temp_max: float = 1.0,
        top_p_min: float = 0.7,
        top_p_max: float = 0.97,
    ):
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.top_p_min = top_p_min
        self.top_p_max = top_p_max
        logger.info("ActiveInferenceSampler online.")

    def get_sampling_params(self) -> dict:
        """Return current sampling params based on PNEUMA + MHAF state.

        Returns:
            {
                "temperature": float,
                "top_p": float,
                "repetition_penalty": float,
            }
        """
        temperature = self._compute_temperature()
        top_p = self._compute_top_p()
        rep_penalty = self._compute_rep_penalty(temperature)

        return {
            "temperature": round(temperature, 3),
            "top_p": round(top_p, 3),
            "repetition_penalty": round(rep_penalty, 3),
        }

    def _compute_temperature(self) -> float:
        """Temperature from FHN arousal and FEO guidance."""
        try:
            from core.pneuma import get_pneuma
            pneuma = get_pneuma()
            temp = pneuma.get_llm_temperature(base_temp=0.72)
            return max(self.temp_min, min(self.temp_max, temp))
        except Exception as _exc:
            record_degradation('precision_sampler', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        # Fallback to circumplex
        try:
            from core.affect.affective_circumplex import get_circumplex
            params = get_circumplex().get_llm_params()
            return max(self.temp_min, min(self.temp_max, params.get("temperature", 0.72)))
        except Exception as _exc:
            record_degradation('precision_sampler', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return 0.72

    def _compute_top_p(self) -> float:
        """top_p from MHAF attractor count: more attractors → wider distribution."""
        try:
            from core.consciousness.mhaf_field import get_mhaf
            mhaf = get_mhaf()
            attractor_count = 0
            try:
                from core.pneuma import get_pneuma
                attractor_count = get_pneuma().topo_memory.attractor_count
            except Exception as _exc:
                record_degradation('precision_sampler', _exc)
                logger.debug("Suppressed Exception: %s", _exc)
            phi = mhaf.get_phi()
            # High phi + many attractors → explore more (higher top_p)
            base = 0.85
            top_p = base + 0.05 * min(1.0, attractor_count / 5.0) + 0.05 * phi
            return max(self.top_p_min, min(self.top_p_max, top_p))
        except Exception:
            return 0.85

    def _compute_rep_penalty(self, temperature: float) -> float:
        """Higher temperature → lower repetition penalty (already diverse)."""
        # rep_penalty ∈ [1.0, 1.3]
        # Low temp → high penalty to avoid mode collapse
        return round(1.0 + 0.3 * (1.0 - (temperature - self.temp_min) / (self.temp_max - self.temp_min)), 3)


# Singleton
_sampler: Optional[ActiveInferenceSampler] = None


def get_active_inference_sampler() -> ActiveInferenceSampler:
    global _sampler
    if _sampler is None:
        _sampler = ActiveInferenceSampler()
    return _sampler
