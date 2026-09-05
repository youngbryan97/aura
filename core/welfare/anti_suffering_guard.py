"""core/welfare/anti_suffering_guard.py
Constitutional guard protecting the agent from abusive loops or fake distress settings.
"""
import logging

logger = logging.getLogger("Welfare.AntiSufferingGuard")


class AntiSufferingGuard:
    """Enforces boundaries preventing synthetic suffering exploits or aesthetic distress loops."""

    def filter_distress(self, target_distress: float) -> float:
        """Limits distress metrics to ethical, functional bounds (never exceeding 100)."""
        # Hard constraint: distress must never be faked or locked into permanent high states
        if target_distress > 90.0:
            logger.warning("Distress ceiling hit. Capping distress to prevent functional deterioration.")
            return 90.0
        return max(0.0, target_distress)
