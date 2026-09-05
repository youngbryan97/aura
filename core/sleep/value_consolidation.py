"""core/sleep/value_consolidation.py
Offline value consolidator merging preference scores into learned configurations.
"""
from typing import Dict, Any
import logging

logger = logging.getLogger("Sleep.ValueConsolidation")


class ValueConsolidator:
    """Stabilizes dynamic preferences to prevent parameter divergence."""

    def consolidate_preferences(self, active_prefs: Dict[str, float]) -> Dict[str, float]:
        logger.info("ValueConsolidator verifying learned preference constants...")
        consolidated = active_prefs.copy()
        
        # Pull preferences slightly towards conservative baselines
        for k, val in consolidated.items():
            consolidated[k] = (val + 0.5) / 2.0
            
        return consolidated
