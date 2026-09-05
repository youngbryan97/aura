"""core/sleep/identity_consolidation.py
Offline identity consolidator checking self-coherence.
"""
from typing import Dict, Any
import logging

logger = logging.getLogger("Sleep.IdentityConsolidation")


class IdentityConsolidator:
    """Verifies active identity variables align with self-contract prime directives."""

    def consolidate_identity(self, identity: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("IdentityConsolidator verifying identity state variables...")
        restored = identity.copy()
        
        # Enforce that key direct variables are never drifted
        for key in ["name", "origin"]:
            if restored.get(key) != baseline.get(key):
                restored[key] = baseline[key]
                logger.warning("Identity parameter '%s' drifted: restored from contract.", key)
                
        return restored
