"""core/values/value_rollback.py
Rolls back preference calibrations if they lead to degradation or failures.
"""
from typing import Dict, List, Any
import logging

logger = logging.getLogger("Values.ValueRollback")


class ValueRollbackManager:
    """Manages restoration of prior stable preference snapshots."""

    def rollback(self, current: Dict[str, float], history: List[Dict[str, Any]]) -> Dict[str, float]:
        if not history:
            logger.info("No prior value snapshots available for rollback.")
            return current
            
        # Restore the most recent stable snapshot
        logger.warning("Rolling back active preferences to last stable checkpoint.")
        return history[-1].get("variables", current)
