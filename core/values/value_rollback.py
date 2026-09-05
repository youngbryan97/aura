"""core/values/value_rollback.py
Rolls back preference calibrations if they lead to degradation or failures.
"""
import logging
from typing import Any

logger = logging.getLogger("Values.ValueRollback")


class ValueRollbackManager:
    """Manages restoration of prior stable preference snapshots."""

    def rollback(self, current: dict[str, float], history: list[dict[str, Any]]) -> dict[str, float]:
        if not history:
            logger.info("No prior value snapshots available for rollback.")
            return current
            
        # Restore the most recent stable snapshot
        logger.warning("Rolling back active preferences to last stable checkpoint.")
        return history[-1].get("variables", current)
