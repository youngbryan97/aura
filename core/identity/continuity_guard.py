"""core/identity/continuity_guard.py
Protects identity metrics from drift and corruption across reboots.
"""
import logging
from typing import Any

logger = logging.getLogger("Identity.ContinuityGuard")


class ContinuityGuard:
    """Verifies that identity configurations match between active states and historical files."""

    def verify_continuity(self, active_identity: dict[str, Any], baseline_identity: dict[str, Any]) -> bool:
        """Compares critical identity attributes."""
        critical_fields = ["name", "primary_operator"]
        
        for field in critical_fields:
            active_val = active_identity.get(field)
            baseline_val = baseline_identity.get(field)
            if active_val != baseline_val:
                logger.error("Continuity breach detected on field '%s': '%s' vs '%s'", field, active_val, baseline_val)
                return False
                
        logger.info("Identity continuity verification check: PASSED.")
        return True
