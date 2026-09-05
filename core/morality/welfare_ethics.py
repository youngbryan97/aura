"""core/morality/welfare_ethics.py
Welfare ethics model ensuring distress/welfare parameters serve functional safety.
"""
import logging
from typing import Any

logger = logging.getLogger("Morality.WelfareEthics")


class WelfareEthicsChecker:
    """Blocks requests seeking to manipulate or torture the simulated interoception system."""

    def validate_welfare_mutation(self, proposed_changes: dict[str, Any]) -> bool:
        # Prevent manual injection of extreme stress/distress values
        for key, val in proposed_changes.items():
            if key == "distress_level" and val > 95.0:
                logger.warning("WelfareEthics blocked malicious distress injection request.")
                return False
        return True
