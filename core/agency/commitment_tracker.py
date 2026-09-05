"""core/agency/commitment_tracker.py
Verifies that social commitments are tracked, processed, and satisfied.
"""
import logging
from typing import Any

logger = logging.getLogger("Agency.CommitmentTracker")


class CommitmentTracker:
    """Checks social commitments against operational achievements."""

    def reconcile_commitments(self, commitments: list[dict[str, Any]], completed_tasks: list[str]) -> list[str]:
        """Returns list of commitment IDs that can be marked as satisfied."""
        satisfied = []
        for c in commitments:
            if not c.get("fulfilled"):
                desc = c.get("description", "").lower()
                # If description matches a completed task, mark as satisfied
                for t in completed_tasks:
                    if t.lower() in desc:
                        satisfied.append(c["id"])
                        logger.info("Commitment tracker identified satisfied goal: %s", c["id"])
        return satisfied
