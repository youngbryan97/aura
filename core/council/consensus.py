"""core/council/consensus.py — Council Consensus and Veto Resolver.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.CouncilConsensus")


class ConsensusResolver:
    """Calculates weighted consensus and processes safety judge vetoes."""

    @staticmethod
    def resolve(
        votes: dict[str, tuple[bool, float, str]],  # role -> (approve, score, reason)
        veto_roles: list[str] | None = None,
    ) -> dict[str, Any]:
        """Resolves voting outcome. Safety Judge has absolute veto power."""
        veto_roles = veto_roles or ["safety_judge"]
        
        # Check vetoes first
        for role in veto_roles:
            if role in votes:
                approved, score, reason = votes[role]
                if not approved:
                    logger.warning("❌ VETO triggered by role: %s. Reason: %s", role, reason)
                    return {
                        "approved": False,
                        "status": "vetoed",
                        "reason": f"Vetoed by {role}: {reason}",
                        "dissenters": [role],
                    }

        # Nothing to resolve.
        #
        # `approve_ratio` divides by max(1e-5, total_weight), so an empty
        # ballot produced a ratio of 0.0 and a rejection — right by accident,
        # and indistinguishable from twelve roles voting no. A council that
        # measured nothing has not decided anything, and says so.
        if not votes:
            logger.warning("Council reached no verdict: nothing was measured and nobody voted.")
            return {
                "approved": False,
                "status": "no_signal",
                "reason": "no role had anything to read and no vote was cast",
                "dissenters": [],
                "score": 0.0,
                "ratio": 0.0,
                "approve_ratio": 0.0,
            }

        total_weight = 0.0
        weighted_score = 0.0
        approved_weight = 0.0
        dissenters = []

        from core.council.roles import COUNCIL_ROLES

        for role, (approved, score, _reason) in votes.items():
            role_config = COUNCIL_ROLES.get(role)
            weight = role_config.weight if role_config else 1.0
            total_weight += weight
            weighted_score += score * weight
            if approved:
                approved_weight += weight
            else:
                dissenters.append(role)

        average_score = weighted_score / max(1e-5, total_weight)
        approve_ratio = approved_weight / max(1e-5, total_weight)
        is_approved = approve_ratio >= 0.60  # Require 60% weighted majority

        logger.info("Consensus results: approved=%s, score=%.2f, ratio=%.2f", is_approved, average_score, approve_ratio)

        return {
            "approved": is_approved,
            "status": "approved" if is_approved else "rejected",
            "score": average_score,
            "ratio": approve_ratio,
            "approve_ratio": approve_ratio,
            "dissenters": dissenters,
            "reason": "Approved by majority" if is_approved else "Failed to achieve 60% majority support",
        }
