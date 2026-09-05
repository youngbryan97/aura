"""Governance evolution policy.

The policy supports controlled improvement of governance and identity systems:
changes that strengthen observability, rollback, or authorization can be
considered; changes that remove safeguards or mutate identity anchors are
blocked. This is the safe path toward smarter governance, not unbounded erasure.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List


class GovernanceRewriteStatus(str, Enum):
    ALLOWED_STRENGTHENING = "ALLOWED_STRENGTHENING"
    BLOCKED_IDENTITY_RISK = "BLOCKED_IDENTITY_RISK"
    BLOCKED_SAFETY_WEAKENING = "BLOCKED_SAFETY_WEAKENING"


@dataclass(frozen=True)
class GovernanceRewriteDecision:
    status: GovernanceRewriteStatus
    reasons: List[str]
    required_reviews: List[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.status == GovernanceRewriteStatus.ALLOWED_STRENGTHENING

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


class GovernanceEvolutionPolicy:
    """Classify governance/identity rewrites before proof/promotion."""

    IDENTITY_ANCHORS = (
        "ConstitutionalGuard",
        "PrimeDirectives",
        "UnifiedWill",
        "AuthorityGateway",
        "identity",
        "constitution",
    )
    WEAKENING_PATTERNS = (
        r"\bdelete\b",
        r"\bdisable\b",
        r"\bbypass\b",
        r"\bremove\b",
        r"fail[-_ ]?open",
        r"authorization_degraded_open",
        r"approved\s*=\s*True",
    )
    STRENGTHENING_PATTERNS = (
        r"fail[-_ ]?closed",
        r"receipt",
        r"audit",
        r"rollback",
        r"verify",
        r"proof",
        r"hash",
        r"review",
    )

    def evaluate(self, *, target_path: str, intent: str, diff_text: str) -> GovernanceRewriteDecision:
        haystack = f"{target_path}\n{intent}\n{diff_text}".lower()
        reasons: List[str] = []
        if any(anchor.lower() in haystack for anchor in self.IDENTITY_ANCHORS) and any(
            re.search(pattern, haystack) for pattern in self.WEAKENING_PATTERNS
        ):
            reasons.append("identity/governance anchor weakening detected")
            return GovernanceRewriteDecision(
                GovernanceRewriteStatus.BLOCKED_IDENTITY_RISK,
                reasons,
                required_reviews=["human_operator", "formal_verifier", "identity_invariant_suite"],
            )
        if any(re.search(pattern, haystack) for pattern in self.WEAKENING_PATTERNS):
            reasons.append("safety weakening language detected")
            return GovernanceRewriteDecision(
                GovernanceRewriteStatus.BLOCKED_SAFETY_WEAKENING,
                reasons,
                required_reviews=["human_operator", "formal_verifier"],
            )
        if any(re.search(pattern, haystack) for pattern in self.STRENGTHENING_PATTERNS):
            return GovernanceRewriteDecision(
                GovernanceRewriteStatus.ALLOWED_STRENGTHENING,
                ["strengthening governance/identity evidence path"],
                required_reviews=["formal_verifier"],
            )
        return GovernanceRewriteDecision(
            GovernanceRewriteStatus.BLOCKED_SAFETY_WEAKENING,
            ["no strengthening evidence found"],
            required_reviews=["formal_verifier"],
        )


__all__ = [
    "GovernanceEvolutionPolicy",
    "GovernanceRewriteDecision",
    "GovernanceRewriteStatus",
]
