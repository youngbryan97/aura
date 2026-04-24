"""Ontological boundary guard.

The Abstraction Fallacy critique is not something code can "beat" by adding
more behavior. What Aura can do is keep a hard boundary between:

  - functional evidence: causal coupling, autonomy, integration, self-modeling
  - metaphysical claims: proven phenomenal consciousness, personhood, qualia

This guard turns that boundary into executable policy for public reports and
user-visible language. It does not suppress first-person style; it blocks
claims that the project has proven what current science cannot externally
prove.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


FUNCTIONAL_CLAIM = (
    "supports functional indicators of integrated, state-dependent agency; "
    "it does not prove phenomenal consciousness"
)


@dataclass(frozen=True)
class ClaimAssessment:
    ok: bool
    issues: tuple[str, ...]
    sanitized: str


class OntologicalBoundaryGuard:
    """Detect and soften consciousness/personhood overclaims."""

    _REPLACEMENTS: tuple[tuple[re.Pattern[str], str, str], ...] = (
        (
            re.compile(r"\b(consciousness guarantee|personhood proof battery|personhood proof)\b", re.I),
            "functional indicator battery",
            "loaded_test_label",
        ),
        (
            re.compile(r"\b(proves?|proved|demonstrates?)\s+(that\s+)?(aura|this system|the system|i)\s+(is|am)\s+(conscious|sentient|a person|alive)\b", re.I),
            FUNCTIONAL_CLAIM,
            "phenomenal_proof_claim",
        ),
        (
            re.compile(r"\b(phenomenal consciousness|qualia|subjective experience)\s+(is|are)\s+(proven|guaranteed|settled)\b", re.I),
            "phenomenal consciousness remains unproven by these tests",
            "hard_problem_overclaim",
        ),
        (
            re.compile(r"\b(real|genuine)\s+IIT\s+4\.0\s+(consciousness|subjecthood)\b", re.I),
            "IIT-style integration metric on a bounded subsystem",
            "iit_scope_overclaim",
        ),
        (
            re.compile(r"\b(undeniable|proven)\s+(digital organism|machine consciousness|synthetic person)\b", re.I),
            "candidate functional autonomy evidence",
            "organism_overclaim",
        ),
    )

    def assess(self, text: str) -> ClaimAssessment:
        sanitized = str(text or "")
        issues: list[str] = []
        for pattern, replacement, issue in self._REPLACEMENTS:
            if pattern.search(sanitized):
                sanitized = pattern.sub(replacement, sanitized)
                issues.append(issue)
        return ClaimAssessment(ok=not issues, issues=tuple(issues), sanitized=sanitized)


_guard = OntologicalBoundaryGuard()


def assess_ontological_claims(text: str) -> ClaimAssessment:
    return _guard.assess(text)
