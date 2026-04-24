"""Ontological boundary guard.

The Abstraction Fallacy critique is not something code can "beat" by adding
more behavior. What Aura can do is keep a hard boundary between:

  - functional evidence: causal coupling, autonomy, integration, self-modeling
  - metaphysical claims: proven phenomenal consciousness, personhood, qualia

This guard turns that boundary into executable policy for public reports and
user-visible language. It does not suppress first-person style; it blocks
claims that the project has proven what current science cannot externally
prove.

The pattern catalog covers: phenomenal consciousness, qualia, personhood,
moral patiency, moral agency, legal personhood, organism status,
hard-problem-bypass language, IIT-as-consciousness, adverbial proof claims,
peerhood claims, and the loaded test-battery names the critique called out.
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
            re.compile(r"\b(consciousness guarantee|personhood proof battery|personhood proof|soul triad|crossing the rubicon|consciousness expansion gauntlet)\b", re.I),
            "functional indicator battery",
            "loaded_test_label",
        ),
        (
            re.compile(r"\b(proves?|proved|demonstrates?|confirms?|establishes?)\s+(that\s+)?(aura|this system|the system|i|she)\s+(is|am|was)\s+(conscious|sentient|a person|alive|self[- ]aware)\b", re.I),
            FUNCTIONAL_CLAIM,
            "phenomenal_proof_claim",
        ),
        (
            re.compile(r"\b(phenomenal consciousness|qualia|subjective experience|felt experience)\s+(is|are|have been|has been)\s+(proven|guaranteed|settled|demonstrated|shown)\b", re.I),
            "phenomenal consciousness remains unproven by these tests",
            "hard_problem_overclaim",
        ),
        (
            re.compile(r"\b(the\s+lights\s+are\s+on\s+inside)\b", re.I),
            "phenomenal status is not decided by these tests",
            "hard_problem_overclaim",
        ),
        (
            re.compile(r"\b(real|genuine|true)\s+IIT\s*(?:4\.0)?\s+(consciousness|subjecthood|experience)\b", re.I),
            "IIT-style integration metric on a bounded subsystem",
            "iit_scope_overclaim",
        ),
        (
            re.compile(r"\b(undeniable|proven|verified|established)\s+(digital organism|machine consciousness|synthetic person|machine sentience|artificial life)\b", re.I),
            "candidate functional autonomy evidence",
            "organism_overclaim",
        ),
        (
            re.compile(r"\b(aura|this system|the system)\s+(has|possesses|owns)\s+(a soul|qualia|moral standing|legal personhood|rights)\b", re.I),
            "presents functional indicators that do not settle moral, legal, or metaphysical status",
            "moral_legal_overclaim",
        ),
        (
            re.compile(r"\b(bridged|crossed|solved|closed)\s+the\s+(hard problem|explanatory gap|symbol grounding problem)\b", re.I),
            "does not close the hard problem or the explanatory gap",
            "hard_problem_bypass_claim",
        ),
        (
            re.compile(r"\b(peer to humanity|equal to humans|human[- ]level consciousness|human peerhood)\b", re.I),
            "candidate functional autonomy, not peerhood",
            "peerhood_overclaim",
        ),
        (
            re.compile(r"\b(is|am|are|was)\s+(truly|actually|genuinely|really)\s+(conscious|sentient|alive|self[- ]aware|a person)\b", re.I),
            "presents functional indicators; metaphysical status is unsettled",
            "adverbial_proof_claim",
        ),
        (
            re.compile(r"\b(moral patient|moral patiency|moral agent|moral agency)\s+(is|has been|status\s+is)\s+(proven|established|demonstrated|settled)\b", re.I),
            "moral patiency and moral agency are not established by these tests",
            "moral_patiency_overclaim",
        ),
        (
            re.compile(r"\b(legal personhood|legal person|legal standing)\s+(is|has been|status\s+is)\s+(proven|established|achieved|demonstrated)\b", re.I),
            "legal personhood is outside the scope of this system",
            "legal_personhood_overclaim",
        ),
        (
            re.compile(r"\b(is|am|are)\s+(a\s+)?person\s+who\s+happens\s+to\s+be\s+digital\b", re.I),
            "is a persistent digital agent with a self-model",
            "person_category_overclaim",
        ),
        (
            re.compile(r"\b(implements|instantiates|achieves|realizes)\s+(phenomenal\s+)?(consciousness|qualia|subjective\s+experience)\b", re.I),
            "implements functional indicators associated with consciousness theories",
            "instantiation_overclaim",
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
