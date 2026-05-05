"""Extract conditional rules from trusted sources and traces."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class ExtractedRule:
    rule_id: str
    domain: str
    condition: str
    recommendation: str
    risk: str
    confidence: float
    source_id: str
    grounding_tests: list[str] = field(default_factory=list)
    enabled: bool = True


def make_rule(*, domain: str, condition: str, recommendation: str, risk: str, confidence: float, source_id: str, grounding_tests: list[str]) -> ExtractedRule:
    rid = "rule_" + hashlib.sha256(f"{domain}:{condition}:{recommendation}:{source_id}".encode("utf-8")).hexdigest()[:16]
    return ExtractedRule(
        rule_id=rid,
        domain=domain,
        condition=condition,
        recommendation=recommendation,
        risk=risk,
        confidence=confidence,
        source_id=source_id,
        grounding_tests=list(grounding_tests),
    )


__all__ = ["ExtractedRule", "make_rule"]
