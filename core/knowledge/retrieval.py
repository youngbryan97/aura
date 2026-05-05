"""Context-matched retrieval for extracted rules."""
from __future__ import annotations

from .rule_extractor import ExtractedRule


class KnowledgeRetriever:
    def __init__(self) -> None:
        self.rules: dict[str, ExtractedRule] = {}

    def add_rule(self, rule: ExtractedRule) -> None:
        self.rules[rule.rule_id] = rule

    def retrieve(self, *, domain: str, context: str) -> list[ExtractedRule]:
        context_lower = context.lower()
        matches = [
            rule
            for rule in self.rules.values()
            if rule.enabled
            and rule.domain == domain
            and any(token in context_lower for token in rule.condition.lower().split())
        ]
        return sorted(matches, key=lambda rule: rule.confidence, reverse=True)


__all__ = ["KnowledgeRetriever"]
