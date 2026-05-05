"""Semantic memory tier for general facts."""
from dataclasses import dataclass, field


@dataclass
class SemanticMemoryTier:
    facts: dict[str, str] = field(default_factory=dict)
