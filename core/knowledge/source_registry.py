"""Knowledge source registry for grounded environment rules."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KnowledgeSource:
    source_id: str
    domain: str
    title: str
    trust_level: float
    allowed_for_eval: bool
    content_hash: str


class KnowledgeSourceRegistry:
    def __init__(self) -> None:
        self.sources: dict[str, KnowledgeSource] = {}

    def register(self, source: KnowledgeSource) -> None:
        self.sources[source.source_id] = source

    def get(self, source_id: str) -> KnowledgeSource:
        return self.sources[source_id]


__all__ = ["KnowledgeSource", "KnowledgeSourceRegistry"]
