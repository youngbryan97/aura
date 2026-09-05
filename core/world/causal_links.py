"""core/world/causal_links.py — Cause and Effect Dependency Mapping.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger("Aura.CausalLinks")


@dataclass
class CausalLink:
    cause_event_id: str
    effect_event_id: str
    confidence: float = 0.50
    mechanism: str = "unknown"  # E.g. "supply_chain", "dependency_drift", "direct_action"


class CausalEngine:
    """Stores and reasons over predictive cause-and-effect relationships."""

    def __init__(self) -> None:
        self.links: List[CausalLink] = []

    def record_link(self, link: CausalLink) -> None:
        self.links.append(link)
        logger.info("🔗 Causal link recorded: %s -> %s (conf: %.2f)", 
                    link.cause_event_id, link.effect_event_id, link.confidence)

    def get_effects(self, cause_event_id: str) -> List[CausalLink]:
        return [lnk for lnk in self.links if lnk.cause_event_id == cause_event_id]

    def get_causes(self, effect_event_id: str) -> List[CausalLink]:
        return [lnk for lnk in self.links if lnk.effect_event_id == effect_event_id]


# Singleton
_causal_engine_instance: CausalEngine | None = None


def get_causal_engine() -> CausalEngine:
    global _causal_engine_instance
    if _causal_engine_instance is None:
        _causal_engine_instance = CausalEngine()
    return _causal_engine_instance
