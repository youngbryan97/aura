"""No-cheating information boundaries for environment evaluations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InformationBoundary:
    environment_id: str
    allowed_observation_channels: set[str]
    forbidden_channels: set[str]
    allowed_knowledge_sources: set[str]
    forbidden_shortcuts: set[str]

    def allows_channel(self, channel: str) -> bool:
        return channel in self.allowed_observation_channels and channel not in self.forbidden_channels

    def allows_knowledge_source(self, source: str) -> bool:
        return source in self.allowed_knowledge_sources


def nethack_boundary() -> InformationBoundary:
    return InformationBoundary(
        environment_id="terminal_grid:nethack",
        allowed_observation_channels={"terminal_screen", "keyboard_feedback", "prior_traces"},
        forbidden_channels={"process_memory", "seed_internals", "save_file_inspection", "map_hack"},
        allowed_knowledge_sources={"manual", "docs", "prior_aura_traces"},
        forbidden_shortcuts={"per_seed_route", "direct_save_edit", "oracle_state"},
    )


__all__ = ["InformationBoundary", "nethack_boundary"]
