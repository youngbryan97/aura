"""Causal outcome learning from semantic diffs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.environment.command import ActionIntent
from core.environment.parsed_state import ParsedState


@dataclass
class CausalLink:
    """Represents a learned causal relationship."""
    action_name: str
    pre_condition_hash: str
    observed_effect: str
    confidence: float = 1.0


class SemanticDiffLearner:
    """Turns state diffs into semantic learning signals."""

    def __init__(self):
        # Maps (action_name, context) -> list[CausalLink]
        self.learned_links: dict[str, list[CausalLink]] = {}

    def compute_diff(self, state_before: ParsedState, state_after: ParsedState) -> list[str]:
        """Computes semantic differences between two states."""
        diffs = []
        
        # In a real environment, this would deeply diff the two parsed states
        # For example, if an entity disappeared, if resources changed significantly, etc.
        if state_before.context_id != state_after.context_id:
            diffs.append(f"context_changed_to_{state_after.context_id}")
            
        # We also ingest explicitly parsed semantic events
        for event in state_after.semantic_events:
            diffs.append(f"event_{event.name}")
            
        return diffs

    def learn_from_transition(
        self,
        state_before: ParsedState,
        action: ActionIntent,
        state_after: ParsedState
    ) -> list[CausalLink]:
        """Extract causal links from an action execution."""
        diffs = self.compute_diff(state_before, state_after)
        links = []
        
        # Simple generic learning rule: Action -> Observed Diff
        for diff in diffs:
            link = CausalLink(
                action_name=action.name,
                pre_condition_hash=state_before.stable_hash()[:8],
                observed_effect=diff,
            )
            links.append(link)
            
            key = action.name
            if key not in self.learned_links:
                self.learned_links[key] = []
            self.learned_links[key].append(link)
            
        return links

__all__ = ["CausalLink", "SemanticDiffLearner"]
