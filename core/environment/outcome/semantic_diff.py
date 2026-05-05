"""Causal outcome learning from semantic diffs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.environment.command import ActionIntent
from core.environment.parsed_state import ParsedState

@dataclass
class SemanticEvent:
    name: str
    details: dict

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

    def compute_diff(self, state_before: ParsedState, state_after: ParsedState) -> list[SemanticEvent]:
        """Computes semantic differences between two states."""
        events = []
        
        # Position diff
        before_pos = state_before.self_state.get("local_coordinates")
        after_pos = state_after.self_state.get("local_coordinates")
        if before_pos and after_pos:
            if before_pos != after_pos:
                events.append(SemanticEvent(name="position_changed", details={"from": before_pos, "to": after_pos}))
            else:
                events.append(SemanticEvent(name="position_unchanged", details={"pos": before_pos}))
                
        # Message checks
        msg = state_after.self_state.get("raw_text", "").lower()
        if "hit a wall" in msg or "blocked" in msg:
            events.append(SemanticEvent(name="blocked_by_wall", details={"message": msg}))
            
        if "die" in msg:
            events.append(SemanticEvent(name="fatal_event", details={"message": msg}))
            
        if state_before.context_id != state_after.context_id:
            events.append(SemanticEvent(name=f"context_changed_to_{state_after.context_id}", details={}))
            
        # We also ingest explicitly parsed semantic events
        for event in state_after.semantic_events:
            events.append(SemanticEvent(name=event.label, details={}))
            
        return events

    def learn_from_transition(
        self,
        state_before: ParsedState,
        action: ActionIntent,
        state_after: ParsedState
    ) -> list[CausalLink]:
        """Extract causal links from an action execution."""
        events = self.compute_diff(state_before, state_after)
        links = []
        
        # Simple generic learning rule: Action -> Observed Diff
        for event in events:
            link = CausalLink(
                action_name=action.name,
                pre_condition_hash=state_before.stable_hash()[:8],
                observed_effect=event.name,
            )
            links.append(link)
            
            key = action.name
            if key not in self.learned_links:
                self.learned_links[key] = []
            self.learned_links[key].append(link)
            
        return links

__all__ = ["CausalLink", "SemanticDiffLearner", "SemanticEvent"]
