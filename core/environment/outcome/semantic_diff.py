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

        # Resource deltas are portable across games, browsers, tools, robots,
        # and service runtimes: health, quota, memory, battery, trust, etc.
        for name in sorted(set(state_before.resources) | set(state_after.resources)):
            before = state_before.resources.get(name)
            after = state_after.resources.get(name)
            if before is None or after is None:
                continue
            delta = float(after.value) - float(before.value)
            if abs(delta) > 0.001:
                direction = "increased" if delta > 0 else "decreased"
                events.append(
                    SemanticEvent(
                        name=f"resource_{name}_{direction}",
                        details={"resource": name, "before": before.value, "after": after.value, "delta": delta},
                    )
                )

        before_ids = set(state_before.observed_ids)
        after_ids = set(state_after.observed_ids)
        for observed_id in sorted(after_ids - before_ids)[:20]:
            events.append(SemanticEvent(name="new_object_or_entity_observed", details={"id": observed_id}))

        if state_before.modal_state and not state_after.modal_state:
            events.append(SemanticEvent(name="modal_cleared", details={"from": state_before.modal_state.text}))
        elif not state_before.modal_state and state_after.modal_state:
            events.append(SemanticEvent(name="modal_opened", details={"to": state_after.modal_state.text}))
            
        # We also ingest explicitly parsed semantic events
        for event in state_after.semantic_events:
            label = event.label
            lowered = label.lower()
            if any(token in lowered for token in ("you die", "dywypi", "possessions identified", "you are dead")):
                events.append(SemanticEvent(name="fatal_event", details={"label": label}))
            elif "welcome to experience level" in lowered:
                events.append(SemanticEvent(name="level_up", details={"label": label}))
            else:
                events.append(SemanticEvent(name=label, details={}))
            
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
