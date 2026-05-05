"""Policy candidate generation for action intents."""
from __future__ import annotations

from typing import Iterable

from core.environment.command import ActionIntent
from core.environment.parsed_state import ParsedState
from core.environment.ontology import Affordance


class CandidateGenerator:
    """Generates candidate actions from the current environment state."""

    def generate(self, parsed_state: ParsedState) -> list[ActionIntent]:
        candidates: list[ActionIntent] = []
        
        # 1. Base default actions (wait, explore)
        candidates.append(ActionIntent(name="wait", risk="safe"))
        candidates.append(ActionIntent(name="explore_frontier", risk="caution"))
        
        # 2. Movement options
        # We assume 4 or 8 directional movement is standard for grid adapters
        for direction in ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]:
            candidates.append(ActionIntent(
                name="move", 
                parameters={"direction": direction}, 
                risk="caution"
            ))
            
        # 3. Object-based affordances
        for obj in parsed_state.objects:
            for affordance in obj.affordances:
                if affordance == "pickup":
                    candidates.append(ActionIntent(name="pickup", parameters={"target_id": obj.object_id}))
                elif affordance == "use":
                    candidates.append(ActionIntent(name="use", parameters={"target_id": obj.object_id}))
                elif affordance == "open":
                    candidates.append(ActionIntent(name="open_door", parameters={"target_id": obj.object_id}))

        # 4. Inventory-based actions
        inventory = getattr(parsed_state, "inventory_items", []) or []
        for item in inventory:
            letter = item.get("letter")
            category = item.get("category")
            if letter:
                if category == "weapon":
                    candidates.append(ActionIntent(name="wield", parameters={"item_letter": letter}))
                elif category == "food":
                    candidates.append(ActionIntent(name="eat", parameters={"item_letter": letter}))
                elif category == "potion":
                    candidates.append(ActionIntent(name="quaff", parameters={"item_letter": letter}))
                elif category == "scroll" or category == "spellbook":
                    candidates.append(ActionIntent(name="read", parameters={"item_letter": letter}))
                
                # General drop action
                candidates.append(ActionIntent(name="drop", parameters={"item_letter": letter}))

        # 5. Information gathering
        candidates.append(ActionIntent(name="search", risk="safe"))
        candidates.append(ActionIntent(name="inventory", risk="safe"))

        return candidates

__all__ = ["CandidateGenerator"]
