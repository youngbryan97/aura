"""Policy candidate generation for action intents.

Generates candidates dynamically from:
- Parsed state affordances, entities, objects, hazards
- Belief graph frontiers and known features
- Recent frame history (loop/failure detection)
- Inventory and resources
"""
from __future__ import annotations

from typing import Iterable

from core.environment.command import ActionIntent
from core.environment.parsed_state import ParsedState
from core.environment.belief_graph import EnvironmentBeliefGraph
from core.environment.ontology import Affordance


class CandidateGenerator:
    """Generates candidate actions from the current environment state."""

    def __init__(self):
        self.failure_counts: dict[str, int] = {}  # (action_name:context) -> count
        self.suppression_threshold: int = 3

    def generate(
        self,
        parsed_state: ParsedState,
        belief: EnvironmentBeliefGraph | None = None,
        recent_frames: list | None = None,
    ) -> list[ActionIntent]:
        candidates: list[ActionIntent] = []

        # 1. Base default actions (wait, explore)
        candidates.append(ActionIntent(name="wait", risk="safe"))
        candidates.append(ActionIntent(name="explore_frontier", risk="caution"))

        # 2. Movement options
        for direction in ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]:
            candidates.append(ActionIntent(
                name="move",
                parameters={"direction": direction},
                risk="caution"
            ))

        # 3. Object-based affordances (from parsed state)
        for obj in parsed_state.objects:
            for affordance in obj.affordances:
                if affordance == "pickup":
                    candidates.append(ActionIntent(name="pickup", parameters={"target_id": obj.object_id}))
                elif affordance == "use":
                    candidates.append(ActionIntent(name="use", parameters={"target_id": obj.object_id}))
                elif affordance == "open":
                    candidates.append(ActionIntent(name="open_door", parameters={"target_id": obj.object_id}))
                elif affordance == "eat":
                    candidates.append(ActionIntent(name="eat", parameters={"target_id": obj.object_id}))

        # 4. Inventory-based actions
        inventory = getattr(parsed_state, "inventory_items", None) or parsed_state.self_state.get("inventory", [])
        for item in inventory:
            letter = item.get("letter") if isinstance(item, dict) else None
            category = item.get("category", "") if isinstance(item, dict) else ""
            if letter:
                if category == "weapon":
                    candidates.append(ActionIntent(name="wield", parameters={"item_letter": letter}))
                elif category == "food":
                    candidates.append(ActionIntent(name="eat", parameters={"item_letter": letter}, risk="safe"))
                elif category == "potion":
                    risk = "risky" if item.get("identified") is False else "caution"
                    tags = {"unknown"} if item.get("identified") is False else set()
                    candidates.append(ActionIntent(name="quaff", parameters={"item_letter": letter}, risk=risk, tags=tags))
                elif category in ("scroll", "spellbook"):
                    candidates.append(ActionIntent(name="read", parameters={"item_letter": letter}))
                candidates.append(ActionIntent(name="drop", parameters={"item_letter": letter}))

        # 5. Entity-based tactical candidates
        for entity in parsed_state.entities:
            if entity.kind == "hostile" or entity.threat_score >= 0.5:
                candidates.append(ActionIntent(
                    name="retreat_to_safety",
                    parameters={"threat_id": entity.entity_id},
                    risk="caution",
                    expected_effect="retreated",
                ))

        # 6. Hazard-aware candidates
        for hazard in parsed_state.hazards:
            candidates.append(ActionIntent(
                name="retreat_to_safety",
                parameters={"hazard_id": hazard.hazard_id if hasattr(hazard, 'hazard_id') else str(hazard)},
                risk="caution",
            ))

        # 7. Information gathering
        candidates.append(ActionIntent(name="search", risk="safe"))
        candidates.append(ActionIntent(name="inventory", risk="safe"))

        # 8. Stairs/transition if belief graph indicates stairs present
        if belief:
            for node in belief.nodes.values():
                if "stair" in node.kind.lower() or "stair" in node.label.lower():
                    direction = "down" if "down" in node.label.lower() else "up"
                    candidates.append(ActionIntent(
                        name="use_stairs",
                        parameters={"direction": direction},
                        risk="caution",
                        expected_effect="level_changed",
                    ))

        # 9. Resource stabilization if homeostasis signals pressure
        resources = parsed_state.resources
        for rname, rstate in resources.items():
            if hasattr(rstate, 'value') and hasattr(rstate, 'max_value'):
                if rstate.value < rstate.max_value * 0.3:
                    candidates.append(ActionIntent(
                        name="stabilize_resource",
                        parameters={"resource": rname},
                        risk="safe",
                        expected_effect=f"{rname}_stabilized",
                    ))

        # 10. Loop recovery: suppress repeated failures
        if recent_frames:
            candidates = self._apply_loop_suppression(candidates, recent_frames, parsed_state.context_id)

        return candidates

    def _apply_loop_suppression(
        self,
        candidates: list[ActionIntent],
        recent_frames: list,
        context_id: str | None,
    ) -> list[ActionIntent]:
        """Suppress candidates that have repeatedly failed in the same context."""
        # Count recent failures per action
        recent_failures: dict[str, int] = {}
        for frame in recent_frames[-10:]:
            if frame.outcome_assessment and frame.outcome_assessment.success_score < 0.3:
                if frame.action_intent:
                    key = f"{frame.action_intent.name}:{context_id}"
                    recent_failures[key] = recent_failures.get(key, 0) + 1

        # Filter out suppressed candidates
        filtered = []
        for c in candidates:
            key = f"{c.name}:{context_id}"
            if recent_failures.get(key, 0) >= self.suppression_threshold:
                continue  # suppressed
            filtered.append(c)

        # If everything was suppressed, add recovery actions
        if not filtered:
            filtered.append(ActionIntent(name="search", risk="safe"))
            filtered.append(ActionIntent(name="wait", risk="safe"))
            filtered.append(ActionIntent(name="inventory", risk="safe"))

        return filtered


__all__ = ["CandidateGenerator"]

