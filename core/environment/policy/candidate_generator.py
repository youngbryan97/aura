"""Policy candidate generation for action intents.

Generates candidates dynamically from:
- Parsed state affordances, entities, objects, hazards
- Belief graph frontiers and known features
- Recent frame history (loop/failure detection)
- Inventory and resources
"""
from __future__ import annotations

from core.environment.command import ActionIntent
from core.environment.parsed_state import ParsedState
from core.environment.belief_graph import EnvironmentBeliefGraph
from core.environment.planning import GridPathPlanner


class CandidateGenerator:
    """Generates candidate actions from the current environment state."""

    def __init__(self):
        self.failure_counts: dict[str, int] = {}  # (action_name:context) -> count
        self.suppression_threshold: int = 3
        self.path_planner = GridPathPlanner()

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

        # 2. Movement options. If a canonical spatial model exists, rank
        # directions by known safe frontier/transition targets and avoid known
        # hazards. Otherwise expose the full reversible movement fan-out.
        for direction in self._movement_directions(parsed_state, belief):
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
                elif affordance in {"navigate", "use_stairs"} and obj.kind == "transition":
                    direction = self._transition_direction(obj.label, obj.properties)
                    if self._is_at_position(parsed_state, obj.position):
                        candidates.append(ActionIntent(
                            name="use_stairs",
                            parameters={"direction": direction},
                            risk="caution",
                            expected_effect="level_changed",
                            target_id=obj.object_id,
                        ))
                    else:
                        move_direction = self._direction_toward(parsed_state, obj.position)
                        if move_direction:
                            candidates.append(ActionIntent(
                                name="move",
                                parameters={"direction": move_direction},
                                risk="caution",
                                expected_effect="approach_transition",
                                target_id=obj.object_id,
                            ))
                elif affordance == "open":
                    candidates.append(ActionIntent(name="open_door", parameters={"target_id": obj.object_id}))
                elif affordance == "eat":
                    candidates.append(ActionIntent(name="eat", parameters={"target_id": obj.object_id}))

        # 4. Inventory-based actions
        inventory = (
            getattr(parsed_state, "inventory_items", None)
            or parsed_state.self_state.get("inventory_items", [])
            or parsed_state.self_state.get("inventory", [])
        )
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

        # 7b. Recent harm or attack evidence should create a general
        # threat-response candidate even if perception did not classify the
        # hostile entity precisely this frame.
        if self._recent_threat_pressure(recent_frames):
            candidates.append(ActionIntent(
                name="retreat_to_safety",
                risk="caution",
                expected_effect="threat_distance_increased",
                tags={"threat_response"},
            ))

        # 8. Stairs/transition if belief graph indicates stairs present
        if belief:
            context = parsed_state.context_id or "default"
            current = self._current_xy(parsed_state) or belief.current_position(context)
            if current is not None:
                for kinds, effect in (
                    ({"transition"}, "approach_transition"),
                    ({"frontier", "unknown"}, "frontier_progress"),
                ):
                    target = belief.nearest_spatial(context_id=context, kinds=kinds, origin=current, min_confidence=0.2)
                    if target:
                        planned = self.path_planner.next_move_intent(
                            belief,
                            context_id=context,
                            goal=(int(target[0]), int(target[1])),
                        )
                        if planned is not None:
                            planned.expected_effect = effect
                            candidates.append(planned)
                            break
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

    def _movement_directions(
        self,
        parsed_state: ParsedState,
        belief: EnvironmentBeliefGraph | None,
    ) -> list[str]:
        all_dirs = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
        if not belief:
            return all_dirs
        context = parsed_state.context_id or "default"
        origin = self._current_xy(parsed_state)
        if origin is None:
            origin = belief.current_position(context)
        if origin is None:
            return all_dirs
        dxdy = {
            "north": (0, -1),
            "south": (0, 1),
            "east": (1, 0),
            "west": (-1, 0),
            "northeast": (1, -1),
            "northwest": (-1, -1),
            "southeast": (1, 1),
            "southwest": (-1, 1),
        }
        safe_dirs: list[str] = []
        unknown_dirs: list[str] = []
        for direction, (dx, dy) in dxdy.items():
            key = (context, origin[0] + dx, origin[1] + dy)
            cell = belief.spatial.get(key)
            if not cell:
                unknown_dirs.append(direction)
                continue
            if cell.get("kind") in {"hazard", "trap", "hostile_entity"} and float(cell.get("confidence", 0.0) or 0.0) >= 0.5:
                continue
            if cell.get("walkable") is False:
                continue
            safe_dirs.append(direction)
        return safe_dirs + [d for d in unknown_dirs if d not in safe_dirs] or all_dirs

    @staticmethod
    def _current_xy(parsed_state: ParsedState) -> tuple[int, int] | None:
        pos = parsed_state.self_state.get("local_coordinates")
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return (int(pos[0]), int(pos[1]))
        return None

    @classmethod
    def _is_at_position(cls, parsed_state: ParsedState, position) -> bool:
        current = cls._current_xy(parsed_state)
        if current is None or not isinstance(position, (list, tuple)) or len(position) < 2:
            return False
        return current == (int(position[0]), int(position[1]))

    @classmethod
    def _direction_toward(cls, parsed_state: ParsedState, position) -> str | None:
        current = cls._current_xy(parsed_state)
        if current is None or not isinstance(position, (list, tuple)) or len(position) < 2:
            return None
        dx = int(position[0]) - current[0]
        dy = int(position[1]) - current[1]
        sx = 1 if dx > 0 else -1 if dx < 0 else 0
        sy = 1 if dy > 0 else -1 if dy < 0 else 0
        return {
            (0, -1): "north",
            (0, 1): "south",
            (1, 0): "east",
            (-1, 0): "west",
            (1, -1): "northeast",
            (-1, -1): "northwest",
            (1, 1): "southeast",
            (-1, 1): "southwest",
        }.get((sx, sy))

    @staticmethod
    def _transition_direction(label: str, properties: dict) -> str:
        text = f"{label} {properties.get('glyph', '')}".lower()
        if "<" in text or "up" in text:
            return "up"
        return "down"

    def _apply_loop_suppression(
        self,
        candidates: list[ActionIntent],
        recent_frames: list,
        context_id: str | None,
    ) -> list[ActionIntent]:
        """Suppress candidates that have repeatedly failed in the same context."""
        # Count recent failures per action
        recent_failures: dict[str, int] = {}
        recent_counts: dict[str, int] = {}
        recent_names: list[str] = []
        for frame in recent_frames[-10:]:
            if frame.action_intent:
                recent_names.append(frame.action_intent.name)
                recent_counts[frame.action_intent.name] = recent_counts.get(frame.action_intent.name, 0) + 1
            if frame.outcome_assessment and frame.outcome_assessment.success_score < 0.3:
                if frame.action_intent:
                    key = f"{frame.action_intent.name}:{context_id}"
                    recent_failures[key] = recent_failures.get(key, 0) + 1

        oscillating_information_loop = self._information_loop(recent_names)

        # Filter out suppressed candidates
        filtered = []
        for c in candidates:
            key = f"{c.name}:{context_id}"
            if recent_failures.get(key, 0) >= self.suppression_threshold:
                continue  # suppressed
            if c.name in {"inventory", "search", "observe", "inspect", "diagnose", "far_look"} and recent_counts.get(c.name, 0) >= self.suppression_threshold:
                continue
            if oscillating_information_loop and c.name in {"inventory", "observe", "inspect", "diagnose", "far_look"}:
                continue
            filtered.append(c)

        # If everything was suppressed, add recovery actions
        if not filtered:
            for direction in self._movement_directions(ParsedState(environment_id="", context_id=context_id), None):
                filtered.append(ActionIntent(name="move", parameters={"direction": direction}, risk="caution", expected_effect="recover_from_information_loop"))
            filtered.append(ActionIntent(name="wait", risk="safe"))

        return filtered

    @staticmethod
    def _information_loop(recent_names: list[str]) -> bool:
        window = recent_names[-8:]
        if len(window) < 4:
            return False
        informational = {"inventory", "observe", "inspect", "diagnose", "far_look", "search", "resolve_modal"}
        if sum(1 for name in window if name in informational) < 5:
            return False
        progress = {"move", "use_stairs", "pickup", "eat", "stabilize_resource", "retreat_to_safety"}
        return not any(name in progress for name in window)

    @staticmethod
    def _recent_threat_pressure(recent_frames: list | None) -> bool:
        if not recent_frames:
            return False
        for frame in recent_frames[-4:]:
            outcome = getattr(frame, "outcome_assessment", None)
            if outcome is None:
                continue
            events = set(getattr(outcome, "observed_events", []) or [])
            if "attacked" in events or any(str(event).startswith("resource_health_decreased") for event in events):
                return True
            if getattr(outcome, "harm_score", 0.0) >= 0.2:
                return True
        return False


__all__ = ["CandidateGenerator"]
