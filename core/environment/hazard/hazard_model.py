"""Hazard and adversarial risk modeling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.environment.parsed_state import ParsedState


@dataclass
class Hazard:
    """Represents an adversarial entity, constraint, or threat."""
    hazard_id: str
    category: str  # e.g. "rate_limit", "hostile_entity", "system_ban"
    threat_level: float = 0.0  # 0.0 to 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    distance: float | None = None  # Generic distance metric (turns, tiles, requests)


class HazardModel:
    """Tracks known hazards and estimates risk probabilities."""

    def __init__(self):
        self.hazards: dict[str, Hazard] = {}

    def update_from_state(self, parsed_state: ParsedState) -> None:
        """Update hazard tracking based on current state observations."""
        new_hazards = {}
        
        # Extract hazard entities
        for entity in getattr(parsed_state, "entities", []) or []:
            props = dict(getattr(entity, "properties", {}) or {})
            is_hazard = bool(props.get("is_hazard", False) or getattr(entity, "threat_score", 0.0) >= 0.5 or getattr(entity, "kind", "") == "hostile")
            if is_hazard:
                hazard_id = getattr(entity, "entity_id", "") or props.get("id", "unknown")
                new_hazards[hazard_id] = Hazard(
                    hazard_id=hazard_id,
                    category=props.get("category", getattr(entity, "kind", "entity")),
                    threat_level=float(getattr(entity, "threat_score", props.get("threat_score", 0.5)) or 0.5),
                    properties={**props, "label": getattr(entity, "label", "")},
                    distance=props.get("distance", None),
                )
                
        for hazard in getattr(parsed_state, "hazards", []) or []:
            hazard_id = getattr(hazard, "hazard_id", "") or getattr(hazard, "id", "unknown")
            new_hazards[hazard_id] = Hazard(
                hazard_id=hazard_id,
                category=getattr(hazard, "kind", "unknown"),
                threat_level=float(getattr(hazard, "severity", 0.5) or 0.5),
                properties=dict(getattr(hazard, "properties", {}) or {}),
            )
                
        for hazard_id, existing in self.hazards.items():
            if hazard_id not in new_hazards:
                existing.threat_level = max(0.05, existing.threat_level * 0.85)
                if existing.threat_level >= 0.2:
                    new_hazards[hazard_id] = existing
        self.hazards = new_hazards

    def estimate_risk(self, action_name: str, parameters: dict[str, Any] | None = None) -> float:
        """Estimates the probability of failure/loss (0.0 to 1.0) for an action."""
        if not self.hazards:
            return 0.0
            
        max_threat = 0.0
        for hazard in self.hazards.values():
            # In a general system, we'd use a causal graph or rule engine.
            # Here, if a hazard is very close, risk is higher.
            local_threat = hazard.threat_level
            if hazard.distance is not None and hazard.distance <= 1:
                local_threat *= 1.5
            
            # Action-specific heuristics
            if action_name in ("move", "request", "execute") and local_threat > 0.5:
                # Interacting or moving while under high threat increases risk
                local_threat *= 1.2
                
            max_threat = max(max_threat, min(local_threat, 1.0))
            
        return max_threat

__all__ = ["Hazard", "HazardModel"]
