"""Base contracts for structured environmental perception.

Environment parsers are the fast, non-LLM front end of embodied cognition.
They convert raw sensory input from any domain -- terminal, browser, UI,
robotics, simulation, media stream -- into a typed state that downstream
belief, risk, planning, and action-gating systems can share.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import abc
import hashlib
import time


@dataclass
class EnvironmentState:
    """A generalized, structured representation of an environment state at a given moment."""
    timestamp: float = field(default_factory=time.time)
    domain: str = "generic"
    context_id: str = "default"
    observation_id: str = ""
    raw_reference: Optional[str] = None
    confidence: float = 1.0

    # Who am I in this environment?
    self_state: Dict[str, Any] = field(default_factory=dict)

    # What are the immediate messages or system communications?
    messages: List[str] = field(default_factory=list)

    # What entities (friends, foes, items, obstacles) do I perceive?
    entities: List[Dict[str, Any]] = field(default_factory=list)

    # What is the layout or topology of my immediate surroundings?
    spatial_info: Dict[str, Any] = field(default_factory=dict)

    # Any active prompts or menus blocking standard interaction?
    active_prompts: List[str] = field(default_factory=list)

    # Has anything significant changed since the last state?
    delta_summary: str = ""

    # Explicit uncertainty / modality channels. These are intentionally
    # lightweight dictionaries so any environment adapter can participate.
    uncertainty: Dict[str, float] = field(default_factory=dict)
    modalities: Dict[str, Any] = field(default_factory=dict)
    action_candidates: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.observation_id:
            self.refresh_observation_id()

    def refresh_observation_id(self) -> str:
        basis = repr(
            (
                self.domain,
                self.context_id,
                self.self_state,
                self.messages[-3:],
                self.entities[:20],
                self.active_prompts,
            )
        )
        self.observation_id = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
        return self.observation_id

    def to_structured_prompt(self) -> str:
        """Converts the structured state into a clean string format for the cognitive engine."""
        lines = ["[ENVIRONMENT STATE]"]
        lines.append(f"DOMAIN: {self.domain}")
        lines.append(f"CONTEXT: {self.context_id}")
        lines.append(f"OBSERVATION: {self.observation_id}")

        if self.self_state:
            lines.append("SELF:")
            for k, v in self.self_state.items():
                lines.append(f"  {k}: {v}")

        if self.messages:
            lines.append("MESSAGES:")
            for m in self.messages:
                lines.append(f"  > {m}")

        if self.entities:
            lines.append("ENTITIES VISIBLE:")
            for e in self.entities:
                lines.append(f"  - {e}")

        if self.active_prompts:
            lines.append("ACTIVE PROMPTS:")
            for p in self.active_prompts:
                lines.append(f"  [!] {p}")

        if self.uncertainty:
            lines.append("UNCERTAINTY:")
            for k, v in sorted(self.uncertainty.items()):
                lines.append(f"  {k}: {float(v):.2f}")

        if self.action_candidates:
            lines.append("ACTION CANDIDATES:")
            for candidate in self.action_candidates[:8]:
                lines.append(f"  - {candidate}")

        return "\n".join(lines)

    def resource_ratio(self, current_key: str, max_key: str) -> Optional[float]:
        """Return a bounded resource ratio if both fields are available."""
        try:
            current = float(self.self_state[current_key])
            maximum = float(self.self_state[max_key])
        except (KeyError, TypeError, ValueError):
            return None
        if maximum <= 0:
            return None
        return max(0.0, min(1.0, current / maximum))

    def has_active_prompt(self) -> bool:
        return bool(self.active_prompts)

    def nearby_entities(self, max_distance: float = 1.0) -> List[Dict[str, Any]]:
        nearby: List[Dict[str, Any]] = []
        for entity in self.entities:
            distance = entity.get("distance")
            if distance is None:
                continue
            try:
                if float(distance) <= max_distance:
                    nearby.append(entity)
            except (TypeError, ValueError):
                continue
        return nearby

    def entity_labels(self) -> List[str]:
        labels: List[str] = []
        for entity in self.entities:
            label = entity.get("label") or entity.get("name") or entity.get("type") or entity.get("glyph")
            if label is not None:
                labels.append(str(label))
        return labels


class EnvironmentParser(abc.ABC):
    """Abstract base class for environment-specific parsers.
    Parsers must be fast (<5ms) and should NOT rely on LLM inference.
    """

    @abc.abstractmethod
    def parse(self, raw_input: Any) -> EnvironmentState:
        """Converts raw input into a structured EnvironmentState."""
        pass
