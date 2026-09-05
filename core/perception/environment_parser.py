"""Base contracts for structured environmental perception.

Environment parsers are the fast, non-LLM front end of embodied cognition.
They convert raw sensory input from any domain -- terminal, browser, UI,
robotics, simulation, media stream -- into a typed state that downstream
belief, risk, planning, and action-gating systems can share.
"""
import abc
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvironmentState:
    """A generalized, structured representation of an environment state at a given moment."""
    timestamp: float = field(default_factory=time.time)
    domain: str = "generic"
    context_id: str = "default"
    observation_id: str = ""
    raw_reference: str | None = None
    confidence: float = 1.0

    # Who am I in this environment?
    self_state: dict[str, Any] = field(default_factory=dict)

    # What are the immediate messages or system communications?
    messages: list[str] = field(default_factory=list)

    # What entities (friends, foes, items, obstacles) do I perceive?
    entities: list[dict[str, Any]] = field(default_factory=list)

    # What is the layout or topology of my immediate surroundings?
    spatial_info: dict[str, Any] = field(default_factory=dict)

    # Any active prompts or menus blocking standard interaction?
    active_prompts: list[str] = field(default_factory=list)

    # Has anything significant changed since the last state?
    delta_summary: str = ""

    # Explicit uncertainty / modality channels. These are intentionally
    # lightweight dictionaries so any environment adapter can participate.
    uncertainty: dict[str, float] = field(default_factory=dict)
    modalities: dict[str, Any] = field(default_factory=dict)
    action_candidates: list[dict[str, Any]] = field(default_factory=list)

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

    def resource_ratio(self, current_key: str, max_key: str) -> float | None:
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

    def nearby_entities(self, max_distance: float = 1.0) -> list[dict[str, Any]]:
        nearby: list[dict[str, Any]] = []
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

    def entity_labels(self) -> list[str]:
        labels: list[str] = []
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


def parser_for_domain(domain: str) -> EnvironmentParser:
    """Pick the structured parser for a perception domain.

    NetHack keeps its game-specific glyph parser; every other terminal-like surface
    (shell, REPL, build log, SSH session) gets the general terminal parser, which scores
    danger from terminal *semantics* rather than game glyphs. Unknown domains fall back to
    the general terminal parser too, since raw text is the common case.
    """
    d = (domain or "").lower()
    if "nethack" in d:
        from core.perception.nethack_parser import NetHackParser
        return NetHackParser()
    from core.perception.general_terminal_parser import GeneralTerminalParser
    return GeneralTerminalParser()
