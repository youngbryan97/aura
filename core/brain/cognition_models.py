from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

# =========================
# AGENT STATE (SHORT-TERM)
# =========================

class AgentState(BaseModel):
    """Represents the fleeting, immediate condition of the agent.
    Short-term, fast-changing internal state.
    """

    model_config = ConfigDict(validate_assignment=True)

    mood: str = Field(
        default="Neutral",
        description="Current emotional valence (Neutral, Curious, Frustrated, etc.)"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence in current actions"
    )
    focus: str = Field(
        default="Idle",
        description="Current primary focus of attention"
    )
    active_goal_id: Optional[str] = Field(
        default=None,
        description="ID of the currently pursued goal"
    )
    energy: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Cognitive energy level"
    )

    def decay_energy(self, amount: float = 0.01) -> None:
        self.energy = max(0.0, self.energy - amount)

    def boost_confidence(self, amount: float = 0.05) -> None:
        self.confidence = min(1.0, self.confidence + amount)

    def lower_confidence(self, amount: float = 0.05) -> None:
        self.confidence = max(0.0, self.confidence - amount)


# =========================
# WORLD MODEL
# =========================

class WorldModel(BaseModel):
    """Represents the agent's beliefs about external reality.
    This is NOT memory; it is the agent's current belief state.
    """

    model_config = ConfigDict(validate_assignment=True)

    facts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Beliefs assumed to be true"
    )
    entities: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Known entities and their attributes"
    )
    rules: List[str] = Field(
        default_factory=list,
        description="Heuristics about how the world behaves"
    )
    uncertainty: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence (0–1) associated with specific facts"
    )

    def update_fact(self, key: str, value: Any, confidence: float = 1.0) -> None:
        self.facts[key] = value
        self.uncertainty[key] = max(0.0, min(1.0, confidence))

    def get_fact(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)

    def contradicts(self, key: str, value: Any) -> bool:
        return key in self.facts and self.facts[key] != value


# =========================
# SELF MODEL
# =========================

class SelfModel(BaseModel):
    """Represents the agent's understanding of itself.
    Long-term identity and self-assessment.
    """

    model_config = ConfigDict(validate_assignment=True)

    identity: str = Field(default="Aura")

    capabilities: Set[str] = Field(
        default_factory=set,
        description="Abilities the agent believes it has"
    )
    limitations: Set[str] = Field(
        default_factory=set,
        description="Known weaknesses or constraints"
    )
    global_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Long-term self-confidence"
    )

    def register_capability(self, capability: str) -> None:
        self.capabilities.add(capability)

    def register_limitation(self, limitation: str) -> None:
        self.limitations.add(limitation)

    def update_confidence(self, success: bool) -> None:
        if success:
            self.global_confidence = min(1.0, self.global_confidence * 1.05)
        else:
            self.global_confidence = max(0.0, self.global_confidence * 0.95)


# =========================
# DRIVES (INTRINSIC MOTIVATION)
# =========================

class Drives(BaseModel):
    """Intrinsic motivational signals that generate goals.
    """

    model_config = ConfigDict(validate_assignment=True)

    curiosity: float = Field(default=0.5, ge=0.0, le=1.0)
    coherence: float = Field(default=0.5, ge=0.0, le=1.0)
    efficiency: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty: float = Field(default=0.5, ge=0.0, le=1.0)
    reward: float = Field(default=0.0)

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, max(0.0, min(1.0, float(value))))