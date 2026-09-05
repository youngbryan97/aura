"""core/organism/life_state.py
System-wide state representation for Aura's canonical organism.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class WelfareState:
    """Affective and homeostatic variables regulating behavior."""
    energy: float = 100.0         # 0.0 to 100.0
    stress: float = 0.0           # 0.0 to 100.0
    health: float = 100.0         # 0.0 to 100.0
    sleep_debt: float = 0.0       # Cumulative hours/ticks awake
    boredom: float = 0.0          # Curiosity motivator
    thermal_load: float = 35.0    # CPU temperature proxy
    distress_level: float = 0.0   # Bounded ethical distress analog
    welfare_index: float = 1.0    # Unified welfare metric (0.0 to 1.0)
    viability_state: str = "healthy"



@dataclass
class BodyState:
    """Proprioceptive and actuator status."""
    is_sleeping: bool = False
    battery_level: float = 100.0
    cpu_usage: float = 10.0
    memory_usage: float = 50.0
    current_focus_app: str = "Terminal"
    clipboard_content: str = ""
    last_user_activity: float = field(default_factory=time.time)
    last_tick_time: float = field(default_factory=time.time)


@dataclass
class CognitiveState:
    """Working memory, attention focus, and goals."""
    active_attention: str = "idle"
    current_goals: List[Dict[str, Any]] = field(default_factory=list)
    pending_actions: List[Dict[str, Any]] = field(default_factory=list)
    inner_monologue: str = ""
    active_scratchpad: str = ""
    uncertainty_score: float = 0.0


@dataclass
class LifeState:
    """Canonical, single-source-of-truth state for the organism."""
    timestamp: float = field(default_factory=time.time)
    tick_count: int = 0
    
    # Sub-states
    welfare: WelfareState = field(default_factory=WelfareState)
    body: BodyState = field(default_factory=BodyState)
    cognition: CognitiveState = field(default_factory=CognitiveState)
    
    # World & Memory structures
    world_model: Dict[str, Any] = field(default_factory=dict)
    autobiographical_memory: List[Dict[str, Any]] = field(default_factory=list)
    active_preferences: Dict[str, float] = field(default_factory=dict)
    commitments: List[Dict[str, Any]] = field(default_factory=list)
    identity: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize current state representation."""
        return {
            "timestamp": self.timestamp,
            "tick_count": self.tick_count,
            "welfare": {k: v for k, v in self.welfare.__dict__.items()},
            "body": {k: v for k, v in self.body.__dict__.items()},
            "cognition": {k: v for k, v in self.cognition.__dict__.items()},
            "world_model": self.world_model,
            "commitments": self.commitments,
            "identity": self.identity
        }
