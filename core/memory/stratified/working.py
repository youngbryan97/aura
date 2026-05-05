"""Working memory tier for current environment runs."""
from dataclasses import dataclass, field


@dataclass
class WorkingMemoryTier:
    active_modal: str = ""
    current_goal: str = ""
    nearby_hazards: list[str] = field(default_factory=list)
    immediate_plan: list[str] = field(default_factory=list)
    last_actions: list[str] = field(default_factory=list)
