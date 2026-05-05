"""Long-horizon episode phase tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EpisodePhase = Literal[
    "boot",
    "orientation",
    "early_exploration",
    "resource_stabilization",
    "progression",
    "recovery",
    "crisis",
    "terminal",
    "postmortem",
]


@dataclass
class EpisodeState:
    run_id: str
    environment_id: str
    phase: EpisodePhase = "boot"
    step_count: int = 0
    context_count: int = 0
    current_objective: str = "orient"
    milestones: list[str] = field(default_factory=list)
    unresolved_risks: list[str] = field(default_factory=list)
    active_constraints: list[str] = field(default_factory=list)


class EpisodeManager:
    def __init__(self, run_id: str, environment_id: str) -> None:
        self.state = EpisodeState(run_id=run_id, environment_id=environment_id)

    def transition(self, *, valid_state: bool = False, stable: bool = False, critical_risk: bool = False, terminal: bool = False) -> EpisodeState:
        if terminal:
            self.state.phase = "terminal"
            return self.state
        if self.state.phase == "terminal":
            self.state.phase = "postmortem"
            return self.state
        if critical_risk:
            self.state.phase = "crisis"
        elif self.state.phase == "boot" and valid_state:
            self.state.phase = "orientation"
        elif self.state.phase == "orientation" and stable:
            self.state.phase = "early_exploration"
        elif self.state.phase == "crisis" and stable:
            self.state.phase = "recovery"
        elif self.state.phase == "recovery" and stable:
            self.state.phase = "progression"
        self.state.step_count += 1
        return self.state


__all__ = ["EpisodePhase", "EpisodeState", "EpisodeManager"]
