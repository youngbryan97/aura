"""core/environment/environment_protocol.py
======================================
Standardized virtual environment interface protocol for Aura environmental grounding.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ActionResult:
    """Structure encapsulating execution results of an environmental action."""
    success: bool
    next_observation: Any  # Instantiated as an Observation subclass
    reward: float  # Normalized score from -1.0 to 1.0
    latency_ms: float  # Action execution duration
    side_effects: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictedOutcome:
    """Structure representing simulated expectations from a predictive world model query."""
    predicted_observation: Any  # Instantiated as an Observation subclass or similar structure
    expected_reward: float
    confidence: float  # Model's confidence score from 0.0 to 1.0


class EnvironmentProtocol(abc.ABC):
    """Abstract interface that all virtual environments must implement for grounding."""

    @abc.abstractmethod
    async def observe(self) -> Any:
        """Returns the current sensory observation from this environment."""
        pass

    @abc.abstractmethod
    async def act(self, action: Any) -> ActionResult:
        """Executes an action in this environment and returns results."""
        pass

    @abc.abstractmethod
    async def predict(self, action: Any) -> PredictedOutcome:
        """Simulates the state transition and rewards of an action without committing side-effects."""
        pass

    @abc.abstractmethod
    async def score(self, result: ActionResult) -> float:
        """Scores the quality or alignment value of an action's outcome."""
        pass

    @abc.abstractmethod
    async def reset(self) -> Any:
        """Resets the environment back to initial states."""
        pass

    @abc.abstractmethod
    async def snapshot(self) -> Dict[str, Any]:
        """Captures a snapshot of the current state of the environment for simulation rollback."""
        pass

    @abc.abstractmethod
    async def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restores the environment state from a prior snapshot."""
        pass
