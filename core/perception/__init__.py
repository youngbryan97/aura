"""Aura PerceptionRuntime — governed multimodal sensors.

The audit calls for camera/audio/subtitle ingestion plus a
shared-attention / silence-policy / movie-session-memory stack, all
mediated by capability tokens issued by Unified Will.

This package exposes the *contract* surface only. Real hardware drivers
are intentionally out of scope here so the runtime can boot without a
camera/microphone present; tests assert that the contract holds and that
sensor enablement is governed.
"""

from .perception_runtime import (
    CapabilityToken,
    MovieSessionMemory,
    PerceptionRuntime,
    SharedAttentionState,
    SilencePolicy,
    SceneEvent,
)
from .action_gateway import ActionDecision, ActionRequest, EnvironmentActionGateway
from .belief_state import EnvironmentBeliefState
from .cognitive_runtime import EmbodiedCognitionRuntime, EmbodiedCognitiveFrame
from .environment_parser import EnvironmentParser, EnvironmentState
from .goal_manager import EmbodiedGoal, EnvironmentGoalManager
from .reflex_layer import DangerAssessment, EnvironmentReflexLayer, RiskProfile
from .skill_graph import EnvironmentSkillGraph, SkillOption

__all__ = [
    "ActionDecision",
    "ActionRequest",
    "CapabilityToken",
    "DangerAssessment",
    "EmbodiedCognitionRuntime",
    "EmbodiedCognitiveFrame",
    "EmbodiedGoal",
    "EnvironmentActionGateway",
    "EnvironmentBeliefState",
    "EnvironmentGoalManager",
    "EnvironmentParser",
    "EnvironmentReflexLayer",
    "EnvironmentSkillGraph",
    "EnvironmentState",
    "MovieSessionMemory",
    "PerceptionRuntime",
    "RiskProfile",
    "SharedAttentionState",
    "SilencePolicy",
    "SceneEvent",
    "SkillOption",
]
