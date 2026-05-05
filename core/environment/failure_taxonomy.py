"""Universal failure classes for environment postmortems."""
from __future__ import annotations

from typing import Literal

FailureClass = Literal[
    "perception_error",
    "belief_error",
    "modal_error",
    "action_compilation_error",
    "gateway_error",
    "authorization_error",
    "execution_error",
    "prediction_error",
    "resource_management_error",
    "planning_error",
    "loop_stagnation",
    "unsafe_irreversible_action",
    "knowledge_gap",
    "learning_error",
    "environment_unavailable",
    "trace_integrity_error",
]

FAILURE_CLASSES = set(FailureClass.__args__)  # type: ignore[attr-defined]

__all__ = ["FailureClass", "FAILURE_CLASSES"]
