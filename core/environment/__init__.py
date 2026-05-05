"""Universal environment operating-system layer for Aura."""
from .adapter import EnvironmentAdapter, EnvironmentCapabilities, ExecutionResult
from .command import ActionIntent, CommandCompiler, CommandSpec, CommandStep
from .environment_kernel import EnvironmentFrame, EnvironmentKernel
from .modal import ModalManager, ModalPolicy, ModalState
from .observation import Observation
from .ontology import Affordance, EntityState, HazardState, ObjectState, ResourceState, SemanticEvent
from .parsed_state import ParsedState

__all__ = [
    "EnvironmentAdapter",
    "EnvironmentCapabilities",
    "ExecutionResult",
    "Observation",
    "ParsedState",
    "EntityState",
    "ObjectState",
    "ResourceState",
    "HazardState",
    "Affordance",
    "SemanticEvent",
    "ModalState",
    "ModalManager",
    "ModalPolicy",
    "ActionIntent",
    "CommandStep",
    "CommandSpec",
    "CommandCompiler",
    "EnvironmentFrame",
    "EnvironmentKernel",
]
