"""Universal environment operating-system layer for Aura."""
from .adapter import EnvironmentAdapter, EnvironmentCapabilities, ExecutionResult
from .command import ActionIntent, CommandCompiler, CommandSpec, CommandStep
from .capability_matrix import CapabilityAuditReport, EnvironmentCapabilityMatrix
from .environment_kernel import EnvironmentFrame, EnvironmentKernel
from .external_validation import ExternalTaskEvidence, ExternalTaskProofGate
from .modal import ModalManager, ModalPolicy, ModalState
from .observation import Observation
from .ontology import Affordance, EntityState, HazardState, ObjectState, ResourceState, SemanticEvent
from .parsed_state import ParsedState
from .startup_policy import StartupPromptDecision, StartupPromptPolicy

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
    "CapabilityAuditReport",
    "EnvironmentCapabilityMatrix",
    "ExternalTaskEvidence",
    "ExternalTaskProofGate",
    "EnvironmentFrame",
    "EnvironmentKernel",
    "StartupPromptDecision",
    "StartupPromptPolicy",
]
