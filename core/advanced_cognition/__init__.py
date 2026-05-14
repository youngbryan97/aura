"""Advanced cognition runtime: transfer, ontology, grounding, and stability."""
from .architecture_evolution import ArchitectureEvolutionGovernor, ArchitectureMutationPlan, MutationTier
from .continual_learning_stability import ContinualLearningStabilityEngine
from .evidence_deliberation import EvidenceDeliberation, ExternalEvidenceDeliberator
from .integration import AdvancedCognitionRuntime, get_advanced_cognition_runtime
from .ontology_invention import OntologyInventionEngine
from .physical_grounding import PhysicalGroundingEngine
from .schemas import ActionCandidate, ActionDecision, Episode, Observation, Outcome, Principle, stable_hash
from .social_cognition import SocialCognitionLayer
from .tiered_action import ActionTier, TieredActionController
from .validation import BenchmarkTask, IndependentValidationLoop
from .world_model import MultiDomainWorldModel, OutcomePrediction
from .zero_shot_transfer import ZeroShotTransferEngine

__all__ = [
    "ActionCandidate",
    "ActionDecision",
    "ArchitectureEvolutionGovernor",
    "ArchitectureMutationPlan",
    "AdvancedCognitionRuntime",
    "BenchmarkTask",
    "ContinualLearningStabilityEngine",
    "EvidenceDeliberation",
    "Episode",
    "ExternalEvidenceDeliberator",
    "IndependentValidationLoop",
    "MutationTier",
    "Observation",
    "OntologyInventionEngine",
    "Outcome",
    "OutcomePrediction",
    "PhysicalGroundingEngine",
    "Principle",
    "SocialCognitionLayer",
    "ActionTier",
    "TieredActionController",
    "MultiDomainWorldModel",
    "ZeroShotTransferEngine",
    "get_advanced_cognition_runtime",
    "stable_hash",
]
