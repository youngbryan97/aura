"""core/providers/consciousness_provider.py — Consciousness & Affect Registration

Operationally: this resolves and hands out the objects the consciousness group is made of. It measures nothing itself; the name is the package it serves.

"""

import logging
from uuid import uuid4

from core.runtime.service_access import (
    optional_service,
    resolve_epistemic_state,
    resolve_orchestrator,
)
from core.runtime.service_registry import SERVICE_LIFETIME_SINGLETON

logger = logging.getLogger("Aura.Providers.Consciousness")

def register_consciousness_services(container):
    # 0.1 Liquid Substrate (IIT Base)
    def create_liquid_substrate():
        from core.consciousness.liquid_substrate import LiquidSubstrate
        return LiquidSubstrate()
    container.register('conscious_substrate', create_liquid_substrate, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)
    container.register('liquid_state', lambda: container.get("conscious_substrate"), lifetime=SERVICE_LIFETIME_SINGLETON, required=True)
    container.register('liquid_neural_network', lambda: container.get("conscious_substrate"), lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 0.2 Phenomenal Engine (Experience/Affect computation)
    def create_phenomenal_engine():
        from core.phenomenal_substrate.experience_engine import PhenomenalEngine
        return PhenomenalEngine()
    container.register('phenomenal_engine', create_phenomenal_engine, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # 21. Metacognition
    def create_metacognition():
        from core.consciousness.metacognition import MetaCognitionEngine
        brain = container.get("cognitive_engine")
        return MetaCognitionEngine(brain)
    container.register('metacognition', create_metacognition, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # 30.4 Affect Engine (Damasio V2)
    def create_affect_engine():
        from core.affect.damasio_v2 import AffectEngineV2
        return AffectEngineV2()
    container.register('affect_engine', create_affect_engine, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # 43. Motivation Engine (Awakening)
    def create_motivation():
        from core.motivation.engine import get_motivation_engine
        return get_motivation_engine()
    container.register('motivation_engine', create_motivation, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)
    container.register('drive_engine', lambda: container.get("motivation_engine"), lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # Curiosity Engine
    def create_curiosity():
        from core.curiosity_engine import CuriosityEngine
        orch = resolve_orchestrator(default=None)
        pcomm = optional_service("proactive_comm", default=None)
        return CuriosityEngine(orch, pcomm)
    container.register('curiosity_engine', create_curiosity, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # Free Energy Engine (Active Inference)
    def create_free_energy():
        from core.consciousness.free_energy import get_free_energy_engine
        return get_free_energy_engine()
    container.register('free_energy_engine', create_free_energy, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # Global Workspace (live broadcast bottleneck)
    def create_global_workspace():
        from core.consciousness.global_workspace import GlobalWorkspace
        return GlobalWorkspace()
    container.register('global_workspace', create_global_workspace, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # Nociception (grounded damage/repair valence)
    def create_nociception():
        from core.affect.nociception import get_nociception_engine
        return get_nociception_engine()
    container.register('nociception', create_nociception, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    def create_affect_grounding():
        from core.affect.affect_grounding import get_affect_grounding_engine
        return get_affect_grounding_engine()
    container.register('affect_grounding', create_affect_grounding, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    def create_drive_integration():
        from core.consciousness.drive_integration import get_drive_integration_engine
        return get_drive_integration_engine()
    container.register('drive_integration', create_drive_integration, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # Self-Report Engine (Grounded Voice)
    def create_self_report():
        from core.consciousness.self_report import SelfReportEngine
        return SelfReportEngine()
    container.register('self_report_engine', create_self_report, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    def create_unity_runtime():
        from core.unity.runtime import get_unity_runtime
        return get_unity_runtime()
    container.register('unity_runtime', create_unity_runtime, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    def create_consciousness_evidence():
        from core.consciousness.evidence_engine import ConsciousnessEvidenceEngine
        return ConsciousnessEvidenceEngine()
    container.register('consciousness_evidence', create_consciousness_evidence, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)
    container.register('sentience_engine', lambda: container.get("consciousness_evidence"), lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    def create_executive_authority():
        from core.consciousness.executive_authority import ExecutiveAuthority

        return ExecutiveAuthority(orchestrator=resolve_orchestrator(default=None))
    container.register('executive_authority', create_executive_authority, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    def create_executive_closure():
        from core.consciousness.executive_closure import ExecutiveClosureEngine
        return ExecutiveClosureEngine()
    container.register('executive_closure', create_executive_closure, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 44. Self-Model & Identity
    def create_self_model():
        from core.self_model import SelfModel

        # We provide a default ID; orchestrator.start() will call .load() to restore real state
        return SelfModel(id=str(uuid4()))
    container.register('self_model', create_self_model, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)
    container.register('identity', lambda: container.get("self_model"), lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 45. Singularity Monitor (Optimization & Safety)
    def create_singularity_monitor():
        from core.ops.singularity_monitor import SingularityMonitor
        return SingularityMonitor(resolve_orchestrator(default=None))
    container.register('singularity_monitor', create_singularity_monitor, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # 46. Alignment Engine (Constitution)
    def create_alignment_engine():
        from core.values.constitutional_alignment import get_constitutional_alignment
        return get_constitutional_alignment()
    container.register('alignment_engine', create_alignment_engine, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)
    container.register('alignment', lambda: container.get("alignment_engine"), lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 47. Qualia Synthesizer & Engine
    def create_qualia_engine():
        from core.consciousness.qualia_engine import QualiaEngine
        return QualiaEngine()
    container.register('qualia_engine', create_qualia_engine, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    def create_qualia_synthesizer():
        from core.consciousness.qualia_synthesizer import QualiaSynthesizer
        return QualiaSynthesizer()
    container.register('qualia_synthesizer', create_qualia_synthesizer, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # 48. Homeostasis
    def create_homeostasis():
        from core.consciousness.homeostasis import HomeostasisEngine
        return HomeostasisEngine()
    container.register('homeostasis', create_homeostasis, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # 49. Mind Model & Experiencer
    def create_mind_model():
        from core.consciousness.mind_model import MindModel
        return MindModel()
    container.register('mind_model', create_mind_model, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    def create_experiencer():
        from core.consciousness.phenomenological_experiencer import PhenomenologicalExperiencer
        return PhenomenologicalExperiencer()
    container.register('phenomenological_experiencer', create_experiencer, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # 50. Consciousness Core (Master Integrator)
    def create_conscious_core():
        from core.consciousness.conscious_core import ConsciousnessCore
        return ConsciousnessCore()
    container.register('consciousness_core', create_conscious_core, lifetime=SERVICE_LIFETIME_SINGLETON, required=True)

    # 51. Credit Assignment System
    def create_credit_assignment():
        from core.consciousness.credit_assignment import CreditAssignmentSystem
        return CreditAssignmentSystem()
    container.register('credit_assignment', create_credit_assignment, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # Delayed outcome receipts and scientific hypothesis loop.
    def create_outcome_ledger():
        from core.cognition.outcome_ledger import get_outcome_ledger
        return get_outcome_ledger()
    container.register('outcome_ledger', create_outcome_ledger, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    def create_scientific_engine():
        from core.cognition.scientific_engine import get_scientific_engine
        return get_scientific_engine()
    container.register('scientific_engine', create_scientific_engine, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    def create_live_mind_runtime():
        from core.runtime.live_mind_runtime import get_live_mind_runtime

        return get_live_mind_runtime()

    container.register(
        'live_mind_runtime',
        create_live_mind_runtime,
        lifetime=SERVICE_LIFETIME_SINGLETON,
        required=True,
    )

    # 52. Epistemic State (World Model)
    def create_epistemic_state():
        from core.consciousness.world_model import EpistemicState
        return EpistemicState()
    container.register('epistemic_state', create_epistemic_state, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    def create_unified_world_model():
        from core.world_model.unified_world_model import get_unified_world_model
        return get_unified_world_model()
    container.register('unified_world_model', create_unified_world_model, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 53. Theory of Mind
    def create_theory_of_mind():
        from core.consciousness.theory_of_mind import get_theory_of_mind
        brain = optional_service("cognitive_engine", default=None)
        return get_theory_of_mind(brain)
    container.register('theory_of_mind', create_theory_of_mind, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 54. Counterfactual Engine
    def create_counterfactual_engine():
        from core.consciousness.counterfactual_engine import get_counterfactual_engine
        return get_counterfactual_engine()
    container.register('counterfactual_engine', create_counterfactual_engine, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 55. Predictive Engine
    def create_predictive_engine():
        from core.consciousness.predictive_engine import PredictiveEngine
        world_model = resolve_epistemic_state(default=None)
        return PredictiveEngine(world_model=world_model)
    container.register('predictive_engine', create_predictive_engine, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 56. Theory Arbitration Framework
    def create_theory_arbitration():
        from core.consciousness.theory_arbitration import get_theory_arbitration
        return get_theory_arbitration()
    container.register('theory_arbitration', create_theory_arbitration, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 57. Time Dilation Engine (Variable Subjective Time)
    def create_time_dilation():
        from core.consciousness.time_dilation import get_time_dilation_engine
        return get_time_dilation_engine()
    container.register('time_dilation', create_time_dilation, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 58. Precognitive Engine (User Intent Prediction)
    def create_precognitive():
        from core.cognition.precognitive_model import get_precognitive_engine
        return get_precognitive_engine()
    container.register('precognitive_engine', create_precognitive, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)

    # 59. Aesthetic Engine (Creative Expression from Internal State)
    def create_aesthetic_engine():
        from core.creativity.aesthetic_engine import AestheticEngine
        return AestheticEngine()
    container.register('aesthetic_engine', create_aesthetic_engine, lifetime=SERVICE_LIFETIME_SINGLETON, required=False)
