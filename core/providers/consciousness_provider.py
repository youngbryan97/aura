"""core/providers/consciousness_provider.py — Consciousness & Affect Registration
"""

import logging
from core.container import ServiceLifetime
from core.runtime.service_access import (
    optional_service,
    resolve_epistemic_state,
    resolve_orchestrator,
)

logger = logging.getLogger("Aura.Providers.Consciousness")

def register_consciousness_services(container):
    # 0.1 Liquid Substrate (IIT Base)
    def create_liquid_substrate():
        from core.consciousness.liquid_substrate import LiquidSubstrate
        return LiquidSubstrate()
    container.register('conscious_substrate', create_liquid_substrate, lifetime=ServiceLifetime.SINGLETON, required=True)
    container.register('liquid_state', lambda: container.get("conscious_substrate"), lifetime=ServiceLifetime.SINGLETON, required=True)
    container.register('liquid_neural_network', lambda: container.get("conscious_substrate"), lifetime=ServiceLifetime.SINGLETON, required=False)

    # 21. Metacognition
    def create_metacognition():
        from core.consciousness.metacognition import MetaCognitionEngine
        brain = container.get("cognitive_engine")
        return MetaCognitionEngine(brain)
    container.register('metacognition', create_metacognition, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 30.4 Affect Engine (Damasio V2)
    def create_affect_engine():
        from core.affect.damasio_v2 import AffectEngineV2
        return AffectEngineV2()
    container.register('affect_engine', create_affect_engine, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 43. Motivation Engine (Awakening)
    def create_motivation():
        from core.motivation.engine import get_motivation_engine
        return get_motivation_engine()
    container.register('motivation_engine', create_motivation, lifetime=ServiceLifetime.SINGLETON, required=True)
    container.register('drive_engine', lambda: container.get("motivation_engine"), lifetime=ServiceLifetime.SINGLETON, required=True)

    # Curiosity Engine
    def create_curiosity():
        from core.curiosity_engine import CuriosityEngine
        orch = resolve_orchestrator(default=None)
        return CuriosityEngine(orch)
    container.register('curiosity_engine', create_curiosity, lifetime=ServiceLifetime.SINGLETON, required=True)

    # Free Energy Engine (Active Inference)
    def create_free_energy():
        from core.consciousness.free_energy import get_free_energy_engine
        return get_free_energy_engine()
    container.register('free_energy_engine', create_free_energy, lifetime=ServiceLifetime.SINGLETON, required=True)

    # Self-Report Engine (Grounded Voice)
    def create_self_report():
        from core.consciousness.self_report import SelfReportEngine
        return SelfReportEngine()
    container.register('self_report_engine', create_self_report, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_consciousness_evidence():
        from core.consciousness.evidence_engine import ConsciousnessEvidenceEngine
        return ConsciousnessEvidenceEngine()
    container.register('consciousness_evidence', create_consciousness_evidence, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('sentience_engine', lambda: container.get("consciousness_evidence"), lifetime=ServiceLifetime.SINGLETON, required=False)

    def create_executive_authority():
        from core.consciousness.executive_authority import ExecutiveAuthority

        return ExecutiveAuthority(orchestrator=resolve_orchestrator(default=None))
    container.register('executive_authority', create_executive_authority, lifetime=ServiceLifetime.SINGLETON, required=False)

    def create_executive_closure():
        from core.consciousness.executive_closure import ExecutiveClosureEngine
        return ExecutiveClosureEngine()
    container.register('executive_closure', create_executive_closure, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 44. Self-Model & Identity
    def create_self_model():
        from core.self_model import SelfModel
        from uuid import uuid4
        # We provide a default ID; orchestrator.start() will call .load() to restore real state
        return SelfModel(id=str(uuid4()))
    container.register('self_model', create_self_model, lifetime=ServiceLifetime.SINGLETON, required=True)
    container.register('identity', lambda: container.get("self_model"), lifetime=ServiceLifetime.SINGLETON, required=False)

    # 45. Singularity Monitor (Optimization & Safety)
    def create_singularity_monitor():
        from core.ops.singularity_monitor import SingularityMonitor
        return SingularityMonitor()
    container.register('singularity_monitor', create_singularity_monitor, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 46. Alignment Engine (Constitution)
    def create_alignment_engine():
        from core.constitutional_alignment import get_constitutional_alignment
        return get_constitutional_alignment()
    container.register('alignment_engine', create_alignment_engine, lifetime=ServiceLifetime.SINGLETON, required=True)
    container.register('alignment', lambda: container.get("alignment_engine"), lifetime=ServiceLifetime.SINGLETON, required=False)

    # 47. Qualia Synthesizer & Engine
    def create_qualia_engine():
        from core.consciousness.qualia_engine import QualiaEngine
        return QualiaEngine()
    container.register('qualia_engine', create_qualia_engine, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_qualia_synthesizer():
        from core.consciousness.qualia_synthesizer import QualiaSynthesizer
        return QualiaSynthesizer()
    container.register('qualia_synthesizer', create_qualia_synthesizer, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 48. Homeostasis
    def create_homeostasis():
        from core.consciousness.homeostasis import HomeostasisEngine
        return HomeostasisEngine()
    container.register('homeostasis', create_homeostasis, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 49. Mind Model & Experiencer
    def create_mind_model():
        from core.consciousness.mind_model import MindModel
        return MindModel()
    container.register('mind_model', create_mind_model, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_experiencer():
        from core.consciousness.phenomenological_experiencer import PhenomenologicalExperiencer
        return PhenomenologicalExperiencer()
    container.register('phenomenological_experiencer', create_experiencer, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 50. Consciousness Core (Master Integrator)
    def create_conscious_core():
        from core.consciousness.conscious_core import ConsciousnessCore
        return ConsciousnessCore()
    container.register('consciousness_core', create_conscious_core, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 51. Credit Assignment System
    def create_credit_assignment():
        from core.consciousness.credit_assignment import CreditAssignmentSystem
        return CreditAssignmentSystem()
    container.register('credit_assignment', create_credit_assignment, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 52. Epistemic State (World Model)
    def create_epistemic_state():
        from core.consciousness.world_model import EpistemicState
        return EpistemicState()
    container.register('epistemic_state', create_epistemic_state, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 53. Theory of Mind
    def create_theory_of_mind():
        from core.consciousness.theory_of_mind import get_theory_of_mind
        brain = optional_service("cognitive_engine", default=None)
        return get_theory_of_mind(brain)
    container.register('theory_of_mind', create_theory_of_mind, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 54. Counterfactual Engine
    def create_counterfactual_engine():
        from core.consciousness.counterfactual_engine import get_counterfactual_engine
        return get_counterfactual_engine()
    container.register('counterfactual_engine', create_counterfactual_engine, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 55. Predictive Engine
    def create_predictive_engine():
        from core.consciousness.predictive_engine import PredictiveEngine
        world_model = resolve_epistemic_state(default=None)
        return PredictiveEngine(world_model=world_model)
    container.register('predictive_engine', create_predictive_engine, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 56. Theory Arbitration Framework
    def create_theory_arbitration():
        from core.consciousness.theory_arbitration import get_theory_arbitration
        return get_theory_arbitration()
    container.register('theory_arbitration', create_theory_arbitration, lifetime=ServiceLifetime.SINGLETON, required=False)
