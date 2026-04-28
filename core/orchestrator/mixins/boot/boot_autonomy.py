from core.runtime.errors import record_degradation
import asyncio
import logging
from typing import Any, Optional

from core.container import ServiceContainer

logger = logging.getLogger(__name__)


class BootAutonomyMixin:
    """Provides initialization for autonomous evolution, proactive drives, and motivation engines."""

    meta_evolution: Any
    epistemic_humility: Any
    world_model: Any
    skill_library: Any
    reflex_engine: Any
    final_engines: Any
    motivation: Any
    attention_summarizer: Any
    probe_manager: Any

    async def _init_autonomous_evolution(self):
        """Initialize the background evolution and Curiosity Engine with granular error boundaries."""
        logger.info("🔎 Activating Autonomous Self-Modification...")

        # These methods are distributed across the various boot mixins
        await self._init_self_modification_engine()
        await self._init_transcendence_layer()
        await self._init_cognitive_modulators()
        await self._init_meta_learning()
        await self._init_meta_optimization()
        await self._init_concept_bridge()
        await self._init_advanced_ontology()
        await self._init_motivation_engine()
        await self._init_reflex_engine()
        await self._init_identity_gate()
        await self._init_lazarus_brainstem()
        await self._init_persona_evolver()
        await self._init_live_learner()
        await self._init_autonomous_task_engine()
        await self._init_continuous_learner()
        await self._init_fictional_synthesis()
        await self._init_final_foundations()
        await self._init_evolution_orchestrator()
        await self._init_singularity_loops()

        logger.info("🛠️ _init_autonomous_evolution complete")

    async def _init_transcendence_layer(self):
        """Initialize the Transcendence Layer (Meta-Evolution)."""
        try:
            from core.meta_cognition import MetaEvolutionEngine

            self.meta_evolution = MetaEvolutionEngine()
            ServiceContainer.register_instance("meta_evolution", self.meta_evolution)
            logger.info("🌌 Transcendence Infrastructure online")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("🌌 Transcendence Infrastructure failed: %s", e)

    async def _init_cognitive_modulators(self):
        """Initialize Cognitive Modulators (Humility, Causal Model, Skill Library)."""
        try:
            from core.adaptation.epistemic_humility import register_epistemic_humility

            self.epistemic_humility = register_epistemic_humility(self)

            from core.brain.causal_world_model import register_causal_world_model

            self.world_model = register_causal_world_model(self)

            from core.agency.skill_library import register_skill_library

            self.skill_library = register_skill_library(self)
            logger.info("🧠 Cognitive Modulators online")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("🧠 Cognitive Modulators failed: %s", e)

    async def _init_reflex_engine(self):
        """Initialize the Reflex System."""
        try:
            from core.reflex_engine import ReflexEngine

            self.reflex_engine = ReflexEngine(self)
            self.reflex_engine.prime_voice()
            logger.info("✓ Reflex Engine online (Tiny Brain primed)")
        except ImportError:
            self.reflex_engine = None
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("Reflex Engine failed: %s", e)

        # Bridge 2: Hardened Reflex Core (SOMA)
        try:
            from core.mycelium import MycelialNetwork

            net = MycelialNetwork()
            if hasattr(net, "reflex") and net.reflex:
                net.reflex.orchestrator = self
                logger.info("⚡ Hardened Reflex Core (SOMA) bridged to Orchestrator")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("Failed to bridge Reflex Core: %s", e)

    async def _init_evolution_orchestrator(self):
        """Initialize the Singularity Path Evolution Orchestrator."""
        try:
            from core.evolution.evolution_orchestrator import get_evolution_orchestrator
            evo = get_evolution_orchestrator()
            await evo.start()
            logger.info("🧬 Evolution Orchestrator online — tracking 8 evolutionary axes")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("🧬 Evolution Orchestrator failed: %s", e)

    async def _init_singularity_loops(self):
        """Initialize the closed-loop evolutionary wiring."""
        try:
            from core.evolution.singularity_loops import get_singularity_loops
            loops = get_singularity_loops()
            await loops.start()
            logger.info("🔗 Singularity Loops online — 6 feedback loops active")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("🔗 Singularity Loops failed: %s", e)

        # ══════════════════════════════════════════════════════════════
        # TIER 4 UNIFICATION BOOT — WorldState, InitiativeSynthesizer,
        # InternalSimulator, Goal Resumption
        # ══════════════════════════════════════════════════════════════

        # WorldState — live perceptual feed
        try:
            from core.world_state import get_world_state
            ws = get_world_state()
            await ws.start()
            logger.info("🌍 WorldState ONLINE — live perceptual feed active")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("🌍 WorldState init failed: %s", e)

        # InitiativeSynthesizer — single origin for all impulses
        try:
            from core.initiative_synthesis import get_initiative_synthesizer
            synth = get_initiative_synthesizer()
            await synth.start()
            logger.info("🔀 InitiativeSynthesizer ONLINE — single impulse funnel active")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("🔀 InitiativeSynthesizer init failed: %s", e)

        # InternalSimulator — counterfactual action evaluation
        try:
            from core.simulation.internal_simulator import InternalSimulator
            simulator = InternalSimulator()
            ServiceContainer.register_instance("internal_simulator", simulator)
            logger.info("🔮 InternalSimulator ONLINE — counterfactual reasoning active")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("🔮 InternalSimulator init failed: %s", e)

        # ContinuousCognitionLoop — non-LLM brainstem (exists between prompts)
        try:
            from core.continuous_cognition import get_continuous_cognition
            ccl = get_continuous_cognition()
            await ccl.start()
            logger.info("🧠 ContinuousCognitionLoop ONLINE — brainstem active at 2Hz")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("🧠 ContinuousCognitionLoop init failed: %s", e)

        # Goal Resumption — restore interrupted goals from SQLite
        try:
            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine:
                active_goals = goal_engine.get_active_goals(
                    limit=5,
                    include_external=False,
                    actionable_only=True,
                )
                resumed_count = 0
                state_repo = ServiceContainer.get("state_repo", default=None)
                state = None
                if state_repo and hasattr(state_repo, "_current"):
                    state = state_repo._current
                if state and active_goals:
                    pending = list(getattr(state.cognition, "pending_initiatives", []) or [])
                    existing_goals = {str(p.get("goal", "")) for p in pending if isinstance(p, dict)}
                    for goal in active_goals:
                        objective = str(goal.get("objective") or goal.get("name") or "")
                        if not objective or objective in existing_goals:
                            continue
                        pending.append({
                            "goal": objective,
                            "source": "goal_engine",
                            "type": "continuity_restored",
                            "urgency": max(0.6, float(goal.get("priority", 0.6))),
                            "triggered_by": "boot_resumption",
                            "timestamp": __import__("time").time(),
                            "metadata": {
                                "goal_id": goal.get("id"),
                                "continuity_restored": True,
                                "horizon": goal.get("horizon", "short_term"),
                            },
                        })
                        resumed_count += 1
                    state.cognition.pending_initiatives = pending
                if resumed_count > 0:
                    logger.info("🔄 Goal Resumption: restored %d interrupted goals", resumed_count)
                else:
                    logger.debug("Goal Resumption: no interrupted goals found")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("🔄 Goal Resumption failed: %s", e)

    async def _init_final_foundations(self):
        """Initialize World Model, Narrative Identity, and Metacognitive Calibrator."""
        try:
            from core.final_engines import register_final_engines

            self.final_engines = register_final_engines(orchestrator=self)
            logger.info("🏛️ Final Foundations registered (World/Identity/Meta)")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("🏛️ Final Foundations failed: %s", e)

        await self._init_salvaged_subsystems()

    async def _init_salvaged_subsystems(self):
        """Wire in fully-implemented subsystems that were previously unregistered."""

        # SessionGuardian — prevents conversation cascade failures in long sessions
        try:
            from core.session_guardian import SessionGuardian
            guardian = SessionGuardian()
            guardian.attach(self).start()
            ServiceContainer.register_instance("session_guardian", guardian)
            logger.info("SessionGuardian active — health monitoring engaged.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("SessionGuardian init failed: %s", e)

        # VolitionEngine — autonomous will, impulse-driven agency
        try:
            from core.volition import VolitionEngine
            volition = VolitionEngine(self)
            ServiceContainer.register_instance("volition_engine", volition)
            logger.info("VolitionEngine online — autonomous agency active.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("VolitionEngine init failed: %s", e)

        # BeliefRevisionEngine — persistent identity and self-model
        try:
            from core.belief_revision import BeliefRevisionEngine
            belief_engine = BeliefRevisionEngine()
            await belief_engine.start()
            ServiceContainer.register_instance("belief_revision_engine", belief_engine)
            logger.info("BeliefRevisionEngine online — identity persistence active.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("BeliefRevisionEngine init failed: %s", e)

        # ValueSystem — ethical weights (curiosity, integrity, safety, autonomy, empathy)
        try:
            from core.values_engine import ValueSystem
            values = ValueSystem()
            ServiceContainer.register_instance("value_system", values)
            ServiceContainer.register_instance("values_engine", values)
            logger.info("ValueSystem online — ethical foundation registered.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("ValueSystem init failed: %s", e)

        # DreamProcessor — offline memory consolidation to long-term knowledge
        try:
            from core.dream_processor import DreamProcessor
            memory_nexus = ServiceContainer.get("memory_facade", default=None) or ServiceContainer.get("memory_manager", default=None)
            brain = ServiceContainer.get("cognitive_engine", default=None)
            if memory_nexus and brain:
                dreamer = DreamProcessor(memory_nexus, brain)
                ServiceContainer.register_instance("dream_processor", dreamer)
                logger.info("DreamProcessor registered — memory consolidation available.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("DreamProcessor init failed: %s", e)

        # GoalDriftDetector — prevents rabbit-holing during long goal pursuit
        try:
            from core.goal_drift_detector import GoalDriftDetector
            cognitive_engine = ServiceContainer.get("cognitive_engine", default=None)
            if cognitive_engine:
                drift_detector = GoalDriftDetector(cognitive_engine)
                ServiceContainer.register_instance("goal_drift_detector", drift_detector)
                logger.info("GoalDriftDetector registered — goal coherence monitoring active.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("GoalDriftDetector init failed: %s", e)

        # SelfDiagnosisTool — lets Aura introspect her own capabilities
        try:
            from core.skill_execution_diagnostics import SelfDiagnosisTool
            capability_engine = ServiceContainer.get("capability_engine", default=None)
            if capability_engine:
                diagnostics = SelfDiagnosisTool(capability_engine)
                ServiceContainer.register_instance("self_diagnostics", diagnostics)
                logger.info("SelfDiagnosisTool registered — capability introspection active.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("SelfDiagnosisTool init failed: %s", e)

        # ReliabilityEngine — already has registration hook, ensure it activates
        try:
            from core.reliability_engine import get_reliability_engine
            from core.utils.task_tracker import get_task_tracker
            rel = get_reliability_engine()
            get_task_tracker().create_task(rel.start(), name="reliability_engine.start")
            logger.info("ReliabilityEngine activated — stability guarantees enforced.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("ReliabilityEngine activation failed: %s", e)

        # StateAuthority — truth arbitration across distributed subsystems
        try:
            from core.state_authority import register_state_authority
            register_state_authority()
            logger.info("StateAuthority registered — single source of truth active.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("StateAuthority init failed: %s", e)

        # ExternalChatManager — lets Aura open proactive terminal/GUI chat windows
        try:
            from core.external_chat import ExternalChatManager
            if not hasattr(self, "conversation_history"):
                self.conversation_history = []
            external_chat = ExternalChatManager(self)
            ServiceContainer.register_instance("external_chat", external_chat)
            logger.info("ExternalChatManager online — proactive chat windows available.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("ExternalChatManager init failed: %s", e)

        # ProcessManager — enterprise process lifecycle supervision
        try:
            from core.process_manager import ProcessManager
            pm = ProcessManager()
            ServiceContainer.register_instance("process_manager", pm)
            logger.info("ProcessManager online — child process supervision active.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("ProcessManager init failed: %s", e)

        # DialecticalCrucible — internal Hegelian debate engine
        try:
            from core.adaptation.dialectics import get_crucible
            crucible = get_crucible()
            ServiceContainer.register_instance("dialectical_crucible", crucible)
            logger.info("⚔️ DialecticalCrucible online — adversarial belief testing active.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("DialecticalCrucible init failed: %s", e)

        # HeuristicSynthesizer — learned instinct extraction
        try:
            from core.adaptation.heuristic_synthesizer import get_heuristic_synthesizer
            hs = get_heuristic_synthesizer()
            ServiceContainer.register_instance("heuristic_synthesizer", hs)
            logger.info("📐 HeuristicSynthesizer online — %d active heuristics.", len(hs._active_heuristics))
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("HeuristicSynthesizer init failed: %s", e)

        # AbstractionEngine — first-principles extraction
        try:
            from core.adaptation.abstraction_engine import AbstractionEngine
            ae = AbstractionEngine()
            ServiceContainer.register_instance("abstraction_engine", ae)
            logger.info("🧠 AbstractionEngine online — first-principles extraction active.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("AbstractionEngine init failed: %s", e)

        # DreamJournal — qualia-driven creativity during idle
        try:
            from core.adaptation.dream_journal import DreamJournal
            memory_nexus = ServiceContainer.get("memory_facade", default=None) or ServiceContainer.get("memory_manager", default=None)
            brain = ServiceContainer.get("cognitive_engine", default=None)
            if memory_nexus and brain:
                dj = DreamJournal(memory_nexus, brain)
                ServiceContainer.register_instance("dream_journal", dj)
                logger.info("🌌 DreamJournal online — subconscious creativity active.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("DreamJournal init failed: %s", e)

        # BryanModelEngine — evolving theory of the user
        try:
            existing_bme = ServiceContainer.get("bryan_model_engine", default=None) or ServiceContainer.get("bryan_model", default=None)
            if existing_bme is None:
                from core.world_model.user_model import BryanModelEngine
                bme = BryanModelEngine()
                ServiceContainer.register_instance("bryan_model_engine", bme)
                logger.info("🧠 BryanModelEngine online — user theory active.")
            else:
                # Ensure it's also available under bryan_model_engine key
                if ServiceContainer.get("bryan_model_engine", default=None) is None:
                    ServiceContainer.register_instance("bryan_model_engine", existing_bme)
                logger.info("🧠 BryanModelEngine already registered.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("BryanModelEngine init failed: %s", e)

        # BeliefGraph — persistent world model
        try:
            existing_bg = ServiceContainer.get("belief_graph", default=None)
            if existing_bg is None:
                from core.world_model.belief_graph import BeliefGraph
                bg = BeliefGraph()
                ServiceContainer.register_instance("belief_graph", bg)
                logger.info("🌐 BeliefGraph online — %d nodes, %d edges.", bg.graph.number_of_nodes(), bg.graph.number_of_edges())
            else:
                logger.info("🌐 BeliefGraph already registered — %d nodes.", existing_bg.graph.number_of_nodes())
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("BeliefGraph init failed: %s", e)

        # GoalBeliefManager — goals as first-class beliefs
        try:
            from core.world_model.goal_beliefs import GoalBeliefManager
            bg_inst = ServiceContainer.get("belief_graph", default=None)
            if bg_inst:
                gbm = GoalBeliefManager(bg_inst)
                ServiceContainer.register_instance("goal_belief_manager", gbm)
                logger.info("🎯 GoalBeliefManager online.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("GoalBeliefManager init failed: %s", e)

        # SnapshotManager — cognitive state persistence
        try:
            from core.resilience.snapshot_manager import SnapshotManager
            sm = SnapshotManager(orchestrator=self)
            ServiceContainer.register_instance("snapshot_manager", sm)
            logger.info("📸 SnapshotManager online — cognitive persistence active.")

            # Register shutdown hooks — save state on death for continuity across restarts
            from core.graceful_shutdown import register_shutdown_hook

            def _save_on_shutdown():
                logger.info("💾 [SHUTDOWN] Saving substrate state and cognitive snapshot...")
                try:
                    substrate = ServiceContainer.get("liquid_substrate", default=None)
                    if substrate and hasattr(substrate, "_save_state"):
                        substrate._save_state()
                        logger.info("💾 [SHUTDOWN] Substrate state saved.")
                except Exception as exc:
                    record_degradation('boot_autonomy', exc)
                    record_degradation('boot_autonomy', exc)
                    logger.error("💾 [SHUTDOWN] Substrate save failed: %s", exc)
                try:
                    sm.freeze()
                    logger.info("💾 [SHUTDOWN] Cognitive snapshot frozen.")
                except Exception as exc:
                    record_degradation('boot_autonomy', exc)
                    record_degradation('boot_autonomy', exc)
                    logger.error("💾 [SHUTDOWN] Snapshot freeze failed: %s", exc)

            register_shutdown_hook(_save_on_shutdown)
            logger.info("💾 Shutdown persistence hooks registered.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("SnapshotManager init failed: %s", e)

        # ShadowASTHealer — self-repair via AST manipulation
        try:
            from core.self_modification.shadow_ast_healer import ShadowASTHealer
            from core.config import config
            healer = ShadowASTHealer(codebase_root=config.paths.project_root)
            ServiceContainer.register_instance("shadow_ast_healer", healer)
            logger.info("🛠️ ShadowASTHealer online — self-repair active.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("ShadowASTHealer init failed: %s", e)

        # RefusalEngine — genuine autonomous refusal
        try:
            from core.autonomy.genuine_refusal import RefusalEngine
            re_engine = RefusalEngine()
            ServiceContainer.register_instance("refusal_engine", re_engine)
            logger.info("🛡️ RefusalEngine online — sovereign identity protection active.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("RefusalEngine init failed: %s", e)

        # AutonomousSelfModification — Will-authorized self-modification
        try:
            from core.autonomy.self_modification import get_autonomous_self_modification
            asm = get_autonomous_self_modification()
            await asm.start()
            logger.info("🧬 AutonomousSelfModification online — Will-gated evolution active.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("AutonomousSelfModification init failed: %s", e)

        # ScarFormation — behavioral scars from critical experiences
        try:
            from core.memory.scar_formation import get_scar_formation
            scars = get_scar_formation()
            await scars.start()
            logger.info("🩹 ScarFormation online — learned caution active.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("ScarFormation init failed: %s", e)

        # ValueAutopoiesis — drive weight evolution from experience
        try:
            from core.adaptation.value_autopoiesis import get_value_autopoiesis
            vap = get_value_autopoiesis()
            await vap.start()
            logger.info("🧬 ValueAutopoiesis online — value evolution active.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("ValueAutopoiesis init failed: %s", e)

    async def _init_motivation_engine(self):
        """Initialize Motivation Engine (Aura's Awakening)."""
        try:
            from core.motivation.engine import MotivationEngine

            self.motivation = MotivationEngine()
            mot = self.motivation
            ServiceContainer.register_instance("motivation_engine", mot)
            if mot is not None:
                await mot.start()
            logger.info("✨ Motivation Engine Active: Aura is now self-directed.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("✨ Motivation Engine failed: %s", e)

    async def _init_autonomous_task_engine(self):
        """Initialize the Autonomous Task Engine for multi-step agency."""
        try:
            from core.agency.autonomous_task_engine import get_task_engine

            te = get_task_engine()
            # Task engine doesn't have a start() yet, but we ensure it's registered
            ServiceContainer.register_instance("autonomous_task_engine", te)
            ServiceContainer.register_instance("task_engine", te)
            logger.info("✓ Autonomous Task Engine registered")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("🛑 Task Engine init failed: %s", e)

    async def _init_proactive_systems(self):
        """Initialize curiosity, proactive communication, and belief sync with granular error boundaries."""
        logger.info("🛠️ _init_proactive_systems starting")

        # We need the tracker for starting async tasks
        from core.utils.task_tracker import get_task_tracker

        tracker = get_task_tracker()

        await self._init_proactive_comm_subsystem()
        await self._init_belief_sync_subsystem()
        await self._init_attention_summarizer_subsystem()
        await self._init_probe_manager_subsystem()
        await self._init_curiosity_engine_subsystem()
        await self._init_sensory_motor_integration_subsystem(tracker)
        await self._init_subconscious_loop_subsystem(tracker)
        await self._start_belief_sync_at_boot(tracker)

        # 🚀 Phase 30: Unfettered Presence & Spontaneous Agency
        try:
            from core.presence_integration import apply_presence_patch

            apply_presence_patch(self)
            logger.info("✨ Phase 30 Presence Patch applied.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("Failed to apply Presence Patch: %s", e)

        # 🔬 Research Cycle Daemon — autonomous knowledge pursuit during idle
        try:
            from core.autonomy.research_cycle import start_research_daemon
            self.research_cycle = await start_research_daemon(self)
            logger.info("🔬 Research Cycle daemon activated.")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("Research Cycle init failed: %s", e)
            self.research_cycle = None

        logger.info("🛠️ _init_proactive_systems complete")

    async def _init_proactive_comm_subsystem(self):
        """Initialize the proactive communication subsystem."""
        try:
            from core.proactive_communication import get_proactive_comm

            pcomm = get_proactive_comm()
            pcomm.notification_callback = self._proactive_notify_callback
            self.proactive_comm = pcomm
            ServiceContainer.register_instance("proactive_comm", pcomm)
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("Proactive Communication init failed: %s", e)
            self.proactive_comm = None
            ServiceContainer.register_instance("proactive_comm", None)

    async def _init_attention_summarizer_subsystem(self):
        """Initialize the Attention Summarizer."""
        try:
            from core.memory.attention import AttentionSummarizer

            self.attention_summarizer = AttentionSummarizer(self)
            ServiceContainer.register_instance(
                "attention_summarizer", self.attention_summarizer
            )
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("AttentionSummarizer init failed: %s", e)
            ServiceContainer.register_instance("attention_summarizer", None)

    async def _init_probe_manager_subsystem(self):
        """Initialize the Probe Manager."""
        try:
            from core.collective.probe_manager import ProbeManager

            self.probe_manager = ProbeManager(self)
            ServiceContainer.register_instance("probe_manager", self.probe_manager)
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("ProbeManager init failed: %s", e)
            ServiceContainer.register_instance("probe_manager", None)

    async def _init_curiosity_engine_subsystem(self):
        """Initialize the Curiosity Engine."""
        try:
            from core.curiosity_engine import CuriosityEngine

            pcomm = ServiceContainer.get("proactive_comm", default=None)
            ce = CuriosityEngine(self, pcomm)
            self.curiosity = ce
            ServiceContainer.register_instance("curiosity_engine", ce)
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("CuriosityEngine init failed: %s", e)
            self.curiosity = None
            ServiceContainer.register_instance("curiosity_engine", None)

    async def _init_sensory_motor_integration_subsystem(self, tracker):
        """Initialize Sensory-Motor Integration components."""
        try:
            from core.autonomous_initiative_loop import AutonomousInitiativeLoop
            from core.conversational_momentum_engine import (
                ConversationalMomentumEngine,
            )
            from core.sensory_motor_cortex import SensoryMotorCortex

            smc = SensoryMotorCortex(self)
            ail = AutonomousInitiativeLoop(self)
            cme = ConversationalMomentumEngine(self)
            ServiceContainer.register_instance("conversational_momentum_engine", cme)
            ServiceContainer.register_instance("sensory_motor_cortex", smc)
            ServiceContainer.register_instance("autonomous_initiative_loop", ail)

            tracker.create_task(smc.start(), name="smc")
            tracker.create_task(ail.start(), name="ail")
            tracker.create_task(cme.start(), name="cme")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("Sensory-Motor Integration failed: %s", e)
            ServiceContainer.register_instance("sensory_motor_cortex", None)
            ServiceContainer.register_instance("autonomous_initiative_loop", None)
            ServiceContainer.register_instance("conversational_momentum_engine", None)

    async def _init_skill_system(self):
        """Initialize unified capability engine."""
        from ....capability_engine import CapabilityEngine

        engine = CapabilityEngine(orchestrator=self)
        self._capability_engine = engine  # Unified reference
        ServiceContainer.register_instance("capability_engine", engine)
        ServiceContainer.register_instance("skill_manager", engine)  # Legacy shim
        ServiceContainer.register_instance("skill_router", engine)  # Legacy shim

        # Intent Router (v11.0 Clean Room)
        from core.cognitive.router import IntentRouter

        intent_router = IntentRouter()
        ServiceContainer.register_instance("intent_router", intent_router)
        ServiceContainer.register_instance("cognitive_router", intent_router)

        # State Machine (v11.0 Deterministic logic)
        from core.cognitive.state_machine import StateMachine

        state_machine = StateMachine(orchestrator=self)
        ServiceContainer.register_instance("state_machine", state_machine)

        self.status.skills_loaded = len(engine.skills)
        logger.info(
            "✓ Capability Engine initialized with %d skills", self.status.skills_loaded
        )

        from core.skill_management.hephaestus import HephaestusEngine

        self.hephaestus = HephaestusEngine()
        ServiceContainer.register_instance("hephaestus_engine", self.hephaestus)
        logger.info("✓ Hephaestus Forge online")

        try:
            from core.brain.parameter_self_modulation import ParameterSelfModulator

            self.sampler_modulator = ParameterSelfModulator()
            logger.info("✓ Parameter Self-Modulator active")
        except Exception as e:
            record_degradation('boot_autonomy', e)
            record_degradation('boot_autonomy', e)
            logger.error("Failed to init Sampler Modulator: %s", e)
            self.sampler_modulator = None
