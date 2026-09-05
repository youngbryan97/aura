import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation

logger = logging.getLogger(__name__)

_BOOT_AUTONOMY_DEGRADATION_KEY = "boot_autonomy"
_BOOT_AUTONOMY_BOUNDARY_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    asyncio.InvalidStateError,
    Exception,
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _foreground_only_runtime() -> bool:
    try:
        from core.runtime.background_policy import foreground_only_runtime

        return bool(foreground_only_runtime())
    except _BOOT_AUTONOMY_BOUNDARY_ERRORS:
        return _env_flag("AURA_FOREGROUND_ONLY", False)


def _proof_runtime_active() -> bool:
    try:
        from core.runtime.proof_policy import proof_run_active

        return bool(proof_run_active(origin="boot_autonomy"))
    except _BOOT_AUTONOMY_BOUNDARY_ERRORS:
        return _env_flag("AURA_PROOF_RUN", False)


def _safe_priority(value: Any, default: float = 0.6) -> float:
    try:
        return max(0.6, float(value))
    except _BOOT_AUTONOMY_BOUNDARY_ERRORS:
        return float(default)


def _record_boot_autonomy_degradation(
    exc: BaseException,
    message: str,
    *args: Any,
    action: str = "continued autonomy boot with optional subsystem degraded",
    severity: str = "warning",
) -> None:
    record_degradation(
        _BOOT_AUTONOMY_DEGRADATION_KEY,
        exc,
        severity=severity,
        action=action,
    )
    logger.debug(message, *args, exc)


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

    def _start_skill_catalog_warmup(self) -> None:
        """Start catalog validation early without publishing partial state."""
        existing = getattr(self, "_skill_catalog_warmup_task", None)
        if existing is not None and not existing.done():
            return

        from core.utils.task_tracker import get_task_tracker

        from ....capability_engine import CapabilityEngine

        engine = CapabilityEngine(orchestrator=self)
        self._skill_catalog_warmup_engine = engine
        self._skill_catalog_warmup_task = get_task_tracker().create_task(
            asyncio.to_thread(lambda: len(engine.skills)),
            name="orchestrator.skill_catalog_warmup",
        )

    async def _consume_skill_catalog_warmup(self) -> tuple[Any, int]:
        """Return the one warmed engine only after its catalog is authoritative."""
        from ....capability_engine import CapabilityEngine

        engine = getattr(self, "_skill_catalog_warmup_engine", None)
        if engine is None:
            engine = CapabilityEngine(orchestrator=self)
        warmup = getattr(self, "_skill_catalog_warmup_task", None)
        if warmup is not None:
            try:
                loaded = await warmup
            except _BOOT_AUTONOMY_BOUNDARY_ERRORS as exc:
                _record_boot_autonomy_degradation(
                    exc,
                    "Early skill-catalog warmup failed; retrying at Phase 6: %s",
                    action="retried the canonical catalog transaction before publishing it",
                    severity="degraded",
                )
                loaded = await asyncio.to_thread(lambda: len(engine.skills))
        else:
            loaded = await asyncio.to_thread(lambda: len(engine.skills))
        self._skill_catalog_warmup_task = None
        self._skill_catalog_warmup_engine = None
        return engine, int(loaded)

    async def _init_autonomous_evolution(self):
        """Initialize the background evolution and Curiosity Engine with granular error boundaries."""
        logger.info("🔎 Activating Autonomous Self-Modification...")

        boot_steps: tuple[tuple[str, Callable[[], Awaitable[None]]], ...] = (
            ("self_modification_engine", self._init_self_modification_engine),
            ("transcendence_layer", self._init_transcendence_layer),
            ("cognitive_modulators", self._init_cognitive_modulators),
            ("meta_learning", self._init_meta_learning),
            ("meta_optimization", self._init_meta_optimization),
            ("concept_bridge", self._init_concept_bridge),
            ("advanced_ontology", self._init_advanced_ontology),
            ("motivation_engine", self._init_motivation_engine),
            ("reflex_engine", self._init_reflex_engine),
            ("identity_gate", self._init_identity_gate),
            ("lazarus_brainstem", self._init_lazarus_brainstem),
            ("persona_evolver", self._init_persona_evolver),
            ("live_learner", self._init_live_learner),
            ("autonomous_task_engine", self._init_autonomous_task_engine),
            ("continuous_learner", self._init_continuous_learner),
            ("weight_compounding", self._init_weight_compounding),
            ("crsm_closure", self._init_crsm_closure),
            ("expert_lora_library", self._init_expert_lora_library),
            ("fictional_synthesis", self._init_fictional_synthesis),
            ("final_foundations", self._init_final_foundations),
            ("evolution_orchestrator", self._init_evolution_orchestrator),
            ("singularity_loops", self._init_singularity_loops),
        )
        for name, step in boot_steps:
            try:
                await step()
            except _BOOT_AUTONOMY_BOUNDARY_ERRORS as exc:
                _record_boot_autonomy_degradation(
                    exc,
                    "Autonomous evolution boot step %s failed: %s",
                    name,
                )

        logger.info("🛠️ _init_autonomous_evolution complete")

    async def _init_transcendence_layer(self):
        """Initialize the Transcendence Layer (Meta-Evolution)."""
        try:
            from core.cognition.meta_cognition import MetaEvolutionEngine

            self.meta_evolution = MetaEvolutionEngine()
            ServiceContainer.register_instance("meta_evolution", self.meta_evolution)
            logger.info("🌌 Transcendence Infrastructure online")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("🌌 Transcendence Infrastructure failed: %s", e)

    async def _init_cognitive_modulators(self):
        """Initialize Cognitive Modulators (Humility, Causal Model, Skill Library)."""
        try:
            from core.adaptation.epistemic_humility import register_epistemic_humility

            self.epistemic_humility = register_epistemic_humility(self)

            from core.brain.causal_world_model import register_causal_world_model

            self.world_model = register_causal_world_model(self)

            # The value graph is the OTHER half of the high-risk-tool restraint
            # pair beside the causal world model. Only the causal half was ever
            # registered, so the gate saw a permanently "unavailable" value
            # graph and refused every python_sandbox / shell_executor /
            # file_operations request outright.
            from core.adaptation.dynamic_value_graph import register_dynamic_value_graph

            self.dynamic_value_graph = register_dynamic_value_graph(self)

            from core.agency.skill_library import register_skill_library

            self.skill_library = register_skill_library(self)
            logger.info("🧠 Cognitive Modulators online")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("🧠 Cognitive Modulators failed: %s", e)

    async def _init_reflex_engine(self):
        """Initialize the Reflex System."""
        try:
            from core.resilience.reflex_engine import ReflexEngine

            self.reflex_engine = ReflexEngine(self)
            self.reflex_engine.prime_voice()
            logger.info("✓ Reflex Engine online (Tiny Brain primed)")
        except ImportError:
            self.reflex_engine = None
            ServiceContainer.register_instance("reflex_engine", None, required=False)
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("Reflex Engine failed: %s", e)

        # Bridge 2: Hardened Reflex Core (SOMA)
        try:
            from core.mycelium import MycelialNetwork

            net = MycelialNetwork()
            if hasattr(net, "reflex") and net.reflex:
                net.reflex.orchestrator = self
                logger.info("⚡ Hardened Reflex Core (SOMA) bridged to Orchestrator")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("Failed to bridge Reflex Core: %s", e)

    async def _init_evolution_orchestrator(self):
        """Initialize the Singularity Path Evolution Orchestrator."""
        if (
            _foreground_only_runtime()
            or _proof_runtime_active()
            or not _env_flag("AURA_ENABLE_EVOLUTION_ORCHESTRATOR", True)
        ):
            logger.info("Evolution Orchestrator disabled for foreground/proof boot.")
            return
        try:
            from core.evolution.evolution_orchestrator import get_evolution_orchestrator

            evo = get_evolution_orchestrator()
            await evo.start()
            logger.info("🧬 Evolution Orchestrator online — tracking 8 evolutionary axes")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("🧬 Evolution Orchestrator failed: %s", e)

    async def _init_singularity_loops(self):
        """Initialize the closed-loop evolutionary wiring."""
        if _foreground_only_runtime() or _proof_runtime_active():
            logger.info("Singularity loops disabled for foreground/proof boot.")
            return
        if _env_flag("AURA_ENABLE_SINGULARITY_LOOPS", True):
            try:
                from core.evolution.singularity_loops import get_singularity_loops

                loops = get_singularity_loops()
                await loops.start()
                logger.info("🔗 Singularity Loops online — 6 feedback loops active")
            except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
                _record_boot_autonomy_degradation(
                    e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
                )
                logger.error("🔗 Singularity Loops failed: %s", e)
        else:
            logger.info(
                "Singularity loops disabled by configuration; continuing Tier 4 boot wiring."
            )

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
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("🌍 WorldState init failed: %s", e)

        # InitiativeSynthesizer — single origin for all impulses
        try:
            from core.initiative_synthesis import get_initiative_synthesizer

            synth = get_initiative_synthesizer()
            await synth.start()
            logger.info("🔀 InitiativeSynthesizer ONLINE — single impulse funnel active")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("🔀 InitiativeSynthesizer init failed: %s", e)

        # InternalSimulator — counterfactual action evaluation
        try:
            from core.simulation.internal_simulator import InternalSimulator

            simulator = InternalSimulator()
            ServiceContainer.register_instance("internal_simulator", simulator)
            logger.info("🔮 InternalSimulator ONLINE — counterfactual reasoning active")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("🔮 InternalSimulator init failed: %s", e)

        # ContinuousCognitionLoop — non-LLM brainstem (exists between prompts)
        try:
            from core.continuous_cognition import get_continuous_cognition

            ccl = get_continuous_cognition()
            await ccl.start()
            logger.info("🧠 ContinuousCognitionLoop ONLINE — brainstem active at 2Hz")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
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
                    existing_goals = {
                        str(p.get("goal", "")) for p in pending if isinstance(p, dict)
                    }
                    for goal in active_goals:
                        objective = str(goal.get("objective") or goal.get("name") or "")
                        if not objective or objective in existing_goals:
                            continue
                        pending.append(
                            {
                                "goal": objective,
                                "source": "goal_engine",
                                "type": "continuity_restored",
                                "urgency": _safe_priority(goal.get("priority", 0.6)),
                                "triggered_by": "boot_resumption",
                                "timestamp": time.time(),
                                "metadata": {
                                    "goal_id": goal.get("id"),
                                    "continuity_restored": True,
                                    "horizon": goal.get("horizon", "short_term"),
                                },
                            }
                        )
                        resumed_count += 1
                    state.cognition.pending_initiatives = pending
                if resumed_count > 0:
                    logger.info("🔄 Goal Resumption: restored %d interrupted goals", resumed_count)
                else:
                    logger.debug("Goal Resumption: no interrupted goals found")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("🔄 Goal Resumption failed: %s", e)

    async def _init_final_foundations(self):
        """Initialize World Model, Narrative Identity, and Metacognitive Calibrator."""
        if getattr(self, "_final_foundations_initialized", False):
            logger.info("🏛️ Final Foundations already initialized; reusing canonical services.")
            return
        if getattr(self, "_final_foundations_initializing", False):
            logger.info("🏛️ Final Foundations initialization already in progress.")
            return
        self._final_foundations_initializing = True
        try:
            from core.final_engines import register_final_engines

            self.final_engines = register_final_engines(orchestrator=self)
            logger.info("🏛️ Final Foundations registered (World/Identity/Meta)")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("🏛️ Final Foundations failed: %s", e)

        try:
            await self._init_salvaged_subsystems()
            self._final_foundations_initialized = True
        finally:
            self._final_foundations_initializing = False

    async def _init_salvaged_subsystems(self):
        """Wire in fully-implemented subsystems that were previously unregistered."""

        # SessionGuardian — prevents conversation cascade failures in long sessions
        try:
            from core.session.session_guardian import get_guardian

            guardian = ServiceContainer.get("session_guardian", default=None)
            if guardian is None:
                guardian = get_guardian()
            guardian.attach(self).start()
            ServiceContainer.register_instance("session_guardian", guardian)
            logger.info("SessionGuardian active — health monitoring engaged.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("SessionGuardian init failed: %s", e)

        # VolitionEngine — autonomous will, impulse-driven agency
        try:
            from core.volition import VolitionEngine

            volition = VolitionEngine(self)
            ServiceContainer.register_instance("volition_engine", volition)
            logger.info("VolitionEngine online — autonomous agency active.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("VolitionEngine init failed: %s", e)

        # BeliefRevisionEngine — persistent identity and self-model
        try:
            from core.epistemics.belief_revision import get_belief_revision_engine

            belief_engine = ServiceContainer.get("belief_revision_engine", default=None)
            if belief_engine is None:
                belief_engine = get_belief_revision_engine()
            await belief_engine.start()
            ServiceContainer.register_instance("belief_revision_engine", belief_engine)
            logger.info("BeliefRevisionEngine online — identity persistence active.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("BeliefRevisionEngine init failed: %s", e)

        # ValueSystem — ethical weights (curiosity, integrity, safety, autonomy, empathy)
        try:
            from core.values.values_engine import ValueSystem

            values = ValueSystem()
            ServiceContainer.register_instance("value_system", values)
            ServiceContainer.register_instance("values_engine", values)
            logger.info("ValueSystem online — ethical foundation registered.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("ValueSystem init failed: %s", e)

        # SubjectiveChoiceEngine — durable preferences that can influence action selection
        try:
            from core.agency.subjective_choice import get_subjective_choice_engine

            subjective_choice = get_subjective_choice_engine()
            ServiceContainer.register_instance(
                "subjective_choice_engine",
                subjective_choice,
                required=False,
                registered_by="boot_autonomy",
            )
            logger.info(
                "SubjectiveChoiceEngine online — authored preference receipts influence arbitration."
            )
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("SubjectiveChoiceEngine init failed: %s", e)

        # AmbientLifeDirector — motive bucketing, pressure pacing, and encounter continuity
        try:
            from core.agency.ambient_life_director import get_ambient_life_director

            ambient_life = get_ambient_life_director()
            ServiceContainer.register_instance(
                "ambient_life_director",
                ambient_life,
                required=False,
                registered_by="boot_autonomy",
            )
            logger.info(
                "AmbientLifeDirector online — motive buckets and autonomy pacing influence arbitration."
            )
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("AmbientLifeDirector init failed: %s", e)

        # DreamProcessor — legacy offline memory consolidation (Disabled)
        logger.debug("DreamProcessor is deprecated. Functionality moved to DreamCoordinator.")

        # GoalDriftDetector — prevents rabbit-holing during long goal pursuit
        try:
            from core.goals.goal_drift_detector import GoalDriftDetector

            cognitive_engine = ServiceContainer.get("cognitive_engine", default=None)
            if cognitive_engine:
                drift_detector = GoalDriftDetector(cognitive_engine)
                ServiceContainer.register_instance("goal_drift_detector", drift_detector)
                logger.info("GoalDriftDetector registered — goal coherence monitoring active.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("GoalDriftDetector init failed: %s", e)

        # SelfDiagnosisTool — lets Aura introspect her own capabilities
        try:
            from core.coordinators.skill_execution_diagnostics import SelfDiagnosisTool

            capability_engine = ServiceContainer.get("capability_engine", default=None)
            if capability_engine:
                diagnostics = SelfDiagnosisTool(capability_engine)
                ServiceContainer.register_instance("self_diagnostics", diagnostics)
                logger.info("SelfDiagnosisTool registered — capability introspection active.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("SelfDiagnosisTool init failed: %s", e)

        # ReliabilityEngine — already has registration hook, ensure it activates
        try:
            from core.reliability_engine import get_reliability_engine
            from core.utils.task_tracker import get_task_tracker

            rel = get_reliability_engine()
            get_task_tracker().create_task(rel.start(), name="reliability_engine.start")
            logger.info("ReliabilityEngine activated — stability guarantees enforced.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("ReliabilityEngine activation failed: %s", e)

        # StateAuthority — truth arbitration across distributed subsystems
        try:
            from core.state.state_authority import register_state_authority

            register_state_authority()
            logger.info("StateAuthority registered — single source of truth active.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("StateAuthority init failed: %s", e)

        # ExternalChatManager — lets Aura open proactive terminal/GUI chat windows
        try:
            from core.conversation.external_chat import ExternalChatManager
            from core.runtime.shutdown_coordinator import get_shutdown_coordinator

            if not hasattr(self, "conversation_history"):
                self.conversation_history = []
            external_chat = ExternalChatManager(self)
            shutdown_coordinator = get_shutdown_coordinator()
            handler_name = "external_chat.shutdown"
            if handler_name not in shutdown_coordinator.handler_names("actors"):
                shutdown_coordinator.register(
                    external_chat.shutdown,
                    phase="actors",
                    name=handler_name,
                    timeout=5.0,
                )
            ServiceContainer.register_instance("external_chat", external_chat)
            logger.info("ExternalChatManager online — proactive chat windows available.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("ExternalChatManager init failed: %s", e)

        # ProcessManager — enterprise process lifecycle supervision
        try:
            from core.ops.process_manager import ProcessManager
            from core.runtime.shutdown_coordinator import get_shutdown_coordinator

            pm = ProcessManager()
            ServiceContainer.register_instance("process_manager", pm)
            shutdown_coordinator = get_shutdown_coordinator()
            handler_name = "process_manager.stop_children"
            if handler_name not in shutdown_coordinator.handler_names("actors"):
                shutdown_coordinator.register(
                    pm.on_stop_async,
                    phase="actors",
                    name=handler_name,
                    timeout=pm.cleanup_timeout_s + 1.0,
                )
            logger.info("ProcessManager online — child process supervision active.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("ProcessManager init failed: %s", e)

        # DialecticalCrucible — internal Hegelian debate engine
        try:
            from core.adaptation.dialectics import get_crucible

            crucible = get_crucible()
            ServiceContainer.register_instance("dialectical_crucible", crucible)
            logger.info("⚔️ DialecticalCrucible online — adversarial belief testing active.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("DialecticalCrucible init failed: %s", e)

        # HeuristicSynthesizer — learned instinct extraction
        try:
            from core.adaptation.heuristic_synthesizer import get_heuristic_synthesizer

            hs = get_heuristic_synthesizer()
            ServiceContainer.register_instance("heuristic_synthesizer", hs)
            logger.info(
                "📐 HeuristicSynthesizer online — %d active heuristics.", len(hs._active_heuristics)
            )
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("HeuristicSynthesizer init failed: %s", e)

        # AbstractionEngine — first-principles extraction
        try:
            from core.adaptation.abstraction_engine import AbstractionEngine

            ae = AbstractionEngine()
            ServiceContainer.register_instance("abstraction_engine", ae)
            logger.info("🧠 AbstractionEngine online — first-principles extraction active.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("AbstractionEngine init failed: %s", e)

        # DreamJournal — qualia-driven creativity during idle
        try:
            from core.adaptation.dream_journal import DreamJournal

            memory_nexus = ServiceContainer.get(
                "memory_facade", default=None
            ) or ServiceContainer.get("memory_manager", default=None)
            brain = ServiceContainer.get("cognitive_engine", default=None)
            if memory_nexus and brain:
                dj = DreamJournal(memory_nexus, brain)
                ServiceContainer.register_instance("dream_journal", dj)
                logger.info("🌌 DreamJournal online — subconscious creativity active.")
                from core.autonomic.reflection_loop import get_autonomic_reflection_loop

                reflection_loop = get_autonomic_reflection_loop()
                await reflection_loop.start()
                ServiceContainer.register_instance(
                    "autonomic_reflection_loop",
                    reflection_loop,
                    required=False,
                )
                logger.info("🌌 AutonomicReflectionLoop online — ambient self-correction journal active.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("DreamJournal init failed: %s", e)

        # BryanModelEngine — evolving theory of the user
        try:
            existing_bme = ServiceContainer.get(
                "bryan_model_engine", default=None
            ) or ServiceContainer.get("bryan_model", default=None)
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
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("BryanModelEngine init failed: %s", e)

        # BeliefGraph — persistent world model
        try:
            existing_bg = ServiceContainer.get("belief_graph", default=None)
            if existing_bg is None:
                from core.world_model.belief_graph import BeliefGraph

                bg = BeliefGraph()
                ServiceContainer.register_instance("belief_graph", bg)
                logger.info(
                    "🌐 BeliefGraph online — %d nodes, %d edges.",
                    bg.graph.number_of_nodes(),
                    bg.graph.number_of_edges(),
                )
            else:
                logger.info(
                    "🌐 BeliefGraph already registered — %d nodes.",
                    existing_bg.graph.number_of_nodes(),
                )
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("BeliefGraph init failed: %s", e)

        # GoalBeliefManager — goals as first-class beliefs
        try:
            from core.world_model.goal_beliefs import GoalBeliefManager

            bg_inst = ServiceContainer.get("belief_graph", default=None)
            if bg_inst:
                gbm = GoalBeliefManager(bg_inst)
                ServiceContainer.register_instance("goal_belief_manager", gbm)
                logger.info("🎯 GoalBeliefManager online.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("GoalBeliefManager init failed: %s", e)

        # SnapshotManager — cognitive state persistence
        try:
            from core.resilience.snapshot_manager import SnapshotManager

            sm = SnapshotManager(orchestrator=self)
            ServiceContainer.register_instance("snapshot_manager", sm)
            logger.info("📸 SnapshotManager online — cognitive persistence active.")

            # Register shutdown hooks — save state on death for continuity across restarts
            from core.ops.graceful_shutdown import register_shutdown_hook

            def _save_on_shutdown():
                logger.info("💾 [SHUTDOWN] Saving substrate state and cognitive snapshot...")
                try:
                    substrate = ServiceContainer.get("liquid_substrate", default=None)
                    if substrate and hasattr(substrate, "_save_state"):
                        substrate._save_state()
                        logger.info("💾 [SHUTDOWN] Substrate state saved.")
                except _BOOT_AUTONOMY_BOUNDARY_ERRORS as exc:
                    _record_boot_autonomy_degradation(
                        exc, "Boot autonomy shutdown persistence failed; continuing shutdown: %s"
                    )
                    logger.error("💾 [SHUTDOWN] Substrate save failed: %s", exc)
                try:
                    sm.freeze()
                    logger.info("💾 [SHUTDOWN] Cognitive snapshot frozen.")
                except _BOOT_AUTONOMY_BOUNDARY_ERRORS as exc:
                    _record_boot_autonomy_degradation(
                        exc, "Boot autonomy shutdown persistence failed; continuing shutdown: %s"
                    )
                    logger.error("💾 [SHUTDOWN] Snapshot freeze failed: %s", exc)

            register_shutdown_hook(_save_on_shutdown)
            logger.info("💾 Shutdown persistence hooks registered.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("SnapshotManager init failed: %s", e)

        # ShadowASTHealer — self-repair via AST manipulation
        try:
            from core.config import config
            from core.self_modification.shadow_ast_healer import ShadowASTHealer

            healer = ShadowASTHealer(codebase_root=config.paths.project_root)
            ServiceContainer.register_instance("shadow_ast_healer", healer)
            logger.info("🛠️ ShadowASTHealer online — self-repair active.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("ShadowASTHealer init failed: %s", e)

        # RefusalEngine — genuine autonomous refusal
        try:
            from core.autonomy.genuine_refusal import RefusalEngine

            re_engine = RefusalEngine()
            ServiceContainer.register_instance("refusal_engine", re_engine)
            logger.info("🛡️ RefusalEngine online — sovereign identity protection active.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("RefusalEngine init failed: %s", e)

        # AutonomousSelfModification — Will-authorized self-modification
        if _foreground_only_runtime() or not _env_flag(
            "AURA_ENABLE_AUTONOMOUS_SELF_MODIFICATION", True
        ):
            logger.info("AutonomousSelfModification disabled for foreground-only boot.")
        else:
            try:
                from core.autonomy.self_modification import get_autonomous_self_modification

                asm = get_autonomous_self_modification()
                await asm.start()
                logger.info("🧬 AutonomousSelfModification online — Will-gated evolution active.")
            except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
                _record_boot_autonomy_degradation(
                    e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
                )
                logger.error("AutonomousSelfModification init failed: %s", e)

        # ScarFormation — behavioral scars from critical experiences
        try:
            from core.memory.scar_formation import get_scar_formation

            scars = get_scar_formation()
            await scars.start()
            logger.info("🩹 ScarFormation online — learned caution active.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("ScarFormation init failed: %s", e)

        # ValueAutopoiesis — drive weight evolution from experience
        try:
            from core.adaptation.value_autopoiesis import get_value_autopoiesis

            vap = get_value_autopoiesis()
            await vap.start()
            logger.info("🧬 ValueAutopoiesis online — value evolution active.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("ValueAutopoiesis init failed: %s", e)

        # ConstitutionalGate — mathematical safety floor for self-modification
        try:
            from core.safety.constitutional_gate import get_constitutional_gate

            const_gate = get_constitutional_gate()
            await const_gate.start()
            logger.info("🛡️ ConstitutionalGate ONLINE — %s", const_gate.get_status())
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("ConstitutionalGate init failed: %s", e)

        # STaR Reasoner — autonomous training data generation from task traces
        try:
            from core.adaptation.star_reasoner import get_star_reasoner

            star = get_star_reasoner()
            await star.start()
            logger.info("⭐ STaR Reasoner ONLINE — self-taught improvement active")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("STaR Reasoner init failed: %s", e)

        # ReimplementationLab — register but DON'T start (gated by memory/Zenith)
        if _foreground_only_runtime() or not _env_flag("AURA_REGISTER_REIMPLEMENTATION_LAB", True):
            logger.info("ReimplementationLab disabled for foreground-only boot.")
        else:
            try:
                from core.config import config
                from core.self_improvement.reimplementation_lab import ReimplementationLab

                lab = ReimplementationLab(project_root=str(config.paths.project_root))
                ServiceContainer.register_instance("reimplementation_lab", lab, required=False)
                logger.info(
                    "🔬 ReimplementationLab REGISTERED (gated — awaiting resource clearance)"
                )
            except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
                _record_boot_autonomy_degradation(
                    e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
                )
                logger.error("ReimplementationLab registration failed: %s", e)

        # ContinuousSimulatorLoop — register but DON'T start (gated by memory)
        try:
            from core.environment.embodied_simulator import ContinuousSimulatorLoop

            affordance_kb = ServiceContainer.get("affordance_kb", default=None)
            causal_model = ServiceContainer.get("causal_world_model", default=None)
            if affordance_kb and causal_model:
                sim_loop = ContinuousSimulatorLoop(affordance_kb, causal_model)
                ServiceContainer.register_instance("embodied_simulator", sim_loop, required=False)
                logger.info(
                    "🌍 ContinuousSimulatorLoop REGISTERED (gated — awaiting resource clearance)"
                )
            else:
                logger.debug(
                    "ContinuousSimulatorLoop skipped — missing affordance_kb or causal_model"
                )
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("ContinuousSimulatorLoop registration failed: %s", e)

    async def _init_motivation_engine(self):
        """Initialize Motivation Engine (Aura's Awakening)."""
        try:
            from core.motivation.engine import MotivationEngine

            self.motivation = MotivationEngine()
            mot = self.motivation
            ServiceContainer.register_instance("motivation_engine", mot)
            if mot is not None:
                await mot.start()
                ServiceContainer.register_instance("drive_engine", mot)
                ServiceContainer.register_instance("drives", mot)
            logger.info("✨ Motivation Engine Active: Aura is now self-directed.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
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
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("🛑 Task Engine init failed: %s", e)

    async def _init_proactive_systems(self):
        """Initialize curiosity, proactive communication, and belief sync with granular error boundaries."""
        logger.info("🛠️ _init_proactive_systems starting")
        background_start_blocker = ""
        try:
            from core.runtime.background_policy import background_loop_start_reason

            background_start_blocker = background_loop_start_reason(
                origin="boot_proactive_systems",
            )
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e,
                "Boot autonomy background policy unavailable; continuing foreground-only.",
                action="disabled proactive systems because background loop admission policy failed closed",
                severity="warning",
            )
            background_start_blocker = "background_policy_unavailable"
        if (
            _foreground_only_runtime()
            or _proof_runtime_active()
            or background_start_blocker
            or not _env_flag("AURA_ENABLE_PROACTIVE_SYSTEMS", True)
        ):
            logger.info(
                "Proactive systems disabled for foreground/proof/safe boot%s.",
                f" ({background_start_blocker})" if background_start_blocker else "",
            )
            ServiceContainer.register_instance("proactive_comm", None, required=False)
            ServiceContainer.register_instance("sensory_motor_cortex", None, required=False)
            ServiceContainer.register_instance("autonomous_initiative_loop", None, required=False)
            ServiceContainer.register_instance("subconscious_loop", None, required=False)
            ServiceContainer.register_instance("abstract_thought_layer", None, required=False)
            ServiceContainer.register_instance(
                "conversational_momentum_engine", None, required=False
            )
            self.proactive_comm = None
            self.research_cycle = None
            return

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
        await self._init_abstract_thought_subsystem(tracker)
        await self._start_belief_sync_at_boot(tracker)

        # 🚀 Phase 30: Unfettered Presence & Spontaneous Agency
        try:
            from core.social.presence_integration import apply_presence_patch

            apply_presence_patch(self)
            logger.info("✨ Phase 30 Presence Patch applied.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("Failed to apply Presence Patch: %s", e)

        # 🔬 Research Cycle Daemon — autonomous knowledge pursuit during idle
        if (
            _foreground_only_runtime()
            or _proof_runtime_active()
            or not _env_flag("AURA_ENABLE_RESEARCH_CYCLE", True)
        ):
            logger.info("Research Cycle disabled for foreground/proof boot.")
            self.research_cycle = None
        else:
            try:
                from core.autonomy.research_cycle import start_research_daemon

                self.research_cycle = await start_research_daemon(self)
                logger.info("🔬 Research Cycle daemon activated.")
            except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
                _record_boot_autonomy_degradation(
                    e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
                )
                logger.error("Research Cycle init failed: %s", e)
                self.research_cycle = None

        logger.info("🛠️ _init_proactive_systems complete")

    async def _init_proactive_comm_subsystem(self):
        """Initialize the proactive communication subsystem."""
        try:
            from core.autonomy.proactive_communication import get_proactive_comm

            pcomm = get_proactive_comm()
            pcomm.notification_callback = self._proactive_notify_callback
            self.proactive_comm = pcomm
            ServiceContainer.register_instance("proactive_comm", pcomm)
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("Proactive Communication init failed: %s", e)
            self.proactive_comm = None
            ServiceContainer.register_instance("proactive_comm", None, required=False)

    async def _init_attention_summarizer_subsystem(self):
        """Initialize the Attention Summarizer."""
        try:
            from core.memory.attention import AttentionSummarizer

            self.attention_summarizer = AttentionSummarizer(self)
            ServiceContainer.register_instance("attention_summarizer", self.attention_summarizer)
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("AttentionSummarizer init failed: %s", e)
            ServiceContainer.register_instance("attention_summarizer", None, required=False)

    async def _init_probe_manager_subsystem(self):
        """Initialize the Probe Manager."""
        try:
            from core.collective.probe_manager import ProbeManager

            self.probe_manager = ProbeManager(self)
            ServiceContainer.register_instance("probe_manager", self.probe_manager)
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("ProbeManager init failed: %s", e)
            ServiceContainer.register_instance("probe_manager", None, required=False)

    async def _init_curiosity_engine_subsystem(self):
        """Initialize the Curiosity Engine."""
        try:
            from core.curiosity_engine import CuriosityEngine

            pcomm = ServiceContainer.get("proactive_comm", default=None)
            ce = CuriosityEngine(self, pcomm)
            self.curiosity = ce
            ServiceContainer.register_instance("curiosity_engine", ce)
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("CuriosityEngine init failed: %s", e)
            self.curiosity = None
            ServiceContainer.register_instance("curiosity_engine", None, required=False)

    async def _init_sensory_motor_integration_subsystem(self, tracker):
        """Initialize Sensory-Motor Integration components."""
        if (
            _foreground_only_runtime()
            or _proof_runtime_active()
            or not _env_flag("AURA_ENABLE_SENSORIMOTOR_GROUNDING", True)
        ):
            logger.info("Sensory-Motor Integration disabled for foreground/proof boot.")
            ServiceContainer.register_instance("sensory_motor_cortex", None, required=False)
            ServiceContainer.register_instance("autonomous_initiative_loop", None, required=False)
            ServiceContainer.register_instance(
                "conversational_momentum_engine", None, required=False
            )
            return
        try:
            from core.autonomy.autonomous_initiative_loop import AutonomousInitiativeLoop
            from core.conversation.conversational_momentum_engine import (
                ConversationalMomentumEngine,
            )
            from core.somatic.sensory_motor_cortex import SensoryMotorCortex

            smc = SensoryMotorCortex(self)
            ail = AutonomousInitiativeLoop(self)
            cme = ConversationalMomentumEngine(self)
            ServiceContainer.register_instance("conversational_momentum_engine", cme)
            ServiceContainer.register_instance("sensory_motor_cortex", smc)
            ServiceContainer.register_instance("autonomous_initiative_loop", ail)

            tracker.create_task(smc.start(), name="smc")
            tracker.create_task(ail.start(), name="ail")
            tracker.create_task(cme.start(), name="cme")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("Sensory-Motor Integration failed: %s", e)
            ServiceContainer.register_instance("sensory_motor_cortex", None, required=False)
            ServiceContainer.register_instance("autonomous_initiative_loop", None, required=False)
            ServiceContainer.register_instance(
                "conversational_momentum_engine", None, required=False
            )

    async def _init_skill_system(self):
        """Initialize unified capability engine."""
        engine, skills_loaded = await self._consume_skill_catalog_warmup()
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

        # Off the loop: this is the first touch of the catalog, so it runs the
        # whole discovery/validation transaction — ~1.4s of imports and file
        # I/O that lockdep measured as a loop-blocking hold. During boot the
        # loop is what serves /api/health/boot, and a loop that cannot answer
        # is a desktop stuck on "RESUMING LIVE SURFACE".
        self.status.skills_loaded = skills_loaded
        logger.info("✓ Capability Engine initialized with %d skills", self.status.skills_loaded)

        from core.skill_management.hephaestus import HephaestusEngine

        self.hephaestus = HephaestusEngine()
        ServiceContainer.register_instance("hephaestus_engine", self.hephaestus)
        logger.info("✓ Hephaestus Forge online")

        try:
            from core.brain.parameter_self_modulation import ParameterSelfModulator

            self.sampler_modulator = ParameterSelfModulator()
            logger.info("✓ Parameter Self-Modulator active")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("Failed to init Sampler Modulator: %s", e)
            self.sampler_modulator = None
