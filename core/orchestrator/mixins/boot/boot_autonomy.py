from .boot_autonomy_salvage import _SalvagesWhatItCan
import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from core.container import ServiceContainer  # noqa: F401  (read at call time by the lifted module)
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


class BootAutonomyMixin(_SalvagesWhatItCan):
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

        from ....capability_engine import CapabilityEngine
        from core.utils.task_tracker import get_task_tracker

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

    #: The modules the salvaged initialiser imports. Loaded on a thread
    #: before the initialiser runs, so the construction below finds them in
    #: sys.modules and pays no disk read on the loop: a module read inside
    #: this initialiser was a 5.7s loop stall on a busy disk (dump,
    #: 2026-09-16 22:56).
    _SALVAGED_SUBSYSTEM_MODULES: tuple[str, ...] = (
        "core.adaptation.abstraction_engine",
        "core.adaptation.dialectics",
        "core.adaptation.dream_journal",
        "core.adaptation.heuristic_synthesizer",
        "core.adaptation.star_reasoner",
        "core.adaptation.value_autopoiesis",
        "core.agency.ambient_life_director",
        "core.agency.subjective_choice",
        "core.autonomic.reflection_loop",
        "core.autonomy.genuine_refusal",
        "core.autonomy.self_modification",
        "core.config",
        "core.conversation.external_chat",
        "core.coordinators.skill_execution_diagnostics",
        "core.environment.embodied_simulator",
        "core.epistemics.belief_revision",
        "core.goals.goal_drift_detector",
        "core.memory.scar_formation",
        "core.ops.graceful_shutdown",
        "core.ops.process_manager",
        "core.reliability_engine",
        "core.resilience.snapshot_manager",
        "core.runtime.shutdown_coordinator",
        "core.safety.constitutional_gate",
        "core.self_improvement.reimplementation_lab",
        "core.self_modification.shadow_ast_healer",
        "core.session.session_guardian",
        "core.state.state_authority",
        "core.utils.task_tracker",
        "core.values.values_engine",
        "core.volition",
        "core.world_model.belief_graph",
        "core.world_model.goal_beliefs",
        "core.world_model.user_model",
    )

    async def _import_salvaged_modules_off_loop(self) -> None:
        import importlib

        def _load() -> None:
            for name in self._SALVAGED_SUBSYSTEM_MODULES:
                try:
                    importlib.import_module(name)
                except Exception as exc:  # noqa: BLE001 — the initialiser reports its own
                    logger.debug("Salvaged module %s did not preload: %s", name, exc)

        await asyncio.to_thread(_load)

    async def _init_salvaged_subsystems(self):
        """Wire in fully-implemented subsystems that were previously unregistered."""
        await self._import_salvaged_modules_off_loop()

        await self._salvage_session_guardian()
        await self._salvage_volition_engine()
        await self._salvage_belief_revision_engine()
        await self._salvage_value_system()
        await self._salvage_subjective_choice_engine()
        await self._salvage_ambient_life_director()
        # DreamProcessor — legacy offline memory consolidation (Disabled)
        logger.debug("DreamProcessor is deprecated. Functionality moved to DreamCoordinator.")
        await self._salvage_goal_drift_detector()
        await self._salvage_self_diagnosis_tool()
        await self._salvage_reliability_engine()
        await self._salvage_state_authority()
        await self._salvage_external_chat_manager()
        await self._salvage_process_manager()
        await self._salvage_dialectical_crucible()
        await self._salvage_heuristic_synthesizer()
        await self._salvage_abstraction_engine()
        await self._salvage_dream_journal()
        await self._salvage_bryan_model_engine()
        await self._salvage_belief_graph()
        await self._salvage_goal_belief_manager()
        await self._salvage_snapshot_manager()
        await self._salvage_shadow_ast_healer()
        await self._salvage_refusal_engine()
        await self._salvage_autonomous_self_modification()
        await self._salvage_scar_formation()
        await self._salvage_value_autopoiesis()
        await self._salvage_constitutional_gate()
        await self._salvage_star_reasoner()
        await self._salvage_reimplementation_lab()
        await self._salvage_continuous_simulator_loop()

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
