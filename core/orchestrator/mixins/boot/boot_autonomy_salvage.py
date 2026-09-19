"""One salvage per subsystem that failed to build.

Lifted whole out of `boot_autonomy`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations


class _SalvagesWhatItCan:
    """Lifted whole out of BootAutonomyMixin; see boot_autonomy.py."""

    async def _salvage_session_guardian(self) -> None:
        # SessionGuardian — prevents conversation cascade failures in long sessions
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_volition_engine(self) -> None:
        # VolitionEngine — autonomous will, impulse-driven agency
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_belief_revision_engine(self) -> None:
        # BeliefRevisionEngine — persistent identity and self-model
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_value_system(self) -> None:
        # ValueSystem — ethical weights (curiosity, integrity, safety, autonomy, empathy)
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_subjective_choice_engine(self) -> None:
        # SubjectiveChoiceEngine — durable preferences that can influence action selection
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_ambient_life_director(self) -> None:
        # AmbientLifeDirector — motive bucketing, pressure pacing, and encounter continuity
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_goal_drift_detector(self) -> None:
        # GoalDriftDetector — prevents rabbit-holing during long goal pursuit
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_self_diagnosis_tool(self) -> None:
        # SelfDiagnosisTool — lets Aura introspect her own capabilities
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_reliability_engine(self) -> None:
        # ReliabilityEngine — already has registration hook, ensure it activates
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_state_authority(self) -> None:
        # StateAuthority — truth arbitration across distributed subsystems
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            _record_boot_autonomy_degradation,
            logger,
        )

        try:
            from core.state.state_authority import register_state_authority

            register_state_authority()
            logger.info("StateAuthority registered — single source of truth active.")
        except _BOOT_AUTONOMY_BOUNDARY_ERRORS as e:
            _record_boot_autonomy_degradation(
                e, "Boot autonomy optional subsystem failed; continuing degraded boot: %s"
            )
            logger.error("StateAuthority init failed: %s", e)

    async def _salvage_external_chat_manager(self) -> None:
        # ExternalChatManager — lets Aura open proactive terminal/GUI chat windows
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_process_manager(self) -> None:
        # ProcessManager — enterprise process lifecycle supervision
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_dialectical_crucible(self) -> None:
        # DialecticalCrucible — internal Hegelian debate engine
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_heuristic_synthesizer(self) -> None:
        # HeuristicSynthesizer — learned instinct extraction
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_abstraction_engine(self) -> None:
        # AbstractionEngine — first-principles extraction
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_dream_journal(self) -> None:
        # DreamJournal — qualia-driven creativity during idle
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_bryan_model_engine(self) -> None:
        # BryanModelEngine — evolving theory of the user
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_belief_graph(self) -> None:
        # BeliefGraph — persistent world model
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_goal_belief_manager(self) -> None:
        # GoalBeliefManager — goals as first-class beliefs
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_snapshot_manager(self) -> None:
        # SnapshotManager — cognitive state persistence
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_shadow_ast_healer(self) -> None:
        # ShadowASTHealer — self-repair via AST manipulation
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_refusal_engine(self) -> None:
        # RefusalEngine — genuine autonomous refusal
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_autonomous_self_modification(self) -> None:
        # AutonomousSelfModification — Will-authorized self-modification
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            _env_flag,
            _foreground_only_runtime,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_scar_formation(self) -> None:
        # ScarFormation — behavioral scars from critical experiences
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_value_autopoiesis(self) -> None:
        # ValueAutopoiesis — drive weight evolution from experience
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_constitutional_gate(self) -> None:
        # ConstitutionalGate — mathematical safety floor for self-modification
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_star_reasoner(self) -> None:
        # STaR Reasoner — autonomous training data generation from task traces
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_reimplementation_lab(self) -> None:
        # ReimplementationLab — register but DON'T start (gated by memory/Zenith)
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _env_flag,
            _foreground_only_runtime,
            _record_boot_autonomy_degradation,
            logger,
        )

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

    async def _salvage_continuous_simulator_loop(self) -> None:
        # ContinuousSimulatorLoop — register but DON'T start (gated by memory)
        from .boot_autonomy import (
            _BOOT_AUTONOMY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_boot_autonomy_degradation,
            logger,
        )

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

