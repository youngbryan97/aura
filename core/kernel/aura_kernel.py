from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from core.consciousness.executive_authority import get_executive_authority
from core.container import ServiceContainer
from core.kernel.bridge import LegacyPhase
from core.kernel.organs import OrganStub
from core.kernel.upgrades_10x import (
    EternalGrowthEngine,
    EternalMemoryPhase,
    GodModeToolPhase,
    NativeMultimodalBridge,
    PerfectEmotionPhase,
    TrueEvolutionPhase,
)
from core.phases.affect_update import AffectUpdatePhase
from core.phases.bonding_phase import BondingPhase
from core.phases.cognitive_integration_phase import CognitiveIntegrationPhase
from core.phases.cognitive_routing_unitary import CognitiveRoutingPhase
from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase
from core.phases.inference_phase import InferencePhase
from core.phases.learning_phase import LearningPhase
from core.phases.motivation_update import MotivationUpdatePhase
from core.phases.phi_consciousness import PhiConsciousnessPhase
from core.phases.repair_phase import RepairPhase
from core.phases.response_generation_unitary import UnitaryResponsePhase
from core.resilience.error_boundary import wrap_phase
from core.runtime.pipeline_blueprint import (
    bind_legacy_runtime_phase_attributes,
    kernel_phase_attribute_order,
    resolve_phase_instances,
)
from core.self_modification.boot_validator import GhostBootValidator
from core.state.aura_state import AuraState
from core.state.state_repository import StateRepository
from core.thought_tracer import tracer
from core.utils.concurrency import RobustLock

from .feedback_observer import FeedbackObserver, TickEntry
from .self_review import SelfReviewPhase
from .shadow_kernel import ShadowExecutionPhase


class KernelStatus(BaseModel):
    running: bool = False
    initialized: bool = False
    cycle_count: int = 0
    message: str = "Standby"

logger = logging.getLogger("Aura.Core.Kernel")

@dataclass(frozen=True)
class KernelConfig:
    max_concurrent_phases: int = 4
    watchdog_timeout_s: float = 240.0
    state_versioning: bool = True
    mirror_frequency: float = 1.0 # Hz

class MirrorSnapshot(BaseModel):
    """Immutable projection of internal state for GUI consumption."""
    version: int
    vitality: float
    mood: str
    curiosity: float
    phi: float
    last_objective: str
    timestamp: float

class AuraKernel:
    """
    The Unitary Organism Kernel.
    Centralizes all state, tasks, and service resolution.
    Enforces the Three Invariants:
    1. Boot-Time Closed Graph
    2. Supervised Task Hierarchy
    3. Monolithic State Vault
    """

    def __init__(self, config: KernelConfig, vault: StateRepository):
        """
        Initialize the kernel with a configuration and state vault.

        All phases, organs, and the feedback observer are instantiated here;
        actual async boot (organ loading, state hydration) happens in boot().
        """
        self.config = config
        self.vault: StateRepository = vault
        self.state: AuraState | None = None 
        self.status = KernelStatus()
        self._running = False
        
        # Pipelines & Supervision
        self._task_group: asyncio.TaskGroup | None = None
        self._phases: list[Any] = []
        self._services: dict[type, Any] = {}
        self._background_tasks: list[asyncio.Task] = []
        
        # Pulse-Mirroring Pattern
        self._mirror_state: MirrorSnapshot | None = None 
        self._gui_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        
        # Organ Stub Pattern
        self.organs: dict[str, OrganStub] = {} # Populated in boot()
        
        # [10X] Phase Singletons
        self.eternal = EternalMemoryPhase(self)
        self.evolution = TrueEvolutionPhase(self, engine=None) # Engine resolved via property lazy-loading
        self.perfect_emotion = PerfectEmotionPhase(self)
        self.godmode_tools = GodModeToolPhase(self)
        self.growth = EternalGrowthEngine(self)
        self.multimodal = NativeMultimodalBridge(self)
        self.evolution_guard = ShadowExecutionPhase(self)

        # Core Kernel Phases
        self.phi_phase = PhiConsciousnessPhase(self)
        self.affect_phase = AffectUpdatePhase(self)
        self.cognitive_integration = CognitiveIntegrationPhase(self)
        self.motivation_phase = MotivationUpdatePhase(self)
        self.routing_phase = CognitiveRoutingPhase(self)
        self.response_phase = UnitaryResponsePhase(self)
        self.learning_phase = LearningPhase(self)
        self.self_review_phase = SelfReviewPhase(self)
        self.inference_phase = InferencePhase(self)
        self.bonding_phase = BondingPhase(self)
        self.repair_phase = RepairPhase(self)
        self.legacy_bridge = LegacyPhase(self)
        self.conversational_dynamics_phase = ConversationalDynamicsPhase(self)

        # [CONSTITUTIONAL UNIFICATION] Shared runtime phases are bootstrapped from one blueprint.
        bind_legacy_runtime_phase_attributes(
            self,
            self,
            include_executive_closure=True,
        )
        # The shared legacy bootstrap wires many common phases, but the response
        # phase must remain the unitary implementation used by the sovereign
        # kernel path rather than the older compatibility generator.
        self.response_phase = UnitaryResponsePhase(self)

        # Feedback Observer
        self.feedback_observer = FeedbackObserver()

        # [ASI Genesis] Self-Review & Boot Verification
        self._boot_validator = GhostBootValidator(Path("."))
        self._auto_fix_engine = None 
        self._guardian = None
        self._lock = RobustLock("AuraKernel.StateLock")
        self.volition_level: int = 3 # 0=Lockdown, 1=Reflective, 2=Perceptive, 3=Agentic [GENESIS DEFAULT]
        # Priority preemption: background ticks yield when a user message is waiting
        import threading as _threading
        self._user_priority_pending: _threading.Event = _threading.Event()
        self._last_tick_completed_at: float = 0.0  # telemetry: set after each tick()

    @staticmethod
    def _normalize_origin(origin: Any) -> str:
        return str(origin or "").strip().lower().replace("-", "_")

    @classmethod
    def _is_user_facing_origin(cls, origin: Any) -> bool:
        normalized = cls._normalize_origin(origin)
        if not normalized:
            return False
        if normalized in {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external"}:
            return True
        tokens = {token for token in normalized.split("_") if token}
        return bool(tokens & {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external"})

    def _finalize_foreground_turn_state(self, *, objective: str, turn_origin: str) -> None:
        if self.state is None:
            return
        if not self._is_user_facing_origin(turn_origin):
            return

        final_origin = self._normalize_origin(getattr(self.state.cognition, "current_origin", ""))
        final_objective = str(getattr(self.state.cognition, "current_objective", "") or "")

        # Preserve a newly seeded background objective that replaced the
        # foreground turn during the same tick.
        if (
            final_objective
            and final_objective != objective
            and not self._is_user_facing_origin(final_origin)
        ):
            logger.debug(
                "Kernel: preserving post-turn background objective '%s' from origin=%s.",
                final_objective[:80],
                final_origin or "unknown",
            )
            return

        self.state.cognition.current_objective = None
        self.state.cognition.current_origin = None

    def _phase_timeout_seconds(self, phase_name: str, *, priority: bool) -> float:
        """Give foreground response generation enough headroom without letting background stalls monopolize the lock.

        Priority turns must protect the foreground lane. They keep generous
        headroom for actual response generation, but all non-response phases
        get tight budgets so a single introspection or consolidation phase
        cannot monopolize the kernel lock and starve chat.
        """
        if not priority:
            if phase_name in {"UnitaryResponsePhase", "ResponseGenerationPhase"}:
                return 12.0
            if phase_name in {
                "EternalMemoryPhase",
                "EternalGrowthEngine",
                "TrueEvolutionPhase",
                "GodModeToolPhase",
                "NativeMultimodalBridge",
                "ShadowExecutionPhase",
            }:
                return 10.0
            return 45.0
        if phase_name in {"UnitaryResponsePhase", "ResponseGenerationPhase"}:
            response_modifiers = getattr(self.state, "response_modifiers", {}) if self.state else {}
            if bool(response_modifiers.get("deep_handoff", False)):
                return 180.0
            return 120.0
        if phase_name == "GodModeToolPhase":
            return 20.0
        if phase_name in {
            "MemoryRetrievalPhase",
            "CognitiveRoutingPhase",
            "ExecutiveClosurePhase",
            "ConversationalDynamicsPhase",
        }:
            return 10.0
        return 8.0

    def _should_skip_priority_phase(self, phase_name: str, *, priority: bool) -> bool:
        """Keep user-facing ticks lean without suppressing explicit tool/task execution."""
        if not priority:
            return False

        background_only = {
            "EternalMemoryPhase",
            "EternalGrowthEngine",
            "TrueEvolutionPhase",
            "NativeMultimodalBridge",
            "ShadowExecutionPhase",
            "PerfectEmotionPhase",
            "PhiConsciousnessPhase",
            "CognitiveIntegrationPhase",
            "InferencePhase",
            "BondingPhase",
            "RepairPhase",
            "MemoryConsolidationPhase",
            "IdentityReflectionPhase",
            "InitiativeGenerationPhase",
            "ConsciousnessPhase",
            "SelfReviewPhase",
            "LearningPhase",
            "LegacyPhase",
            "GodModeToolPhase",
        }
        if phase_name not in background_only:
            return False

        if phase_name == "GodModeToolPhase":
            response_modifiers = getattr(self.state, "response_modifiers", {}) if self.state else {}
            intent_type = str(response_modifiers.get("intent_type", "") or "").upper()
            return intent_type not in {"SKILL", "TASK"}

        return True

    def _spawn_background_task(self, coro: Any, *, name: str) -> asyncio.Task:
        """Create a supervised kernel-owned background task and retain it for shutdown/restart handling."""
        try:
            from core.utils.task_tracker import get_task_tracker

            task = get_task_tracker().create_task(coro, name=name)
        except Exception:
            task = asyncio.create_task(coro, name=name)
            try:
                task._aura_supervised = True
                task._aura_task_tracker = "AuraKernel"
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)
        self._background_tasks.append(task)
        return task

    @property
    def auto_fix_engine(self):
        """Lazy-load the AutonomousSelfModificationEngine."""
        if self._auto_fix_engine is None:
            try:
                # Harden: Use absolute imports and handle missing components gracefully
                from core.container import ServiceContainer
                from core.self_modification.self_modification_engine import (
                    AutonomousSelfModificationEngine,
                )
                
                # Check for brain independently of cog_engine if needed
                cog_engine = ServiceContainer.get("cognitive_engine", default=None)
                if not cog_engine:
                     # Attempt to resolve via type if string lookup fails
                     from core.brain.llm.llm_router import IntelligentLLMRouter
                     cog_engine = ServiceContainer.get(IntelligentLLMRouter, default=None)

                if cog_engine:
                    self._auto_fix_engine = AutonomousSelfModificationEngine(
                        cognitive_engine=cog_engine,
                        code_base_path="."
                    )
                    logger.info("🧬 [ASI] AutonomousSelfModificationEngine initialized.")
                else:
                    logger.debug("⚠️ [ASI] LLM engine not found for SME initialization.")
            except Exception as e:
                logger.error("❌ [ASI] SME initialization failed: %s", e)
        return self._auto_fix_engine

    def set_volition_level(self, level: int):
        """
        Sets the system volition level (0-3).
        Updates the SubstrateGovernor frequency scaling.
        """
        old_level = self.volition_level
        self.volition_level = max(0, min(3, level))
        
        logger.info("🔥 [GENESIS] Volition Level shifted: %d -> %d", old_level, self.volition_level)
        
        # Update governor if available
        gov = ServiceContainer.get("substrate_governor", default=None)
        if gov:
            try:
                # We assume the governor will be updated to handle this
                gov.apply_volition_profile(self.volition_level)
            except Exception as e:
                logger.error("Failed to update SubstrateGovernor with new volition: %s", e)

    async def boot(self) -> None:
        """
        Deterministic, closed-graph boot sequence.
        Fails fast if dependencies are missing.
        """
        logger.info("🛡️ Kernel Boot sequence initiated...")

        try:
            from core.runtime.runtime_hygiene import get_runtime_hygiene

            await get_runtime_hygiene().start()
        except Exception as hygiene_exc:
            logger.debug("Kernel boot runtime hygiene install skipped: %s", hygiene_exc)
        
        # Initialize Lock Watchdog before anything else
        try:
            from core.resilience.lock_watchdog import get_lock_watchdog
            get_lock_watchdog().start()
        except Exception as e:
            logger.error(f"Failed to start LockWatchdog: {e}")

        try:
            # 1. Register Services (Explicit, Typed)
            self._register_all_services()
            
            # 2. Initialize Organs first (Sync population)
            self._initialize_organs()
            
            # 3. Warm up Organs in parallel using a short-lived TaskGroup
            async with asyncio.TaskGroup() as tg:
                for organ in self.organs.values():
                    tg.create_task(self._supervise_organ_load(organ))
            
            # 4. Validate Dependency Graph (AFTER organs are populated)
            await self._validate_dependency_graph()
            
            # 5. Instantiate Phases (Ordering is critical)
            self._setup_phases()
            
            # 6. Start Supervised Background Tasks (Outside startup TaskGroup)
            self._background_tasks = []
            self._spawn_background_task(self._supervise_background_loops(), name="aura.supervisor")
            
            # 7. Initialize/Load State from Vault
            await self._load_initial_state()

            # 8. [RUBICON] Boot Motor Cortex, Pre-Linguistic Engine, Feedback Processor
            await self._boot_rubicon_layers()

            self._running = True
            self.status.running = True

            logger.info("✅ AuraKernel booted — Unitary Organism online.")

            # Record boot in Cognitive Ledger
            try:
                from core.resilience.cognitive_ledger import (
                    Transition,
                    TransitionType,
                    compute_state_hash,
                    get_cognitive_ledger,
                )
                ledger = get_cognitive_ledger()
                ledger.append(Transition.create(
                    ttype=TransitionType.BOOT,
                    subsystem="kernel",
                    cause="boot_complete",
                    payload={
                        "organs": list(self.organs.keys()),
                        "phases": len(self._phases),
                        "volition": self.volition_level,
                    },
                    prior_hash=compute_state_hash(self.state) if self.state else "genesis",
                ))
                ServiceContainer.register_instance("cognitive_ledger", ledger)
            except Exception as _le:
                logger.debug("Ledger boot record failed (non-critical): %s", _le)
            
            # Verify LLM resolution
            try:
                llm_organ = self.organs.get("llm")
                if llm_organ:
                    # We log the class name to confirm if it's IntelligentLLMRouter or MockLLM
                    logger.info("LLM organ instance: %s", llm_organ.instance.__class__.__name__)
            except Exception as e:
                logger.warning(f"Failed to log LLM instance class: {e}")

        except Exception as e:
            logger.critical("🛑 Kernel Boot FATAL ERROR: %s", e)
            raise SystemExit(1) from e

    def _register_all_services(self):
        """
        [HARDENING] Explicit registry only. No string-based lookups.
        """
        logger.debug("Registering core services...")
        self._services[StateRepository] = self.vault
        self._services[AffectUpdatePhase] = self.affect_phase
        self._services[MotivationUpdatePhase] = self.motivation_phase
        self._services[PhiConsciousnessPhase] = self.phi_phase
        self._services[CognitiveRoutingPhase] = self.routing_phase
        self._services[UnitaryResponsePhase] = self.response_phase
        self._services[EternalMemoryPhase] = self.eternal
        self._services[LegacyPhase] = self.legacy_bridge
        self._services[GodModeToolPhase] = self.godmode_tools
        self._services[FeedbackObserver] = self.feedback_observer
        
        # Register LLMRouter if available
        try:
            from core.brain.llm.llm_router import IntelligentLLMRouter as LLMRouter
            # Assuming the router is available via container or created here
            # For the unitary kernel, we want it explicit.
            router = ServiceContainer.get("llm_router", default=None)
            if router:
                self._services[LLMRouter] = router
        except ImportError:
            logger.debug("LLMRouter not found — skipping explicit registration.")

        logger.info("✅ Registered %d core services.", len(self._services))

    async def _validate_dependency_graph(self):
        """
        Refuses to start if the organism is not 'closed'.
        Ensures all required organs are mapped and phases are instantiated.
        """
        logger.info("🛡️ Validating Organism Integrity (Closed-Graph)...")
        
        required_organs = {"llm", "memory", "metabolism", "vision", "voice", "neural", "cookie", "prober", "tricorder", "ice_layer", "omni_tool", "continuity"}
        
        # Fix: Check both presence AND load status (instance not None)
        missing = [o for o in required_organs if o not in self.organs]
        broken = [o for o in required_organs if o in self.organs and self.organs[o].instance is None]
        
        if missing:
            raise RuntimeError(f"CRITICAL: Missing core organs in boot graph: {missing}")
        
        if broken:
            # Harder validation: Refuse boot if LLM is missing and no Mock available
            # (Though OrganStub usually provides a Mock, we verify here)
            for b_organ in broken:
                logger.error(f"🛑 CRITICAL ORGAN FAILURE: {b_organ} instance is None.")
            if "llm" in broken:
                 raise RuntimeError("Kernel cannot start: LLM organ is dysfunctional.")
            
        logger.info("✓ Dependency graph validated.")

    def _initialize_organs(self):
        """
        Populate organ stubs synchronously; loading is handled by caller's TaskGroup.
        """
        self.organs = {
            "llm":       OrganStub("llm", self),
            "vision":     OrganStub("vision", self),
            "memory":     OrganStub("memory", self),
            "voice":      OrganStub("voice", self),
            "metabolism": OrganStub("metabolism", self),
            "neural":     OrganStub("neural", self),
            "cookie":     OrganStub("cookie", self),
            "prober":     OrganStub("prober", self),
            "tricorder":  OrganStub("tricorder", self),
            "ice_layer":  OrganStub("ice_layer", self),
            "omni_tool":  OrganStub("omni_tool", self),
            "continuity": OrganStub("continuity", self)
        }

    async def _supervise_organ_load(self, organ: OrganStub):
        """Supervises the async loading of a hardware organ."""
        try:
            await organ.load()
            try:
                self._gui_queue.put_nowait({"type": "ORGAN_READY", "name": organ.name})
            except asyncio.QueueFull:
                 # Non-blocking failsafe
                 pass
            logger.info("🫀 Organ %s is READY", organ.name)
        except Exception as e:
            logger.error("⚠️ Organ %s failed to load: %s", organ.name, e)

    def _setup_phases(self):
        """
        Defines the immutable phase pipeline.
        Affective Primacy -> Metadata -> Cognition -> Evolution.
        """
        # Phase Pipeline Definition
        # [CONSTITUTIONAL UNIFICATION] This is now the SOLE phase pipeline.
        # MindTick's parallel pipeline has been collapsed into this sequence.
        # Ordering follows the natural cognitive flow:
        #   Soma → Perception → Memory → Affect → Executive → Cognition → Response
        #   → Consolidation → Reflection → Initiative → Consciousness → Review
        self._phases = resolve_phase_instances(self, kernel_phase_attribute_order())


    def get(self, service_type: Any, default: Any = "_K_SENTINEL") -> Any:
        """
        Service retrieval with legacy string support and default value.
        [Lineage] Allows phases to query for required organs or registries.
        """
        # 1. Try Typed lookup in local registry
        svc = self._services.get(service_type)
        if svc is not None:
            return svc
            
        # 2. Try String lookup (fallback for legacy phases)
        if isinstance(service_type, str):
            # Attempt to find by string name or class name in local registry
            for s_type, s_inst in self._services.items():
                if getattr(s_type, "__name__", "") == service_type:
                    return s_inst
            
            # Fallback to ServiceContainer for broader resonance
            res = ServiceContainer.get(service_type, default=default)
            if res != default:
                return res

        # 3. Fallback for structural type matching
        if isinstance(service_type, type):
            for _s_type, s_inst in self._services.items():
                try:
                    if issubclass(s_inst.__class__, service_type):
                        return s_inst
                except TypeError:
                    continue
        
        # 4. Final Fallback: Return default if specified, else raise
        if default != "_K_SENTINEL":
             return default
             
        raise RuntimeError(f"Service {service_type} not registered at boot")

    async def _load_initial_state(self) -> None:
        """Loads state from vault or creates a fresh one if empty."""
        try:
            async def _maybe_await(result):
                if inspect.isawaitable(result):
                    return await result
                return result

            # Ensure DB is ready when the supplied vault supports explicit initialization.
            initialize = getattr(self.vault, "initialize", None)
            if callable(initialize):
                await _maybe_await(initialize())

            get_current = getattr(self.vault, "get_current", None)
            if callable(get_current):
                state = await _maybe_await(get_current())
            else:
                state = getattr(self.vault, "state", None) or getattr(self.vault, "current_state", None)
            if state is None:
                logger.info("🌱 No existing state found. Creating fresh AuraState.")
                from core.state.aura_state import AuraState
                state = AuraState()
                # Warm up the vault with the initial state when the supplied vault
                # exposes a durable commit path. Older/mock vault shims may not.
                commit = getattr(self.vault, "commit", None)
                if callable(commit):
                    await _maybe_await(commit(state, cause="genesis"))
            
            try:
                from core.continuity import get_continuity

                continuity = get_continuity()
                continuity.load()
                state = continuity.apply_to_state(state)
            except Exception as continuity_exc:
                logger.debug("Continuity hydration skipped: %s", continuity_exc)

            self.state = state
            logger.info("🧬 State successfully initialized (version %d)", self.state.version)
        except Exception as e:
            logger.error("❌ Failed to initialize state: %s", e, exc_info=True)
            raise RuntimeError(f"Kernel state initialization failed: {e}") from e

    async def _boot_rubicon_layers(self) -> None:
        """[RUBICON] Boot the Motor Cortex, Pre-Linguistic Engine, and Feedback Processor.

        These three subsystems form the "Crossing the Rubicon" layer:
          - Motor Cortex: 50ms reflex loop, independent of cognitive tick
          - Pre-Linguistic Engine: structured decisions before LLM generation
          - Feedback Processor: structured action feedback -> affect + body schema

        All are fail-safe: if any fails to boot, the system degrades gracefully.
        """
        # 1. Feedback Processor (must be online before motor cortex)
        try:
            from core.somatic.action_feedback import get_feedback_processor
            fp = get_feedback_processor()
            await fp.start()
            logger.info("[RUBICON] FeedbackProcessor ONLINE")
        except Exception as exc:
            logger.warning("[RUBICON] FeedbackProcessor boot failed (degraded): %s", exc)

        # 2. Motor Cortex (independent 50ms reflex loop)
        try:
            from core.somatic.motor_cortex import get_motor_cortex
            mc = get_motor_cortex()
            await mc.start()
            self._spawn_background_task(
                self._motor_cortex_watchdog(mc),
                name="motor_cortex_watchdog",
            )
            logger.info("[RUBICON] MotorCortex ONLINE -- 50ms reflex loop active")
        except Exception as exc:
            logger.warning("[RUBICON] MotorCortex boot failed (degraded): %s", exc)

        # 3. Pre-Linguistic Decision Engine
        try:
            from core.cognition.pre_linguistic import get_pre_linguistic
            pl = get_pre_linguistic()
            await pl.start()
            logger.info("[RUBICON] PreLinguisticEngine ONLINE")
        except Exception as exc:
            logger.warning("[RUBICON] PreLinguisticEngine boot failed (degraded): %s", exc)

    async def _motor_cortex_watchdog(self, mc: Any) -> None:
        """Watchdog that restarts the motor cortex loop if it dies."""
        while self._running:
            await asyncio.sleep(10.0)
            try:
                if mc._running and (mc._task is None or mc._task.done()):
                    logger.warning("[RUBICON] Motor cortex loop died -- restarting")
                    mc._task = asyncio.create_task(mc._run_loop(), name="motor_cortex_loop")
            except Exception as exc:
                logger.debug("[RUBICON] Motor cortex watchdog error: %s", exc)

    async def tick(self, objective: str, priority: bool = False) -> TickEntry | None:
        """
        The Unitary Cognitive Cycle.
        [Lineage] Now using state.derive() for every phase transition.
        Returns a TickEntry containing the causal chain metrics.
        """
        # Use local state for type safety and consistency throughout the tick
        state = self.state
        if state is None:
            raise RuntimeError("Kernel ticked before state initialization")
        turn_origin = self._normalize_origin(getattr(getattr(state, "cognition", None), "current_origin", ""))

        # [PRIORITY PREEMPTION] Signal that a user-facing tick is waiting.
        # Background ticks check this flag between phases and yield early.
        if priority:
            self._user_priority_pending.set()

        # [DEADLOCK PREVENTION] Use robust lock for the tick
        # [STABILITY v50] Reduced from 135→45s. Background ticks now yield
        # within 5s when a priority request is pending, so we don't need
        # a huge lock timeout. 45s covers the worst-case phase-in-progress
        # plus a comfortable margin for 32B inference startup.
        if not await self._lock.acquire_robust(timeout=45.0, max_retries=3):
            logger.error("🛑 CRITICAL: Could not acquire Kernel lock for tick. Possible deadlock. Objective: '%s'", objective)
            if self.status:
                logger.error("Kernel Status: %s, Cycle: %s", self.status.message, self.status.cycle_count)
            return None

        try:
            # Priority request acquired the lock — clear the pending flag
            if priority:
                self._user_priority_pending.clear()

            start_time = time.time()
            logger.info("🌀 Unitary Tick Initiated: '%s' (priority=%s)", objective, priority)
            
            # 1. Feedback Loop: Begin
            entry = self.feedback_observer.begin_tick(
                state,
                objective,
                origin=str(getattr(state.cognition, "current_origin", "") or ""),
                priority=bool(priority),
            )
            
            # Initial derivation for the tick itself
            state = await state.derive_async(f"tick_start: {objective[:50]}", origin="tick")
            # Clear stale skill modifiers from previous ticks so prior skill
            # results (e.g. clock) don't leak into unrelated messages.
            for _stale_key in (
                "last_skill_run", "last_skill_ok", "last_skill_result_payload",
                "matched_skills", "intent_type", "precomputed_grounded_reply",
                "last_task_outcome", "last_task_id", "auto_browse_urls",
            ):
                state.response_modifiers.pop(_stale_key, None)
            self.state = state

            # CASIE: Score user objective for strategy
            tricorder = self.organs.get("tricorder")
            if tricorder and tricorder.instance:
                casie = tricorder.instance.score_user_message(objective)
                logger.info(f"🎭 [CASIE] Strategy: {casie['strategy']} - {casie['description']}")
                
            # [SEVERANCE] Apply Persona Masking to Cognitive Cycle
            partition = state.context_partition
            if state.partition_mask:
                logger.info(f"🎭 [SEVERANCE] Executing in {partition} partition. Field masking ACTIVE.")
                
            state.cognition.current_objective = objective
            get_executive_authority().record_objective_binding(
                state,
                objective,
                source="aura_kernel.tick",
                mode="unitary_tick",
                reason="kernel_tick_bound",
            )

            # Linear Pipeline execution
            volition = self.volition_level

            # Phases that only belong in background autonomous ticks.
            # Running them during a user-facing (priority) tick blocks the response
            # for up to 60s per phase and is never needed for conversation.
            for phase in self._phases:
                phase_name = phase.__class__.__name__

                # [PRIORITY PREEMPTION] If a user message is now waiting for the
                # kernel lock, yield immediately after the current phase completes.
                if not priority and self._user_priority_pending.is_set():
                    logger.info("⚡ Background tick yielding to priority user request — aborting remaining phases after %s.", phase_name)
                    break

                # Skip background-only phases during user-facing ticks so the
                # response pipeline runs without waiting for slow autonomous work.
                if self._should_skip_priority_phase(phase_name, priority=priority):
                    continue

                # Volition-based Gating
                # Level 0: Lockdown (Standard pipeline only)
                # Level 1: Reflective (Adds Self-Review)
                # Level 2: Perceptive (Adds Learning/Repair)
                # Level 3: Agentic (Adds Growth/Evolution)
                if volition < 3 and isinstance(phase, (EternalGrowthEngine, TrueEvolutionPhase)):
                    continue
                if volition < 2 and isinstance(phase, (LearningPhase, RepairPhase, BondingPhase)):
                    continue
                if volition < 1 and isinstance(phase, SelfReviewPhase):
                    continue

                # Strict Lineage: Each phase execution derives a new state version.
                # Use asyncio.shield() so that if the outer task is cancelled the
                # inner phase coroutine is NOT cancelled — preventing CancelledError
                # from reaching MLX workers and triggering unnecessary worker reboots.
                # Each MLX call has its own internal timeout (45 s for background),
                # so phases will complete or fail on their own without kernel-level
                # cancellation.
                try:
                    phase_task = asyncio.create_task(
                        wrap_phase(
                            phase_name,
                            phase.execute,
                            self.state,
                            objective=objective,
                            priority=priority,
                        ),
                        name=f"AuraKernel.{phase_name}",
                    )
                    # Shield the task: TimeoutError aborts our wait but keeps the
                    # task alive so the worker is not disturbed mid-generation.
                    try:
                        phase_timeout = self._phase_timeout_seconds(phase_name, priority=priority)

                        # [STABILITY v50] FAST PREEMPTION: When a priority user
                        # request is pending, cap background phase budgets at 5s
                        # so the user doesn't wait 45s+ for a background tick to
                        # finish. Response phases get a hard 5s cap; other phases
                        # get 8s. This is the #1 fix for kernel lock contention.
                        if not priority and self._user_priority_pending.is_set():
                            if phase_name in {"UnitaryResponsePhase", "ResponseGenerationPhase"}:
                                phase_timeout = min(phase_timeout, 5.0)
                            else:
                                phase_timeout = min(phase_timeout, 8.0)

                        result = await asyncio.wait_for(
                            asyncio.shield(phase_task),
                            timeout=phase_timeout,
                        )
                        self.state = result
                    except TimeoutError:
                        logger.error("⏰ Phase '%s' timed out after %.0fs — skipping", phase_name, phase_timeout)
                        if not priority and phase_name in {"UnitaryResponsePhase", "ResponseGenerationPhase"}:
                            logger.info(
                                "⚡ Background tick ending early after %s timeout so stale response generation does not pin the foreground lane.",
                                phase_name,
                            )
                            break
                        if not priority and self._user_priority_pending.is_set():
                            logger.info(
                                "⚡ Background tick releasing kernel lock after timed-out %s for a waiting priority request.",
                                phase_name,
                            )
                            break
                        # Let the shielded task finish in the background; do not cancel it.
                        continue
                except Exception as phase_err:
                    logger.error("🔥 Phase '%s' raised unexpected error: %s", phase_name, phase_err, exc_info=True)
                    # Don't let a single phase crash the entire tick — skip and continue
                    continue

                if self.state is None:
                     raise RuntimeError(f"Phase {phase_name} returned None state")

                self.state.updated_at = time.time()

            # Flush deferred storage side-effects (eternal_append, db_write, etc.)
            # [STABILITY v53] Timeout guard — storage intents can hang on slow I/O
            try:
                await asyncio.wait_for(self._process_storage_intents(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("⚠️ [STABILITY] Storage intents timed out (10s) — skipping for this tick.")

            # ── CONSTITUTIONAL CLOSURE ──────────────────────────────────────
            # Stamp this tick's arbitration into the canonical state before commit.
            # Every committed state is self-documenting about the decision chain.
            try:
                self.state.cognition.last_kernel_cycle_id = entry.tick_id if entry else None
                self.state.cognition.last_action_source = self.state.cognition.current_origin or "kernel"

                from core.executive.executive_core import get_executive_core
                _exec = get_executive_core()
                if _exec is not None:
                    _exec_stats = _exec.get_stats() if hasattr(_exec, "get_stats") else {}
                    self.state.cognition.kernel_decision_count = int(
                        _exec_stats.get("approved", 0) or 0
                    )
                    self.state.cognition.kernel_veto_count = int(
                        _exec_stats.get("rejected", 0) or 0
                    )
                    _recent = _exec_stats.get("recent_decisions", []) or []
                    self.state.cognition.last_veto_reasons = [
                        str(d.get("reason", ""))
                        for d in _recent
                        if isinstance(d, dict) and d.get("outcome") == "rejected"
                    ][-5:]
            except Exception as _cc_err:
                logger.debug("Constitutional closure stamp skipped: %s", _cc_err)
            # ────────────────────────────────────────────────────────────────

            # Persistence
            # [STABILITY v53] Timeout guard — vault commit can hang on slow disk/network
            try:
                await asyncio.wait_for(self._commit_vault(objective), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("⚠️ [STABILITY] Vault commit timed out (10s) — state not persisted this tick.")

            # Cognitive Ledger: record this tick as a structured transition
            try:
                from core.resilience.cognitive_ledger import (
                    Transition,
                    TransitionType,
                    compute_state_hash,
                    get_cognitive_ledger,
                )
                ledger = get_cognitive_ledger()
                state_hash = compute_state_hash(self.state)
                ledger.append(Transition.create(
                    ttype=TransitionType.TICK_COMPLETE,
                    subsystem="kernel",
                    cause=objective[:120] if objective else "tick",
                    payload={
                        "phi": round(self.state.phi, 4),
                        "valence": round(self.state.affect.valence, 3),
                        "mode": self.state.cognition.current_mode.value,
                        "response_len": len(self.state.cognition.last_response or ""),
                        "cycle": self.status.cycle_count,
                    },
                    prior_hash=state_hash,
                    confidence=1.0 - (self.state.free_energy if hasattr(self.state, "free_energy") else 0.0),
                ))
            except Exception as _ledger_err:
                logger.debug("Ledger tick record failed (non-critical): %s", _ledger_err)

            # Visual Update
            await self._pulse_mirror()
            
            # 2. Feedback Loop: End
            response = self.state.cognition.last_response
            self.feedback_observer.end_tick(entry, response, self.state, start_time)
            
            # Record phase health in StabilityGuardian
            try:
                if self._guardian is None:
                    from core.container import ServiceContainer
                    self._guardian = ServiceContainer.get("stability_guardian", default=None)
                
                if self._guardian:
                    self._guardian.record_tick_health(entry)
            except Exception as e:
                logger.debug(f"StabilityGuardian: Health record skipped: {e}")
            
            # Log the loop summary
            logger.info("LOOP| %s", entry.summary())
            
            # Issue #42: Structured Thought Trace
            try:
                trace_response = response
                trace_outcome = "SUCCESS" if self.state else "FAILURE"
                trace_meta: dict[str, Any] = {}
                modifiers = dict(getattr(self.state, "response_modifiers", {}) or {}) if self.state else {}
                task_outcome = str(modifiers.get("last_task_outcome", "") or "").strip().lower()
                if task_outcome == "started":
                    trace_outcome = "IN_PROGRESS"
                elif task_outcome in {"failed", "capability_gap", "denied"}:
                    trace_outcome = "FAILURE"
                elif task_outcome == "completed":
                    trace_outcome = "SUCCESS"
                elif "last_skill_run" in modifiers:
                    trace_outcome = "SUCCESS" if modifiers.get("last_skill_ok") else "FAILURE"

                has_action_marker = bool(
                    re.search(r"\[(?:SKILL_RESULT|SKILL|ACTION|TOOL|SKILL_INVOCATION)\s*:", str(response or ""), re.IGNORECASE)
                )
                if has_action_marker and not modifiers.get("last_skill_ok") and task_outcome != "completed":
                    trace_outcome = "UNGROUNDED_ACTION"
                    trace_meta["grounding_warning"] = ["marker_without_verified_execution"]

                try:
                    from core.phases.action_grounding import check_unverified_action_claims

                    receipts = []
                    if modifiers.get("last_skill_ok") and modifiers.get("last_skill_run"):
                        receipts.append({"skill": str(modifiers.get("last_skill_run"))})
                    unverified_claims = check_unverified_action_claims(str(response or ""), skill_receipts=receipts)
                    if unverified_claims:
                        trace_outcome = "UNGROUNDED_ACTION"
                        trace_meta["grounding_warning"] = unverified_claims[:4]
                except Exception:
                    pass

                tracer.log_cycle(
                    objective=objective,
                    context=getattr(self.state, "cognition", {}).__dict__ if self.state else {},
                    thought={"last_response": trace_response, **trace_meta},
                    outcome=trace_outcome,
                )
            except Exception as e:
                logger.debug("Tracer failed: %s", e)

            self._finalize_foreground_turn_state(objective=objective, turn_origin=turn_origin)

            # Record completion timestamp for telemetry staleness detection
            self._last_tick_completed_at = time.time()

            return entry
        finally:
            if self._lock.locked():
                self._lock.release()

    def print_loop(self, n: int = 5):
        """Print the last N ticks of the causal chain."""
        self.feedback_observer.print_loop(n)

    def loop_state(self) -> dict:
        """Get the current live state of the feedback loop."""
        return self.feedback_observer.get_current_loop_state()

    async def _dispatch_pending_initiatives(self):
        """
        Retired compatibility hook.
        Pending initiatives are objective proposals now and must be promoted or
        suppressed through ExecutiveAuthority, not consumed as spontaneous speech.
        """
        logger.debug("AuraKernel._dispatch_pending_initiatives is retired; no action taken.")

    async def _commit_vault(self, objective: str):
        """Persist state to vault. Non-fatal on failure — the tick still returns."""
        commit = getattr(self.vault, "commit", None)
        if not callable(commit):
            return
        try:
            await commit(self.state, self.state.transition_cause or f"tick: {objective}")
        except (BrokenPipeError, ConnectionError, OSError) as e:
            logger.warning("Vault commit failed (pipe/connection): %s — state not persisted this tick.", e)
        except Exception as e:
            logger.warning("Vault commit failed: %s — state not persisted this tick.", e)

    async def _process_storage_intents(self):
        """
        [ZENITH] Functional Purity Guard.
        Processes deferred side-effects generated during the phase pipeline.
        Uses thread offload to avoid blocking the event loop.
        """
        cognition = getattr(self.state, "cognition", None)
        intents = list(getattr(cognition, "pending_intents", []) or [])
        
        async def _append_to_file(path: str, payload: dict):
            # Offload blocking write to thread pool
            def _sync_write():
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
            await asyncio.to_thread(_sync_write)

        for intent in intents:
            try:
                t = intent.get("type")
                if t == "db_write":
                    # vASI: Direct Vault commitment for intentional state shifts
                    cause = intent.get("cause", "autonomous_intent")
                    commit = getattr(self.vault, "commit", None)
                    if callable(commit) and self.state:
                        await commit(self.state, cause=cause)
                elif t == "eternal_append":
                    path = intent.get("path")
                    payload = intent.get("payload")
                    if path and payload:
                        await _append_to_file(path, payload)
                        logger.debug("✅ Eternal Vault: Appended state record to %s", path)
            except Exception as e:
                logger.exception("Failed to process storage intent: %s", e)

        # Clear intents after processing
        self.state.cognition.pending_intents = []

    async def _pulse_mirror(self):
        """
        [ZENITH] Atomic Snapshot Swap.
        Eliminates race conditions by creating a deep-copy projection 
        instead of sharing live state objects.
        """
        if not self._running:
            return

        try:
            snapshot = MirrorSnapshot(
                version=getattr(self.state, 'version', 0),
                vitality=getattr(self.state, 'vitality', 1.0),
                mood=getattr(self.state, 'mood', 'neutral'),
                curiosity=getattr(getattr(self.state, 'affect', None), 'curiosity', 0.5),
                phi=getattr(self.state, 'phi', 0.1),
                last_objective=(getattr(getattr(self.state, 'cognition', None), 'current_objective', 'Unknown') or 'Unknown')[:80],
                timestamp=time.time()
            )
            # Atomic swap
            self._mirror_state = snapshot
            # Optional: Keep non-blocking queue for legacy GUI listeners
            try:
                self._gui_queue.put_nowait(snapshot)
            except asyncio.QueueFull:
                try:
                    self._gui_queue.get_nowait()
                except asyncio.QueueEmpty:
                    # nothing to discard
                    logger.debug("Mirror queue empty during purge.")
                try:
                    self._gui_queue.put_nowait(snapshot)
                except asyncio.QueueFull:
                    # if still full, log and drop
                    logger.debug("Mirror queue still full after purge; dropping snapshot")
        except Exception as e:
            logger.error("Mirror projection failed: %s", e)

    async def shutdown(self):
        """Graceful shutdown of all organs and background tasks."""
        if (
            not self._running
            and not self._background_tasks
            and not any(getattr(organ, "instance", None) is not None for organ in self.organs.values())
        ):
            return
        
        logger.info("🛑 [KERNEL] Initiating graceful shutdown...")
        self._running = False
        self.status.running = False
        
        # 1. Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks = []

        # 2. Shutdown organs
        for name, organ in self.organs.items():
            try:
                if hasattr(organ, "shutdown"):
                    await organ.shutdown()
                logger.info("🫀 Organ %s shut down.", name)
            except Exception as e:
                logger.error("Error shutting down organ %s: %s", name, e)

        logger.info("✅ [KERNEL] Shutdown complete.")

    async def hot_reboot(self):
        """
        [ASI Genesis] Recursive Recursive Self-Improvement trigger.
        Re-initializes the phase pipeline and re-loads code without stopping the process.
        """
        logger.info("⚡ [ASI] Initiating Hot Reboot (Bytecode-Aware)...")
        
        # 1. Stop background loops
        for task in self._background_tasks:
            task.cancel()
        
        # 2. Bytecode-aware Module Reloading
        import importlib
        import sys
        
        if not hasattr(self, "_module_mtimes"):
            self._module_mtimes = {}

        # Identify core modules currently in the pipeline
        modules_to_check = set()
        for phase in self._phases:
            mod_name = phase.__class__.__module__
            if mod_name.startswith("core."):
                modules_to_check.add(mod_name)
        
        # Add the kernel's own sub-packages if they changed
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("core.phases") or mod_name.startswith("core.kernel"):
                modules_to_check.add(mod_name)

        reloaded_count = 0
        for mod_name in sorted(list(modules_to_check)):
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, "__file__") and mod.__file__:
                try:
                    mtime = await asyncio.to_thread(lambda path=mod.__file__: Path(path).stat().st_mtime)
                    # If file is newer than our last record, reload it
                    if mtime > self._module_mtimes.get(mod_name, 0):
                        logger.info("♻️ [REBOOT] Reloading modified module: %s", mod_name)
                        importlib.reload(mod)
                        self._module_mtimes[mod_name] = mtime
                        reloaded_count += 1
                except Exception as e:
                    logger.warning("⚠️ [REBOOT] Could not reload %s: %s", mod_name, e)

        # 3. Re-setup phases (In case of code modifications)
        self._setup_phases()
        
        # 4. Re-start background loops
        self._background_tasks = []
        self._spawn_background_task(self._supervise_background_loops(), name="aura.supervisor")
        
        logger.info("✅ [ASI] Hot Reboot complete. %d modules reloaded. New logic is active.", reloaded_count)

    async def _supervise_background_loops(self):
        """
        [CF-7] Actual supervision — detect and restart dead background tasks.
        Also runs the ResourceGovernor periodically for long-term stability.
        """
        _governor = None
        _governor_interval = 60  # seconds
        _last_govern = 0

        while True:
            await asyncio.sleep(5)

            # ── Task supervision ──
            for task in list(self._background_tasks):
                if task.done():
                    exc = None
                    try:
                        exc = task.exception()
                    except (asyncio.CancelledError, Exception) as e:
                        logger.debug(f"Background supervisor: Task exception ignored: {e}")

                    logger.error(
                        "⚠️ Background task '%s' died unexpectedly: %s",
                        task.get_name(), exc
                    )
                    self._background_tasks.remove(task)

                    # Restart critical tasks
                    if task.get_name() == "vault_mutation_consumer":
                        logger.info("🔄 Restarting StateRepository mutation consumer...")
                        self._spawn_background_task(
                            self.vault._mutation_consumer(),
                            name="vault_mutation_consumer"
                        )

            # ── Resource governance (every 60s) ──
            now = time.time()
            if now - _last_govern >= _governor_interval:
                _last_govern = now
                try:
                    if _governor is None:
                        from core.resilience.resource_governor import ResourceGovernor
                        _governor = ResourceGovernor(kernel=self)
                    report = await _governor.govern()
                    mem_status = report.get("memory", {}).get("status", "")
                    if mem_status in ("warning", "emergency"):
                        logger.warning("⚠️ ResourceGovernor: memory pressure detected — %s", report)
                except Exception as e:
                    logger.debug("ResourceGovernor cycle failed (non-critical): %s", e)
    
    async def stop(self):
        """Graceful shutdown of the kernel."""
        import inspect
        stack = inspect.stack()
        stack_str = "\n".join([f"  {s.filename}:{s.lineno} in {s.function}" for s in stack])
        logger.info("🛑 [ASI] Kernel stop requested. Called by:\n%s", stack_str)
        self._running = False
        for task in self._background_tasks:
            task.cancel()

    def _extract_thought(self) -> str | None:
        """Extracts the resulting thought from the final state."""
        return self.state.cognition.last_response if self.state else None
