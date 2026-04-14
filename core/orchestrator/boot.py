"""Orchestrator Boot Mixin"""
import asyncio
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

from core.sovereignty.integrity_guard import IntegrityGuard
from core.continuity import get_continuity
from core.resilience.cognitive_wal import cognitive_wal
from contextlib import asynccontextmanager
try:
    from core.master_moral_integration import integrate_complete_moral_and_sensory_systems
except ImportError:
    integrate_complete_moral_and_sensory_systems = None

from .initializers.core_baseline import init_enterprise_layer
from .initializers.hardening import init_hardening_layer
from .initializers.cognitive_sensory import init_cognitive_sensory_layer
from .initializers.pathways import register_core_pathways

from core.config import config, Environment
from core.container import ServiceContainer
from core.mind_tick import MindTick
from core.state.state_repository import StateRepository
try:
    from core.agency.skill_library import SkillLibrary
except ImportError:
    SkillLibrary = Any

# Top-level imports moved to methods for boot performance
from .orchestrator_types import SystemStatus

from .mixins.boot.boot_sensory import BootSensoryMixin
from .mixins.boot.boot_cognitive import BootCognitiveMixin
from .mixins.boot.boot_identity import BootIdentityMixin
from .mixins.boot.boot_resilience import BootResilienceMixin
from .mixins.boot.boot_autonomy import BootAutonomyMixin
from .mixins.boot.boot_background import BootBackgroundMixin

logger = logging.getLogger(__name__)


class OrchestratorBootMixin(
    BootSensoryMixin,
    BootCognitiveMixin,
    BootIdentityMixin,
    BootResilienceMixin,
    BootAutonomyMixin,
    BootBackgroundMixin,
):
    """Mixin handling initialization of subsystems and core architecture."""
    # Type hints for attributes provided by RobustOrchestrator
    status: SystemStatus
    start_time: float
    output_gate: Any
    affect: Any
    memory: Any
    agency: Any
    state_repo: Any
    message_queue: Any
    reply_queue: Any
    conversation_history: list[dict[str, Any]]
    
    _last_thought_time: float
    _last_pulse: float
    _last_health_check: float
    _last_boredom_impulse: float
    _last_reflection_impulse: float
    _last_heartbeat_write: float
    _last_user_interaction_time: float
    _current_thought_task: Optional[asyncio.Task]
    _private_archive: list
    _last_self_initiated_contact: float
    boredom: float
    _active_metabolic_tasks: set[str]
    _stop_event: Any
    _lock: Any
    _history_lock: Any
    _task_lock: Any
    _extension_lock: Any
    stats: Dict[str, Any]

    reasoning_queue: Optional[Any]
    reflex_engine: Optional[Any]
    lazarus: Optional[Any]
    persona_evolver: Optional[Any]
    self_modifier: Optional[Any]
    meta_evolution: Optional[Any]
    epistemic_humility: Optional[Any]
    
    # Core Attributes
    terminal_monitor: Any
    ast_guard: Any
    capability_engine: Any
    _capability_engine: Any
    fictional_engines: Any
    latent_distiller: Any
    meta_learning: Any
    learning_engine: Any
    _learning_engine: Any
    hooks: Any
    self_preservation: Any
    backup_system: Any
    stability_guardian: Any
    research_cycle: Any
    self_model: Any
    personhood: Any
    voice: Any
    
    async def emit_spontaneous_message(self, message: str, modality: str = "chat", origin: str = "system"):
        from .mixins.autonomy import AutonomyMixin

        return await AutonomyMixin.emit_spontaneous_message(
            self,
            message,
            modality=modality,
            origin=origin,
        )
    world_model: Optional[Any]
    skill_library: Optional[Any]
    rsi_lab: Optional[Any]
    concept_bridge: Optional[Any]
    cryptolalia_decoder: Optional[Any]
    ontology_genesis: Optional[Any]
    morphic_forking: Optional[Any]
    motivation: Optional[Any]
    belief_sync: Optional[Any]
    attention: Optional[Any]
    attention_summarizer: Optional[Any]
    probe_manager: Optional[Any]
    cognitive_engine: Optional[Any]
    dream_cycle: Optional[Any]
    meaning_substrate: Optional[Any]
    hallucination_filter: Optional[Any]
    dream_engine: Optional[Any]
    continuous_learner: Optional[Any]
    react_loop: Optional[Any]
    _autonomous_action_times: deque

    def setup(self):
        """Standardized Bootstrap Phase (Synchronous)."""
        # v10.1 HARDENING: Set markers FIRST so we're ready even if partial init happens
        self.status.running = False
        self.status.last_error = None
        self.status.healthy = True

        # Initialize Output Gate
        from core.utils.output_gate import get_output_gate
        self.output_gate = get_output_gate(self)
        
        # Register self as orchestrator for dependency resolution
        ServiceContainer.register_instance("orchestrator", self)

        try:
            from core.consciousness.executive_authority import get_executive_authority

            self.executive_authority = get_executive_authority(self)
        except Exception as exc:
            logger.debug("Executive authority bootstrap skipped: %s", exc)

        try:
            from core.constitution import get_constitutional_core

            self.constitutional_core = get_constitutional_core(self)
        except Exception as exc:
            logger.debug("Constitutional core bootstrap skipped: %s", exc)

        # [PATCH 23] Integrity Guard (File Verification)
        try:
            from core.sovereignty.integrity_guard import IntegrityGuard
            guard = IntegrityGuard(self)
            score = guard.verify_sovereignty()
            if score < 1.0:
                logger.warning("🛡️ [BOOT] Integrity score degraded: %.2f", score)
        except Exception as e:
            logger.error("🛡️ [BOOT] Integrity check failed: %s", e)
        
        # Initialize internal state markers on wall-clock time so the rest of the
        # runtime can compare them consistently with persisted timestamps.
        now = time.time()
        self.start_time = now
        self._last_thought_time = now
        self._last_pulse = now
        self._last_health_check = now
        self._last_boredom_impulse = now
        self._last_reflection_impulse = now
        self._last_heartbeat_write = now
        self._last_user_interaction_time = now
        self.conversation_history = []
        self._current_thought_task = None
        self._autonomous_action_times = deque()
        
        # 🟢 Sovereign State Initialization
        self._private_archive = []
        self._last_self_initiated_contact = 0.0
        
        # Sub-Coordinators (Decomposition Phase)
        from .coordinators.agency import AgencyCoordinator
        from .coordinators.memory import MemoryCoordinator
        from .coordinators.affect import AffectCoordinator
        
        # v5.0.1 FIX: Register Facades early so coordinators can resolve them immediately
        try:
            from core.memory.memory_facade import MemoryFacade
            mem_facade = MemoryFacade(orchestrator=self)
            mem_facade.setup()
            ServiceContainer.register_instance("memory_facade", mem_facade)
            
            from core.agency.agency_facade import AgencyFacade
            agency_facade = AgencyFacade(orchestrator=self)
            # Agency setup is usually async or involves skills, so we just register the instance
            # to satisfy the ServiceContainer. Coordinators will call setup if needed or
            # the async path will handle deep init.
            ServiceContainer.register_instance("agency_facade", agency_facade)
            ServiceContainer.register_instance("agency_core", agency_facade)
            
            from core.affect.affect_facade import AffectFacade
            affect_facade = AffectFacade(orchestrator=self)
            # AffectFacade also likely needs skeletal setup to be usable for status
            ServiceContainer.register_instance("affect_facade", affect_facade)
            
            logger.info("✓ [BOOT] All Core Facades (Memory, Agency, Affect) registered during synchronous setup.")
        except Exception as e:
            logger.warning("⚠️ [BOOT] Early Facade registration encountered issues: %s", e)

        self.agency = AgencyCoordinator(self)
        self.memory = MemoryCoordinator(self)
        self.affect = AffectCoordinator(self)
        self.affect.reset_boredom()

        # Register Coordinators in ServiceContainer for health checks
        ServiceContainer.register_instance("agency_coordinator", self.agency)
        ServiceContainer.register_instance("memory_coordinator", self.memory)
        ServiceContainer.register_instance("affect_coordinator", self.affect)

        logger.info("--- RobustOrchestrator Boot Sequence Complete ---")
        
        # UPSO State Layer (Moved to _init_basic_state for early boot access)
        # self.state_repo = StateRepository()
        # self.mind_tick = MindTick(self)
        
        # v14.1 FIX: Ensure queues exist for processing
        if not hasattr(self, 'message_queue') or self.message_queue is None:
            from core.utils.queues import PriorityBackpressuredQueue
            self.message_queue = PriorityBackpressuredQueue(maxsize=100)
        if not hasattr(self, 'reply_queue') or self.reply_queue is None:
            from core.tagged_reply_queue import TaggedReplyQueue
            self.reply_queue = TaggedReplyQueue(maxsize=50)
        
        # Reset stats safely (preserves keys)
        if hasattr(self, 'stats') and isinstance(self.stats, dict):
            for k in list(self.stats.keys()):
                if isinstance(self.stats[k], (int, float)):
                    self.stats[k] = 0
                elif isinstance(self.stats[k], list):
                    self.stats[k].clear()
        
        # Reset timing markers for immediate test availability
        self._last_heartbeat_write = 0.0
        self._last_user_interaction_time = 0.0
        self._last_thought_time = 0.0
        # H-02 Fix: Remove fake uptime for production monitor accuracy
        self.start_time = time.time()
        self.boredom = 0
        
        logger.info("🛡️ [BOOT] Synchronous bootstrap phase complete.")

    async def _async_init_subsystems(self):
        """Modularized subsystem initialization (Async)."""
        from core.utils.concurrency import RobustLock
        if not hasattr(self, "_boot_lock"):
            self._boot_lock = RobustLock()

        async with self._boot_lock:
            t1_s = time.perf_counter()
            if self.status.initialized:
                logger.debug("🛡️ _async_init_subsystems: Already initialized. skipping.")
                return

            try:
                logger.info("🚀 [BOOT] Starting Async Subsystem Initialization (Modular)...")
                lightweight_test_boot = bool(os.environ.get("PYTEST_CURRENT_TEST")) and not bool(
                    os.environ.get("AURA_FULL_TEST_BOOT")
                )

                # --- Phase 1: Sync & Threading (FIXES SENTINEL RACE) ---
                self.setup()
                if hasattr(self, "_async_init_threading"):
                    self._async_init_threading()

                try:
                    from core.runtime.runtime_hygiene import get_runtime_hygiene

                    self.runtime_hygiene = get_runtime_hygiene()
                    await self.runtime_hygiene.start()
                    ServiceContainer.register_instance("runtime_hygiene", self.runtime_hygiene)
                    logger.info("🧹 Runtime hygiene installed (tasks, threads, processes, memory).")
                except Exception as hygiene_exc:
                    logger.error("⚠️ Runtime hygiene bootstrap failed: %s", hygiene_exc, exc_info=True)
                
                await init_enterprise_layer(self)

                # --- PHASE 2: Resilience Foundation & State ---
                await self._start_state_vault_actor()
                await self._start_meta_evolution()

                if hasattr(self, "state_repo") and self.state_repo:
                    await asyncio.wait_for(self.state_repo.initialize(), timeout=15.0)

                from core.resilience.database_coordinator import get_db_coordinator

                db_coord = get_db_coordinator()
                await asyncio.wait_for(db_coord.start(), timeout=10.0)
                ServiceContainer.register_instance("database_coordinator", db_coord)

                # --- PHASE 3: Cognitive & Sensory (Modular) ---
                # PHASE 3.5 MOVED UP: Inference Gate (Isolated MLX Actor)
                # CRITICAL: InferenceGate MUST be registered BEFORE build_router_from_config
                # so the router can inject the gate as the MLX endpoint client.
                try:
                    from core.brain.inference_gate import InferenceGate
                    self._inference_gate = InferenceGate(self)
                    if lightweight_test_boot:
                        self._inference_gate._initialized = True
                        logger.info("🧪 Lightweight pytest boot: deferring InferenceGate warmup.")
                    else:
                        await self._inference_gate.initialize()
                    ServiceContainer.register_instance("inference_gate", self._inference_gate)
                    logger.info("✅ [BOOT] InferenceGate registered and initialized.")
                except Exception as gate_err:
                    logger.error("⚠️ [BOOT] InferenceGate init failed: %s. Creating cloud-only gate.", gate_err, exc_info=True)
                    # ALWAYS create a gate — cloud fallback is better than Legacy Pipeline
                    from core.brain.inference_gate import InferenceGate
                    self._inference_gate = InferenceGate(self)
                    self._inference_gate._initialized = True  # Skip MLX, use cloud only
                    ServiceContainer.register_instance("inference_gate", self._inference_gate)
                    logger.warning("⚠️ [BOOT] InferenceGate running in CLOUD-ONLY mode.")
                
                # INVARIANT: _inference_gate must NEVER be None after this point
                if not self._inference_gate:
                    logger.critical("🛑 [BOOT] CRITICAL: _inference_gate is STILL None after init! Force-creating.")
                    from core.brain.inference_gate import InferenceGate
                    self._inference_gate = InferenceGate(self)
                    self._inference_gate._initialized = True

                # Now build the LLM router — it will find the InferenceGate in ServiceContainer
                from core.brain.llm_health_router import build_router_from_config

                ServiceContainer.register_instance(
                    "llm_router", build_router_from_config(config)
                )

                await self._init_voice_subsystem()
                await self._init_cognitive_architecture()
                await self._init_language_services()

                def _spawn_boot_task(coro: Any, name: str) -> asyncio.Task:
                    try:
                        from core.utils.task_tracker import get_task_tracker

                        return get_task_tracker().create_task(coro, name=name)
                    except Exception:
                        return asyncio.create_task(coro, name=name)

                # ZENITH LOCKDOWN: Start Deadlock Watchdog
                if hasattr(self, "_deadlock_watchdog") and not lightweight_test_boot:
                    self._deadlock_watchdog_task = _spawn_boot_task(
                        self._deadlock_watchdog(),
                        "orchestrator.deadlock_watchdog",
                    )

                # --- PHASE 4: Identity & Self-Model ---
                from core.self_model import SelfModel

                self.self_model = await SelfModel.load()
                ServiceContainer.register_instance("self_model", self.self_model)
                ServiceContainer.register_instance("identity", self.self_model)

                await self._init_identity_systems()

                # --- PHASE 5: Resilience Guardians ---
                await self._init_system_guardians()
                await self._init_resilience()
                self._initialize_self_preservation()
                
                # --- PHASE 5.5: Unitary Kernel Interface ---
                from core.kernel.kernel_interface import KernelInterface
                await KernelInterface.attach_to_orchestrator(self)

                # --- PHASE 6: Skill System & Mycelium ---
                await self._init_skill_system()

                from core.mycelium import MycelialNetwork

                mycelium = ServiceContainer.get("mycelial_network", default=None)
                if not mycelium:
                    mycelium = MycelialNetwork()
                    ServiceContainer.register_instance("mycelial_network", mycelium)
                mycelium.establish_connection("system", "core_logic", priority=1.0)
                mycelium.establish_connection(
                    "core_logic", "skill_execution", priority=1.0
                )
                mycelium.establish_connection("personality", "cognition", priority=1.0)

                async def mycelium_ui_delivery(msg: str):
                    await self.emit_spontaneous_message(msg, modality="both")

                mycelium.set_ui_callback(mycelium_ui_delivery)

                from .initializers.pathways import register_core_pathways

                register_core_pathways(mycelium)
                mycelium.establish_unification_hyphae()

                if not lightweight_test_boot:
                    _spawn_boot_task(
                        mycelium.pulse_check(),
                        "orchestrator.mycelium.pulse_check",
                    )

                async def _background_mapping():
                    try:
                        from core.config import config

                        await asyncio.to_thread(
                            mycelium.map_infrastructure,
                            str(config.paths.base_dir),
                            scan_dirs=["core", "skills"],
                        )
                        mycelium.establish_consciousness_hyphae()
                    except Exception as e:
                        logger.error("🍄 [MYCELIUM] Mapping failed: %s", e)

                if not lightweight_test_boot:
                    _spawn_boot_task(
                        _background_mapping(),
                        "orchestrator.mycelium.background_mapping",
                    )

                # Phase 5: Supplementary Deep Hardening (Claude Feedback)
                # Ensure cognitive core is ready before marking initialized
                await self._init_cognitive_core()

                await self._init_sovereign_scanner()

                if lightweight_test_boot:
                    self.status.initialized = True
                    logger.info("🧪 Lightweight pytest boot: deferred runtime subsystems skipped.")
                    return
                
                # Trace Mode Integration
                if os.environ.get("AURA_TRACE_MODE") == "1":
                    logger.info("🕵️ TRACE MODE ENABLED: Millisecond-level logging active.")
                
                logger.info("✓ Step 1 Complete (%.3fs)", time.perf_counter() - t1_s)

                # Step 2: Adaptive & Sensory Systems (DEFERRED)
                logger.info("⚡ BOOT: Deferring Step 2 Sensory init...")
                
                # EARLY REGISTRATION of BeliefSync to satisfy Audit Check
                from core.collective.belief_sync import BeliefSync
                self.belief_sync = BeliefSync(self)
                ServiceContainer.register_instance("belief_sync", self.belief_sync)
                
                _spawn_boot_task(self._init_sensory_systems(), "orchestrator.init_sensory_systems")
                _spawn_boot_task(self._init_autonomous_evolution(), "orchestrator.init_autonomous_evolution")
                _spawn_boot_task(self._init_react_loop(), "orchestrator.init_react_loop")
                _spawn_boot_task(self._init_metabolism(), "orchestrator.init_metabolism")
                _spawn_boot_task(self._init_proactive_systems(), "orchestrator.init_proactive_systems")
                _spawn_boot_task(self._init_fictional_synthesis(), "orchestrator.init_fictional_synthesis")
                _spawn_boot_task(self._init_final_foundations(), "orchestrator.init_final_foundations")
                # Phase 32: Lazarus Protocol Heartbeat
                _spawn_boot_task(self._cognitive_heartbeat_task(), "orchestrator.cognitive_heartbeat")
                
                # Step 3 & 4: (DEFERRED)
                async def _final_steps():
                    # Step 3: Consciousness & Logic Integration
                    if hasattr(self, 'substrate') and self.substrate:
                        await self.substrate.start()
                        
                    await self._integrate_systems()
                    
                    # Step 4: State Recovery & Persistence
                    load_fn = getattr(self, "_load_state", lambda: None)
                    wal_fn = getattr(self, "_recover_wal_state", lambda: asyncio.sleep(0))
                    drift_fn = getattr(self, "_calculate_temporal_drift", lambda: None)
                    
                    await asyncio.gather(
                        asyncio.to_thread(load_fn),
                        wal_fn(),
                        asyncio.to_thread(drift_fn)
                    )

                # Start Memory Defragmenter
                try:
                    from core.memory.semantic_defrag import start_defrag_scheduler
                    _spawn_boot_task(start_defrag_scheduler(), "orchestrator.semantic_defrag_scheduler")
                except ImportError as _e:
                    logger.debug('Ignored ImportError in boot.py: %s', _e)
                
                # Start Cognitive Loop Service (Skipped in Skeletal Mode)
                if not config.skeletal_mode:
                    from core.cognitive_loop import CognitiveLoop
                    self.cognitive_loop = CognitiveLoop(self)
                    try:
                        await asyncio.wait_for(self.cognitive_loop.start(), timeout=10.0)
                        logger.info("🧠 Cognitive Loop started.")
                    except asyncio.TimeoutError:
                        logger.error("🛑 Cognitive Loop boot TIMEOUT.")
                    except Exception as e:
                        logger.error("❌ Cognitive Loop failed: %s", e)
                    ServiceContainer.register_instance("cognitive_loop", self.cognitive_loop)
                else:
                    logger.info("💀 Skeletal Mode: Cognitive Loop initialization skipped.")
                    ServiceContainer.register_instance("cognitive_loop", None)

                # Start UPSO MindTick (Phase 2) (Skipped in Skeletal Mode)
                if not config.skeletal_mode:
                    tick = getattr(self, "mind_tick", None)
                    if tick and hasattr(tick, "start"):
                        try:
                            await asyncio.wait_for(tick.start(), timeout=10.0)
                            logger.info("💓 MindTick: Unified cognitive rhythm online.")
                        except asyncio.TimeoutError:
                            logger.error("🛑 MindTick boot TIMEOUT.")
                        except Exception as e:
                            logger.error("❌ MindTick failed: %s", e)
                else:
                    logger.info("💀 Skeletal Mode: MindTick activation skipped.")

                # Start Memory Governor Service (Phase 2)
                from core.collective.swarm_protocol import SwarmProtocol
                from core.resilience.hotfix_engine import HotfixEngine
                from core.resilience.memory_governor import MemoryGovernor
                from core.resilience.metrics_exporter import MetricsExporter
                self.memory_governor = MemoryGovernor(self)
                gov = self.memory_governor
                if gov:
                    try:
                        await asyncio.wait_for(gov.start(), timeout=10.0)
                        logger.info("🛡️ Memory Governor started.")
                    except asyncio.TimeoutError:
                        logger.error("🛑 Memory Governor TIMEOUT.")
                    except Exception as e:
                        logger.error("❌ Memory Governor failed: %s", e)
                    ServiceContainer.register_instance("memory_governor", gov)

                # Start Prometheus Metrics (Phase 3)
                try:
                    self.metrics_exporter = MetricsExporter(port=9090)
                    await self.metrics_exporter.start()
                    ServiceContainer.register_instance("metrics_exporter", self.metrics_exporter)
                except (ImportError, ModuleNotFoundError) as e:
                    logger.warning(f"📈 [BOOT] Metrics Exporter skipped: {e}")
                except Exception as e:
                    logger.error(f"📈 [BOOT] Metrics Exporter failed to start: {e}")
                
                # Phase 4 Advanced Features (Unified Meta-Cognition & Resilience Shards)
                try:
                    from core.orchestrator.meta_cognition_shard import MetaCognitionShard
                    metacog = MetaCognitionShard(self)
                    metacog.start()
                    ServiceContainer.register_instance("meta_cognition_shard", metacog)
                    self.meta_cognition = metacog
                    logger.info("🧠 Meta-Cognition Shard initialized and started.")
                except Exception as e:
                    logger.error(f"🛑 Failed to init Meta-Cognition Shard: {e}")

                try:
                    from core.resilience.healing_swarm import HealingSwarmService
                    healer = HealingSwarmService(self)
                    healer.start()
                    ServiceContainer.register_instance("healing_swarm", healer)
                    self.healing_service = healer
                    logger.info("🛡️ Healing Swarm Service initialized and started.")
                except Exception as e:
                    logger.error(f"🛑 Failed to init Healing Swarm: {e}")
                
                self.hotfix_engine = HotfixEngine(self)
                ServiceContainer.register_instance("hotfix_engine", self.hotfix_engine)
                
                self.swarm = SwarmProtocol()
                ServiceContainer.register_instance("swarm_protocol", self.swarm)
                # Trigger infrastructure mapping (Soul Graph)
                mycelium.setup()
                
                logger.info("🛡️ [ORCHESTRATOR] Subsystems synchronously initialized.")

                # Swarm Protocol start moved to proactive systems (v26.3 Unified)

                _spawn_boot_task(_final_steps(), "orchestrator.final_steps")
                
                # ── Startup Health Check ──────────────────────────────
                # Report status of each critical service so boot issues are visible
                critical_services = [
                    "cognitive_engine", "capability_engine", "mycelial_network",
                    "voice_engine", "database_coordinator", "liquid_substrate",
                ]
                boot_warnings = []
                for svc_name in critical_services:
                    svc = ServiceContainer.get(svc_name, default=None)
                    status_str = "[ OK ]" if svc else "[WARN]"
                    if not svc:
                        boot_warnings.append(svc_name)
                    logger.info("  %s %s", status_str, svc_name)
                
                if boot_warnings:
                    logger.warning("⚠️  BOOT: %d services unavailable: %s", 
                                  len(boot_warnings), ", ".join(boot_warnings))
                else:
                    logger.info("✅ All critical services online")
                
                # ── Startup Validation ────────────────────────────────
                # Always mark initialized BEFORE validation to prevent
                # _boot_lock deadlock if start() is called again. Validation failures
                # set healthy=False for degraded mode instead of returning early.
                self.status.initialized = True
                
                from core.resilience.startup_validator import StartupValidator
                validator = StartupValidator(self)
                is_safe = await validator.validate_all()
                if not is_safe:
                    logger.critical("🚨 STARTUP VALIDATION FAILED. Entering DEGRADED mode (not deadlocking).")
                    self.status.healthy = False
                    # Continue — do NOT return. Deadlocking here blocks the entire system.
                else:
                    logger.info("✅ BOOT COMPLETE: System fully initialized.")

                # --- UPSO Phase 1: Post-Boot State Commit ---
                try:
                    state = await self.state_repo.get_current()
                    if state:
                        await self.state_repo.commit(state.derive("online"), "online")
                        logger.info("💾 UPSO: Online state committed.")
                    else:
                        logger.warning("⚠️ UPSO: No state found to commit online.")
                except Exception as e:
                    logger.error("UPSO: Failed to commit online state: %s", e)
                
                # ── Only auto-start voice capture after validation passes ──────────────
                if self.status.healthy:
                    try:
                        ve = ServiceContainer.get("voice_engine", default=None)
                        should_auto_listen = False
                        if ve:
                            if hasattr(ve, "should_auto_listen"):
                                should_auto_listen = ve.should_auto_listen()
                            else:
                                should_auto_listen = bool(getattr(ve, "auto_listen_enabled", False) and getattr(ve, "microphone_enabled", False))
                        if should_auto_listen:
                            success = await ve.start_listening()
                            if success:
                                logger.info("🎙️ Voice capture auto-started (server-side sounddevice)")
                            else:
                                logger.warning("🎙️ Voice capture failed to auto-start — will retry on demand")
                        else:
                            logger.info("🎙️ Voice capture deferred. Mic will start only after explicit enablement.")
                    except Exception as e:
                        logger.warning("🎙️ Voice auto-start skipped: %s", e)
                
                # Swarm Protocol start moved to proactive systems (v26.3 Unified)
                # ── Immune System Post-Boot Scan ─────────────────
                try:
                    immune = ServiceContainer.get("immune_system", default=None)
                    if immune and hasattr(immune, 'post_boot_scan'):
                        await immune.post_boot_scan(self)
                except Exception as scan_err:
                    logger.warning("Immune post-boot scan failed: %s", scan_err)

                # ── Final Success State ──────────────────────────
                self.status.initialized = True
                if self.status.healthy:
                    logger.info("✅ BOOT COMPLETE: System fully initialized.")
                else:
                    logger.warning("⚠️ BOOT COMPLETE: System initialized in degraded mode.")
                
            except Exception as e:
                logger.error("BOOT ENCOUNTERED ISSUES (Recovering...): %s", e, exc_info=True)
                self.status.add_error(str(e))
                # IMMORTAL BOOT: We still mark as initialized if core components are likely to run
                self.status.initialized = True 
                self.status.healthy = False
                logger.warning("⚠️ BOOT: Entering degraded state. Cycle starting despite errors.")
