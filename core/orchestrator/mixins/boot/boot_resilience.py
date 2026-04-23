import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from core.container import ServiceContainer
from core.config import config
from core.state.state_repository import StateRepository
from core.mind_tick import MindTick
from core.utils.concurrency import RobustLock, LOCK_SENTINEL

logger = logging.getLogger(__name__)


class BootResilienceMixin:
    """Provides initialization for state, resilience, threading, and recovery systems."""

    status: Any
    start_time: float
    auto_fix_enabled: bool
    state_repo: Any
    mind_tick: Any
    stats: Dict[str, Any]
    stealth_mode: Any
    conversation_history: list
    peers: Dict[str, Any]
    hooks: Any
    logger: logging.Logger
    swarm: Any
    substrate: Any
    health_monitor: Any
    state_manager: Any
    loop: Optional[asyncio.AbstractEventLoop]
    _lock: Any
    _stop_event: Any
    _history_lock: Any
    _task_lock: Any
    _extension_lock: Any
    _input_lock: Any
    _task_registry: Any
    ast_guard: Any
    watchdog: Any
    lazarus: Any
    episodic_memory: Any
    tool_learner: Any

    def _init_basic_state(
        self, config_path: Optional[Path], auto_fix_enabled: Optional[bool]
    ):
        """Initialize basic status, timing, and configuration."""
        from collections import deque
        from core.utils.hook_manager import HookManager
        from core.orchestrator.orchestrator_types import SystemStatus
        get_stealth_mode = lambda: False  # privacy_stealth removed

        now = time.time()
        self.status = SystemStatus(
            start_time=now,
            initialized=False,
            running=False,
            agency=0.8,
            curiosity=0.5,
            healthy=True,
        )
        self.start_time = now

        if config_path:
            logger.debug("Custom config_path provided but not yet supported")

        if auto_fix_enabled is None:
            auto_fix_enabled = config.security.auto_fix_enabled
        self.auto_fix_enabled = auto_fix_enabled

        # [BOOT FIX] Initialize core state attributes early for ResilientBoot access
        self.state_repo = StateRepository(is_vault_owner=False)
        try:
            # Check if already registered to avoid ContainerError if locked
            if not ServiceContainer.has("state_repository"):
                ServiceContainer.register_instance("state_repository", self.state_repo)
        except Exception as e:
            logger.debug("Skipping state_repository registration in boot mixin: %s", e)
        self.mind_tick = MindTick(self)

        self.stats = {
            "goals_processed": 0,
            "errors_encountered": 0,
            "modifications_made": 0,
            "average_cycle_time": 0.0,
        }

        # Internal State Variables
        self.start_time = time.time()
        self.stealth_mode = get_stealth_mode()
        self.conversation_history = []
        self._thread = None
        self._autonomous_task = None

        # Initialize sync primitives (Phase 16 Sync)
        self._init_threading()
        from core.supervisor.registry import TaskRegistry

        self._task_registry = TaskRegistry()
        self._current_thought_task = None
        self.peers = {}  # Phase 16

        self.hooks = HookManager()
        self.logger = logging.getLogger("Aura.Orchestrator")

        from core.collective.delegator import AgentDelegator

        self.swarm = AgentDelegator(self)

        # Phase XI: Liquid Substrate Continuous Consciousness
        from core.consciousness.liquid_substrate import LiquidSubstrate

        self.substrate = LiquidSubstrate()
        ServiceContainer.register_instance("liquid_substrate", self.substrate)

        from core.managers.health_monitor import HealthMonitor
        from core.resilience.state_manager import StateManager

        self.health_monitor = HealthMonitor(max_consecutive_errors=15)
        self.state_manager = StateManager()

        # Initialize timing attributes
        now_mono = time.monotonic()
        self._last_thought_time = now_mono
        self._last_boredom_impulse = now_mono
        self._last_reflection_impulse = now_mono
        self._last_pulse = now_mono
        self._last_health_check = now_mono
        self._last_volition_poll = 0
        self._last_persona_persist = now_mono  # Phase 26.3
        self._active_metabolic_tasks = set()
        
        self._input_hash_cache = deque(maxlen=20)
        self._input_lock = (
            None  # Initialized in _async_init_threading
        )

    def _init_threading(self):
        """Initialize sync primitives to sentinel for safety. Actual init happens in async start."""
        from core.utils.concurrency import LOCK_SENTINEL

        self.loop = None
        self._lock = LOCK_SENTINEL
        self._stop_event = LOCK_SENTINEL
        self._history_lock = LOCK_SENTINEL
        self._task_lock = LOCK_SENTINEL
        self._extension_lock = LOCK_SENTINEL
        self._input_lock = LOCK_SENTINEL

    def _async_init_threading(self):
        """Initialize asyncio objects within the running event loop."""
        from concurrent.futures import ThreadPoolExecutor
        import os

        # v51: We isolate cognitive I/O from system I/O to prevent starvation.
        # If the default pool fills up with DB reads, Aura's heartbeat will still fire.
        cog_executor = ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) + 4),
            thread_name_prefix="Aura_Cognition"
        )
        try:
            asyncio.get_running_loop().set_default_executor(cog_executor)
        except RuntimeError:
            pass  # No running loop at boot time; executor will be picked up when loop starts

        # Fix #3: Do NOT cache the loop.
        # Resolving via get_running_loop() ensures thread-safety.
        self._lock = RobustLock("Orchestrator.GlobalLock")
        self._stop_event = asyncio.Event()
        self._history_lock = RobustLock("Orchestrator.HistoryLock")
        self._task_lock = RobustLock("Orchestrator.TaskLock")
        self._extension_lock = RobustLock("Orchestrator.ExtensionLock")
        self._input_lock = RobustLock("Orchestrator.InputLock")

    async def _init_resilience(self):
        """Initialize health monitoring and state management."""
        self.status.dependencies_ok = False
        from core.security.ast_guard import ASTGuard

        self.ast_guard = ASTGuard()

        # Sovereign Watchdog
        try:
            from core.resilience.sovereign_watchdog import SovereignWatchdog

            self.watchdog = SovereignWatchdog(self)
            await self.watchdog.start()
            ServiceContainer.register_instance("sovereign_watchdog", self.watchdog)
            logger.info("🛡️  Sovereign Watchdog ACTIVE")
        except Exception as e:
            logger.error("Failed to initialize Sovereign Watchdog: %s", e)

        logger.info(
            "🛡️  Resilience Foundation mapped (Integrations deferred to _integrate_systems)"
        )

    async def _init_meta_optimization(self):
        """Bridge 1: Meta-Optimization Loop."""
        try:
            modifier = getattr(self, "self_modifier", None)
            if modifier and hasattr(modifier, "optimizer"):
                opt = getattr(modifier, "optimizer", None)
                if opt:
                    logger.info(
                        "🚀 Meta-Optimization Loop registered in Self-Modification Engine"
                    )
        except Exception as e:
            logger.error("Meta-Optimization registration failed: %s", e)

    async def _start_state_vault_actor(self):
        """Initializes and starts the StateVaultActor via the Supervision Tree (Phase 3)."""
        try:
            from core.state.vault import vault_process_entry
            from core.supervisor.tree import ActorSpec

            # Check if already started (e.g. by ResilientBoot)
            bus = (
                getattr(self, "_actor_bus", None)
                or getattr(self, "actor_bus", None)
                or ServiceContainer.get("actor_bus", default=None)
            )
            sup = getattr(self, "supervisor", None) or ServiceContainer.get("supervisor", default=None)
            if sup and hasattr(sup, "is_actor_running") and sup.is_actor_running("state_vault"):
                parent_pipe = sup.get_actor_pipe("state_vault")
                if bus and parent_pipe and not getattr(bus, "has_actor", lambda *_: False)("state_vault"):
                    bus.add_actor("state_vault", parent_pipe)
                    logger.info("🛡️  StateVaultActor transport rebound from existing supervisor actor.")
                logger.info("🛡️  StateVaultActor already active. Skipping redundant start.")
                return
            if bus and getattr(bus, "has_actor", lambda *_: False)("state_vault"):
                logger.info("🛡️  StateVaultActor already active. Skipping redundant start.")
                return

            # 1. Register with Supervisor
            spec = ActorSpec(
                name="state_vault",
                target=vault_process_entry,
                args=(
                    str(config.paths.data_dir / "aura_state.db"),
                ),  # Pipe is added by supervisor.start_actor
                restart_policy="permanent",  # State Vault must always be up
            )

            if not sup:
                logger.error(
                    "❌ Cannot start StateVaultActor: Supervisor Tree not available."
                )
                return

            sup.add_actor(spec)

            # 2. Start Actor and get the control pipe
            # supervisor.start_actor returns parent_conn
            parent_pipe = sup.start_actor("state_vault")

            if bus and not getattr(bus, "has_actor", lambda *_: False)("state_vault"):
                bus.add_actor("state_vault", parent_pipe)
                logger.info("🛡️  StateVaultActor transport registered with ActorBus")

            # 3. Wait for readiness using the same request protocol as runtime IPC.
            logger.info("⏳ Waiting for StateVaultActor to be ready...")
            import uuid

            ready = False
            for attempt in range(20):
                try:
                    if bus and getattr(bus, "has_actor", lambda *_: False)("state_vault"):
                        resp = await bus.request(
                            "state_vault",
                            "ping",
                            {"source": "boot_resilience", "attempt": attempt + 1},
                            timeout=2.0,
                        )
                        if isinstance(resp, dict) and resp.get("type") == "pong":
                            ready = True
                            logger.info(
                                "📡 StateVaultActor responded to handshake (Attempt %d)",
                                attempt + 1,
                            )
                            break
                    else:
                        req_id = str(uuid.uuid4())
                        ping_msg = {
                            "type": "ping",
                            "payload": {"source": "boot_resilience", "attempt": attempt + 1},
                            "request_id": req_id,
                            "trace_id": str(uuid.uuid4()),
                            "is_request": True,
                        }
                        parent_pipe.send(json.dumps(ping_msg))

                        polled = await asyncio.to_thread(parent_pipe.poll, 2.0)
                        if polled:
                            resp_raw = await asyncio.to_thread(parent_pipe.recv)
                            resp = json.loads(resp_raw)
                            payload = resp.get("payload") if isinstance(resp, dict) else None
                            if (
                                resp.get("response_to") == req_id
                                and isinstance(payload, dict)
                                and payload.get("type") == "pong"
                            ):
                                ready = True
                                logger.info(
                                    "📡 StateVaultActor responded to fallback handshake (Attempt %d)",
                                    attempt + 1,
                                )
                                break
                    logger.debug("Vault ping attempt %d failed. Retrying...", attempt + 1)
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning("Vault readiness check error: %s", e)
                    await asyncio.sleep(0.5)

            if not ready:
                logger.error(
                    "❌ StateVaultActor failed to respond to ping. Boot proceeding in degraded mode."
                )

            # 4. Register pipe with Actor Bus for Orchestrator communication
            if not bus:
                # Fallback: if bus wasn't found initially but we just started it, try to find it now
                bus = getattr(self, "_actor_bus", None) or getattr(self, "actor_bus", None)
                if bus and not getattr(bus, "has_actor", lambda *_: False)("state_vault"):
                    bus.add_actor("state_vault", parent_pipe)
                    logger.info("🛡️  StateVaultActor transport registered with ActorBus (fallback)")

        except Exception as e:
            logger.error("Failed to start StateVaultActor: %s", e)

    def _initialize_resilience_systems(self):
        """Initialize diagnostics and immune systems."""
        try:
            from core.resilience.diagnostics_agent import DiagnosticsAgent

            diag = DiagnosticsAgent(self)
            ServiceContainer.register_instance("diagnostics", diag)

            # Phase 5: Replaced ImmuneSystem with AutonomicCore
            from core.autonomic.core_monitor import AutonomicCore

            core = AutonomicCore(self)
            ServiceContainer.register_instance("autonomic_core", core)

            logger.info("🛡️  Resilience & Autonomic Core active")

            # Final fallback registration for HUD and dependency resolution
            if not ServiceContainer.has("episodic_memory"):
                # memory is usually a coordinator instance or fachada
                mem_inst = getattr(self, "memory", None)
                ServiceContainer.register_instance("episodic_memory", getattr(mem_inst, "episodic", mem_inst))
                
            if not ServiceContainer.has("liquid_state"):
                # HUD in server.py expects 'liquid_state' — should be the CTRNN substrate
                substrate = ServiceContainer.get("liquid_substrate", default=None) or getattr(self, "substrate", None)
                if substrate:
                    ServiceContainer.register_instance("liquid_state", substrate)
                else:
                    # Last resort fallback
                    affect = getattr(self, "affect", getattr(self, "memory", None))
                    ServiceContainer.register_instance("liquid_state", affect)
                
            logger.info(
                "🛡️ [BOOT] Resilience foundation established. healthy=%s running=%s",
                self.status.healthy,
                self.status.running,
            )
        except Exception as e:
            logger.error("Failed to register resilience systems: %s", e)

    async def _init_lazarus_brainstem(self):
        """Initialize the Lazarus Brainstem (Phase 12 Recovery)."""
        try:
            from core.brain.llm.lazarus_brainstem import LazarusBrainstem

            self.lazarus = LazarusBrainstem(self)
            ServiceContainer.register_instance("lazarus", self.lazarus)
            logger.info(
                "✓ Lazarus Brainstem active (emergency recovery protocols armed)"
            )
        except Exception as e:
            logger.error("Failed to init Lazarus Brainstem: %s", e)
            self.lazarus = None

    def _initialize_self_preservation(self):
        """Integrate self-preservation into the core loop."""
        try:
            from core.safety.self_preservation_safe import integrate_safe_backup

            integrate_safe_backup(self)
            logger.info("🛡️  Self-Preservation Instincts Enabled (Survival Protocol Active)")

            # Connect Unity Embodiment
            try:
                # v26 FIX: High-performance severing of phantom limb
                if getattr(config.security, "unity_enabled", False):
                    embodiment = ServiceContainer.get("embodiment", default=None)

                    if embodiment and hasattr(embodiment, "connect_unity"):
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(embodiment.connect_unity())
                            logger.info("🎨 Unity Embodiment connection initiated")
                        except RuntimeError:
                            logger.debug("No running loop; Unity Embodiment deferred.")
                else:
                    logger.info("🎨 Embodiment: Headless mode active (Unity bridge disabled)")

                # 2. Embodiment System (The 'Body')
                if getattr(self, "embodied", False):
                    logger.info("✓ Embodiment System synchronized.")

                # 2.1 Cognitive Augmentation (The 'Integration')
                if self.cognitive_engine and getattr(self, "consciousness_core", None):
                    try:
                        from core.consciousness.integration import (
                            ConsciousnessAugmentor,
                        )

                        aug_consciousness = ConsciousnessAugmentor(
                            self.consciousness_core
                        )
                        if self.cognitive_engine:
                            self.cognitive_engine.register_augmentor(aug_consciousness)

                        # Add Sovereign Internet Awareness
                        from core.brain.llm.web_augmentor import SovereignWebAugmentor

                        aug_web = SovereignWebAugmentor()
                        if self.cognitive_engine:
                            self.cognitive_engine.register_augmentor(aug_web)

                    except Exception as e:
                        logger.debug("Cognitive augmentor registration failed: %s", e)
            except Exception as ue:
                logger.warning("Unity Embodiment connection deferred: %s", ue)

        except Exception as e:
            logger.error("Failed to integrate self-preservation: %s", e)

        # ── v5.0 New Systems ────────────────────────────────────────────

        # Episodic Memory
        try:
            from core.memory.episodic_memory import get_episodic_memory

            vectors = None
            try:
                from core.container import get_container

                vectors = get_container().get("memory_vector")
            except Exception as _e:
                logger.debug("memory_vector lookup failed (non-critical): %s", _e)
            self.episodic_memory = get_episodic_memory(vector_memory=vectors)
            ServiceContainer.register_instance("episodic_memory", self.episodic_memory)
            logger.info("✓ Episodic Memory initialized and registered (autobiographical recall)")
        except Exception as e:
            logger.error("Failed to init Episodic Memory: %s", e)
            self.episodic_memory = None

        # Tool Learning System
        try:
            from core.memory.learning.tool_learning import tool_learner

            self.tool_learner = tool_learner
            logger.info("✓ Tool Learning System initialized")
        except Exception as e:
            logger.error("Failed to init Tool Learning: %s", e)
            self.tool_learner = None

        # Wire new systems into Self-Model
        try:
            if self.self_model:
                from core.world_model.belief_graph import belief_graph

                self.self_model.attach_subsystems(
                    belief_graph=belief_graph,
                    episodic_memory=getattr(self, "episodic_memory", None),
                    goal_hierarchy=getattr(self, "goal_hierarchy", None),
                    tool_learner=getattr(self, "tool_learner", None),
                )
                logger.info(
                    "✓ Self-Model wired (beliefs, memory, goals, tool learning)"
                )
        except Exception as e:
            logger.warning("Self-Model wiring deferred: %s", e)

    async def _init_system_guardians(self):
        """Initialize MemoryGuard, ResilienceEngine, SovereignPruner, and SystemGovernor."""
        from core.guardians.memory_guard import MemoryGuard

        memory_guard = MemoryGuard()
        await memory_guard.start()
        ServiceContainer.register_instance("memory_guard", memory_guard)

        from core.soma.resilience_engine import ResilienceEngine

        resilience = ResilienceEngine(self)
        await resilience.start()
        ServiceContainer.register_instance("soma", resilience)
        ServiceContainer.register_instance("resilience_engine", resilience)

        from core.memory.sovereign_pruner import SovereignPruner

        pruner = SovereignPruner(self)
        ServiceContainer.register_instance("sovereign_pruner", pruner)

        from core.guardians.governor import SystemGovernor

        system_governor = SystemGovernor()
        await system_governor.start()
        ServiceContainer.register_instance("system_governor", system_governor)

        from core.resilience.stability_guardian import StabilityGuardian
        self.stability_guardian = StabilityGuardian(self)
        await self.stability_guardian.start()
        ServiceContainer.register_instance("stability_guardian", self.stability_guardian)

        logger.info("🛡️  MemoryGuard, SystemGovernor, StabilityGuardian and Resilience Engines active")

    def _initialize_logging(self):
        """Initialize logging system"""
        import logging.config
        from core.utils.sanitizer import PIIFilter

        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": config.logging.format,
                },
            },
            "filters": {"pii_stripper": {"()": PIIFilter}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": config.logging.level,
                    "filters": ["pii_stripper"],
                },
            },
            "loggers": {
                "": {  # Root logger
                    "handlers": ["console"],
                    "level": config.logging.level,
                },
            },
        }

        # Add file handler if enabled
        if config.logging.file_output:
            logging_config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": config.paths.log_file,
                "maxBytes": config.logging.max_file_size_mb * 1024 * 1024,
                "backupCount": config.logging.backup_count,
                "formatter": "default",
                "level": config.logging.level,
                "filters": ["pii_stripper"],
            }
            logging_config["loggers"][""]["handlers"].append("file")

        try:
            logging.config.dictConfig(logging_config)
        except Exception as e:
            # Fallback basic logging if file paths fail
            logging.basicConfig(level=logging.INFO)
            logger.error("Failed to setup complex logging: %s", e)

        # Log startup
        logger.info("=" * 60)
        logger.info("Aura Autonomous Engine Starting")
        logger.info("=" * 60)
        logger.info("Log level: %s", config.logging.level)
        logger.info("Log file: %s", config.paths.log_file)

    def _check_dependencies(self):
        """Verify core package availability via ServiceContainer."""
        logger.info("Verifying Core Dependencies...")
        core_pkgs = ["psutil", "yaml", "pydantic", "fastapi", "uvicorn"]
        missing = [pkg for pkg in core_pkgs if not ServiceContainer.check_package(pkg)]

        if missing:
            logger.warning("⚠️  Missing optional/utility packages: %s", ", ".join(missing))
            self.status.dependencies_ok = False
        else:
            logger.info("✓ Core Dependencies verified.")
            self.status.dependencies_ok = True

    def _calculate_temporal_drift(self):
        """Calculate time elapsed since the last recorded heartbeat."""
        try:
            # H-22 FIX: Use config.paths.home_dir for definitive heartbeat path
            heartbeat_path = config.paths.home_dir / "heartbeat"
            if heartbeat_path.exists():
                last_heartbeat = float(heartbeat_path.read_text())
                drift = time.time() - last_heartbeat
                if drift > 3600:  # If more than 1 hour has passed
                    logger.info(
                        "⏳ TEMPORAL DRIFT: Recovered from %.2f hours of downtime.",
                        drift / 3600,
                    )
                    self.status.temporal_drift_s = drift

                    if hasattr(self, "reply_queue") and self.reply_queue:
                        msg = f"[RECOVERY] Resuming interrupted thought: {drift/3600:.1f} hours. Resuming."
                        try:
                            if hasattr(self, "emit_spontaneous_message"):
                                asyncio.create_task(
                                    self.emit_spontaneous_message(
                                        msg,
                                        modality="chat",
                                        origin="recovery",
                                        urgency=0.7,
                                        metadata={
                                            "visible_presence": True,
                                            "initiative_activity": True,
                                            "trigger": "temporal_drift_recovery",
                                            "voice": False,
                                        },
                                    )
                                )
                            elif getattr(self, "output_gate", None):
                                asyncio.create_task(
                                    self.output_gate.emit(
                                        msg,
                                        origin="recovery",
                                        target="secondary",
                                        metadata={
                                            "autonomous": True,
                                            "spontaneous": True,
                                            "trigger": "temporal_drift_recovery",
                                            "voice": False,
                                        },
                                    )
                                )
                            else:
                                from core.health.degraded_events import record_degraded_event

                                record_degraded_event(
                                    "boot_resilience",
                                    "recovery_message_suppressed_without_output_gate",
                                    detail=msg[:120],
                                    severity="warning",
                                    classification="background_degraded",
                                    context={"drift_hours": round(drift / 3600, 3)},
                                )
                        except Exception as exc:
                            logger.debug("Recovery output routing failed: %s", exc)
        except Exception as e:
            logger.error("Failed to calculate temporal drift: %s", e)

    def _init_global_registration(self):
        """Register orchestrator in the global container (Singleton Aware)."""
        # Use simple ServiceContainer here
        container = ServiceContainer

        # Direct registration to avoid circular resolution check
        logger.debug("Registering orchestrator instance...")
        container.register_instance("orchestrator", self)

        container.register_instance("health_monitor", self.health_monitor)

        if hasattr(self, "swarm"):
            container.register_instance("agent_delegator", self.swarm)
            container.register_instance("swarm", self.swarm)  # Legacy alias

        if getattr(self, "vector_memory", None):
            container.register_instance("vector_memory", self.vector_memory)
            container.register_instance("memory_vector", self.vector_memory)

        # Register telemetry engines
        if hasattr(self, "drive_controller"):
            container.register_instance("drive_engine", self.drive_controller)
        if hasattr(self, "curiosity"):
            container.register_instance("curiosity_engine", self.curiosity)
        # Metadata registration complete
        logger.info("✓ Global registration complete")

        # Capability Mapping
        try:
            from ...capability_map import get_capability_map

            self.capability_map = get_capability_map()
            engine = ServiceContainer.get("capability_engine", default=None)
            self.capability_map.ping_all(engine)
        except Exception:
            self.capability_map = None

    async def _recover_wal_state(self):
        """Recover any interrupted cognitive intents from the WAL."""
        try:
            from core.resilience import cognitive_wal

            pending = cognitive_wal.recover_state()
            if pending:
                logger.info(
                    "💾 WAL: Found %s interrupted thoughts. Resuming...", len(pending)
                )
                for intent in pending:
                    msg = f"[RECOVERY] Resuming interrupted thought: {intent.get('action')} -> {intent.get('target')}"
                    logger.info(msg)
        except Exception as e:
            logger.error("WAL recovery failed: %s", e)

    def _initialize_skills(self):
        """Initialize skill registry via SkillManager."""
        logger.info("Initializing skills...")
        try:
            if not getattr(self, "skill_manager", None):
                logger.warning("Skill Manager not yet available. Skipping discovery.")
                return
            self.skill_manager.discover_skills()
            self.status.skills_loaded = self.skill_manager.skills_loaded
            logger.info(
                "✓ Loaded %s skills and registered with Router",
                self.status.skills_loaded,
            )
        except Exception as e:
            logger.error("Failed to initialize skills: %s", e)

    async def _init_sovereign_scanner(self):
        """Initialize the high-speed autonomous integrity guard."""
        try:
            from core.sovereignty.integrity_guard import IntegrityGuard

            self.integrity_guard = IntegrityGuard(orchestrator=self)
            await self.integrity_guard.start()
            ServiceContainer.register_instance("integrity_guard", self.integrity_guard)
            ServiceContainer.register_instance(
                "sovereign_scanner", self.integrity_guard
            )  # Compatibility
            logger.info("✓ Integrity Guard initialized and running")
        except Exception as e:
            logger.error("Failed to initialize Integrity Guard: %s", e)
            self.integrity_guard = None
