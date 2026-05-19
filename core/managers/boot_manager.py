"""BootManager service for Aura - Full Extraction.
Handles the procedural initialization of subsystems, decoupling the boot sequence from the core orchestrator.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from core.capability_engine import CapabilityEngine
from core.config import config
from core.container import ServiceContainer
from core.orchestrator.types import SystemStatus
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

# privacy_stealth module removed from public repo

logger = logging.getLogger("Aura.BootManager")


def _record_boot_manager_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("boot_manager", exc, severity=severity, action=action)


class BootManager:
    """Manages the startup sequence for Aura.
    
    This class replaces OrchestratorBootMixin by providing a standalone service
    for subsystem orchestration and wiring.
    """

    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.logger = logger

    def _record_degradation(
        self,
        exc: BaseException,
        *,
        component: str,
        action: str,
        severity: str = "warning",
        mark_status_error: bool = False,
    ) -> None:
        _record_boot_manager_degradation(exc, action=action, severity=severity)
        status = getattr(self.orchestrator, "status", None)
        if status is not None:
            metrics = getattr(status, "health_metrics", None)
            if isinstance(metrics, dict):
                degraded = metrics.setdefault("boot_degraded_components", [])
                if isinstance(degraded, list) and len(degraded) < 32:
                    degraded.append(
                        {
                            "component": component,
                            "severity": severity,
                            "action": action,
                            "error": type(exc).__name__,
                        }
                    )
            if mark_status_error and hasattr(status, "add_error"):
                status.add_error(f"{component}: {exc}")

    async def initialize(self, config_path: Path | None = None, auto_fix_enabled: bool | None = None) -> bool:
        """Executes the full boot sequence."""
        # 1. Sync Init (Basic State)
        self._init_basic_state(config_path, auto_fix_enabled)
        self._init_queues()
        self._init_threading()
        
        # 2. Async Init (Subsystems)
        return await self._async_init_subsystems()

    def _init_basic_state(self, config_path: Path | None, auto_fix_enabled: bool | None):
        self.orchestrator.status = SystemStatus()
        self.orchestrator.start_time = time.time()
        self.orchestrator.status.start_time = self.orchestrator.start_time
        
        if auto_fix_enabled is None:
            auto_fix_enabled = config.security.auto_fix_enabled
        self.orchestrator.auto_fix_enabled = auto_fix_enabled
        
        self.orchestrator.stats = {
            "goals_processed": 0,
            "errors_encountered": 0,
            "modifications_made": 0,
            "average_cycle_time": 0.0,
        }
        
        self.orchestrator._extensions_initialized = False
        self.orchestrator.boredom = 0
        self.orchestrator.stealth_mode = False
        self.orchestrator.conversation_history = []
        self.orchestrator._thread = None
        self.orchestrator._autonomous_task = None
        self.orchestrator._current_thought_task = None
        self.orchestrator.peers = {} 
        
        from core.utils.hook_manager import HookManager
        self.orchestrator.hooks = HookManager()
        
        from core.collective.delegator import AgentDelegator
        self.orchestrator.swarm = AgentDelegator(self.orchestrator)
        
        from core.managers.health_monitor import HealthMonitor
        from core.resilience.state_manager import StateManager
        self.orchestrator.health_monitor = HealthMonitor(max_consecutive_errors=15)
        self.orchestrator.state_manager = StateManager()
        
        # Timing
        now = time.time()
        self.orchestrator._last_thought_time = now
        self.orchestrator._last_boredom_impulse = now
        self.orchestrator._last_reflection_impulse = now
        self.orchestrator._last_pulse = now
        self.orchestrator._last_health_check = now
        self.orchestrator._last_volition_poll = 0
        self.orchestrator._active_metabolic_tasks = set()

    def _init_queues(self):
        self.orchestrator.message_queue = asyncio.Queue(maxsize=100)
        self.orchestrator.reply_queue = asyncio.Queue(maxsize=100)
        
        from core.managers.drive_controller import DriveController
        self.orchestrator.drive_controller = DriveController(self.orchestrator)

    def _init_threading(self):
        self.orchestrator.loop = None
        self.orchestrator._lock = None
        self.orchestrator._stop_event = None
        self.orchestrator._history_lock = None
        self.orchestrator._task_lock = None
        self.orchestrator._extension_lock = None

    async def _async_init_subsystems(self):
        if self.orchestrator.status.initialized:
            return True
            
        self.logger.info("🧠 BOOT: Starting Subsystem Initialization...")
        try:
            self._async_init_threading()
            self._init_resilience()
            self._init_skill_system()
            self._init_cognitive_core()
            self._init_sensory_systems()
            self._init_autonomous_evolution()
            self._init_metabolism()
            self._init_strategic_planning()
            
            await self._integrate_systems()
            self.orchestrator._load_state()
            await self._recover_wal_state()
            self._calculate_temporal_drift()

            from core.memory.semantic_defrag import start_defrag_scheduler
            get_task_tracker().create_task(start_defrag_scheduler())
            
            self.logger.info("✅ BOOT COMPLETE: Orchestrator architecture online")
            self.orchestrator.status.initialized = True
            degraded = self.orchestrator.status.health_metrics.get("boot_degraded_components", [])
            self.orchestrator.status.dependencies_ok = not bool(degraded)
            self.orchestrator.status.health_metrics["boot_completed_at"] = time.time()
            return True
        except (ImportError, AttributeError, RuntimeError) as e:
            self._record_degradation(
                e,
                component="boot_sequence",
                action="failed boot closed and left orchestrator uninitialized",
                severity="critical",
                mark_status_error=True,
            )
            self.logger.error("BOOT FAILED: %s", e, exc_info=True)
            self.orchestrator.status.initialized = False
            self.orchestrator.status.dependencies_ok = False
            return False

    def _async_init_threading(self):
        try:
            self.orchestrator.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.orchestrator.loop = asyncio.get_running_loop()
        self.orchestrator._lock = asyncio.Lock()
        self.orchestrator._stop_event = asyncio.Event()
        self.orchestrator._history_lock = asyncio.Lock()
        self.orchestrator._task_lock = asyncio.Lock()
        self.orchestrator._extension_lock = asyncio.Lock()

    def _init_resilience(self):
        from core.security.ast_guard import ASTGuard
        self.orchestrator.ast_guard = ASTGuard()
        try:
            from infrastructure.watchdog import get_watchdog
            watchdog = get_watchdog()
            watchdog.register_component("orchestrator", timeout=60.0)
            watchdog.start()
            self.orchestrator._watchdog = watchdog
        except (ImportError, AttributeError, RuntimeError) as e:
            self._record_degradation(
                e,
                component="system_watchdog",
                action="continued boot without system watchdog registration",
                severity="degraded",
            )
            self.logger.warning("Failed to initialize System Watchdog: %s", e)

    def _init_skill_system(self):
        engine = CapabilityEngine(orchestrator=self.orchestrator)
        ServiceContainer.register_instance("capability_engine", engine)
        ServiceContainer.register_instance("skill_manager", engine)
        self.orchestrator.status.skills_loaded = len(engine.skills)
        self.logger.info("✓ Capability Engine initialized")

    def _init_cognitive_core(self):
        ce = self.orchestrator.cognitive_engine
        if ce and hasattr(ce, 'wire'):
            engine = ServiceContainer.get("capability_engine", default=None)
            ce.wire(engine, engine)

    def _init_sensory_systems(self):
        try:
            from core.senses.ears import SovereignEars
            from core.senses.screen_vision import LocalVision
            ears = SovereignEars()
            ServiceContainer.register_instance("ears", ears)
            vision = LocalVision()
            ServiceContainer.register_instance("vision", vision)
            
            from core.terminal_monitor import get_terminal_monitor
            self.orchestrator.terminal_monitor = get_terminal_monitor()
            
            from core.brain.reasoning_queue import get_reasoning_queue
            self.orchestrator.reasoning_queue = get_reasoning_queue()
        except (ImportError, AttributeError, RuntimeError) as e:
            self._record_degradation(
                e,
                component="sensory_systems",
                action="continued boot without local ears/vision/reasoning queue wiring",
                severity="degraded",
            )
            self.logger.error("Senses init failed: %s", e)

    def _init_autonomous_evolution(self):
        try:
            from core.self_modification.self_modification_engine import (
                AutonomousSelfModificationEngine,
            )
            self.orchestrator.self_modifier = AutonomousSelfModificationEngine(
                self.orchestrator.cognitive_engine,
                code_base_path=str(config.paths.base_dir),
                auto_fix_enabled=self.orchestrator.auto_fix_enabled
            )
            if config.security.auto_fix_enabled:
                self.orchestrator.self_modifier.start_monitoring()
        except (ImportError, AttributeError, RuntimeError) as e:
            self._record_degradation(
                e,
                component="self_modification",
                action="continued boot with autonomous self-modification disabled",
                severity="degraded",
            )
            self.logger.error("Self-mod failed: %s", e)

    def _init_metabolism(self):
        try:
            from core.ops.metabolic_monitor import MetabolicMonitor
            monitor = MetabolicMonitor(ram_threshold_mb=3072, cpu_threshold=85.0)
            monitor.start()
            ServiceContainer.register_instance("metabolic_monitor", monitor)
        except (ImportError, AttributeError, RuntimeError) as e:
            self._record_degradation(
                e,
                component="metabolic_monitor",
                action="continued boot without metabolic monitor process guard",
                severity="degraded",
            )
            self.logger.error("Metabolism init failed: %s", e)

    def _init_strategic_planning(self):
        try:
            from core.data.project_store import ProjectStore
            from core.neural_feed import NeuralFeed
            from core.strategic_planner import StrategicPlanner
            feed = NeuralFeed()
            ServiceContainer.register_instance("neural_feed", feed)
            store = ProjectStore(str(config.paths.data_dir / "projects.db"))
            planner = StrategicPlanner(self.orchestrator.cognitive_engine, store)
            ServiceContainer.register_instance("strategic_planner", planner)
        except (ImportError, AttributeError, RuntimeError) as e:
            self._record_degradation(
                e,
                component="strategic_planning",
                action="continued boot without strategic planner registration",
                severity="degraded",
            )
            self.logger.error("Strategy init failed: %s", e)

    async def _integrate_systems(self):
        # Implementation of integration methods from mixin
        try:
            from core.master_moral_integration import integrate_complete_moral_and_sensory_systems
            integrate_complete_moral_and_sensory_systems(self.orchestrator)
        except (ImportError, AttributeError, RuntimeError) as e:
            self._record_degradation(
                e,
                component="moral_sensory_integration",
                action="continued boot without complete moral/sensory integration layer",
                severity="degraded",
            )
            self.logger.error("Moral integration failed: %s", e)

        # Autonomic Core
        try:
            from core.autonomic.core_monitor import AutonomicCore
            core = AutonomicCore(self.orchestrator)
            ServiceContainer.register_instance("autonomic_core", core)
        except (ImportError, AttributeError, RuntimeError) as e:
            self._record_degradation(
                e,
                component="autonomic_core",
                action="continued boot without autonomic core registration",
                severity="degraded",
            )
            self.logger.error("Autonomic Core failed: %s", e)
            
        # Advanced Cognition Layer
        try:
            from core.cognitive_integration_layer import CognitiveIntegrationLayer
            cognition = CognitiveIntegrationLayer(orchestrator=self.orchestrator, base_data_dir=str(config.paths.data_dir))
            await cognition.initialize()
            ServiceContainer.register_instance("cognitive_integration", cognition)
            self.orchestrator.cognition = cognition
        except (ImportError, AttributeError, RuntimeError) as e:
            self._record_degradation(
                e,
                component="advanced_cognition",
                action="continued boot without advanced cognitive integration layer",
                severity="degraded",
            )
            self.logger.error("Advanced cognition failed: %s", e)

    async def _recover_wal_state(self):
        from core.resilience.cognitive_wal import cognitive_wal
        try:
            pending = cognitive_wal.recover_state()
            if pending:
                self.logger.info("💾 WAL: Found %d interrupted thoughts", len(pending))
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            self._record_degradation(
                e,
                component="cognitive_wal",
                action="continued boot without replaying interrupted cognitive WAL state",
                severity="degraded",
            )
            self.logger.error("WAL recovery failed: %s", e)

    def _calculate_temporal_drift(self):
        try:
            heartbeat_path = Path("data/heartbeat.txt")
            if heartbeat_path.exists():
                last_heartbeat = float(heartbeat_path.read_text())
                drift = time.time() - last_heartbeat
                self.orchestrator.status.temporal_drift_s = drift
        except OSError as _exc:
            self._record_degradation(
                _exc,
                component="temporal_drift",
                action="continued boot without persisted heartbeat drift measurement",
                severity="warning",
            )
            logger.debug("Exception caught during execution", exc_info=True)
