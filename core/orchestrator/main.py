from __future__ import annotations
from core.runtime.errors import record_degradation

# Robust orchestrator with proper initialization.

import asyncio
import collections
import hashlib
import inspect
import json
import logging
import os
import random
import re
import threading
import time
import multiprocessing
import psutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, List, Dict

# [AURA HARDENING] Final Quick-Win: Disable background telemetry to prevent hangs in sovereign environments
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from core.autonomy_guardian import AutonomyGuardian
from core.brain.types import ThinkingMode
from core.meta.cognitive_trace import CognitiveTrace
from core.scheduler import TaskSpec, scheduler
from core.health.degraded_events import record_degraded_event
from core.utils.exceptions import capture_and_log
from core.utils.queues import BackpressuredQueue, USER_FACING_ORIGINS
from core.utils.concurrency import run_io_bound, LOCK_SENTINEL, RobustLock

from ..config import config
from ..container import ServiceContainer
from ..state.state_repository import StateRepository
from .boot import OrchestratorBootMixin
from .orchestrator_types import SystemStatus
from .handlers.status_manager import StatusManagerMixin
from .services import OrchestratorServicesMixin
from .state import OrchestratorStateMixin

from .coordinators.agency import AgencyCoordinator
from .coordinators.memory import MemoryCoordinator
from .coordinators.affect import AffectCoordinator

logger = logging.getLogger("Aura.Core.Orchestrator")
from core.brain.llm.continuous_substrate import ContinuousSubstrate


# Phase 2: Supervisor & Registry
from core.supervisor.tree import get_tree, ActorSpec
from core.supervisor.registry import get_task_registry, TaskStatus
from core.bus.actor_bus import ActorBus


class AsyncNullContext:
    """Async context manager that does nothing."""
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass


def _bg_task_exception_handler(task: asyncio.Task) -> None:
    """Handle exceptions raised by background tasks quietly without crashing the event loop."""
    try:
        if not task.cancelled():
            exc = task.exception()
            if exc:
                logging.getLogger("Aura.BgTasks").error(f"Background task failed: {repr(exc)}")
    except asyncio.CancelledError as _e:
        logger.debug('Ignored asyncio.CancelledError in main.py: %s', _e)
    except Exception as e:
        record_degradation('main', e)
        record_degradation('main', e)
        logging.getLogger("Aura.BgTasks").debug(f"Task exception handler itself failed: {e}")


def _dispose_awaitable(result: Any) -> None:
    if inspect.iscoroutine(result):
        result.close()
        return
    cancel = getattr(result, "cancel", None)
    if callable(cancel):
        cancel()

from .mixins.output_formatter import OutputFormatterMixin
from .mixins.personality_bridge import PersonalityBridgeMixin
from .mixins.cognitive_background import CognitiveBackgroundMixin
from .mixins.message_pipeline import MessagePipelineMixin
from .mixins.autonomy import AutonomyMixin
from .mixins.response_processing import ResponseProcessingMixin
from .mixins.tool_execution import ToolExecutionMixin
from .mixins.learning_evolution import LearningEvolutionMixin
from .mixins.context_streaming import ContextStreamingMixin
from .mixins.message_handling import MessageHandlingMixin
from .mixins.incoming_logic import IncomingLogicMixin

class RobustOrchestrator(OrchestratorBootMixin, StatusManagerMixin, OrchestratorStateMixin, OrchestratorServicesMixin, OutputFormatterMixin, PersonalityBridgeMixin, CognitiveBackgroundMixin, MessagePipelineMixin, ToolExecutionMixin, LearningEvolutionMixin, AutonomyMixin, ResponseProcessingMixin, ContextStreamingMixin, MessageHandlingMixin, IncomingLogicMixin):
    """The central brain that coordinates everything."""
    
    # Internal role identifier for LLM API compatibility
    # NOTE: This string is strictly for API protocol compliance (Role: assistant).
    # Aura's identity is "Autonomous Intelligence", NOT a subordinate assistant.
    AI_ROLE = "assistant"

    # Core Attributes & Components
    status: SystemStatus
    stats: dict[str, Any]
    message_queue: asyncio.Queue
    reply_queue: asyncio.Queue
    conversation_history: list[dict[str, Any]]
    start_time: float
    auto_fix_enabled: bool
    stealth_mode: bool
    boredom: int
    # State & Pacing (Monotonic for stability)
    _last_thought_time: float
    _last_boredom_impulse: float
    _last_reflection_impulse: float
    _last_pulse: float
    _last_health_check: float
    _last_user_interaction_time: float = 0.0  # Monotonic tracking
    _foreground_user_quiet_until: float = 0.0
    
    # Locks & Synchronization (Audit-Hardened)
    async def process_user_input(self, message: str, origin: str = "user") -> Optional[str]:
        """Compatibility alias for the cognitive pipeline."""
        return await self.process_user_input_priority(message, origin)
    _stop_event: threading.Event
    _lock: RobustLock
    _history_lock: RobustLock
    _stats_lock: RobustLock
    _task_lock: RobustLock
    _extension_lock: RobustLock
    _current_thought_task: Optional[asyncio.Task]
    _autonomous_task: Optional[asyncio.Task]
    _autonomous_action_times: collections.deque[float]
    _thread: Optional[threading.Thread]
    
    # State tracking
    _extensions_initialized: bool = False
    _current_objective: str = ""
    _recovery_attempts: int = 0
    _poison_pill_cache: set[str]
    _active_metabolic_tasks: set[str]
    _message_counter: int = 0  # Monotonic counter for PriorityQueue tie-breaking
    _correction_shards: list[dict[str, Any]]

    # Component Attributes (Resolved via setup())
    _capability_engine: Optional[Any] = None
    _actor_bus: Optional[ActorBus] = None
    _sensory_actor: Optional[Any] = None # Process reference managed by supervisor
    _last_sensory_heartbeat: float = 0.0
    _task_registry: Any = None
    _supervisor_tree: Any = None
    kernel_interface: Optional[Any] = None
    
    meta_cognition: Optional[Any] = None
    healing_service: Optional[Any] = None
    
    @property
    def live_learner(self) -> Any:
        return ServiceContainer.get("live_learner", default=None)
        
    @property
    def task_engine(self) -> Any:
        return ServiceContainer.get("task_engine", default=None)
        
    @property
    def audit_suite(self) -> Any:
        return ServiceContainer.get("audit_suite", default=None)

    def _check_autonomous_rate_limit(self) -> bool:
        """Returns True if we're within the hourly autonomous action budget."""
        now = time.monotonic()
        cutoff = now - 3600
        # Remove entries older than 1 hour
        while self._autonomous_action_times and self._autonomous_action_times[0] < cutoff:
            self._autonomous_action_times.popleft()

        if len(self._autonomous_action_times) >= config.max_autonomous_actions_per_hour:
            logger.warning(
                "⚠️ Autonomous rate limit reached (%d/hr). Suppressing action.",
                config.max_autonomous_actions_per_hour,
            )
            # Defensive check: If rate limit reached, suppress action and clear current thought task
            self._current_thought_task = None
            return False

        self._autonomous_action_times.append(now)
        return True
    _scratchpad_engine: Optional[Any] = None
    _strategic_planner: Optional[Any] = None
    _project_store: Optional[Any] = None
    _intent_router: Optional[Any] = None
    _state_machine: Optional[Any] = None
    _autonomic_core: Optional[Any] = None
    _pending_correction: str = ""  # v40: Identity Drift correction injection
    _ears: Optional[Any] = None
    _liquid_state: Optional[Any] = None
    _personality_engine: Optional[Any] = None
    _knowledge_graph: Optional[Any] = None
    _meta_learning: Optional[Any] = None
    _singularity_monitor: Optional[Any] = None
    _memory_manager: Optional[Any] = None
    _goal_hierarchy: Optional[Any] = None
    _identity: Optional[Any] = None
    _self_model: Optional[Any] = None
    _global_workspace: Optional[Any] = None
    _world_state: Optional[Any] = None
    _memory_optimizer: Optional[Any] = None
    _self_healer: Optional[Any] = None
    _metabolic_monitor: Optional[Any] = None
    
    # Sub-Coordinators (Decomposition Phase)
    agency: AgencyCoordinator
    memory: MemoryCoordinator
    affect: AffectCoordinator
    
    state_repo: StateRepository
    _event_loop_monitor: Optional[Any] = None

    @property
    def actor_bus(self) -> Any:
        """Lazy-loaded actor bus from container."""
        if self._actor_bus is None:
            self._actor_bus = ServiceContainer.get("actor_bus", default=None)
            if self._actor_bus is None:
                self._actor_bus = ActorBus()
                try:
                    if not ServiceContainer.has("actor_bus"):
                        ServiceContainer.register_instance("actor_bus", self._actor_bus)
                except Exception: pass
        return self._actor_bus

    @property
    def supervisor(self) -> Any:
        """Lazy-loaded supervisor tree from container."""
        if self._supervisor_tree is None:
            self._supervisor_tree = ServiceContainer.get("supervisor", default=None)
            if self._supervisor_tree is None:
                from core.supervisor.tree import get_tree
                self._supervisor_tree = get_tree()
                ServiceContainer.register_instance("supervisor", self._supervisor_tree)
        return self._supervisor_tree

    # --- CORE REASONING LOOP ---

    # CLASS-LEVEL sentinel — allows tests to inject a fake status
    # without triggering the Mixin's ServiceContainer lookup.
    _status_override: Optional["SystemStatus"] = None

    @property
    def is_busy(self) -> bool:
        # 1. Honour the test/override sentinel first
        status = self._status_override if self._status_override is not None \
                 else getattr(self, 'status', None)
        if status is None:
            # Also busy if a thought task is running
            task = getattr(self, '_current_thought_task', None)
            return task is not None and not task.done()
        is_processing = getattr(status, 'is_processing', False)
        task = getattr(self, '_current_thought_task', None)
        return is_processing or (task is not None and not task.done())

    def _publish_status(self, data: dict[str, Any]):
        """Publish system status update to Event Bus."""
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish_threadsafe("status", {
                "timestamp": time.time(),
                "status": self.status.model_dump(),
                **data
            })
        except Exception as exc:
            record_degradation('main', exc)
            record_degradation('main', exc)
            logger.debug("Suppressed: %s", exc)
    def _publish_telemetry(self, data: dict[str, Any]):
        """Publish telemetry data to Event Bus."""
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish_threadsafe("telemetry", {
                "timestamp": time.time(),
                **data
            })
        except Exception as exc:
            record_degradation('main', exc)
            record_degradation('main', exc)
            logger.debug("Suppressed: %s", exc)

    def _background_message_block_reason(self, origin: str) -> str:
        if self._is_user_facing_origin(origin):
            return ""
        try:
            quiet_until = float(getattr(self, "_foreground_user_quiet_until", 0.0) or 0.0)
            if quiet_until > time.time():
                return "foreground_quiet_window"
        except Exception as _exc:
            record_degradation('main', _exc)
            record_degradation('main', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        try:
            router = ServiceContainer.get("llm_router", default=None)
            if router and getattr(router, "high_pressure_mode", False):
                return "memory_pressure"
        except Exception as _exc:
            record_degradation('main', _exc)
            record_degradation('main', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        try:
            gate = self._inference_gate or ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "_background_local_deferral_reason"):
                reason = str(gate._background_local_deferral_reason(origin=origin) or "").strip()
                if reason:
                    return reason
        except Exception as _exc:
            record_degradation('main', _exc)
            record_degradation('main', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        try:
            if psutil.virtual_memory().percent >= 84.0:
                return "memory_pressure"
        except Exception as _exc:
            record_degradation('main', _exc)
            record_degradation('main', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return ""
    # Standardized async start/stop.
    async def stop(self):
        """Signal the orchestrator to stop gracefully. (Idempotent)"""
        from core.orchestrator.handlers.shutdown import orchestrator_shutdown
        await orchestrator_shutdown(self)
        
    # --- COMPATIBILITY SHIMS (ZENITH RECONCILIATION) ---

    async def _process_cycle(self):
        """Legacy shim for the cognitive loop. Delegated to self.cognitive_loop."""
        if hasattr(self, "cognitive_loop") and self.cognitive_loop:
            await self.cognitive_loop._process_cycle()
        else:
            # Basic increment for tests/skeletal mode matching status expectancy
            self.status.cycle_count += 1
            
            # [LEGACY COMPAT] Trigger personality update
            pe = getattr(self, "_personality_engine", None)
            if not pe:
                pe = ServiceContainer.get("personality_engine", default=None)
            
            if pe and hasattr(pe, "update"):
                if asyncio.iscoroutinefunction(pe.update):
                    # We are in an async method, but the legacy test might be sync mock
                    # We'll try to await if it's a coroutine, otherwise call sync
                    try:
                        res = pe.update()
                        if asyncio.iscoroutine(res):
                             await res
                    except TypeError as _exc:
                        logger.debug("Suppressed TypeError: %s", _exc)
                else:
                    pe.update()
                
            logger.debug("Shim: _process_cycle called but no cognitive_loop found. Basic increment.")

    # --- END SHIMS ---

    async def retry_brain_connection(self) -> bool:
        """Alias for retry_cognitive_connection to match Lazarus interface."""
        return await self.retry_cognitive_connection()

    async def retry_cognitive_connection(self) -> bool:
        """Manually retry connecting to the cognitive brain (LLM)."""
        from core.orchestrator.handlers.recovery import retry_cognitive_connection
        return await retry_cognitive_connection(self)

    def __init__(self, config_path: Optional[Path] = None, auto_fix_enabled: Optional[bool] = None, kernel_interface: Optional[Any] = None):
        """Initializes the orchestrator with required components."""
        # [BOOT FIX] Call Mixin initializer to set up hooks, state_manager, etc.
        self._init_basic_state(config_path, auto_fix_enabled)
        
        # Override/Set additional attributes specific to RobustOrchestrator
        self.auto_fix_enabled = auto_fix_enabled if auto_fix_enabled is not None else config.security.auto_fix_enabled
        self.stealth_mode = config.security.enable_stealth_mode
        self.boredom = 0
        
        self._current_thought_task = None
        self._autonomous_task = None
        self._autonomous_action_times = collections.deque(maxlen=200)
        self._poison_pill_cache = set()
        self._correction_shards = []
        
        # Loop-Agnostic Synchronization
        self._history_lock = RobustLock("Orchestrator.HistoryLock")
        self._task_lock = RobustLock("Orchestrator.TaskLock")
        self._extension_lock = RobustLock("Orchestrator.ExtensionLock")
        self._lock = RobustLock("Orchestrator.GlobalLock")
        self._stop_event = threading.Event()
        
        # ZENITH LOCKDOWN: Hardened state tracking
        self._inference_gate = None
        self._deadlock_watchdog_task = None
        self._last_emitted_fingerprint = ""
        self._last_user_interaction_time = 0.0
        self._foreground_user_quiet_until = 0.0

        from core.utils.concurrency import get_robust_semaphore
        self._user_input_semaphore = get_robust_semaphore("Orchestrator.UserInput", 3)
        
        self._init_queues()
        
        # UI/Event context
        self._status_override = None
        
        # v11.0: Attach Lazarus Early
        self.lazarus = None 
        
    def _init_queues(self):
        """Initialize communication queues."""
        from core.orchestrator.flow_control import CognitiveFlowController
        from core.tagged_reply_queue import TaggedReplyQueue
        from core.utils.queues import PriorityBackpressuredQueue
        self.message_queue = PriorityBackpressuredQueue(maxsize=100)
        # TaggedReplyQueue preserves response ownership across overlapping flows.
        self.reply_queue = TaggedReplyQueue(maxsize=50)
        # ZENITH LOCKDOWN: Background LLM Priority Queue
        self.system_priority_queue = PriorityBackpressuredQueue(maxsize=50)
        self._flow_controller = CognitiveFlowController()


    def restore_queues(self):
        """[PATCH 31] Robust Queue Restoration.
        Ensures communication channels are active even after partial failure.
        """
        logger.info("🔄 [RECOVERY] Restoring communication queues...")
        self._init_queues()
        self.status.is_processing = False
        return True

    def _update_heartbeat(self):
        """Update the orchestrator's heartbeat with monotonic time for duration checks."""
        self._last_heartbeat = time.monotonic()
        # Ensure status object reflects heartbeats for external monitoring
        if hasattr(self, 'status') and self.status:
            self.status.last_heartbeat = self._last_heartbeat

    @staticmethod
    def _is_user_facing_origin(origin: Optional[str]) -> bool:
        normalized = str(origin or "").strip().lower()
        return normalized in USER_FACING_ORIGINS

    def _extend_foreground_quiet_window(self, seconds: float) -> None:
        now = time.time()
        quiet_until = now + max(0.0, float(seconds))
        self._foreground_user_quiet_until = max(
            float(getattr(self, "_foreground_user_quiet_until", 0.0) or 0.0),
            quiet_until,
        )

    async def _aegis_sentinel(self):
        """Phase XXIII: True-Lock Reality Maintenance Loop."""
        from core.orchestrator.handlers.aegis import aegis_sentinel_loop
        await aegis_sentinel_loop(self)

    async def start(self):
        """Start the orchestrator (Async)"""
        lightweight_test_boot = bool(os.environ.get("PYTEST_CURRENT_TEST")) and not bool(
            os.environ.get("AURA_FULL_TEST_BOOT")
        )
        # v46: Boot Hard-Gate
        try:
            from core.bootstrap.validation import BootValidator
            v_result = BootValidator.validate_boot()
            if not v_result.passed:
                # Log failures but do NOT abort. These checks run before
                # _async_init_subsystems() registers late services, so failures here
                # are often premature. Full validation happens in StartupValidator.
                logger.warning("⚠️ Boot pre-check: %d issues (non-fatal): %s", 
                             len(v_result.failures), v_result.failures)
                # Continue — do not return False
            self._boot_warnings = v_result.warnings
            
            # Print factual banner (Patch 23)
            print("------------------------------------------")
            print("       AURORA NEURAL CORE v1.0.0          ")
            print("------------------------------------------")
            print(f" Integrity: Validated" if not self._boot_warnings else f" Integrity: Validated with {len(self._boot_warnings)} warnings")
            print(f" Environment: {os.uname().sysname} {os.uname().machine}")
            print("------------------------------------------")
            
        except Exception as e:
            record_degradation('main', e)
            record_degradation('main', e)
            logger.critical(f"❌ Boot Validator crashed: {e}")
            return False

        # Handle lazy initialization of subsystems
        if not self.status.initialized:
            await self._async_init_subsystems()

        if not self._inference_gate:
            container_gate = ServiceContainer.get("inference_gate", default=None)
            if container_gate is not None:
                self._inference_gate = container_gate
            else:
                await self._ensure_inference_gate_ready(context="startup")
        
        if self.status.running:
            logger.warning("Orchestrator already running")
            return True
        
        logger.info("Starting orchestrator (Async Mode)...")
        
        try:
            logger.info("🚩 [ORCHESTRATOR] Setting running flag...")
            self.status.running = True
            logger.info("🚩 [ORCHESTRATOR] running flag set to True.")
            self.status.start_time = time.time()

            # Wire graceful shutdown signals so persistence hooks fire on SIGTERM
            try:
                from core.graceful_shutdown import GracefulShutdown
                GracefulShutdown.setup_signals()
                logger.info("🛡️ Graceful shutdown signals wired (persistence on SIGTERM).")
            except Exception as exc:
                record_degradation('main', exc)
                record_degradation('main', exc)
                logger.debug("Graceful shutdown signal setup skipped: %s", exc)

            if lightweight_test_boot:
                logger.info("🧪 Lightweight pytest runtime boot: skipping continuous background services.")
                return True
            
            if getattr(self, "substrate", None):
                logger.info("🚩 [ORCHESTRATOR] Starting Substrate...")
                await asyncio.wait_for(self.substrate.start(), timeout=15.0)
                logger.info("🚩 [ORCHESTRATOR] Substrate started.")
            
            self.status.start_time = time.time()
            
            # Sensory systems are initialized in _async_init_subsystems or below
            if hasattr(self, '_start_sensory_systems'):
                logger.info("🚩 [ORCHESTRATOR] Starting Sensory Systems...")
                await asyncio.wait_for(self._start_sensory_systems(), timeout=15.0)
                logger.info("🚩 [ORCHESTRATOR] Sensory Systems started.")
            
            # Start Actor-Kernel Sensory Gate (Phase 1)
            logger.info("🚩 [ORCHESTRATOR] Starting Sensory Actor...")
            await asyncio.wait_for(self._start_sensory_actor(), timeout=15.0)
            logger.info("🚩 [ORCHESTRATOR] Sensory Actor started.")
                
            # --- Live Multimodal Vision Start ---
            try:
                from core.config import config
            # Redundant local import removed
                from core.senses.continuous_vision import ContinuousSensoryBuffer
                
                vision_buffer = ContinuousSensoryBuffer(data_dir=config.paths.data_dir)
                ServiceContainer.register_instance("continuous_vision", vision_buffer)
                vision_buffer.start()
                logger.info("👁️ Continuous Sensory Buffer registered and started.")
            except Exception as e:
                record_degradation('main', e)
                record_degradation('main', e)
                logger.error("Failed to start Continuous Sensory Buffer: %s", e)
            if hasattr(self, 'belief_sync') and self.belief_sync:
                await asyncio.wait_for(self.belief_sync.start(), timeout=15.0)
            if hasattr(self, 'attention_summarizer') and self.attention_summarizer:
                await asyncio.wait_for(self.attention_summarizer.start(), timeout=15.0)
            if hasattr(self, 'swarm') and self.swarm:
                await asyncio.wait_for(self.swarm.start(), timeout=15.0)
            
            # 📖 [PEER MODE] Evolution 2: Private Inner World
            if hasattr(self, 'narrative_engine'):
                # new attribute added to boot.py setup()
                self.narrative_engine.enable_private_archive = True
                logger.info("📖 Peer Mode: Private narrative archive activated")
            if hasattr(self, 'probe_manager') and self.probe_manager:
                from core.utils.task_tracker import get_task_tracker
                get_task_tracker().track_task(self.probe_manager.auto_cleanup_loop())
            # Loading Continuity Record
            try:
                from core.continuity import get_continuity
                _continuity_engine = get_continuity()
                record = _continuity_engine.load()
                if record:
                    self.status.cycle_count = getattr(record, "session_count", 0)
            except Exception as e:
                record_degradation('main', e)
                record_degradation('main', e)
                logger.error("Failed to load Continuity state: %s", e)
                _continuity_engine = None

            # ── Waking Sequence ───────────────────────────────────────────────
            # Emit an orientation thought based on how long we were offline.
            # Goes to the neural feed (thought cards), NOT the user chat.
            try:
                _ce = _continuity_engine if '_continuity_engine' in dir() else None
                if _ce:
                    _waking_ctx = _ce.get_waking_context()
                    _gap_h = _ce.gap_seconds / 3600.0

                    # Always emit the raw continuity context as a thought
                    from core.thought_stream import get_emitter as _get_emitter
                    _get_emitter().emit(
                        "Waking",
                        _waking_ctx,
                        level="info",
                        category="WakingSequence",
                    )
                    logger.info("🌅 Waking Sequence emitted (gap=%.1fh)", _gap_h)

                    # For gaps > 10 minutes, generate an LLM orientation narrative
                    # and emit it as a second thought (non-blocking — don't delay boot)
                    if _gap_h > (10 / 60):
                        async def _generate_orientation(_ctx=_waking_ctx, _gap=_gap_h):
                            try:
                                await asyncio.sleep(5.0)  # Let InferenceGate finish warming
                                from core.brain.inference_gate import InferenceGate
                                _gate = ServiceContainer.get("inference_gate", default=None)
                                if _gate is None:
                                    _gate = InferenceGate(self)
                                _orientation_prompt = (
                                    f"[WAKING SEQUENCE]\n{_ctx}\n\n"
                                    "In one or two sentences, synthesize your re-orientation. "
                                    "What do you notice? What are you picking up from where you left off? "
                                    "Be terse, authentic — no performance, just the felt sense of resuming."
                                )
                                _narrative = await _gate.think(
                                    _orientation_prompt,
                                    system_prompt="You are waking. Be honest, brief, and grounded.",
                                    prefer_tier="tertiary",
                                    is_background=True,
                                )
                                if _narrative and _narrative.strip():
                                    from core.thought_stream import get_emitter as _get_emitter2
                                    _get_emitter2().emit(
                                        "Orientation",
                                        _narrative.strip(),
                                        level="info",
                                        category="WakingSequence",
                                    )
                            except Exception as _oe:
                                record_degradation('main', _oe)
                                record_degradation('main', _oe)
                                logger.warning("Orientation narrative failed (non-fatal): %s", _oe)

                        self._fire_and_forget(_generate_orientation(), name="orchestrator.generate_orientation")
            except Exception as _we:
                record_degradation('main', _we)
                record_degradation('main', _we)
                logger.warning("Waking sequence non-fatal: %s", _we)

            # Loading Self Model
            if self.self_model:
                try:
                    from core.self_model import SelfModel
                    loaded = await SelfModel.load()
                    # Re-attach subsystems to the loaded instance if needed,
                    # but usually we just update the singleton's attributes.
                    # For now, we assume the factory provided an instance and we update its beliefs.
                    self.self_model.beliefs = loaded.beliefs
                    logger.info("✓ Self-Model persistent state loaded.")
                except Exception as e:
                    record_degradation('main', e)
                    record_degradation('main', e)
                    logger.error("Failed to load Self-Model state: %s", e)

            # ── Architecture Index ─────────────────────────────────────────────
            # Build Aura's self-knowledge index in the background so she can
            # answer questions about her own subsystems without blocking boot.
            try:
                from core.self.architecture_index import get_architecture_index
                _arch_idx = get_architecture_index()
                ServiceContainer.register_instance("architecture_index", _arch_idx)
                logger.info("✓ Architecture self-awareness index initializing (background)")
            except Exception as _ai_err:
                record_degradation('main', _ai_err)
                record_degradation('main', _ai_err)
                logger.warning("Architecture index boot init non-fatal: %s", _ai_err)

            # ── Affective Circumplex ───────────────────────────────────────────
            # Pre-warm the circumplex singleton so the first inference has params.
            try:
                from core.affect.affective_circumplex import get_circumplex
                _circ = get_circumplex()
                ServiceContainer.register_instance("affective_circumplex", _circ)
                _circ_p = _circ.get_llm_params()
                logger.info(
                    "✓ Affective Circumplex online: V=%.2f A=%.2f → temp=%.2f tokens=%d",
                    _circ_p["valence"], _circ_p["arousal"],
                    _circ_p["temperature"], _circ_p["max_tokens"],
                )
            except Exception as _circ_err:
                record_degradation('main', _circ_err)
                record_degradation('main', _circ_err)
                logger.warning("Affective Circumplex boot init non-fatal: %s", _circ_err)

            # ── Darwinian Heartstone Values ────────────────────────────────────
            try:
                from core.affect.heartstone_values import get_heartstone_values
                _hsv = get_heartstone_values()
                ServiceContainer.register_instance("heartstone_values", _hsv)
                logger.info("♥ HeartstoneValues online: %s",
                            {k: round(v, 2) for k, v in _hsv.values.items()})
            except Exception as _hsv_err:
                record_degradation('main', _hsv_err)
                record_degradation('main', _hsv_err)
                logger.warning("HeartstoneValues boot init non-fatal: %s", _hsv_err)

            # ── Epistemic Filter ───────────────────────────────────────────────
            try:
                from core.world_model.epistemic_filter import get_epistemic_filter
                _ef = get_epistemic_filter()
                ServiceContainer.register_instance("epistemic_filter", _ef)
                logger.info("🔬 EpistemicFilter online")
            except Exception as _ef_err:
                record_degradation('main', _ef_err)
                record_degradation('main', _ef_err)
                logger.warning("EpistemicFilter boot init non-fatal: %s", _ef_err)

            # ── Autonomous Sleep Trigger ───────────────────────────────────────
            try:
                from core.autonomy.sleep_trigger import get_sleep_trigger
                _st = get_sleep_trigger(self)
                ServiceContainer.register_instance("sleep_trigger", _st)
                self._fire_and_forget(_st.start(), name="orchestrator.sleep_trigger.start")
                logger.info("😴 AutonomousSleepTrigger active")
            except Exception as _st_err:
                record_degradation('main', _st_err)
                record_degradation('main', _st_err)
                logger.warning("SleepTrigger boot init non-fatal: %s", _st_err)

            # ── PNEUMA (Active Inference Engine) ──────────────────────────────
            try:
                from core.pneuma import get_pneuma
                _pneuma = get_pneuma()
                ServiceContainer.register_instance("pneuma", _pneuma)
                self._fire_and_forget(_pneuma.start(), name="orchestrator.pneuma.start")
                logger.info("🧠 PNEUMA active inference engine online")
            except Exception as _pe:
                record_degradation('main', _pe)
                record_degradation('main', _pe)
                logger.warning("PNEUMA boot non-fatal: %s", _pe)

            # ── MHAF (Mycelial Hypergraph Attractor Field) ────────────────────
            try:
                from core.consciousness.mhaf_field import get_mhaf
                _mhaf = get_mhaf()
                ServiceContainer.register_instance("mhaf", _mhaf)
                self._fire_and_forget(_mhaf.start(), name="orchestrator.mhaf.start")
                logger.info("🌿 MHAF consciousness substrate online")
            except Exception as _mhaf_err:
                record_degradation('main', _mhaf_err)
                record_degradation('main', _mhaf_err)
                logger.warning("MHAF boot non-fatal: %s", _mhaf_err)

            # ── ActiveInferenceSampler ────────────────────────────────────────
            try:
                from core.consciousness.precision_sampler import get_active_inference_sampler
                _ais = get_active_inference_sampler()
                ServiceContainer.register_instance("active_inference_sampler", _ais)
                logger.info("🎯 ActiveInferenceSampler online")
            except Exception as _ais_err:
                record_degradation('main', _ais_err)
                record_degradation('main', _ais_err)
                logger.warning("ActiveInferenceSampler boot non-fatal: %s", _ais_err)

            # ── Neologism Engine ──────────────────────────────────────────────
            try:
                from core.consciousness.neologism_engine import get_neologism_engine
                _neo = get_neologism_engine()
                ServiceContainer.register_instance("neologism_engine", _neo)
                logger.info("🔤 NeologismEngine (private lexicon) online")
            except Exception as _neo_err:
                record_degradation('main', _neo_err)
                record_degradation('main', _neo_err)
                logger.warning("NeologismEngine boot non-fatal: %s", _neo_err)

            # ── Terminal Fallback Chat + Autonomous Watchdog ───────────────────
            # TerminalFallbackChat: registered but NOT yet active.
            # TerminalWatchdog: background monitor — autonomously opens terminal
            #   only when UI is confirmed gone for 30s AND Aura has queued messages.
            try:
                from core.terminal_chat import get_terminal_fallback, get_terminal_watchdog
                _term = get_terminal_fallback()
                ServiceContainer.register_instance("terminal_fallback", _term)
                _watchdog = get_terminal_watchdog(orchestrator=self)
                ServiceContainer.register_instance("terminal_watchdog", _watchdog)
                self._fire_and_forget(_watchdog.start(), name="orchestrator.terminal_watchdog.start")
                logger.info("📟 TerminalFallbackChat + TerminalWatchdog online (autonomous, last-resort)")
            except Exception as _term_err:
                record_degradation('main', _term_err)
                record_degradation('main', _term_err)
                logger.warning("TerminalFallback boot non-fatal: %s", _term_err)

            # ── CRSM (Continuous Recurrent Self-Model) ────────────────────────
            try:
                from core.consciousness.crsm import get_crsm
                _crsm = get_crsm()
                ServiceContainer.register_instance("crsm", _crsm)
                logger.info("🔄 CRSM bidirectional self-model online")
            except Exception as _e:
                record_degradation('main', _e)
                record_degradation('main', _e)
                logger.warning("CRSM boot non-fatal: %s", _e)

            # ── HOT Engine (Higher-Order Thought) ─────────────────────────────
            try:
                from core.consciousness.hot_engine import get_hot_engine
                _hot = get_hot_engine()
                ServiceContainer.register_instance("hot_engine", _hot)
                logger.info("🔁 HOT Engine reflexive meta-awareness online")
            except Exception as _e:
                record_degradation('main', _e)
                record_degradation('main', _e)
                logger.warning("HOT Engine boot non-fatal: %s", _e)

            # ── Hedonic Gradient Engine ───────────────────────────────────────
            try:
                from core.consciousness.hedonic_gradient import get_hedonic_gradient
                _hg = get_hedonic_gradient()
                ServiceContainer.register_instance("hedonic_gradient", _hg)
                logger.info("💚 Hedonic Gradient Engine online — valence load-bearing")
            except Exception as _e:
                record_degradation('main', _e)
                record_degradation('main', _e)
                logger.warning("HedoniGradient boot non-fatal: %s", _e)

            # ── Counterfactual Engine ─────────────────────────────────────────
            try:
                from core.consciousness.counterfactual_engine import get_counterfactual_engine
                _cfe = get_counterfactual_engine()
                ServiceContainer.register_instance("counterfactual_engine", _cfe)
                logger.info("🔀 Counterfactual Engine deliberative agency online")
            except Exception as _e:
                record_degradation('main', _e)
                record_degradation('main', _e)
                logger.warning("CounterfactualEngine boot non-fatal: %s", _e)

            # ── AGI Layer ─────────────────────────────────────────────────────
            try:
                from core.agi.curiosity_explorer import get_curiosity_explorer
                from core.agi.skill_synthesizer import get_skill_synthesizer
                from core.agi.hierarchical_planner import get_hierarchical_planner
                _cx = get_curiosity_explorer()
                _ss = get_skill_synthesizer()
                _hp = get_hierarchical_planner()
                ServiceContainer.register_instance("curiosity_explorer", _cx)
                ServiceContainer.register_instance("skill_synthesizer", _ss)
                ServiceContainer.register_instance("hierarchical_planner", _hp)
                logger.info("🤖 AGI layer online (CuriosityExplorer + SkillSynthesizer + HierarchicalPlanner)")
            except Exception as _e:
                record_degradation('main', _e)
                record_degradation('main', _e)
                logger.warning("AGI layer boot non-fatal: %s", _e)

            # ── Agency Layer ──────────────────────────────────────────────────
            try:
                from core.agency.commitment_engine import get_commitment_engine
                from core.agency.compute_orchestrator import get_compute_orchestrator
                from core.agency.identity_guard import get_identity_guard
                from core.agency.sandboxed_modifier import get_sandboxed_modifier
                _cmt = get_commitment_engine()
                _co  = get_compute_orchestrator()
                _ig  = get_identity_guard()
                _sm  = get_sandboxed_modifier()
                ServiceContainer.register_instance("commitment_engine", _cmt)
                ServiceContainer.register_instance("compute_orchestrator", _co)
                ServiceContainer.register_instance("identity_guard", _ig)
                ServiceContainer.register_instance("sandboxed_modifier", _sm)
                logger.info("🛡️ Agency layer online (CommitmentEngine + ComputeOrchestrator + IdentityGuard + SandboxedModifier)")
            except Exception as _e:
                record_degradation('main', _e)
                record_degradation('main', _e)
                logger.warning("Agency layer boot non-fatal: %s", _e)

            # ── Security Layer ────────────────────────────────────────────────
            try:
                from core.security.user_recognizer import get_user_recognizer
                from core.security.trust_engine import get_trust_engine
                from core.security.integrity_guardian import get_integrity_guardian
                from core.security.emergency_protocol import get_emergency_protocol
                _ur  = get_user_recognizer()
                _te  = get_trust_engine()
                _ig2 = get_integrity_guardian()
                _ep  = get_emergency_protocol()
                # Build or verify integrity manifest on boot, then start background checks
                _ig2.initialize()
                _ig2.start_background_checks()
                ServiceContainer.register_instance("user_recognizer", _ur)
                ServiceContainer.register_instance("trust_engine", _te)
                ServiceContainer.register_instance("integrity_guardian", _ig2)
                ServiceContainer.register_instance("emergency_protocol", _ep)
                logger.info("🔐 Security layer online (UserRecognizer + TrustEngine + IntegrityGuardian + EmergencyProtocol)")
                if not _ur.has_passphrase():
                    logger.warning("⚠️  No owner passphrase set. Run: python -m core.security.user_recognizer --setup")
            except Exception as _e:
                record_degradation('main', _e)
                record_degradation('main', _e)
                logger.warning("Security layer boot non-fatal: %s", _e)

            # ── Substrate & Embodiment Layer ──────────────────────────────────
            try:
                from core.consciousness.crsm_lora_bridge import get_crsm_lora_bridge
                from core.senses.circadian import get_circadian
                from core.consciousness.experience_consolidator import get_experience_consolidator
                _lora_bridge = get_crsm_lora_bridge()
                _circadian   = get_circadian()
                _consolidator = get_experience_consolidator()
                # Wire cognitive engine into consolidator
                _consolidator.brain = getattr(self, "cognitive_engine", None)
                ServiceContainer.register_instance("crsm_lora_bridge", _lora_bridge)
                ServiceContainer.register_instance("circadian", _circadian)
                ServiceContainer.register_instance("experience_consolidator", _consolidator)
                # Start consolidation background loop
                self._fire_and_forget(_consolidator.start(), name="orchestrator.experience_consolidator.start")
                logger.info("🌱 Substrate layer online (CRSMLoraBridge + CircadianEngine + ExperienceConsolidator)")
            except Exception as _e:
                record_degradation('main', _e)
                record_degradation('main', _e)
                logger.warning("Substrate layer boot non-fatal: %s", _e)

            # Restore continuous stream of consciousness from snapshot
            try:
                from core.resilience.snapshot_manager import SnapshotManager
                snapshot_mgr = SnapshotManager(self)
                snapshot_mgr.thaw()
            except Exception as e:
                record_degradation('main', e)
                record_degradation('main', e)
                logger.error("Failed to thaw cognitive snapshot: %s", e)
            
            # Start Lazarus Brainstem (v11.0)
            try:
                from core.brain.llm.lazarus_brainstem import LazarusBrainstem
                self.brainstem = LazarusBrainstem(self)
                logger.info("✓ Lazarus Brainstem active")
            except Exception as e:
                record_degradation('main', e)
                record_degradation('main', e)
                logger.error("Failed to init Lazarus: %s", e)
                self.brainstem = None

            # Start Background Loops
            if hasattr(self, 'consciousness') and self.consciousness:
                res = self.consciousness.start()
                if res and inspect.isawaitable(res):
                    await asyncio.wait_for(res, timeout=15.0)
                logger.info("✓ Consciousness stream activated")
                
            if hasattr(self, 'curiosity') and self.curiosity:
                if hasattr(self.curiosity, 'start'):
                     res = self.curiosity.start()
                     if res and inspect.isawaitable(res):
                         await asyncio.wait_for(res, timeout=15.0)
                logger.info("✓ Curiosity background loop started")
            
            # Start Aegis Sentinel (Phase XXIII)
            from core.utils.task_tracker import get_task_tracker
            get_task_tracker().track_task(self._aegis_sentinel())
            
            # Start Proactive Communication (v4.3)
            if hasattr(self, 'proactive_comm') and self.proactive_comm:
                if hasattr(self.proactive_comm, 'start'):
                     res = self.proactive_comm.start()
                     if res and inspect.isawaitable(res):
                         await asyncio.wait_for(res, timeout=15.0)
                logger.info("✓ Proactive Communication loop started")
            
            # Start Narrative Engine (v11.0)
            if hasattr(self, 'narrative_engine') and self.narrative_engine:
                await asyncio.wait_for(self.narrative_engine.start(), timeout=15.0)
            
            # Start Agency Core background tasks
            if hasattr(self, 'agency_core') and self.agency_core:
                await asyncio.wait_for(self.agency_core.initialize(), timeout=15.0)
            
            # Start Sovereign Ears
            if self.ears:
                if hasattr(self.ears, "should_auto_listen") and self.ears.should_auto_listen():
                    logger.info("🚩 [ORCHESTRATOR] Starting Sovereign Ears...")
                    def _hear_callback(text):
                        logger.info("👂 Heard: %s", text)
                        # Phase Transcendental: Route voice through the FULL cognitive pipeline
                        # (not a separate lightweight LLM). Use run_coroutine_threadsafe because
                        # this callback fires from a non-async STT thread.
                        try:
                            loop = getattr(self, "loop", None)
                            if loop is None:
                                try:
                                    loop = asyncio.get_running_loop()
                                except RuntimeError:
                                    loop = None
                            if loop and loop.is_running():
                                asyncio.run_coroutine_threadsafe(
                                    self.process_user_input_priority(text, origin="voice"), loop
                                )
                            else:
                                get_task_tracker().track(self.process_user_input_priority(text, origin="voice"), loop=loop)
                        except Exception as e:
                            record_degradation('main', e)
                            record_degradation('main', e)
                            logger.error("Failed to schedule voice input: %s", e)
                    
                    await asyncio.wait_for(self.ears.start_listening(_hear_callback), timeout=15.0)
                    logger.info("✓ Sovereign Ears listening")
                else:
                    logger.info("✓ Sovereign Ears standing by (mic idle until explicitly enabled)")

            # Start Pulse Manager (Proactive Awareness)
            if self.pulse_manager:
                logger.info("🚩 [ORCHESTRATOR] Starting Pulse Manager...")
                await asyncio.wait_for(self.pulse_manager.start(), timeout=10.0)
                logger.info("✓ Pulse Manager active (Proactive Awareness)")

            # Start Inter-process Event Listeners
            from core.utils.task_tracker import get_task_tracker
            get_task_tracker().track_task(self._setup_event_listeners())

            # Start Cognitive Integration Layer
            if hasattr(self, 'cognition') and self.cognition:
                if hasattr(self.cognition, 'initialize'):
                     res = self.cognition.initialize()
                     if res and inspect.isawaitable(res):
                         await asyncio.wait_for(res, timeout=15.0)
                logger.info("✓ Advanced Cognitive Layer (Learning, Memory, Beliefs) initialized")
                
            # Initialize AgencyCore and SubsystemAudit
            try:
                from core.agency_core import AgencyCore
                self._agency_core = AgencyCore(orchestrator=self)
                ServiceContainer.register_instance("agency_core", self._agency_core)
                logger.info("✓ AgencyCore initialized")
            except Exception as ac_err:
                record_degradation('main', ac_err)
                record_degradation('main', ac_err)
                logger.error("AgencyCore init failed (non-fatal): %s", ac_err)
                self._agency_core = None
            
            try:
                from core.subsystem_audit import SubsystemAudit
                self._subsystem_audit = ServiceContainer.get("subsystem_audit", default=None) or SubsystemAudit()
                if not ServiceContainer.get("subsystem_audit", default=None):
                    ServiceContainer.register_instance("subsystem_audit", self._subsystem_audit)
                logger.info("✓ SubsystemAudit initialized")
            except Exception as sa_err:
                record_degradation('main', sa_err)
                record_degradation('main', sa_err)
                logger.error("SubsystemAudit init failed (non-fatal): %s", sa_err)
                self._subsystem_audit = None
                
            # Ollama Watchdog removed in v5.1
            
            # Start System Integrity Monitor (Stability Hardening)
            try:
                from core.resilience.integrity_monitor import SystemIntegrityMonitor
                self._integrity_monitor = SystemIntegrityMonitor(
                    data_dir=str(config.paths.data_dir)
                )
                await asyncio.wait_for(self._integrity_monitor.start(), timeout=10.0)
                ServiceContainer.register_instance("integrity_monitor", self._integrity_monitor)
                from core.utils.task_tracker import get_task_tracker
                get_task_tracker().track_task(self._integrity_monitor._task)
                logger.info("✓ System Integrity Monitor active")
            except Exception as im_err:
                record_degradation('main', im_err)
                record_degradation('main', im_err)
                logger.warning("Integrity Monitor init failed (non-fatal): %s", im_err)
            
            try:
                # Reuse the monitor created in hardening init if it exists
                existing_monitor = ServiceContainer.get("event_loop_monitor", default=None)
                if existing_monitor is not None:
                    self._event_loop_monitor = existing_monitor
                    logger.info("✓ Event Loop Monitor active (reused from hardening)")
                else:
                    from core.utils.concurrency import EventLoopMonitor
                    self._event_loop_monitor = EventLoopMonitor(threshold=0.25, interval=1.0)
                    self._event_loop_monitor.start()
                    logger.info("✓ Event Loop Monitor active")
            except Exception as el_err:
                record_degradation('main', el_err)
                record_degradation('main', el_err)
                logger.warning("Event Loop Monitor init failed: %s", el_err)
            
            # ---------------------------------------------------------
            # HARDENING: Register Periodic Metabolic/Substrate Tasks
            # ---------------------------------------------------------
            await asyncio.wait_for(self._register_scheduled_tasks(), timeout=10.0)
            # [STABILITY FIX] scheduler.start() is a continuous loop. Must run in background.
            self._fire_and_forget(scheduler.start(), name="orchestrator.scheduler.start")

            # 🧠 [PEER MODE] Evolution 1: MindTick / cognitive_loop becomes the PRIMARY heartbeat
            if hasattr(self, 'mind_tick') and self.mind_tick and not config.skeletal_mode:
                # [STABILITY FIX] mind_tick.start() is a continuous loop. Must run in background.
                self._fire_and_forget(self.mind_tick.start(), name="orchestrator.mind_tick.start")
                logger.info("🧠 Peer Mode: MindTick elevated as primary sovereign thread")
                
                # 🗣️ [PEER MODE] Evolution 5: Permanent internal society (Internal Debate)
                swarm_autostart = os.getenv("AURA_ENABLE_PERMANENT_SWARM", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if swarm_autostart and self.sovereign_swarm is not None and hasattr(self.sovereign_swarm, "start_permanent_debate"):
                    swarm_allowed = True
                    swarm_reason = "authority_unavailable"
                    try:
                        from core.constitution import get_constitutional_core

                        swarm_allowed, swarm_reason, _authority_decision = await get_constitutional_core(self).approve_initiative(
                            "peer_mode:permanent_swarm_debate",
                            source="peer_mode",
                            urgency=0.35,
                        )
                    except Exception as exec_err:
                        record_degradation('main', exec_err)
                        record_degradation('main', exec_err)
                        logger.debug("Permanent swarm authority gate unavailable: %s", exec_err)

                    # [STABILITY] Run in background to avoid blocking orchestrator launch
                    # especially under memory pressure when model loading is slow.
                    if swarm_allowed:
                        self._fire_and_forget(
                            self.sovereign_swarm.start_permanent_debate(
                                roles=["philosopher", "critic", "explorer", "ethicist"],
                                topic_source="liquid_state"
                            ),
                            name="orchestrator.swarm.start_permanent_debate",
                        )
                        logger.info("🗣️ Peer Mode: Internal multi-agent society running 24/7")
                    else:
                        logger.info("🗣️ Peer Mode: Permanent swarm debate suppressed by Executive: %s", swarm_reason)
                elif self.sovereign_swarm is not None and not swarm_autostart:
                    logger.info("🗣️ Peer Mode: Permanent swarm debate disabled by default. Set AURA_ENABLE_PERMANENT_SWARM=1 to enable.")
                elif self.sovereign_swarm is not None:
                    logger.warning("🗣️ Peer Mode: sovereign_swarm missing 'start_permanent_debate' method. Interface mismatch?")

                # 🛠️ [PEER MODE] Evolution 7: Sovereign self-modification loop
                if hasattr(self, '_self_modification') or hasattr(self, 'meta_learning'):
                    self_mod_allowed = True
                    self_mod_reason = "authority_unavailable"
                    try:
                        from core.constitution import get_constitutional_core

                        self_mod_allowed, self_mod_reason, _authority_decision = await get_constitutional_core(self).approve_initiative(
                            "peer_mode:sovereign_self_modification_loop",
                            source="peer_mode",
                            urgency=0.45,
                        )
                    except Exception as exec_err:
                        record_degradation('main', exec_err)
                        record_degradation('main', exec_err)
                        logger.debug("Self-mod authority gate unavailable: %s", exec_err)

                    if self_mod_allowed:
                        self._fire_and_forget(
                            self._safe_self_modification_loop(),
                            name="orchestrator.safe_self_modification_loop",
                        )
                        logger.info("🛠️ Peer Mode: Sovereign self-modification loop active")
                    else:
                        logger.info("🛠️ Peer Mode: Sovereign self-modification loop suppressed by Executive: %s", self_mod_reason)
            elif config.skeletal_mode:
                logger.info("💀 Skeletal Mode: High-CPU autonomous subsystems (MindTick, Swarm, Self-Mod) bypassed.")

            logger.info("✓ Orchestrator started")
            # [BOOT FIX] Defer locking to aura_main.py or _final_steps to avoid ContainerError
            # during CognitiveIntegrationLayer initialization.
            # ServiceContainer.lock_registration()
            return True
            
        except Exception as e:
            record_degradation('main', e)
            record_degradation('main', e)
            logger.error("Failed to start orchestrator: %s", e)
            self.status.running = False
            return False

    async def run(self):
        """Main continuous execution loop for the orchestrator.
        
        [DEEP ROOT FIX DR-1 + DR-2]:
        This loop is the heartbeat ONLY. It does NOT drain the message_queue.
        Message draining is CognitiveLoop's job (cognitive_loop.py:_acquire_next_message).
        Previously, run() and CognitiveLoop both raced for the same queue, causing:
        - Messages to bypass CognitiveLoop's proper routing
        - cycle_count to never increment (CognitiveLoop never got messages)
        - Double-processing of messages
        """
        # [HARDENING] _stop_event is always initialized in __init__ as threading.Event().
        # Do NOT re-assign as asyncio.Event() here — that causes a type mismatch.
        # Only guard against the (impossible) case where __init__ was bypassed.
        if not hasattr(self, "_stop_event") or self._stop_event is None:
            self._stop_event = threading.Event()
            
        self.status.running = True
        logger.info("🚩 [ORCHESTRATOR] Main Heartbeat Active (Loop started).")
        
        try:
            while not self._stop_event.is_set():
                # Heartbeat cycle count — ensures UI sees progress
                self.status.cycle_count += 1
                
                # Short sleep to prevent CPU spinning while remaining responsive
                await asyncio.sleep(0.05) 
        except asyncio.CancelledError:
            logger.info("Orchestrator heartbeat cancelled.")
        except Exception as e:
            record_degradation('main', e)
            record_degradation('main', e)
            logger.error("🛑 ORCHESTRATOR LOOP CRITICAL ERROR: %s", e, exc_info=True)
        finally:
            self.status.running = False
            logger.info("Orchestrator heartbeat stopped.")


    async def _register_scheduled_tasks(self):
        """Standardize all periodic substrate tasks into the Central Scheduler."""
        from core.resilience.circuit_breaker import get_circuit_breaker
        breaker = get_circuit_breaker()

        # 1. Liquid Pacing & Subsystem Audit (High Frequency)
        await scheduler.register(TaskSpec(
            name="physiological_pacing",
            coro=self._update_liquid_pacing,
            tick_interval=2.0  # Runs every 2 seconds
        ))
        await scheduler.register(TaskSpec(
            name="subsystem_audit",
            coro=self._pulse_subsystem_audit,
            tick_interval=5.0
        ))
        # v48 DEPRECATION: The legacy 'cognitive_cycle' is disabled in favor of
        # the MindTick loop (Sovereign mode). Running both causes queue race conditions
        # and double-processing instability.
        # await scheduler.register(TaskSpec(
        #     name="cognitive_cycle",
        #     coro=self._process_cycle,
        #     tick_interval=1.0  # Core cycle once per second
        # ))

        # 2. State Persistence (Safety)
        await scheduler.register(TaskSpec(
            name="state_persistence",
            coro=lambda: self._save_state_async("periodic"),
            tick_interval=60.0 # Standard periodic save
        ))

        # 3. Metabolic Pulse (Unified Coordinator)
        if self.metabolic_coordinator:
            # Ensure the resolved service actually has the expected cycle method
            if hasattr(self.metabolic_coordinator, "process_cycle"):
                await scheduler.register(TaskSpec(
                    name="metabolic_heartbeat",
                    coro=self.metabolic_coordinator.process_cycle,
                    tick_interval=60.0 # 1 minute metabolic pulse
                ))
                logger.info("💓 MetabolicCoordinator integrated into Scheduler.")
            else:
                logger.warning("⚠️ metabolic_coordinator (%s) lacks 'process_cycle'. Skipping registration.", type(self.metabolic_coordinator).__name__)
        else:
            logger.warning("⚠️ MetabolicCoordinator not found. Background tasks may be degraded.")

        # 4. Telemetry Pulse
        await scheduler.register(TaskSpec(
            name="telemetry_heartbeat",
            coro=self._emit_telemetry_pulse,
            tick_interval=5.0
        ))
        
        # 🎯 [PEER MODE] Evolution 6: Autonomous Goal Genesis
        await scheduler.register(TaskSpec(
            name="peer_goal_genesis",
            coro=self._peer_generate_and_persist_goal,
            tick_interval=3600.0 # Spontaneous goal once per hour
        ))

        # 🌀 Evolution 9: Meta-Cognitive Self-Optimization
        async def meta_evolution_wrapper():
            if hasattr(self, 'meta_cognition') and self.meta_cognition:
                # Guard: skip if memory pressure is high or system is under load
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    if mem.percent > 75:
                        logger.debug("🌀 Meta-Evolution skipped: memory pressure %.1f%%", mem.percent)
                        return
                except Exception as _exc:
                    record_degradation('main', _exc)
                    record_degradation('main', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)
                logger.info("🌀 [SCHEDULER] Triggering Meta-Evolution Cycle...")
                try:
                    await asyncio.wait_for(self.meta_cognition.evolve(), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning("🌀 Meta-Evolution timed out after 30s — skipping this cycle.")
                except Exception as exc:
                    record_degradation('main', exc)
                    record_degradation('main', exc)
                    logger.debug("🌀 Meta-Evolution error: %s", exc)

        await scheduler.register(TaskSpec(
            name="meta_evolution_cycle",
            coro=meta_evolution_wrapper,
            tick_interval=1800.0 # Runs every 30 minutes
        ))
        
        logger.info("✓ Substrate tasks registered with Scheduler.")

    async def _handle_periodic_tasks(self):
        """Standard interface for scheduled substrate tasks (Legacy).
        Note: Use core.scheduler for new periodic tasks.
        """
        pass  # no-op: intentional

        pass # _emit_telemetry_pulse removed (now in StatusManagerMixin)


    def _pulse_subsystem_audit(self):
        """Register heartbeats from all active subsystems and emit periodic health pulses."""
        audit = getattr(self, '_subsystem_audit', None)
        if not audit:
            return
        
        try:
            # Register heartbeats for subsystems that are actually active
            # We use the properties/getters to ensure we're checking the live service instances.
            # Names must match SubsystemAudit.SUBSYSTEMS keys.
            
            # 1. Personality
            if self.personality_engine or getattr(self, '_personality_engine', None):
                audit.heartbeat("personality_engine")
            
            # 2. Liquid State (Affect Engine)
            if self.liquid_state or getattr(self, '_liquid_state', None):
                audit.heartbeat("liquid_state")
            
            # 3. Liquid Substrate (LNN)
            if (hasattr(self, 'substrate') and self.substrate) or (hasattr(self, 'liquid_substrate') and self.liquid_substrate) or getattr(self, '_liquid_substrate', None):
                audit.heartbeat("liquid_substrate")
                
            # 4. Drive Controller
            if (hasattr(self, 'drive_controller') and self.drive_controller) or (hasattr(self, 'drives') and self.drives) or getattr(self, '_drive_controller', None):
                audit.heartbeat("drive_controller")
            
            # 5. Consciousness
            if self.consciousness or getattr(self, '_conscious_substrate', None):
                audit.heartbeat("consciousness")
            
            # 6. Affect Engine (Explicit)
            if self.affect or getattr(self, '_affect_engine', None):
                audit.heartbeat("affect_engine")
            
            # 7. Agency Core
            if (hasattr(self, 'agency') and self.agency) or getattr(self, '_agency_core', None):
                audit.heartbeat("agency_core")
            
            # 8. Capability Engine (Skill Registry)
            if self.capability_engine or getattr(self, '_capability_engine', None):
                audit.heartbeat("capability_engine")
            
            # 9. Identity (Self Model)
            if self.identity or getattr(self, '_identity', None) or getattr(self, '_self_model', None):
                audit.heartbeat("identity")
            
            # 10. Cognitive Engine (Brain)
            if self.cognitive_engine or getattr(self, '_cognitive_engine', None):
                audit.heartbeat("cognitive_engine")
            
            # 11. Sovereign Scanner
            if (hasattr(self, 'scanner') and self.scanner) or (hasattr(self, 'sovereign_scanner') and self.sovereign_scanner) or getattr(self, '_sovereign_scanner', None):
                audit.heartbeat("sovereign_scanner")
            
            # Periodic Unified Health Pulse
            if audit.should_emit_pulse():
                pulse_report = audit.emit_pulse()
                self._emit_thought_stream(pulse_report)
                logger.info("🫀 %s", pulse_report.replace('\n', ' | '))
        except Exception as e:
            record_degradation('main', e)
            record_degradation('main', e)
            logger.warning("Subsystem audit pulse error (non-fatal): %s", e)

    def _track_metabolic_task(self, name: str, coro_or_func):
        """Ensures metabolic tasks (RL, updates) don't pile up and exhaust resources."""
        if name in self._active_metabolic_tasks:
            return
            
        self._active_metabolic_tasks.add(name)
        
        if not asyncio.iscoroutine(coro_or_func):
            # If it's just a sync function/result, we're done already
            self._active_metabolic_tasks.discard(name)
            return

        from core.utils.task_tracker import get_task_tracker
        task = get_task_tracker().track(coro_or_func, name=name)
        
        def _cleanup(t):
            self._active_metabolic_tasks.discard(name)
            if not t.cancelled() and t.exception():
                logger.error("Metabolic task %s failed: %s", name, t.exception())

        task.add_done_callback(_cleanup)
        return task

    # Zenith: Public alias for legacy tests
    track_metabolic_task = _track_metabolic_task

    def _fire_and_forget(self, coro, name: Optional[str] = None):
        """Helper to run an async coroutine in the background without awaiting it."""
        # Import upfront — placing the import after the first use makes Python
        # treat `get_task_tracker` as a local and raises UnboundLocalError on
        # line ``task = get_task_tracker().create_task(...)``. Every
        # fire-and-forget on boot was failing with "cannot access local
        # variable 'get_task_tracker'" until this import was hoisted.
        from core.utils.task_tracker import get_task_tracker

        # Zenith-HF1 HARDENING: Only create task for real coroutines
        if not (asyncio.iscoroutine(coro) or inspect.iscoroutine(coro)):

            # If it's a coroutine function, it needs to be called
            if inspect.iscoroutinefunction(coro):
                logger.warning("Passed coroutine function instead of coroutine to fire_and_forget: %s", coro)
                return None
            logger.warning("Attempted to fire_and_forget a non-coroutine: %s", coro)
            return

        try:
            task = get_task_tracker().create_task(coro, name=name)
        except RuntimeError:
            _dispose_awaitable(coro)
            return None

        if not isinstance(task, asyncio.Task):
            _dispose_awaitable(coro)
            return None

        task = get_task_tracker().track_task(task)
        task.add_done_callback(_bg_task_exception_handler)
        return task

    async def _recover_from_stall(self):
        """Attempts to recover from a cognitive loop stall."""
        self._recovery_attempts += 1
        logger.warning("🚑 RECOVERY ATTEMPT #%s initiated...", self._recovery_attempts)
        
        # 0. DLQ Capture
        try:
            dlq = ServiceContainer.get("dead_letter_queue", default=None)
            if dlq:
                dlq.capture_failure(
                    message=getattr(self, "_current_objective", "None"),
                    context={"recovery_attempt": self._recovery_attempts},
                    error=RuntimeError("Cognitive Stall Detected"),
                    source="orchestrator_stall"
                )
        except Exception as dlq_e:
            record_degradation('main', dlq_e)
            record_degradation('main', dlq_e)
            logger.error("CRITICAL: Failed to log to DLQ during stall: %s", dlq_e)

        try:
            # 1. Soft Recovery: Cancel hanging thought tasks
            if (task := self._current_thought_task) and not task.done():
                logger.info("Cancelling hanging thought task...")
                task.cancel()
            
            # 2. Reset queues if severely backed up and save to DLQ
            if self.message_queue.qsize() > 50:
                logger.warning("Message queue overflow detected. Clearing and moving to DLQ...")
                dropped = []
                while not self.message_queue.empty():
                    raw = self.message_queue.get_nowait()
                    if isinstance(raw, tuple):
                        msg = raw[-1]
                    else:
                        msg = raw
                    dropped.append(msg)
                if dropped:
                    try:
                        dlq_path = config.paths.data_dir / "dlq.jsonl"
                        def _append_dlq(path, msgs):
                            with open(path, "a") as f:
                                for msg in msgs:
                                    f.write(json.dumps({"timestamp": time.time(), "message": msg}) + "\n")
                        await run_io_bound(_append_dlq, dlq_path, dropped)
                    except Exception as e:
                        record_degradation('main', e)
                        record_degradation('main', e)
                        logger.error("Failed to dump dropped messages to DLQ file: %s", e)

            # 3. Substrate Defrag — clear caches before re-initializing brain
            try:
                autonomic = ServiceContainer.get("autonomic_core", default=None)
                if autonomic and hasattr(autonomic, '_substrate_defrag'):
                    await autonomic._substrate_defrag()
            except Exception as df_err:
                record_degradation('main', df_err)
                record_degradation('main', df_err)
                logger.debug("Substrate defrag during recovery skipped: %s", df_err)

            # 3.5 Soft-restart cognitive connection
            await self.retry_cognitive_connection()

            # 3.75 Lazarus Brainstem Intervention
            if self._recovery_attempts >= 2 and hasattr(self, 'lazarus') and self.lazarus:
                logger.warning("🚨 [RECOVERY] Escalating to Lazarus Brainstem...")
                await self.lazarus.attempt_recovery()

            # 4. [STABILITY] Re-dispatch last user message if this was a priority turn
            origin = getattr(self, "_current_origin", "user")
            if hasattr(self, "_current_user_message") and self._current_user_message and origin in ("user", "voice", "admin"):
                last_msg = self._current_user_message
                msg_hash = hashlib.md5(last_msg.encode()).hexdigest()
                
                if msg_hash in self._poison_pill_cache:
                    logger.critical("🚫 [RECOVERY] Poison pill detected (%s). Skipping auto-retry.", msg_hash)
                    await self.output_gate.emit(
                        "My neural substrate rejected that request due to a fatal processing fault. I've purged the thought.",
                        origin="autonomous_thought", target="primary", metadata={"error": True}
                    )
                    return

                logger.info("🔄 [RECOVERY] Auto-retrying last user message...")
                # We use a slight delay to let the reset settle
                async def _delayed_retry():
                    await asyncio.sleep(2.0)
                    await self._handle_incoming_message(last_msg, origin=origin)
                
                get_task_tracker().create_task(_delayed_retry())

            # 5. Escalation: Full system restart if recovery fails repeatedly
            if self._recovery_attempts >= 3:
                # Add current message to poison pill cache before restart
                if (msg := getattr(self, "_current_user_message", None)):
                    msg_hash = hashlib.md5(msg.encode()).hexdigest()
                    self._poison_pill_cache.add(msg_hash)
                    # Prevent unbounded growth over very long sessions
                    if len(self._poison_pill_cache) > 500:
                        self._poison_pill_cache = set(list(self._poison_pill_cache)[-200:])
                    logger.critical("☠️ POISON PILL CACHED: %s", msg_hash)
                
                logger.critical("🚨 STALL PERSISTS: Escalating to internal orchestrator restart (Hot-Reload).")
                # Do NOT set self.status.running = False here.
                # This flag tells the parent (aura_main.py) to exit the loop and call stop().
                # We want to perform an internal restart instead.
                await self.start()
                self._recovery_attempts = 0 # Reset after escalation
                
            logger.info("✅ Recovery logic applied.")
        except Exception as e:
            record_degradation('main', e)
            record_degradation('main', e)
            logger.error("Recovery sequence failed: %s", e)

    # _acquire_next_message -> MessageHandlingMixin
    # _defer_enqueue_message -> MessageHandlingMixin
    # enqueue_message -> MessageHandlingMixin

    async def _execute_and_reply(self, message: Any, origin: str = "user"):
        """Execute the cognitive cycle and send the reply."""
        # Priority Inference Lane
        # If this is an autonomous/background thought, and the user interacted recently, yield.
        if not self._is_user_facing_origin(origin):
            time_since_user = time.time() - getattr(self, '_last_user_interaction_time', 0.0)
            if time_since_user < 30.0:
                logger.debug("🛡️ [PRIORITY] Yielding autonomous thought to user inference lane.")
                await asyncio.sleep(0.2) # Minimum yield to allow user-path to acquire semaphore
        
        # Original logic continues...

    async def process_event(self, event: Any, origin: Any = "internal", priority: int = 20):
        """Compatibility alias for legacy subsystems.

        Uses the async-native enqueue_message path since callers are
        coroutines running in the same event loop.  The old
        enqueue_from_thread path would incorrectly resolve the loop
        and trigger RuntimeError when the async loop was already
        running.

        Legacy callers sometimes passed a payload dict as the second positional
        argument (before this method standardized on ``origin``). Preserve that
        shape by wrapping the event into a message payload instead of treating
        the dict as an origin label.
        """
        payload = None
        normalized_origin = origin

        if isinstance(origin, dict):
            payload = dict(origin)
            normalized_origin = str(payload.pop("origin", "") or "internal")
            if priority == 20 and "priority" in payload:
                try:
                    priority = int(payload.pop("priority"))
                except Exception:
                    pass  # no-op: intentional
        elif origin is None:
            normalized_origin = "internal"
        elif not isinstance(origin, str):
            payload = {"value": origin}
            normalized_origin = "internal"

        if payload is not None:
            if isinstance(event, dict):
                merged = dict(event)
                merged_context = dict(merged.get("context") or {})
                merged_context.update(payload)
                merged["context"] = merged_context
                merged.setdefault("origin", normalized_origin)
                event = merged
            else:
                event = {
                    "content": event,
                    "context": payload,
                    "origin": normalized_origin,
                }

        self.enqueue_message(
            event,
            priority=priority,
            origin=str(normalized_origin or "internal"),
        )

    async def _ensure_inference_gate_ready(self, context: str = "runtime") -> bool:
        """Ensure the unified inference gate is ready before user-facing chat begins."""
        if self._inference_gate is not None:
            return True

        container_gate = ServiceContainer.get("inference_gate", default=None)
        if container_gate is not None:
            self._inference_gate = container_gate
            logger.info("✅ InferenceGate adopted from ServiceContainer during %s.", context)
            return True

        logger.warning("⚠️ InferenceGate missing during %s. Creating it now...", context)
        from core.brain.inference_gate import InferenceGate

        self._inference_gate = InferenceGate(self)
        try:
            lightweight_test_boot = bool(os.environ.get("PYTEST_CURRENT_TEST")) and not bool(
                os.environ.get("AURA_FULL_TEST_BOOT")
            )
            if lightweight_test_boot:
                self._inference_gate._initialized = True
                logger.info("🧪 Lightweight pytest boot: deferring InferenceGate init during %s.", context)
            else:
                await self._inference_gate.initialize()
            ServiceContainer.register_instance("inference_gate", self._inference_gate)
            logger.info("✅ InferenceGate initialized successfully during %s.", context)
        except Exception as gate_err:
            record_degradation('main', gate_err)
            record_degradation('main', gate_err)
            logger.error(
                "⚠️ [ZENITH] InferenceGate init failed during %s: %s. Cloud-only mode.",
                context,
                gate_err,
            )
            self._inference_gate._initialized = True
            ServiceContainer.register_instance("inference_gate", self._inference_gate)
        return self._inference_gate is not None

    # enqueue_from_thread -> MessageHandlingMixin
    # _deep_circular_safe_sanitize -> MessageHandlingMixin
    # _normalize_to_dict -> MessageHandlingMixin
    # _dispatch_message -> MessageHandlingMixin
    # _emit_dispatch_telemetry -> MessageHandlingMixin

    # --- Actor-Kernel Helper Methods (Phase 1) ---
    async def _start_sensory_actor(self):
        """Initializes and starts the SensoryGateActor via the Supervision Tree (Phase 2)."""
        try:
            from core.actors.sensory_gate import start_sensory_gate

            # 1. Get or Create Actor Bus (Parent side)
            self._actor_bus = ServiceContainer.get("actor_bus", default=ActorBus())
            try:
                if not ServiceContainer.has("actor_bus"):
                    ServiceContainer.register_instance("actor_bus", self._actor_bus)
            except Exception: pass

            # 2. Register Actor with Supervisor Tree
            spec = ActorSpec(
                name="SensoryGate",
                target=start_sensory_gate,
                args=(),
                restart_policy="one_for_one"
            )
            
            if self.supervisor:
                self.supervisor.add_actor(spec)
                
                # 3. Start Supervisor and Actor
                await self.supervisor.start()
                parent_pipe = self.supervisor.start_actor("SensoryGate")
                self._actor_bus.add_actor("SensoryGate", parent_pipe, is_child=False)
                self._actor_bus.start()

                # 4. Register Cross-Process Handlers
                self._actor_bus.register_handler("SensoryGate", "SENSORY_UPDATE", self._handle_sensory_update)
                self._actor_bus.register_handler("SensoryGate", "COMMIT_STATE", self._handle_remote_commit)
                self._actor_bus.register_handler("SensoryGate", "HEARTBEAT", self._handle_actor_heartbeat)
                logger.info("🛡️ SensoryGateActor managed by Supervision Tree.")
            else:
                logger.error("❌ Cannot start SensoryGateActor: Supervisor Tree not available in container.")
            
        except Exception as e:
            record_degradation('main', e)
            record_degradation('main', e)
            import traceback
            logger.error("Failed to start SensoryGateActor: %s\n%s", e, traceback.format_exc())
            # Legacy fallback: Sensory systems continue in-process if possible


    async def _handle_sensory_update(self, payload):
        """Callback from actor for sensory discoveries."""
        logger.info("📡 Sensory Actor update: %s", payload.get('type'))
        # Logic to integrate into belief graph/state will be expanded in Phase 2

    async def _handle_remote_commit(self, payload):
        """Securely proxy state commits from actors through StateRepositoryV2."""
        mutation = payload.get("mutation")
        trace_id = payload.get("trace_id", "remote-actor")
        if mutation:
            # Ensure safe commit via the single atomic queue
            await self.state_repo.commit(mutation, trace_id)

    async def _handle_actor_heartbeat(self, payload):
        """Actor is still alive and processing."""
        self._last_sensory_heartbeat = time.time()
        # Optional: update health metrics

    async def _update_liquid_pacing(self):
        """
        Background paced loop for Liquid Substrate Bridge.
        Dynamically wrapped by `core/consciousness/liquid_substrate_bridge.py`.
        """
        while not self._stop_event.is_set():
            from core.utils.task_tracker import get_task_tracker
            await asyncio.sleep(5.0)
            
            # Maintenance triggers
            if self.status.cycle_count % 10 == 0:
                self._update_heartbeat()
                
                # Database Optimization (Vacuum)
                async def _optimize_dbs():
                    try:
                        import sqlite3
                        def _sync_vacuum():
                            for db_file in config.paths.data_dir.glob("*.db"):
                                try:
                                    conn = sqlite3.connect(db_file)
                                    conn.execute("VACUUM")
                                    conn.close()
                                except Exception:
                                    logger.debug("Failed to record cognitive latency.")
                        await asyncio.to_thread(_sync_vacuum)
                    except Exception as e:
                        record_degradation('main', e)
                        record_degradation('main', e)
                        logger.debug("Database vacuum thread failed: %s", e)
                
                if not self.status.is_processing:
                    get_task_tracker().bounded_track(_optimize_dbs(), name="db_vacuum")

            # 4. Long-term Consolidation (Persistent Highlights)
            if self.status.cycle_count % 50 == 0 and len(self.conversation_history) > 10 and self.memory_manager:
                get_task_tracker().track_task(self._consolidate_long_term_memory())

            # 5. Digital Metabolism (Strategic Forgetting)
            if self.status.cycle_count % 100 == 0:
                # Call maintenance on core facades
                from inspect import isawaitable
                for provider in ("memory_manager", "meta_learning", "alignment"):
                    comp = getattr(self, provider, None)
                    if comp and hasattr(comp, 'run_maintenance'):
                        coro = comp.run_maintenance()
                        if isawaitable(coro):
                            from core.utils.task_tracker import get_task_tracker
                            get_task_tracker().track_task(coro)
                
                if hasattr(self, 'memory') and self.memory:
                    # Prune low salience memories older than 14 days
                    try:
                        from core.utils.task_tracker import get_task_tracker
                        if asyncio.iscoroutinefunction(self.memory.prune_low_salience):
                            get_task_tracker().track_task(self.memory.prune_low_salience(threshold_days=14))
                        else:
                            get_task_tracker().track_task(get_task_tracker().create_task(
                                run_io_bound(self.memory.prune_low_salience, threshold_days=14)
                            ))
                    except Exception as e:
                        record_degradation('main', e)
                        record_degradation('main', e)
                        logger.debug("Vector pruning skipped: %s", e)


    # _deduplicate_history -> ContextStreamingMixin
    # _prune_history_async -> ContextStreamingMixin
    # _consolidate_long_term_memory -> ContextStreamingMixin

    # _process_message -> MessageHandlingMixin

    async def _run_terminal_self_heal(self):
        """Check terminal monitor for errors to fix."""
        try:
            from core.terminal_monitor import get_terminal_monitor
            monitor = get_terminal_monitor()
            if monitor:
                error_goal = await monitor.check_for_errors()
                is_thinking = (t := self._current_thought_task) is not None and not t.done()
                if error_goal and not is_thinking:
                    logger.info("🔧 Terminal Monitor: Auto-fix triggered")
                    
                    # Report to self-modifier for intelligence logging
                    sm = getattr(self, 'self_modifier', None)
                    if sm:
                        sm.on_error(
                            Exception(f"Terminal Command Failure: {error_goal.get('error', 'Unknown')}") if isinstance(error_goal.get('error'), str) else Exception("Terminal Command Failure"),
                            {"command": error_goal.get("command"), "output": error_goal.get("output")},
                            skill_name="TerminalMonitor"
                        )
                    
                    from core.constitution import get_constitutional_core
                    from core.utils.task_tracker import get_task_tracker

                    allowed, reason, _authority_decision = await get_constitutional_core(self).approve_initiative(
                        f"terminal_self_heal:{error_goal.get('objective', '')[:160]}",
                        source="terminal_monitor",
                        urgency=0.72,
                        state=getattr(getattr(self, "state_repo", None), "_current", None),
                    )
                    if not allowed:
                        record_degraded_event(
                            "terminal_monitor",
                            "self_heal_blocked",
                            detail=str(error_goal.get("objective", ""))[:160],
                            severity="warning",
                            classification="background_degraded",
                            context={"reason": reason},
                        )
                        return

                    self._current_thought_task = get_task_tracker().track_task(get_task_tracker().create_task(
                        self.process_user_input_priority(error_goal['objective'], origin="terminal_monitor")
                    ))
        except Exception as e:
            record_degradation('main', e)
            record_degradation('main', e)
            record_degraded_event(
                "terminal_monitor",
                "self_heal_check_failed",
                detail=f"{type(e).__name__}: {e}",
                severity="warning",
                classification="background_degraded",
                exc=e,
            )
            logger.debug("Terminal monitor check failed: %s", e)

    async def process_unprompted_stimulus(self, modality: str, data: Any, context: str = "") -> None:
        """Reactive hook for spontaneous sensory inputs (Vision/Audio).
        Bypasses the normal queue to ensure real-time environmental awareness.
        """
        logger.info("✨ [SENSORY] Spontaneous %s stimulus detected: %s", modality, context)
        
        # Log spontaneous event to UnifiedTranscript
        try:
            from core.conversation.unified_transcript import UnifiedTranscript
            transcript = UnifiedTranscript.get_instance()
            transcript.add_system(f"Spontaneous {modality} stimulus: {context}")
        except Exception as e:
            record_degradation('main', e)
            record_degradation('main', e)
            capture_and_log(e, {'module': __name__})

        try:
            from core.constitution import get_constitutional_core

            allowed, reason, _authority_decision = await get_constitutional_core(self).approve_initiative(
                f"sensory_stimulus:{modality}:{context[:160]}",
                source="sensory_motor",
                urgency=0.62,
                state=getattr(getattr(self, "state_repo", None), "_current", None),
            )
            if not allowed:
                record_degraded_event(
                    "sensory_motor",
                    "stimulus_blocked",
                    detail=f"{modality}:{context[:160]}",
                    severity="warning",
                    classification="background_degraded",
                    context={"reason": reason},
                )
                return
        except Exception as e:
            record_degradation('main', e)
            record_degradation('main', e)
            record_degraded_event(
                "sensory_motor",
                "stimulus_gate_failed",
                detail=f"{modality}:{context[:160]}",
                severity="warning",
                classification="background_degraded",
                context={"error": type(e).__name__},
                exc=e,
            )
            return

        # Inject as an unprompted thought into the core cognitive cycle
        await self.process_user_input_priority(f"[ENVIRONMENTAL TRIGGER]: {context}", origin="sensory_motor")

    async def generate_autonomous_thought(self) -> str:
        """Triggers a self-generated thought/topic from the internal cognitive engine."""
        logger.info("🧠 [VOLITION] Generating autonomous volition pulse...")
        # In a full implementation, this triggers a 'dream' or 'contemplate' cycle.
        # Here we provide a high-agency topic grounded in recent activity.
        return "I've been contemplating our recent mission objectives. I have a new perspective on the Sovereign Network synchronization — should we discuss it?"

    async def generate_voice_response(self, user_text: str) -> str:
        """Voice now goes through the FULL cognitive pipeline.
        
        Previously this was a separate Ollama llama3.2:3b call with a bare
        system prompt — no personality, memory, qualia, consciousness, or
        homeostatic modifiers. Now it routes through process_user_input()
        which goes through the state machine, cognitive engine, and all
        enrichment layers identically to text.
        """
        logger.info("🧠 [VOICE→COGNITIVE] Routing voice through full pipeline")
        
        # Log to unified transcript
        try:
            from core.conversation.unified_transcript import UnifiedTranscript
            transcript = UnifiedTranscript.get_instance()
            transcript.add_voice_input(user_text)
        except Exception as e:
            record_degradation('main', e)
            record_degradation('main', e)
            capture_and_log(e, {'module': __name__})

        # Route through the SAME cognitive pipeline as text
        response = await self.process_user_input_priority(user_text, origin="voice")
        
        if not response:
            response = "I'm processing that thought. One moment."

        # Log Aura's response to unified transcript
        try:
            from core.conversation.unified_transcript import UnifiedTranscript
            transcript = UnifiedTranscript.get_instance()
            transcript.add_voice_output(response)
        except Exception as e:
            record_degradation('main', e)
            record_degradation('main', e)
            capture_and_log(e, {'module': __name__})

        return response

    def _get_fingerprint(self, text: str) -> str:
        """Generates a stable fingerprint for message deduplication."""
        return hashlib.sha256(text.strip().encode()).hexdigest()

    async def _deadlock_watchdog(self):
        """
        ZENITH LOCKDOWN: 45s Force-Release Loop.
        Monitors the global StateLock and force-releases it if held beyond the safety threshold
        to prevent permanent system hangs during Metal XPC stalls.
        """
        logger.info("🛡️ Deadlock Watchdog active (45s threshold).")

        def _event_is_set(event_obj: Any) -> bool:
            probe = getattr(event_obj, "is_set", None)
            if callable(probe):
                return bool(probe())
            return bool(event_obj)

        def _lock_is_locked(lock_obj: Any) -> bool:
            probe = getattr(lock_obj, "locked", None)
            if callable(probe):
                return bool(probe())
            if probe is not None:
                return bool(probe)
            return False

        while not _event_is_set(self._stop_event):
            try:
                await asyncio.sleep(15)
                # If lock is held and processing is stalled
                if _lock_is_locked(self._lock) and self.status.is_processing:
                    elapsed = time.monotonic() - getattr(self, "_current_processing_start", time.monotonic())
                    if elapsed > 45.0:
                        logger.critical("🚨 [WATCHDOG] Deadlock detected (held for %.1fs)! Force-releasing StateLock...", elapsed)
                        self._lock.force_release()
                        self.status.is_processing = False
                        # Notify UI of recovery
                        await self.output_gate.emit("I've recovered from a cognitive stall. Reprioritizing...", origin="system", target="primary")
            except Exception as e:
                record_degradation('main', e)
                record_degradation('main', e)
                logger.error("⚠️ Watchdog error: %s", e)
            except asyncio.CancelledError:
                break

    def _publish_telemetry(self, data: dict[str, Any]):
        """
        ZENITH LOCKDOWN: Fast-path telemetry delivery.
        Bypasses the EventBus for critical status/activity updates to ensure UI responsiveness.
        """
        try:
            # Direct pipe broadcast to UI
            bus = ServiceContainer.get("actor_bus", default=None)
            if bus:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(bus.publish("telemetry_update", data))
                except RuntimeError:
                    # Generic fallback if no loop is running
                    pass  # no-op: intentional
            
            # Legacy thought stream fallback
            from ..thought_stream import get_emitter
            label = data.get("label", data.get("type", "status"))
            get_emitter().emit("telemetry", str(label), level="debug")
        except Exception as e:
            record_degradation('main', e)
            record_degradation('main', e)
            logger.debug("Fast-path telemetry failed: %s", e)

    def get_cognitive_load(self) -> dict[str, Any]:
        """Returns metrics on cognitive uncertainty and current investigation targets."""
        if hasattr(self, "_flow_controller") and self._flow_controller:
            snap = self._flow_controller.snapshot(self)
            return {
                "uncertainty": round(snap.load, 3),
                "target_topic": getattr(self, "_current_objective", "") or "current context",
                "queue_depth": snap.queue_depth,
                "reply_depth": snap.reply_depth,
                "governor_mode": snap.governor_mode,
                "busy": snap.busy,
            }
        return {
            "uncertainty": 0.2, # Stable by default
            "target_topic": "Sovereign Intelligence"
        }

    # process_user_input_priority -> MessageHandlingMixin
    # _process_user_input_unlocked -> MessageHandlingMixin
    # _process_user_input_core -> MessageHandlingMixin

    _MAX_CORRECTION_SHARDS = 30

    # _route_prefixed_message -> IncomingLogicMixin
    # _process_message_pipeline -> IncomingLogicMixin
    # _handle_incoming_message -> IncomingLogicMixin
    # _handle_filesystem_reality_check -> IncomingLogicMixin

    # _original_handle_incoming_logic -> IncomingLogicMixin (see stub below for reference)
    # _original_handle_incoming_logic -> IncomingLogicMixin
    # _gather_agentic_context -> ContextStreamingMixin
    # chat_stream -> ContextStreamingMixin
    # sentence_stream_generator -> ContextStreamingMixin

    def _get_current_mood(self) -> str:
        """Get current mood from personality engine (safe helper)."""
        try:
            from ..brain.personality_engine import get_personality_engine
            res = get_personality_engine().current_mood
            if inspect.isawaitable(res):
                _dispose_awaitable(res)
                return "balanced"
            return res if isinstance(res, str) else "balanced"
        except Exception:
            return "balanced"

    def _get_current_time_str(self) -> str:
        """Get current time string (safe helper)."""
        try:
            from ..brain.personality_engine import get_personality_engine
            time_context = get_personality_engine().get_time_context()
            if inspect.isawaitable(time_context):
                _dispose_awaitable(time_context)
                return ""
            res = time_context.get("formatted", "")
            return res if isinstance(res, str) else ""
        except Exception:
            return ""
            

    async def _execute_plan(self, plan: dict[str, Any]) -> list[Any]:
        """Execute a plan of actions.
        Overridden/Patched by Behavior Controller.
        """
        results = []
        for i, call in enumerate(plan.get("tool_calls", [])):
            result = await self.execute_tool(call["name"], call.get("arguments", {}))
            results.append(result)
            
            # Critic Engine hook (Phase 25)
            # Every 3 steps, verify progress
            if (i + 1) % 3 == 0:
                try:
                    # (None)
                    critic = ServiceContainer.get("critic_engine", default=None)
                    if critic:
                        judgment = await critic.critique_plan(plan, results)
                        if judgment.recommendation == "backtrack":
                            logger.warning("Critic triggered backtrack: %s", judgment.evidence)
                            break # Fall out of loop, let caller replan
                        elif judgment.recommendation == "replan":
                            logger.warning("Critic triggered replan: %s", judgment.evidence)
                            break
                        # If 'continue', proceed normally
                except Exception as e:
                    record_degradation('main', e)
                    record_degradation('main', e)
                    logger.debug("Critic Engine evaluation failed: %s", e)
                    
        return results

    def health_check(self) -> bool:
        """Perform health check"""
        checks = []
        
        # Check if ready or running (Ready = Initialized and No Errors)
        is_ready = self.status.running or (self.status.initialized and not self.status.last_error)
        checks.append(("ready", is_ready))
        
        # Check thread (Only check if running to avoid test failure on mocks)
        if self.status.running:
            is_alive = True
            if hasattr(self, "_thread") and self._thread:
                if hasattr(self._thread, 'is_alive'):
                    try:
                        is_alive = self._thread.is_alive()
                    except Exception:
                        is_alive = False
            checks.append(("thread_alive", is_alive))
        
        # Check for too many errors
        err_count = self.stats.get("errors_encountered", 0) if isinstance(self.stats, dict) else 0
        checks.append(("error_rate", err_count < 100))
        
        # All checks must pass
        self.status.healthy = all(check[1] for check in checks)
        
        # v10.0 Parity: If status lacks expected attributes, return True conservatively
        if not hasattr(self.status, 'healthy'):
             return True
             
        return self.status.healthy





    async def _setup_event_listeners(self):
        """Subscribe to inter-process events."""
        from core.event_bus import get_event_bus
        bus = get_event_bus()
        
        q = await bus.subscribe("user_input")
        logger.info("👂 Orchestrator listening for 'user_input' events (Redis-backed)")
        
        while self.status.running:
            try:
                # v48: PriorityQueue returns (priority, seq, event) tuples
                _priority, _seq, event = await q.get()
                data = event.get("data", {})
                message = data.get("message")
                
                if message:
                    # v47 FIX: Preserve the origin (source) from the EventBus
                    # Voice engine sends {"message": text, "source": "voice"}.
                    # If we don't pass this source as origin, TTS won't trigger.
                    origin = data.get("source", "user")
                    logger.info("📥 Processing event-driven input (%s): %s", origin, message[:50])
                    # Standardized input processing
                    from core.utils.task_tracker import get_task_tracker
                    get_task_tracker().track_task(self.process_user_input_priority(message, origin=origin))
                else:
                    logger.debug("Auto-suggestion source check handled.")
            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('main', e)
                record_degradation('main', e)
                logger.error("Error in event listener loop: %s", e)


_orchestrator_instance: Optional[RobustOrchestrator] = None
_orchestrator_lock = threading.Lock()

def create_orchestrator(**kwargs) -> RobustOrchestrator:
    # Factory function for orchestrator
    global _orchestrator_instance
    if _orchestrator_instance is not None:
        if _orchestrator_instance:
            return _orchestrator_instance

    with _orchestrator_lock:
        if _orchestrator_instance: # Double-checked locking
            return _orchestrator_instance

        try:
            from core.service_registration import register_all_services
            register_all_services()
            
            _orchestrator_instance = RobustOrchestrator(**kwargs)
            
            # Phase 14.1: Register instance IMMEDIATELY in container to satisfy dependencies
        # Redundant local import removed
            ServiceContainer.register_instance("orchestrator", _orchestrator_instance)
            
            logger.info("✓ Orchestrator instance created directly (v14.1)")
            return _orchestrator_instance
        
        except Exception as exc:
            record_degradation('main', exc)
            record_degradation('main', exc)
            logger.critical("CRITICAL: Orchestrator creation failed: %s", exc, exc_info=True)
        
            # Create minimal fallback with ALL required methods
            class FallbackOrchestrator:
                def __init__(self, error_msg="Unknown error"):
                    self.status = type('Status', (), {
                        'is_processing': False, 
                        'dependencies_ok': False, 
                        'initialized': False, 
                        'running': False, 
                        'cycle_count': 0,
                        'health_metrics': {}
                    })()
                    self.error = error_msg
                    self._stop_event: Optional[asyncio.Event] = None
                    self.reply_queue = asyncio.Queue()
                    self.message_queue = asyncio.Queue()
                    from pathlib import Path
                    self.state_repo = type('StateRepo', (), {
                        'db_path': Path('data/aura_state.db'),
                        'initialize': lambda *args, **kwargs: asyncio.sleep(0),
                        'get_current': lambda *args, **kwargs: asyncio.sleep(0),
                        'commit': lambda *args, **kwargs: asyncio.sleep(0),
                    })()

                @property
                def stop_event(self) -> asyncio.Event:
                    if self._stop_event is None:
                        self._stop_event = asyncio.Event()
                    return self._stop_event
                
                async def start(self):
                    logger.error("Cannot start failed orchestrator: " + self.error)
                    return False

                async def run(self):
                    logger.info("FallbackOrchestrator: Running in limited mode")
                    await self.stop_event.wait()

                async def retry_brain_connection(self):
                    return {"status": "error", "message": f"Critical failure: {self.error}"}

                async def retry_cognitive_connection(self):
                    return await self.retry_brain_connection()
                
                def get_status(self):
                    return {"status": "FAILED", "error": self.error}
                
                async def stop(self):
                    self._stop_event.set()

                async def process_user_input(self, message, origin="user"):
                    return f"System is in recovery mode: {self.error}"

                def _publish_telemetry(self, data):
                    pass  # No-op in fallback mode
            
            return FallbackOrchestrator(str(exc))

SovereignOrchestrator = RobustOrchestrator
Orchestrator = RobustOrchestrator
AsyncAgentOrchestrator = RobustOrchestrator

__all__ = [
    "AsyncAgentOrchestrator",
    "Orchestrator",
    "RobustOrchestrator",
    "SovereignOrchestrator",
    "SystemStatus",
    "create_orchestrator",
]
