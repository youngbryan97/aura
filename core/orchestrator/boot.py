"""Orchestrator Boot Mixin"""

import asyncio
import logging
import os
import time
from collections import deque
from typing import Any

from core.runtime.errors import record_degradation

try:
    from core.morality.master_moral_integration import integrate_complete_moral_and_sensory_systems
except ImportError:
    integrate_complete_moral_and_sensory_systems = None

from core.config import config
from core.container import ServiceContainer

from .initializers.core_baseline import init_enterprise_layer

try:
    from core.agency.skill_library import SkillLibrary
except ImportError:
    SkillLibrary = Any

# Top-level imports moved to methods for boot performance
from .mixins.boot.boot_autonomy import BootAutonomyMixin
from .mixins.boot.boot_background import BootBackgroundMixin
from .mixins.boot.boot_cognitive import BootCognitiveMixin
from .mixins.boot.boot_identity import BootIdentityMixin
from .mixins.boot.boot_resilience import BootResilienceMixin
from .mixins.boot.boot_sensory import BootSensoryMixin
from .orchestrator_types import SystemStatus

#: Returned by an extracted block that did NOT return early. A unique
#: object, so no value a block legitimately returns can be mistaken for it.
_SEAM_FELL_THROUGH = object()

logger = logging.getLogger(__name__)


async def _await_startup_io(
    start: Any,
    *,
    what: str,
    env_var: str,
    default_s: float,
) -> None:
    """Await a disk-bound startup step, retrying once before giving up.

    How long opening a database takes is a property of the disk and of how much
    history is already in it — not of whether the runtime is healthy. These
    steps were awaited on hardcoded budgets (15s for the state repository, 10s
    for the database coordinator) with no operator knob, and a TimeoutError here
    propagates out of boot: the whole runtime fails to start.

    Observed live 2026-07-26 on a busy host: the state repository's 15-second
    budget expired at boot.py:478 and the TimeoutError ended the boot. Aura did
    not come up at all. A slow disk must not be indistinguishable from
    a broken one, so the budget is tunable, generous by default, and a first
    timeout buys a second attempt at double the budget before the failure is
    allowed to be fatal. The final failure is recorded with the elapsed time so
    the cause is attributable rather than a bare traceback.
    """
    try:
        budget = max(5.0, float(os.getenv(env_var, "") or default_s))
    except (TypeError, ValueError):
        budget = default_s

    started = time.monotonic()
    for attempt, timeout_s in enumerate((budget, budget * 2.0), start=1):
        try:
            await asyncio.wait_for(start(), timeout=timeout_s)
            if attempt > 1:
                logger.warning(
                    "🐢 %s came up on attempt %d after %.1fs — the disk is slow, "
                    "not broken.",
                    what, attempt, time.monotonic() - started,
                )
            return
        except TimeoutError:
            if attempt == 1:
                logger.warning(
                    "🐢 %s did not initialize within %.0fs; retrying once with "
                    "%.0fs before treating a slow disk as a failed boot.",
                    what, timeout_s, timeout_s * 2.0,
                )
                continue
            elapsed = time.monotonic() - started
            record_degradation(
                "orchestrator_boot",
                TimeoutError(f"{what} initialize exceeded {elapsed:.1f}s"),
                severity="critical",
                action=(
                    f"failed boot after {what} exceeded its startup budget twice; "
                    f"raise {env_var} if this host is simply slow"
                ),
            )
            raise


def _record_boot_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("boot", exc, severity=severity, action=action)


def _health_contract_boot_log(level: Any, *, initialized: bool, running: bool) -> tuple[int, str]:
    level_name = str(getattr(level, "value", level)).lower()
    runtime_ready = bool(initialized and running)
    if not runtime_ready and level_name in {"dead", "critical", "degraded"}:
        phase = "BOOT CORE COMPLETE" if initialized else "BOOTING"
        return (
            logging.INFO,
            f"⏳ HEALTH CONTRACT: {phase} — runtime services are still registering",
        )
    if level_name == "dead":
        return logging.CRITICAL, "🚨 HEALTH CONTRACT: DEAD — no critical services alive"
    if level_name == "critical":
        return logging.CRITICAL, "🚨 HEALTH CONTRACT: CRITICAL — some critical services missing"
    if level_name == "degraded":
        return logging.WARNING, "⚠️ HEALTH CONTRACT: DEGRADED — important services missing"
    return logging.INFO, "✅ HEALTH CONTRACT: All critical + important services online"


def _final_boot_health_log(
    contract: dict[str, Any],
    *,
    initialized: bool,
    running: bool,
) -> tuple[int, str]:
    """Classify final initialization separately from live runtime readiness."""
    status = str(contract.get("status") or "dead")
    critical_keys = [
        str(item.get("container_key") or "")
        for item in contract.get("failures", {}).get("critical", [])
        if isinstance(item, dict) and item.get("container_key")
    ]
    important_keys = [
        str(item.get("container_key") or "")
        for item in contract.get("failures", {}).get("important", [])
        if isinstance(item, dict) and item.get("container_key")
    ]
    probe_blockers = [
        str(item)
        for item in contract.get("probe_blockers", [])
        if str(item).strip()
    ]
    blockers = list(dict.fromkeys(critical_keys + important_keys + probe_blockers))

    if initialized and not running:
        if blockers == ["inference_gate"]:
            return (
                logging.INFO,
                "⏳ BOOT CORE COMPLETE: core systems initialized; Cortex prewarm "
                "is still pending, so launcher readiness remains gated.",
            )
        detail = f" ({', '.join(blockers)})" if blockers else ""
        return (
            logging.INFO,
            "⏳ BOOT CORE COMPLETE: initialization succeeded; runtime readiness "
            f"remains gated while services register{detail}.",
        )

    return _health_contract_boot_log(
        status,
        initialized=initialized,
        running=running,
    )


async def _skip_the_background_subsystems_in_foreground_only(
    *,
    self: Any,
) -> Any:
    """Stop here when the run is foreground-only.

    Moved out of ``OrchestratorBootMixin._async_init_subsystems`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 1 name(s) and hands back
    0.
    """
    async def _block() -> Any:
        if os.getenv("AURA_FOREGROUND_ONLY", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        } or os.getenv("AURA_ENABLE_SELF_HEALING", "1").lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            logger.info("Healing Swarm disabled for foreground-only boot.")
        else:
            try:
                from core.resilience.healing_swarm import HealingSwarmService

                healer = HealingSwarmService(self)
                healing_started = healer.start()
                if healing_started:
                    ServiceContainer.register_instance("healing_swarm", healer)
                    self.healing_service = healer
                    logger.info("🛡️ Healing Swarm Service initialized and started.")
                else:
                    logger.info("Healing Swarm deferred by runtime background policy.")

                if healing_started:
                    # ── Wire IncidentManager → HealingSwarm ──────────────
                    # Close the "log and limp on" gap: when degradation
                    # events escalate to CRITICAL/EMERGENCY, trigger
                    # autonomous repair instead of just recording.
                    try:
                        from core.resilience.incident_manager import get_incident_manager

                        def _incident_to_repair(incident):
                            """Bridge: IncidentManager alert → HealingSwarm repair."""
                            import asyncio as _aio

                            try:
                                _aio.get_running_loop()
                            except RuntimeError:
                                return  # No event loop — can't schedule repair
                            from core.utils.task_tracker import get_task_tracker

                            get_task_tracker().create_task(
                                healer.attempt_repair(
                                    incident.category,
                                    {
                                        "status": incident.severity.value,
                                        "description": incident.description[:200],
                                        "root_cause": incident.root_cause_hint,
                                        "occurrences": incident.occurrence_count,
                                    },
                                ),
                                name=f"heal.{incident.category[:40]}",
                            )
                            logger.info(
                                "🔗 [INCIDENT→HEAL] Dispatched repair for %s (severity=%s, occurrences=%d)",
                                incident.category,
                                incident.severity.value,
                                incident.occurrence_count,
                            )

                        get_incident_manager().register_alert_callback(_incident_to_repair)
                        logger.info("🔗 IncidentManager → HealingSwarm alert bridge active.")
                    except (ImportError, AttributeError, RuntimeError) as bridge_err:
                        _record_boot_degradation(
                            bridge_err,
                            action="continued healing swarm without incident alert bridge",
                            severity="degraded",
                        )
                        logger.warning(
                            "⚠️ IncidentManager→HealingSwarm bridge failed: %s", bridge_err
                        )
            except (ImportError, AttributeError, RuntimeError) as e:
                _record_boot_degradation(
                    e,
                    action="continued boot without healing swarm service",
                    severity="degraded",
                )
                logger.error("🛑 Failed to init Healing Swarm: %s", e)
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response


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
    _current_thought_task: asyncio.Task | None
    _private_archive: list
    _last_self_initiated_contact: float
    boredom: float
    _active_metabolic_tasks: set[str]
    _stop_event: Any
    _lock: Any
    _history_lock: Any
    _task_lock: Any
    _extension_lock: Any
    stats: dict[str, Any]

    reasoning_queue: Any | None
    reflex_engine: Any | None
    lazarus: Any | None
    persona_evolver: Any | None
    self_modifier: Any | None
    meta_evolution: Any | None
    epistemic_humility: Any | None

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

    async def emit_spontaneous_message(
        self,
        message: str,
        modality: str = "chat",
        origin: str = "system",
        *,
        urgency: float | None = None,
        metadata=None,
    ):
        from .mixins.autonomy import AutonomyMixin

        return await AutonomyMixin.emit_spontaneous_message(
            self,
            message,
            modality=modality,
            origin=origin,
            urgency=urgency,
            metadata=metadata,
        )

    world_model: Any | None
    skill_library: Any | None
    rsi_lab: Any | None
    concept_bridge: Any | None
    cryptolalia_decoder: Any | None
    ontology_genesis: Any | None
    morphic_forking: Any | None
    motivation: Any | None
    belief_sync: Any | None
    attention: Any | None
    attention_summarizer: Any | None
    probe_manager: Any | None
    cognitive_engine: Any | None
    dream_cycle: Any | None
    meaning_substrate: Any | None
    hallucination_filter: Any | None
    dream_engine: Any | None
    continuous_learner: Any | None
    react_loop: Any | None
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
        ServiceContainer.register_instance("output_gate", self.output_gate)

        # Register self as orchestrator for dependency resolution
        ServiceContainer.register_instance("orchestrator", self)

        try:
            from core.consciousness.executive_authority import get_executive_authority

            self.executive_authority = get_executive_authority(self)
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_boot_degradation(
                exc,
                action="continued setup with executive authority unavailable",
                severity="degraded",
            )
            logger.error("Executive authority bootstrap failed: %s", exc, exc_info=True)

        try:
            from core.constitution import get_constitutional_core

            self.constitutional_core = get_constitutional_core(self)
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_boot_degradation(
                exc,
                action="continued setup with constitutional core unavailable",
                severity="degraded",
            )
            logger.error("Constitutional core bootstrap failed: %s", exc, exc_info=True)

        # [PATCH 23] Integrity Guard (File Verification)
        try:
            from core.sovereignty.integrity_guard import IntegrityGuard

            guard = IntegrityGuard(self)
            score = guard.verify_sovereignty()
            if score < 1.0:
                logger.warning("🛡️ [BOOT] Integrity score degraded: %.2f", score)
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_boot_degradation(
                e,
                action="continued setup without sovereignty integrity score",
                severity="degraded",
            )
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
        from .coordinators.affect import AffectCoordinator
        from .coordinators.agency import AgencyCoordinator
        from .coordinators.memory import MemoryCoordinator

        # v5.0.1 FIX: Register Facades early so coordinators can resolve them immediately
        try:
            from core.memory.memory_facade import MemoryFacade
            from core.memory.memory_write_gateway import get_memory_write_gateway

            ServiceContainer.register_instance(
                "memory_write_gateway",
                get_memory_write_gateway(),
            )

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

            from core.executive.authority_gateway import get_authority_gateway
            from core.executive.standing_authority import get_standing_authority_manager
            from core.will import get_will

            get_will()
            get_standing_authority_manager()
            get_authority_gateway()

            logger.info(
                "✓ [BOOT] Core facades and governance gates registered during synchronous setup."
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_boot_degradation(
                e,
                action="continued setup with early facade registration partially degraded",
            )
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
        if not hasattr(self, "message_queue") or self.message_queue is None:
            from core.utils.queues import PriorityBackpressuredQueue

            self.message_queue = PriorityBackpressuredQueue(maxsize=100)
        if not hasattr(self, "reply_queue") or self.reply_queue is None:
            from core.conversation.tagged_reply_queue import TaggedReplyQueue

            self.reply_queue = TaggedReplyQueue(maxsize=50)

        # Reset stats safely (preserves keys)
        if hasattr(self, "stats") and isinstance(self.stats, dict):
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
            self._boot_lock = RobustLock(
                "Orchestrator.AsyncBootLock",
                watchdog_threshold_s=900.0,
                force_release_on_stall=False,
                timeout_s=900.0,
            )

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

                # Boot flight recorder: one mark per phase; summary + artifact
                # at ready. A 13-minute live boot once left no evidence of
                # which phase ate the time.
                from core.runtime.boot_profile import get_boot_profiler

                boot_profiler = get_boot_profiler()
                boot_profiler.mark("pre_subsystem_init")

                # --- Phase 1: Sync & Threading (FIXES SENTINEL RACE) ---
                self.setup()
                if hasattr(self, "_async_init_threading"):
                    self._async_init_threading()
                try:
                    from core.executive.standing_authority import (
                        get_standing_authority_manager,
                    )

                    await asyncio.wait_for(
                        get_standing_authority_manager().initialize(),
                        timeout=10.0,
                    )
                    logger.info("✓ [BOOT] Standing authority loaded with durable budgets and revocations.")
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as authority_exc:
                    _record_boot_degradation(
                        authority_exc,
                        action=(
                            "continued boot with autonomous tool execution fail-closed until "
                            "standing-authority state recovers"
                        ),
                        severity="degraded",
                    )
                    logger.error(
                        "⚠️ [BOOT] Standing-authority state unavailable: %s",
                        authority_exc,
                    )
                boot_profiler.mark("sync_bootstrap_and_threading")

                try:
                    from core.runtime.runtime_hygiene import get_runtime_hygiene

                    self.runtime_hygiene = get_runtime_hygiene()
                    await self.runtime_hygiene.start()
                    ServiceContainer.register_instance("runtime_hygiene", self.runtime_hygiene)
                    logger.info("🧹 Runtime hygiene installed (tasks, threads, processes, memory).")
                except (ImportError, AttributeError, RuntimeError) as hygiene_exc:
                    _record_boot_degradation(
                        hygiene_exc,
                        action="continued async boot without runtime hygiene watchdog",
                        severity="degraded",
                    )
                    logger.error(
                        "⚠️ Runtime hygiene bootstrap failed: %s", hygiene_exc, exc_info=True
                    )

                await init_enterprise_layer(self)
                boot_profiler.mark("runtime_hygiene_and_enterprise_layer")

                # --- PHASE 2: Resilience Foundation & State ---
                await self._start_state_vault_actor()
                await self._start_meta_evolution()

                if hasattr(self, "state_repo") and self.state_repo:
                    await _await_startup_io(
                        self.state_repo.initialize,
                        what="state repository",
                        env_var="AURA_STATE_REPO_INIT_TIMEOUT_S",
                        default_s=60.0,
                    )

                from core.resilience.database_coordinator import get_db_coordinator

                db_coord = get_db_coordinator()
                await _await_startup_io(
                    db_coord.start,
                    what="database coordinator",
                    env_var="AURA_DB_COORDINATOR_START_TIMEOUT_S",
                    default_s=45.0,
                )
                ServiceContainer.register_instance("database_coordinator", db_coord)
                boot_profiler.mark("state_vault_and_databases")

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
                    if getattr(self._inference_gate, "_initialized", False):
                        logger.info("✅ [BOOT] Local InferenceGate registered and initialized.")
                    else:
                        receipt_reader = getattr(
                            self._inference_gate, "initialization_receipt", None
                        )
                        receipt = receipt_reader() if callable(receipt_reader) else {}
                        reason = str(
                            receipt.get("reason")
                            or getattr(self._inference_gate, "_init_error", "")
                            or "local_initialization_incomplete"
                        )
                        init_error = RuntimeError(reason)
                        _record_boot_degradation(
                            init_error,
                            action=(
                                "registered unready local inference gate for bounded local retry"
                            ),
                            severity="degraded",
                        )
                        logger.error(
                            "⚠️ [BOOT] Local InferenceGate is registered but unready (%s). "
                            "Foreground demand will retry local initialization.",
                            reason,
                        )
                except (ImportError, AttributeError, RuntimeError) as gate_err:
                    _record_boot_degradation(
                        gate_err,
                        action="kept inference local and unready after gate initialization failed",
                        severity="degraded",
                    )
                    logger.error(
                        "⚠️ [BOOT] Local InferenceGate init failed: %s. "
                        "Remote inference is retired; the local gate remains retryable.",
                        gate_err,
                        exc_info=True,
                    )
                    if self._inference_gate is not None:
                        ServiceContainer.register_instance(
                            "inference_gate", self._inference_gate
                        )

                boot_profiler.mark("inference_gate")

                # Preserve a structural local gate when construction is possible. Never
                # forge readiness: callers can distinguish an unready retryable gate from
                # a gate that completed initialization.
                if not self._inference_gate:
                    logger.critical(
                        "🛑 [BOOT] Local InferenceGate is absent after initialization. "
                        "Attempting one bounded local reconstruction."
                    )
                    try:
                        from core.brain.inference_gate import InferenceGate

                        self._inference_gate = InferenceGate(self)
                        ServiceContainer.register_instance(
                            "inference_gate", self._inference_gate
                        )
                    except (ImportError, AttributeError, RuntimeError) as gate_err:
                        _record_boot_degradation(
                            gate_err,
                            action="continued boot without a constructible local inference gate",
                            severity="error",
                        )
                        logger.critical(
                            "🛑 [BOOT] Local InferenceGate reconstruction failed: %s.",
                            gate_err,
                            exc_info=True,
                        )

                # Now build the LLM router — it will find the InferenceGate in ServiceContainer
                from core.brain.llm_health_router import build_router_from_config

                ServiceContainer.register_instance("llm_router", build_router_from_config(config))

                await self._init_voice_subsystem()
                boot_profiler.mark("voice_subsystem")
                await self._init_cognitive_architecture()
                boot_profiler.mark("cognitive_architecture")

                # --- PHASE 3.1: Narrative Thread Activation ---
                narrative_thread = ServiceContainer.get("narrative_thread", default=None)
                if narrative_thread:
                    await narrative_thread.start()
                    logger.info("🎬 NarrativeThread activated.")
                else:
                    logger.warning("⚠️ NarrativeThread not found in ServiceContainer.")

                await self._init_language_services()
                boot_profiler.mark("narrative_and_language_services")

                def _spawn_boot_task(coro: Any, name: str) -> asyncio.Task:
                    from core.utils.task_tracker import get_task_tracker

                    try:
                        return get_task_tracker().create_task(coro, name=name)
                    except (RuntimeError, AttributeError, TypeError, ValueError):
                        return get_task_tracker().create_task(coro, name=name)

                # Discovery and isolated validation form one immutable catalog
                # transaction. Start it before identity/guardian boot, but do
                # not publish the engine until its Phase 6 owner consumes it.
                self._start_skill_catalog_warmup()

                # ZENITH LOCKDOWN: Start Deadlock Watchdog
                if hasattr(self, "_deadlock_watchdog") and not lightweight_test_boot:
                    self._deadlock_watchdog_task = _spawn_boot_task(
                        self._deadlock_watchdog(),
                        "orchestrator.deadlock_watchdog",
                    )

                # --- PHASE 4: Identity & Self-Model ---
                from core.brain.identity import IdentityService
                from core.self_model import SelfModel

                self.self_model = await SelfModel.load()
                ServiceContainer.register_instance("self_model", self.self_model)
                ServiceContainer.register_instance("identity", self.self_model)
                identity_service = ServiceContainer.get("identity_service", default=None)
                if identity_service is None:
                    identity_service = IdentityService()
                    ServiceContainer.register_instance("identity_service", identity_service)
                self.identity_service = identity_service

                await self._init_identity_systems()
                boot_profiler.mark("identity_and_self_model")

                # --- PHASE 5: Resilience Guardians ---
                await self._init_system_guardians()
                await self._init_resilience()
                self._initialize_self_preservation()
                boot_profiler.mark("guardians_and_resilience")

                # --- PHASE 5.2: Homeostate convergence (declared runtime baseline) ---
                # Salt-style desired-state engine: converge the runtime baseline
                # once at boot, then keep the degradation beacon + reactor live
                # for event-driven re-convergence. Never blocks or fails boot.
                if not lightweight_test_boot:
                    from core.runtime.homeostate import start_homeostate_runtime

                    homeostate_summary = await start_homeostate_runtime()
                    if homeostate_summary.get("ok"):
                        logger.info(
                            "✅ [BOOT] Homeostate baseline converged (changed=%s failed=%s); beacon+reactor live.",
                            homeostate_summary.get("baseline_changed"),
                            homeostate_summary.get("baseline_failed"),
                        )
                    else:
                        logger.warning(
                            "⚠️ [BOOT] Homeostate runtime unavailable: %s",
                            homeostate_summary.get("error"),
                        )
                boot_profiler.mark("homeostate")

                # --- PHASE 5.5: Unitary Kernel Interface ---
                from core.kernel.kernel_interface import KernelInterface

                await KernelInterface.attach_to_orchestrator(self)
                boot_profiler.mark("kernel_interface")

                # --- PHASE 6: Skill System & Mycelium ---
                await self._init_skill_system()
                boot_profiler.mark("skill_system")

                # --- PHASE 6.5: Capability Engine & Desktop Agency Boot ---
                try:
                    from core.capabilities import boot_capabilities
                    await boot_capabilities()
                    logger.info("✅ [BOOT] Desktop agency capabilities booted successfully.")
                except (ImportError, AttributeError, RuntimeError, TypeError, OSError) as cap_err:
                    _record_boot_degradation(
                        cap_err,
                        action="continued boot with degraded capabilities",
                        severity="error"
                    )
                    logger.error("⚠️ [BOOT] Failed to boot desktop agency capabilities: %s", cap_err, exc_info=True)

                from core.mycelium import MycelialNetwork

                mycelium = ServiceContainer.get("mycelial_network", default=None)
                if not mycelium:
                    mycelium = MycelialNetwork()
                    ServiceContainer.register_instance("mycelial_network", mycelium)
                mycelium.establish_connection("system", "core_logic", priority=1.0)
                mycelium.establish_connection("core_logic", "skill_execution", priority=1.0)
                mycelium.establish_connection("personality", "cognition", priority=1.0)

                async def mycelium_ui_delivery(msg: str):
                    await self.emit_spontaneous_message(msg, modality="both")

                mycelium.set_ui_callback(mycelium_ui_delivery)

                from .initializers.pathways import register_core_pathways

                register_core_pathways(mycelium)
                mycelium.establish_unification_hyphae()
                mycelium.establish_consciousness_hyphae()

                proof_boot_active = False
                try:
                    from core.runtime.proof_policy import proof_run_active

                    proof_boot_active = proof_run_active(origin="orchestrator_boot")
                except (ImportError, AttributeError, RuntimeError):
                    proof_boot_active = os.getenv("AURA_PROOF_RUN", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }

                if not lightweight_test_boot and not proof_boot_active:
                    _spawn_boot_task(
                        mycelium.pulse_check(),
                        "orchestrator.mycelium.pulse_check",
                    )

                boot_profiler.mark("capabilities_and_mycelium")

                # Phase 5: Supplementary Deep Hardening (Claude Feedback)
                # Ensure cognitive core is ready before marking initialized
                await self._init_cognitive_core()
                boot_profiler.mark("cognitive_core")

                await self._init_sovereign_scanner()
                boot_profiler.mark("sovereign_scanner")

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
                _spawn_boot_task(
                    self._init_autonomous_evolution(), "orchestrator.init_autonomous_evolution"
                )
                _spawn_boot_task(self._init_react_loop(), "orchestrator.init_react_loop")
                _spawn_boot_task(self._init_metabolism(), "orchestrator.init_metabolism")
                _spawn_boot_task(
                    self._init_proactive_systems(), "orchestrator.init_proactive_systems"
                )
                # Phase 32: Lazarus Protocol Heartbeat
                _spawn_boot_task(
                    self._cognitive_heartbeat_task(), "orchestrator.cognitive_heartbeat"
                )

                # Step 3 & 4: (DEFERRED)
                async def _final_steps():
                    # Step 3: Consciousness & Logic Integration
                    if hasattr(self, "substrate") and self.substrate:
                        await self.substrate.start()

                    await self._integrate_systems()

                    # Step 4: State Recovery & Persistence
                    load_fn = getattr(self, "_load_state", lambda: None)
                    wal_fn = getattr(self, "_recover_wal_state", lambda: asyncio.sleep(0))
                    drift_fn = getattr(self, "_calculate_temporal_drift", lambda: None)

                    await asyncio.gather(
                        asyncio.to_thread(load_fn), wal_fn(), asyncio.to_thread(drift_fn)
                    )

                # Start Memory Defragmenter
                try:
                    from core.memory.semantic_defrag import start_defrag_scheduler

                    _spawn_boot_task(
                        start_defrag_scheduler(), "orchestrator.semantic_defrag_scheduler"
                    )
                except ImportError as _e:
                    logger.debug("Ignored ImportError in boot.py: %s", _e)

                # Start Cognitive Loop Service (Skipped in Skeletal Mode)
                if not config.skeletal_mode:
                    from core.cognition.cognitive_loop import CognitiveLoop

                    self.cognitive_loop = CognitiveLoop(self)
                    try:
                        await asyncio.wait_for(self.cognitive_loop.start(), timeout=10.0)
                        logger.info("🧠 Cognitive Loop started.")
                    except TimeoutError:
                        logger.error("🛑 Cognitive Loop boot TIMEOUT.")
                    except asyncio.CancelledError:
                        raise
                    except (RuntimeError, AttributeError) as e:
                        _record_boot_degradation(
                            e,
                            action="continued boot with cognitive loop unavailable",
                            severity="degraded",
                        )
                        logger.error("❌ Cognitive Loop failed: %s", e)
                    ServiceContainer.register_instance("cognitive_loop", self.cognitive_loop)
                else:
                    logger.info("💀 Skeletal Mode: Cognitive Loop initialization skipped.")
                    ServiceContainer.register_instance("cognitive_loop", None)

                boot_profiler.mark("cognitive_loop_start")

                # Start UPSO MindTick (Phase 2) (Skipped in Skeletal Mode)
                if not config.skeletal_mode:
                    tick = getattr(self, "mind_tick", None)
                    if tick and hasattr(tick, "start"):
                        try:
                            await asyncio.wait_for(tick.start(), timeout=10.0)
                            logger.info("💓 MindTick: Unified cognitive rhythm online.")
                        except TimeoutError:
                            logger.error("🛑 MindTick boot TIMEOUT.")
                        except asyncio.CancelledError:
                            raise
                        except (RuntimeError, AttributeError) as e:
                            _record_boot_degradation(
                                e,
                                action="continued boot with MindTick rhythm unavailable",
                                severity="degraded",
                            )
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
                    except TimeoutError:
                        logger.error("🛑 Memory Governor TIMEOUT.")
                    except asyncio.CancelledError:
                        raise
                    except (RuntimeError, AttributeError) as e:
                        _record_boot_degradation(
                            e,
                            action="continued boot with memory governor unavailable",
                            severity="degraded",
                        )
                        logger.error("❌ Memory Governor failed: %s", e)
                    ServiceContainer.register_instance("memory_governor", gov)

                # Out-of-band memory watchdog: a daemon thread that keeps
                # enforcing RSS/swap ceilings even when the event loop is
                # wedged (the in-loop governor goes blind exactly when a
                # swap spiral stalls the loop).
                try:
                    from core.resilience.memory_watchdog import start_memory_watchdog

                    self.memory_watchdog = start_memory_watchdog(
                        loop=asyncio.get_running_loop(),
                        governor=gov,
                    )
                    ServiceContainer.register_instance(
                        "memory_watchdog", self.memory_watchdog
                    )
                    try:
                        from core.runtime.shutdown_coordinator import (
                            get_shutdown_coordinator,
                        )

                        get_shutdown_coordinator().register(
                            self.memory_watchdog.stop,
                            phase="task_supervisor",
                            name="memory_watchdog.stop",
                            timeout=5.0,
                        )
                    except (ImportError, RuntimeError, ValueError, TypeError) as e:
                        _record_boot_degradation(
                            e,
                            action="continued boot without memory watchdog shutdown hook",
                            severity="warning",
                        )
                    logger.info("🛡️ Memory Watchdog started (out-of-band).")
                except (ImportError, RuntimeError, AttributeError, TypeError) as e:
                    _record_boot_degradation(
                        e,
                        action="continued boot without out-of-band memory watchdog",
                        severity="degraded",
                    )
                    logger.error("❌ Memory Watchdog failed: %s", e)

                boot_profiler.mark("mind_tick_and_memory_guardians")

                # Start Prometheus Metrics (Phase 3)
                try:
                    self.metrics_exporter = MetricsExporter(port=9090)
                    await self.metrics_exporter.start()
                    ServiceContainer.register_instance("metrics_exporter", self.metrics_exporter)
                except (ImportError, ModuleNotFoundError) as e:
                    logger.warning("📈 [BOOT] Metrics Exporter skipped: %s", e)
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    _record_boot_degradation(
                        e,
                        action="continued boot without metrics exporter",
                    )
                    logger.error("📈 [BOOT] Metrics Exporter failed to start: %s", e)

                # Phase 4 Advanced Features (Unified Meta-Cognition & Resilience Shards)
                try:
                    from core.orchestrator.meta_cognition_shard import MetaCognitionShard

                    metacog = MetaCognitionShard(self)
                    metacog.start()
                    ServiceContainer.register_instance("meta_cognition_shard", metacog)
                    self.meta_cognition = metacog
                    logger.info("🧠 Meta-Cognition Shard initialized and started.")
                except (ImportError, AttributeError, RuntimeError) as e:
                    _record_boot_degradation(
                        e,
                        action="continued boot without meta-cognition shard",
                        severity="degraded",
                    )
                    logger.error("🛑 Failed to init Meta-Cognition Shard: %s", e)

                _seam_early_response = await _skip_the_background_subsystems_in_foreground_only(
                    self=self,
                )
                if _seam_early_response is not _SEAM_FELL_THROUGH:
                    return _seam_early_response

                try:
                    # Incident Narrator: receipt-backed synthesis of Aura's own
                    # forensics (stall dumps, degraded events, sentinel ring,
                    # boot profile) into causal narratives — serves the
                    # /system/incidents endpoint and grounded "why were you
                    # slow?" self-reports on the conversation lane.
                    from core.observability.incident_narrator import get_incident_narrator

                    await get_incident_narrator().start()
                except (ImportError, AttributeError, RuntimeError) as e:
                    _record_boot_degradation(
                        e,
                        action="continued boot without incident narrator",
                        severity="warning",
                    )
                    logger.warning("Incident narrator unavailable: %s", e)

                try:
                    # Black-box flight recorder (roadmap A5): opens this
                    # boot's crash-survivable mind-moment ring; if the
                    # previous run died without a clean shutdown, extracts
                    # its last recorded moments into a governed death report
                    # for the narrator and the waking sequence.
                    from core.runtime.flight_recorder import get_flight_recorder

                    death_report = await get_flight_recorder().start()
                    if death_report:
                        logger.warning(
                            "🛬 Previous run ended uncleanly — %s",
                            death_report.get("narrative", "death report recovered"),
                        )
                except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as e:
                    _record_boot_degradation(
                        e,
                        action="continued boot without flight recorder",
                        severity="warning",
                    )
                    logger.warning("Flight recorder unavailable: %s", e)

                self.hotfix_engine = HotfixEngine(self)
                ServiceContainer.register_instance("hotfix_engine", self.hotfix_engine)

                from core.collective.delegator import AgentDelegator

                self.swarm_protocol = SwarmProtocol()
                ServiceContainer.register_instance("swarm_protocol", self.swarm_protocol)

                delegator = ServiceContainer.get("agent_delegator", default=None)
                if delegator is None or not hasattr(delegator, "delegate_debate"):
                    delegator = AgentDelegator(orchestrator=self)
                    ServiceContainer.register_instance("agent_delegator", delegator)
                self.agent_delegator = delegator
                self.swarm = delegator
                ServiceContainer.register_instance("swarm", delegator)
                # The Mycelium is the sole owner of mapping scheduling.
                # Foreground policy defers the repo-wide AST scan and setup()
                # is idempotent when another caller already requested it.
                mapping_scheduled = mycelium.setup()
                logger.info(
                    "🍄 [MYCELIUM] Infrastructure mapping state: %s.",
                    "scheduled"
                    if mapping_scheduled
                    else mycelium.get_infrastructure_report()["mapping_state"],
                )

                logger.info("🛡️ [ORCHESTRATOR] Subsystems synchronously initialized.")

                # Bring health-contract important services online before the
                # formal boot verdict is emitted. These are not decorative:
                # compute allocation, reaping, hypervision, and loop-lag
                # monitoring define whether Aura can survive a long run.
                try:
                    from core.agency.compute_orchestrator import get_compute_orchestrator

                    self.compute_orchestrator = get_compute_orchestrator()
                    ServiceContainer.register_instance(
                        "compute_orchestrator",
                        self.compute_orchestrator,
                    )
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    _record_boot_degradation(
                        exc,
                        action="left compute orchestrator unavailable so health contract can fail honestly",
                        severity="critical",
                    )
                    logger.error("ComputeOrchestrator boot failed: %s", exc)

                try:
                    from core.orchestrator.initializers.hardening import init_hardening_layer

                    await init_hardening_layer(self)
                except asyncio.CancelledError:
                    raise
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    _record_boot_degradation(
                        exc,
                        action="left hardening supervisors unavailable so health contract can fail honestly",
                        severity="critical",
                    )
                    logger.error("Hardening supervisors failed to start: %s", exc)

                try:
                    delegator = ServiceContainer.get("agent_delegator", default=None)
                    if delegator and hasattr(delegator, "start"):
                        start_result = delegator.start()
                        if asyncio.iscoroutine(start_result):
                            await asyncio.wait_for(start_result, timeout=15.0)
                        self.agent_delegator = delegator
                except asyncio.CancelledError:
                    raise
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                    _record_boot_degradation(
                        exc,
                        action="continued boot with agent delegator unavailable; optional health check will report it",
                        severity="warning",
                    )
                    logger.warning("AgentDelegator early start failed: %s", exc)

                # Swarm Protocol start moved to proactive systems (v26.3 Unified)

                _spawn_boot_task(_final_steps(), "orchestrator.final_steps")

                # ── Canonical Scheduler Heartbeat ───────────────────────
                # The health contract treats scheduler liveness as a critical
                # runtime probe. Start the scheduler before the contract is
                # evaluated so boot cannot report healthy on a mere heartbeat.
                try:
                    from core.scheduler import scheduler

                    ServiceContainer.register_instance("scheduler", scheduler, required=False)
                    await asyncio.wait_for(scheduler.start(), timeout=5.0)
                    if not scheduler.is_alive():
                        raise RuntimeError("scheduler start returned without live main loop")
                    logger.info("✓ Scheduler heartbeat active before health contract.")
                except asyncio.CancelledError:
                    raise
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                    _record_boot_degradation(
                        exc,
                        action="left scheduler unavailable so runtime health contract can fail closed",
                        severity="critical",
                    )
                    logger.error("Scheduler heartbeat failed before health contract: %s", exc)

                # ── Runtime Health Contract ───────────────────────────
                # Evaluate the formal health contract that defines what
                # MUST be alive vs what's optional enrichment.
                try:
                    from core.runtime.health_contract import (
                        HealthLevel,
                        evaluate_health,
                        log_health_report,
                    )

                    runtime_ready_for_health_log = bool(
                        getattr(self.status, "initialized", False)
                        and getattr(self.status, "running", False)
                    )
                    verdict = log_health_report() if runtime_ready_for_health_log else evaluate_health()
                    log_level, message = _health_contract_boot_log(
                        verdict.level,
                        initialized=bool(getattr(self.status, "initialized", False)),
                        running=bool(getattr(self.status, "running", False)),
                    )
                    logger.log(log_level, message)
                    if not runtime_ready_for_health_log and verdict.critical_failures:
                        pending = [
                            status.requirement.container_key
                            for status in verdict.critical_failures
                        ]
                        logger.info(
                            "⏳ HEALTH CONTRACT DETAIL: boot pending critical liveness=%s",
                            pending,
                        )
                    if verdict.level in (HealthLevel.DEAD, HealthLevel.CRITICAL):
                        self.status.healthy = False
                except (ImportError, AttributeError, RuntimeError) as hc_err:
                    _record_boot_degradation(
                        hc_err,
                        action="marked runtime unhealthy because health contract evaluation failed",
                        severity="critical",
                    )
                    self.status.healthy = False
                    logger.error("Health contract evaluation failed: %s", hc_err)

                # ── Startup Validation ────────────────────────────────
                # Always mark initialized BEFORE validation to prevent
                # _boot_lock deadlock if start() is called again. Validation failures
                # set healthy=False for degraded mode instead of returning early.
                self.status.initialized = True

                from core.resilience.startup_validator import StartupValidator

                validator = StartupValidator(self)
                is_safe = await validator.validate_all()
                if not is_safe:
                    logger.critical(
                        "🚨 STARTUP VALIDATION FAILED. Entering DEGRADED mode (not deadlocking)."
                    )
                    self.status.healthy = False
                    # Continue — do NOT return. Deadlocking here blocks the entire system.
                else:
                    logger.info("✅ Startup validation complete; final runtime health check pending.")

                # --- UPSO Phase 1: Post-Boot State Commit ---
                try:
                    state = await self.state_repo.get_current()
                    if state:
                        await self.state_repo.commit(state.derive("online"), "online")
                        logger.info("💾 UPSO: Online state committed.")
                    else:
                        logger.warning("⚠️ UPSO: No state found to commit online.")
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    _record_boot_degradation(
                        e,
                        action="continued boot after online UPSO state commit failed",
                        severity="degraded",
                    )
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
                                should_auto_listen = bool(
                                    getattr(ve, "auto_listen_enabled", False)
                                    and getattr(ve, "microphone_enabled", False)
                                )
                        if should_auto_listen:
                            success = await ve.start_listening()
                            if success:
                                logger.info(
                                    "🎙️ Voice capture auto-started (server-side sounddevice)"
                                )
                            else:
                                logger.warning(
                                    "🎙️ Voice capture failed to auto-start — will retry on demand"
                                )
                        else:
                            logger.info(
                                "🎙️ Voice capture deferred. Mic will start only after explicit enablement."
                            )
                    except (ImportError, AttributeError, RuntimeError) as e:
                        _record_boot_degradation(
                            e,
                            action="continued boot with voice capture deferred to on-demand retry",
                        )
                        logger.warning("🎙️ Voice auto-start skipped: %s", e)

                # Swarm Protocol start moved to proactive systems (v26.3 Unified)
                # ── Immune System Post-Boot Scan ─────────────────
                try:
                    immune = ServiceContainer.get("immune_system", default=None)
                    if immune and hasattr(immune, "post_boot_scan"):
                        await immune.post_boot_scan(self)
                except (ImportError, AttributeError, RuntimeError) as scan_err:
                    _record_boot_degradation(
                        scan_err,
                        action="continued boot without immune post-boot scan",
                        severity="degraded",
                    )
                    logger.warning("Immune post-boot scan failed: %s", scan_err)

                # ── Final Success State ──────────────────────────
                self.status.initialized = True
                try:
                    self.status.healthy = bool(self.health_check())
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as final_health_err:
                    _record_boot_degradation(
                        final_health_err,
                        action="marked boot unhealthy because final runtime health check failed",
                        severity="critical",
                    )
                    self.status.healthy = False
                    logger.error("Final boot health check failed: %s", final_health_err)

                if self.status.healthy:
                    logger.info("✅ BOOT COMPLETE: System fully initialized.")
                else:
                    try:
                        from core.runtime.health_contract import runtime_health_report

                        contract = runtime_health_report()
                        log_level, message = _final_boot_health_log(
                            contract,
                            initialized=bool(getattr(self.status, "initialized", False)),
                            running=bool(getattr(self.status, "running", False)),
                        )
                        logger.log(log_level, message)
                    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                        _record_boot_degradation(
                            exc,
                            action="could not classify final boot readiness failure",
                            severity="error",
                        )
                        logger.warning("⚠️ BOOT COMPLETE: System initialized in degraded mode.")

            except (ImportError, AttributeError, RuntimeError) as e:
                _record_boot_degradation(
                    e,
                    action="marked boot initialized but unhealthy after recoverable boot failure",
                    severity="critical",
                )
                logger.error("BOOT ENCOUNTERED ISSUES (Recovering...): %s", e, exc_info=True)
                self.status.add_error(str(e))
                # IMMORTAL BOOT: We still mark as initialized if core components are likely to run
                boot_profiler.mark("meta_healing_and_boot_tail")
                self.status.initialized = True
                self.status.healthy = False
                logger.warning("⚠️ BOOT: Entering degraded state. Cycle starting despite errors.")


async def boot_orchestrator(*, headless: bool = True, skip_gui: bool = True, **kwargs: Any) -> Any:
    """Compatibility boot helper used by legacy headless tests.

    The canonical factory lives in ``core.orchestrator.main``. This shim
    preserves the old import while delegating to the existing orchestrator
    factory instead of introducing a second boot path.
    """
    from core.orchestrator.main import create_orchestrator

    orchestrator = create_orchestrator(**kwargs)
    setup = getattr(orchestrator, "setup", None)
    status = getattr(orchestrator, "status", None)
    if status is None and callable(setup):
        setup()
    if skip_gui:
        orchestrator._skip_gui = True
    orchestrator._headless = bool(headless)
    return orchestrator
