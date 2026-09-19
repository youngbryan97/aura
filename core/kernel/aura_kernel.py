from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import sys  # noqa: F401  (read at call time by the lifted module)
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from core.consciousness.executive_authority import (
    get_executive_authority,  # noqa: F401  (read at call time by the lifted module)
)
from core.container import ServiceContainer  # noqa: F401  (read at call time by the lifted module)
from core.introspection.thought_tracer import tracer
from core.kernel.bridge import LegacyPhase
from core.kernel.organs import OrganStub
from core.kernel.upgrades_10x import (
    EternalGrowthEngine,  # noqa: F401  (read at call time by the lifted module)
    EternalMemoryPhase,
    GodModeToolPhase,
    NativeMultimodalBridge,
    PerfectEmotionPhase,
    TrueEvolutionPhase,  # noqa: F401  (read at call time by the lifted module)
)
from core.phases.affect_update import AffectUpdatePhase
from core.phases.bonding_phase import (
    BondingPhase,  # noqa: F401  (read at call time by the lifted module)
)
from core.phases.cognitive_integration_phase import CognitiveIntegrationPhase
from core.phases.cognitive_routing_unitary import CognitiveRoutingPhase
from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase
from core.phases.inference_phase import InferencePhase
from core.phases.learning_phase import (
    LearningPhase,  # noqa: F401  (read at call time by the lifted module)
)
from core.phases.motivation_update import MotivationUpdatePhase
from core.phases.phi_consciousness import PhiConsciousnessPhase
from core.phases.repair_phase import (
    RepairPhase,  # noqa: F401  (read at call time by the lifted module)
)
from core.phases.response_generation_unitary import UnitaryResponsePhase
from core.phases.unity_binding import UnityBindingPhase
from core.resilience.error_boundary import wrap_phase
from core.runtime.cognitive_provenance import (  # noqa: F401  (read at call time by the lifted module)
    begin_transformation,
    close_tick,
    open_tick,
)
from core.runtime.errors import record_degradation
from core.runtime.pipeline_blueprint import (
    bind_legacy_runtime_phase_attributes,
    kernel_phase_attribute_order,
    resolve_phase_instances,
)
from core.runtime.shutdown_coordinator import (
    is_shutdown_requested,  # noqa: F401  (read at call time by the lifted module)
)
from core.self_modification.boot_validator import GhostBootValidator
from core.state.aura_state import AuraState
from core.state.state_repository import StateRepository
from core.utils.concurrency import RobustLock
from core.utils.task_tracker import (
    get_task_tracker,  # noqa: F401  (read at call time by the lifted module)
)

from .aura_kernel_lifecycle import _TicksAndShutsDown
from .feedback_observer import FeedbackObserver, TickEntry
from .self_review import SelfReviewPhase  # noqa: F401  (read at call time by the lifted module)
from .shadow_kernel import ShadowExecutionPhase


class KernelStatus(BaseModel):
    running: bool = False
    initialized: bool = False
    cycle_count: int = 0
    message: str = "Standby"


logger = logging.getLogger("Aura.Core.Kernel")


@dataclass(frozen=True)
class _KernelFileWriteDecision:
    receipt_id: str
    domain: str
    source: str


def _kernel_file_write_decision(source: str) -> _KernelFileWriteDecision:
    safe_source = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", source.strip() or "aura_kernel")
    return _KernelFileWriteDecision(
        receipt_id=f"kernel-file-write:{safe_source}:{time.time_ns()}",
        domain="file_write",
        source=safe_source,
    )


_KERNEL_OPTIONAL_PERCEPTION_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _pass_instrumentation():
    """The process-wide pass instrumentation (core/pipeline/pass_manager.py).

    Imported lazily so the kernel keeps no import-time dependency on the
    pass machinery, and so a broken instrumentation module degrades to a
    no-op rather than taking the tick loop with it.
    """
    try:
        from core.pipeline.pass_manager import get_instrumentation

        return get_instrumentation()
    except Exception:  # noqa: BLE001 — degrade to a no-op, never to a broken tick
        logger.debug("pass instrumentation unavailable", exc_info=True)
        return _NullInstrumentation()


class _NullInstrumentation:
    """Fallback so the tick loop's contract holds even with no instrumentation."""

    @staticmethod
    def should_run(name: str) -> tuple[bool, int, str]:
        return True, 0, ""


def _begin_pass_run(label: str) -> None:
    """Number this tick's passes from 1, or do nothing if unavailable.

    Wrapped rather than trusting ``_pass_instrumentation`` to be total: a
    debugging aid must never be the reason a tick fails.
    """
    try:
        begin = getattr(_pass_instrumentation(), "begin_run", None)
        if begin is not None:
            begin(label)
    except Exception:  # noqa: BLE001 — a debug aid may never break a tick
        logger.debug("pass run label %s not recorded", label, exc_info=True)


def _record_pass(
    name: str,
    ordinal: int,
    duration_s: float,
    *,
    skipped: bool,
    reason: str = "",
    error: str = "",
) -> None:
    try:
        from core.pipeline.pass_manager import PassRecord, get_instrumentation

        get_instrumentation().after_pass(
            PassRecord(
                name=name,
                ordinal=ordinal,
                duration_s=duration_s,
                skipped=skipped,
                reason=reason,
                error=error,
            )
        )
    except Exception:  # noqa: BLE001 — instrumentation never breaks a tick
        logger.debug("pass record failed for %s", name, exc_info=True)


def _record_kernel_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("aura_kernel", exc, severity=severity, action=action)


@dataclass(frozen=True)
class KernelConfig:
    max_concurrent_phases: int = 4
    watchdog_timeout_s: float = 240.0
    state_versioning: bool = True
    mirror_frequency: float = 1.0  # Hz


class MirrorSnapshot(BaseModel):
    """Immutable projection of internal state for GUI consumption."""

    version: int
    vitality: float
    mood: str
    curiosity: float
    phi: float
    last_objective: str
    timestamp: float


async def _await_phase_completion(task, *, phase_name, priority, origin, budget_s):
    """Wait for a phase, with the budget the caller is allowed.

    A module function rather than a method: it reads no instance state, and
    AuraKernel is close enough to the size ratchet's ceiling that a helper
    which does not need to be on the class should not be.
    """
    from core.runtime.turn_origin import a_person_is_waiting

    if (
        priority
        and phase_name in {"UnitaryResponsePhase", "ResponseGenerationPhase"}
        and a_person_is_waiting(origin)
    ):
        from core.brain.llm_health_router import _await_while_it_is_working

        return await _await_while_it_is_working(
            task, budget_s=budget_s, user_facing=True, person_is_waiting=True,
        )
    return await asyncio.wait_for(task, timeout=budget_s)


class AuraKernel(_TicksAndShutsDown):
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
        self._shutdown_lock = asyncio.Lock()
        self._shutdown_complete = False
        self._shutdown_process_runtime_owner: bool | None = None

        # Pipelines & Supervision
        self._task_group: asyncio.TaskGroup | None = None
        self._phases: list[Any] = []
        self._services: dict[type, Any] = {}
        self._background_tasks: list[asyncio.Task] = []

        # Pulse-Mirroring Pattern
        self._mirror_state: MirrorSnapshot | None = None
        self._gui_queue: asyncio.Queue = asyncio.Queue(maxsize=32)

        # Organ boot registry
        self.organs: dict[str, OrganStub] = {}  # Populated in boot()

        # [10X] Phase Singletons
        self.eternal = EternalMemoryPhase(self)
        self.evolution = TrueEvolutionPhase(
            self, engine=None
        )  # Engine resolved via property lazy-loading
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
        self.unity_phase = UnityBindingPhase(self)
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
        self.routing_phase = CognitiveRoutingPhase(self)
        self.response_phase = UnitaryResponsePhase(self)

        # Feedback Observer
        self.feedback_observer = FeedbackObserver()

        # Evidence-bounded self-review and boot verification.
        self._boot_validator = GhostBootValidator(Path("."))
        self._auto_fix_engine = None
        self._guardian = None
        self._lock = RobustLock("AuraKernel.StateLock")
        self.volition_level: int = (
            3  # 0=Lockdown, 1=Reflective, 2=Perceptive, 3=Agentic [DEFAULT]
        )
        # Priority preemption: background ticks yield when a user message is waiting
        import threading as _threading

        self._user_priority_pending: _threading.Event = _threading.Event()
        self._last_tick_completed_at: float = 0.0  # telemetry: set after each tick()
        self._phase_runtime_samples: deque[dict[str, Any]] = deque(maxlen=256)

    @staticmethod
    def _normalize_origin(origin: Any) -> str:
        return str(origin or "").strip().lower().replace("-", "_")

    @classmethod
    def _is_user_facing_origin(cls, origin: Any) -> bool:
        normalized = cls._normalize_origin(origin)
        if not normalized:
            return False
        if normalized in {
            "user",
            "voice",
            "admin",
            "api",
            "gui",
            "ws",
            "websocket",
            "direct",
            "external",
            "benchmark",
        }:
            return True
        tokens = {token for token in normalized.split("_") if token}
        return bool(
            tokens
            & {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external", "benchmark"}
        )

    def _finalize_foreground_turn_state(self, *, objective: str, turn_origin: str) -> None:
        from core.goals.objective_lifecycle import finalize_foreground_turn_state

        receipt = finalize_foreground_turn_state(
            self.state,
            objective=objective,
            origin=turn_origin,
        )
        closure = ServiceContainer.get("executive_closure", default=None)
        if closure is not None and hasattr(closure, "complete_foreground_turn"):
            closure.complete_foreground_turn(objective, turn_origin)
        if receipt.get("preserved_background"):
            logger.debug(
                "Kernel: preserved a post-turn background objective after closing %s.",
                receipt.get("objective_digest") or "foreground turn",
            )

    def _phase_timeout_seconds(self, phase_name: str, *, priority: bool) -> float:
        """Give foreground response generation enough headroom without letting background stalls monopolize the lock.

        Priority turns must protect the foreground lane. They keep generous
        headroom for actual response generation, but all non-response phases
        get tight budgets so a single introspection or consolidation phase
        cannot monopolize the kernel lock and starve chat.
        """
        if not priority:
            if phase_name in {"UnitaryResponsePhase", "ResponseGenerationPhase"}:
                # [STABILITY] Embodied control needs more than the standard 12s background cap
                # even if it's not a 'priority' tick, as the generative action is the goal.
                cognition = getattr(self.state, "cognition", None) if self.state else None
                objective = str(getattr(cognition, "current_objective", "") or "").lower()
                if "[embodied control contract]" in objective:
                    return 60.0
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
                return 210.0
            return 180.0
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
        """Keep user-facing ticks lean without suppressing explicit tool/task execution.

        This is the mechanism behind Aura's two rates, and the set it consults
        lives in core/runtime/pipeline_blueprint.py rather than inline here.
        Kept inline, nothing outside the kernel could read which phases a user
        turn actually runs — so every description of the pipeline was written
        from the blueprint's length, and "29 phases per turn" is what that
        produced. It is closer to eleven, and the rest run on MindTick's
        background pass over the same kernel.
        """
        if not priority:
            return False

        from core.runtime.pipeline_blueprint import BACKGROUND_ONLY_PHASES

        if phase_name not in BACKGROUND_ONLY_PHASES:
            return False

        if phase_name == "GodModeToolPhase":
            response_modifiers = getattr(self.state, "response_modifiers", {}) if self.state else {}
            intent_type = str(response_modifiers.get("intent_type", "") or "").upper()
            return intent_type not in {"SKILL", "TASK"}

        return True

    async def _execute_phase_with_timing(
        self,
        phase: Any,
        phase_name: str,
        entry: TickEntry,
        *,
        objective: str,
        priority: bool,
    ) -> AuraState:
        """Run one phase and retain attributable latency on every terminal path."""
        started = time.perf_counter()
        # The provenance seam. Every kernel phase passes through here, so this
        # is where the state is digested before and after — measured around the
        # phase rather than reported by it, which is the difference between a
        # causal record and a phase asserting it behaved.
        transformation = begin_transformation(phase_name, self.state)
        phase_error = ""
        result_state = self.state
        try:
            result_state = await wrap_phase(
                phase_name,
                phase.execute,
                self.state,
                objective=objective,
                priority=priority,
            )
            return result_state
        except BaseException as exc:  # noqa: BLE001 — recorded, then re-raised
            phase_error = repr(exc)
            raise
        finally:
            try:
                transformation.complete(
                    result_state,
                    error=phase_error,
                    inputs={"objective": objective[:120], "priority": bool(priority)},
                )
            except Exception as exc:  # noqa: BLE001 — provenance never breaks a tick
                logger.debug("provenance receipt failed for %s: %s", phase_name, exc)
            duration_ms = (time.perf_counter() - started) * 1000.0
            rounded_ms = round(duration_ms, 3)
            phase_durations = getattr(entry, "phase_durations_ms", None)
            if not isinstance(phase_durations, dict):
                phase_durations = {}
                entry.phase_durations_ms = phase_durations
            phase_durations[phase_name] = rounded_ms
            sample = {
                "timestamp": time.time(),
                "phase": phase_name,
                "duration_ms": rounded_ms,
                "priority": bool(priority),
                "objective": objective[:120],
            }
            self._phase_runtime_samples.append(sample)

            default_budget_ms = 5000.0 if priority else 1500.0
            try:
                budget_ms = float(
                    os.getenv("AURA_KERNEL_SLOW_PHASE_MS", str(default_budget_ms))
                )
            except (TypeError, ValueError):
                budget_ms = default_budget_ms
            if duration_ms > max(1.0, budget_ms):
                logger.warning(
                    "Kernel phase latency exceeded budget: phase=%s duration_ms=%.1f "
                    "budget_ms=%.1f priority=%s",
                    phase_name,
                    duration_ms,
                    budget_ms,
                    priority,
                )

    def phase_runtime_status(self, *, window_s: float = 300.0) -> dict[str, Any]:
        """Return bounded, attributable phase latency evidence for proof surfaces."""
        now = time.time()
        samples = [
            dict(sample)
            for sample in self._phase_runtime_samples
            if now - float(sample.get("timestamp", now)) <= max(1.0, window_s)
        ]
        if not samples:
            return {
                "sample_count": 0,
                "slowest_phase": None,
                "slowest_duration_ms": 0.0,
                "latest": [],
            }
        slowest = max(samples, key=lambda sample: float(sample.get("duration_ms", 0.0)))
        return {
            "sample_count": len(samples),
            "slowest_phase": str(slowest.get("phase", "")),
            "slowest_duration_ms": float(slowest.get("duration_ms", 0.0)),
            "latest": samples[-32:],
        }

    async def _record_dream_fragment(self, objective: str, phase: Any, phase_name: str) -> None:
        """Record a dream fragment (interrupted background cognition) to disk for offline dreaming consolidation.

        Written on the async lane. Every caller is a preemption path — a user
        request is already waiting on the kernel lock — so this was a blocking
        append plus fsync at the exact moment the loop had somebody waiting on
        it. The yield was supposed to be the fast part.
        """
        try:
            from core.config import config
            from core.governance_context import governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            fragment_file = config.paths.data_dir / "dream_fragments.jsonl"
            completed_index = self._phases.index(phase) if phase in self._phases else -1
            completed_phases = (
                [p.__class__.__name__ for p in self._phases[: completed_index + 1]]
                if completed_index != -1
                else []
            )
            affect = getattr(self.state, "affect", None) if self.state else None
            metabolism = getattr(self.state, "metabolism", None) if self.state else None
            fragment_entry = {
                "timestamp": time.time(),
                "objective": objective,
                "preempted_at_phase": phase_name,
                "completed_phases": completed_phases,
                "state_snapshot": {
                    "phi": getattr(self.state, "phi", 0.0) if self.state else 0.0,
                    "valence": getattr(affect, "valence", 0.0) if affect else 0.0,
                    "energy": getattr(metabolism, "energy", 1.0) if metabolism else 1.0,
                },
            }
            source = "kernel.preemption.dream_fragment"
            async with governed_scope(_kernel_file_write_decision(source)):
                await get_file_write_gateway().append_text_async(
                    fragment_file,
                    json.dumps(fragment_entry) + "\n",
                    source=source,
                )
            logger.info("🌙 Registered preempted background tick as a dream fragment for offline consolidation.")
        except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
            _record_kernel_degradation(
                exc,
                action="continued after preempted dream fragment persistence failed",
                severity="warning",
            )
            logger.debug("Failed to write dream fragment: %s", exc)

    def _spawn_background_task(self, coro: Any, *, name: str) -> asyncio.Task:
        """Create a supervised kernel-owned background task and retain it for shutdown/restart handling."""
        try:
            from core.utils.task_tracker import get_task_tracker

            task = get_task_tracker().create_task(coro, name=name)
        except (ImportError, AttributeError, RuntimeError):
            task = get_task_tracker().create_task(coro, name=name)
            try:
                task._aura_supervised = True
                task._aura_task_tracker = "AuraKernel"
            except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                _record_kernel_degradation(
                    _exc,
                    action="created background task without AuraKernel supervision metadata",
                )
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
                        cognitive_engine=cog_engine, code_base_path="."
                    )
                    logger.info("🧬 [SELF-REPAIR] AutonomousSelfModificationEngine initialized.")
                else:
                    logger.debug("⚠️ [SELF-REPAIR] LLM engine not found for SME initialization.")
            except (ImportError, AttributeError, RuntimeError) as e:
                _record_kernel_degradation(
                    e,
                    action="left autonomous self-modification engine unavailable after initialization failed",
                )
                logger.error("❌ [SELF-REPAIR] SME initialization failed: %s", e)
        return self._auto_fix_engine

    def set_volition_level(self, level: int):
        """
        Sets the system volition level (0-3).
        Updates the SubstrateGovernor frequency scaling.
        """
        old_level = self.volition_level
        self.volition_level = max(0, min(3, level))

        logger.info("🔥 [VOLITION] Level shifted: %d -> %d", old_level, self.volition_level)

        # Update governor if available
        gov = ServiceContainer.get("substrate_governor", default=None)
        if gov:
            try:
                # We assume the governor will be updated to handle this
                gov.apply_volition_profile(self.volition_level)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                _record_kernel_degradation(
                    e,
                    action="kept volition level update but skipped substrate governor profile application",
                )
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
        except (ImportError, AttributeError, RuntimeError) as hygiene_exc:
            _record_kernel_degradation(
                hygiene_exc,
                action="continued kernel boot without runtime hygiene watchdog",
                severity="error",
            )
            logger.error(
                "Kernel boot runtime hygiene install failed: %s", hygiene_exc, exc_info=True
            )

        # Initialize Lock Watchdog before anything else
        try:
            from core.resilience.lock_watchdog import get_lock_watchdog

            get_lock_watchdog().start()
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_kernel_degradation(
                e,
                action="continued kernel boot without lock watchdog",
                severity="error",
            )
            logger.error("Failed to start LockWatchdog: %s", e)

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

            # 9. Boot PerceptionDaemon
            try:
                from core.perception.perception_daemon import get_perception_daemon
                daemon = get_perception_daemon()
                await daemon.start()
                logger.info("📡 [PERCEPTION] PerceptionDaemon ONLINE")
            except _KERNEL_OPTIONAL_PERCEPTION_ERRORS as e:
                _record_kernel_degradation(
                    e,
                    action="continued kernel boot without active PerceptionDaemon",
                    severity="error"
                )
                logger.error("Failed to start PerceptionDaemon: %s", e)

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
                ledger.append(
                    Transition.create(
                        ttype=TransitionType.BOOT,
                        subsystem="kernel",
                        cause="boot_complete",
                        payload={
                            "organs": list(self.organs.keys()),
                            "phases": len(self._phases),
                            "volition": self.volition_level,
                        },
                        prior_hash=compute_state_hash(self.state) if self.state else "genesis",
                    )
                )
                ServiceContainer.register_instance("cognitive_ledger", ledger)
            except (ImportError, AttributeError, RuntimeError) as _le:
                _record_kernel_degradation(
                    _le,
                    action="completed boot without cognitive ledger boot transition",
                )
                logger.debug("Ledger boot record failed (non-critical): %s", _le)

            # Verify LLM resolution
            try:
                llm_organ = self.organs.get("llm")
                if llm_organ:
                    # We log the class name to confirm if it's IntelligentLLMRouter or MockLLM
                    logger.info("LLM organ instance: %s", llm_organ.instance.__class__.__name__)
            except (OSError, ConnectionError, TimeoutError) as e:
                _record_kernel_degradation(
                    e,
                    action="completed boot without logging resolved LLM organ class",
                )
                logger.warning("Failed to log LLM instance class: %s", e)

        except (ImportError, AttributeError, RuntimeError) as e:
            _record_kernel_degradation(
                e,
                action="failed closed kernel boot after required boot graph failed",
                severity="critical",
            )
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
        self._services[UnityBindingPhase] = self.unity_phase
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

        required_organs = {
            "llm",
            "memory",
            "metabolism",
            "vision",
            "voice",
            "neural",
            "cookie",
            "prober",
            "tricorder",
            "ice_layer",
            "omni_tool",
            "continuity",
        }

        # Fix: Check both presence AND load status (instance not None)
        missing = [o for o in required_organs if o not in self.organs]
        broken = [
            o for o in required_organs if o in self.organs and self.organs[o].instance is None
        ]

        if missing:
            raise RuntimeError(f"CRITICAL: Missing core organs in boot graph: {missing}")

        if broken:
            # Harder validation: refuse boot when a required runtime organ did not load.
            for b_organ in broken:
                logger.error("🛑 CRITICAL ORGAN FAILURE: %s instance is None.", b_organ)
            if "llm" in broken:
                raise RuntimeError("Kernel cannot start: LLM organ is dysfunctional.")

        logger.info("✓ Dependency graph validated.")

    def _initialize_organs(self):
        """
        Populate organ stubs synchronously; loading is handled by caller's TaskGroup.
        """
        self.organs = {
            "llm": OrganStub("llm", self),
            "vision": OrganStub("vision", self),
            "memory": OrganStub("memory", self),
            "voice": OrganStub("voice", self),
            "metabolism": OrganStub("metabolism", self),
            "neural": OrganStub("neural", self),
            "cookie": OrganStub("cookie", self),
            "prober": OrganStub("prober", self),
            "tricorder": OrganStub("tricorder", self),
            "ice_layer": OrganStub("ice_layer", self),
            "omni_tool": OrganStub("omni_tool", self),
            "continuity": OrganStub("continuity", self),
        }

    async def _supervise_organ_load(self, organ: OrganStub):
        """Supervises the async loading of a hardware organ."""
        try:
            await organ.load()
            if (
                organ.name == "ice_layer"
                and organ.instance is not None
                and not organ.fallback_used
            ):
                ServiceContainer.register_instance(
                    "ice_layer",
                    organ.instance,
                    required=True,
                    owner="aura_kernel",
                    registered_by="AuraKernel._supervise_organ_load",
                    required_for="authority_containment",
                    failure_policy="fail_closed",
                )
            try:
                self._gui_queue.put_nowait({"type": "ORGAN_READY", "name": organ.name})
            except asyncio.QueueFull:
                # Non-blocking failsafe
                pass  # no-op: intentional
            logger.info("🫀 Organ %s is READY", organ.name)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_kernel_degradation(
                e,
                action="marked organ unavailable after supervised organ load failed",
                severity="error",
            )
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
                state = getattr(self.vault, "state", None) or getattr(
                    self.vault, "current_state", None
                )
            if state is None:
                logger.info("🌱 No existing state found. Creating fresh AuraState.")
                from core.state.aura_state import AuraState

                state = AuraState()
                # Warm up the vault with the initial state when the supplied vault
                # exposes a durable commit path. Lightweight vault adapters may not.
                commit = getattr(self.vault, "commit", None)
                if callable(commit):
                    await _maybe_await(commit(state, cause="genesis"))

            try:
                from core.continuity import get_continuity

                continuity = get_continuity()
                continuity.load()
                state = continuity.apply_to_state(state)
            except (ImportError, AttributeError, RuntimeError) as continuity_exc:
                _record_kernel_degradation(
                    continuity_exc,
                    action="continued state initialization without continuity hydration",
                    severity="error",
                )
                logger.error("Continuity hydration failed: %s", continuity_exc, exc_info=True)

            self.state = state
            logger.info("🧬 State successfully initialized (version %d)", self.state.version)
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_kernel_degradation(
                e,
                action="failed closed kernel state initialization after vault load failed",
                severity="critical",
            )
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
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_kernel_degradation(
                exc,
                action="continued rubicon boot without feedback processor",
                severity="error",
            )
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
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_kernel_degradation(
                exc,
                action="continued rubicon boot without motor cortex loop",
                severity="error",
            )
            logger.warning("[RUBICON] MotorCortex boot failed (degraded): %s", exc)

        # 3. Pre-Linguistic Decision Engine
        try:
            from core.cognition.pre_linguistic import get_pre_linguistic

            pl = get_pre_linguistic()
            await pl.start()
            logger.info("[RUBICON] PreLinguisticEngine ONLINE")
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_kernel_degradation(
                exc,
                action="continued rubicon boot without pre-linguistic engine",
                severity="error",
            )
            logger.warning("[RUBICON] PreLinguisticEngine boot failed (degraded): %s", exc)

    async def _motor_cortex_watchdog(self, mc: Any) -> None:
        """Watchdog that restarts the motor cortex loop if it dies."""
        while self._running:
            try:
                await asyncio.sleep(10.0)
                if mc._running and (mc._task is None or mc._task.done()):
                    logger.warning("[RUBICON] Motor cortex loop died -- restarting")
                    mc._task = get_task_tracker().create_task(
                        mc._run_loop(), name="motor_cortex_loop"
                    )
            except asyncio.CancelledError:
                if not self._running or is_shutdown_requested():
                    break
                logger.warning("Motor cortex watchdog spuriously cancelled. Ignoring.")
                continue
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                _record_kernel_degradation(
                    exc,
                    action="kept motor cortex watchdog alive after restart probe failed",
                )
                logger.debug("[RUBICON] Motor cortex watchdog error: %s", exc)
                await asyncio.sleep(1.0)

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
        turn_origin = self._normalize_origin(
            getattr(getattr(state, "cognition", None), "current_origin", "")
        )
        try:
            from core.continuity import _is_generic_continuity_reentry_goal

            if not priority and _is_generic_continuity_reentry_goal(objective):
                state.cognition.current_objective = None
                state.cognition.pending_initiatives = [
                    item
                    for item in list(getattr(state.cognition, "pending_initiatives", []) or [])
                    if not _is_generic_continuity_reentry_goal(
                        item.get("goal", "") if isinstance(item, dict) else str(item)
                    )
                ]
                logger.debug("Kernel: skipped generic continuity re-entry bookkeeping objective.")
                return None
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_kernel_degradation(
                exc,
                action="continued tick after continuity re-entry scrub failed",
            )
            logger.debug("Kernel continuity objective scrub failed: %s", exc)

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
            logger.error(
                "🛑 CRITICAL: Could not acquire Kernel lock for tick. Possible deadlock. Objective: '%s'",
                objective,
            )
            if self.status:
                logger.error(
                    "Kernel Status: %s, Cycle: %s", self.status.message, self.status.cycle_count
                )
            return None

        return await self._tick_body(objective, priority, turn_origin, state)

    async def _trace_the_tick(self, objective: str, response: Any) -> None:
        """Issue #42: the structured thought trace for one tick, graded by what happened."""
        try:
            trace_response = response
            trace_outcome = "SUCCESS" if self.state else "FAILURE"
            trace_meta: dict[str, Any] = {}
            modifiers = (
                dict(getattr(self.state, "response_modifiers", {}) or {}) if self.state else {}
            )
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
                re.search(
                    r"\[(?:SKILL_RESULT|SKILL|ACTION|TOOL|SKILL_INVOCATION)\s*:",
                    str(response or ""),
                    re.IGNORECASE,
                )
            )
            if (
                has_action_marker
                and not modifiers.get("last_skill_ok")
                and task_outcome != "completed"
            ):
                trace_outcome = "UNGROUNDED_ACTION"
                trace_meta["grounding_warning"] = ["marker_without_verified_execution"]

            try:
                from core.phases.action_grounding import check_unverified_action_claims

                receipts = []
                if modifiers.get("last_skill_ok") and modifiers.get("last_skill_run"):
                    receipts.append({"skill": str(modifiers.get("last_skill_run"))})
                unverified_claims = check_unverified_action_claims(
                    str(response or ""), skill_receipts=receipts
                )
                if unverified_claims:
                    trace_outcome = "UNGROUNDED_ACTION"
                    trace_meta["grounding_warning"] = unverified_claims[:4]
            except (ImportError, AttributeError, RuntimeError):
                pass  # no-op: intentional

            await tracer.log_cycle_async(
                objective=objective,
                context=getattr(self.state, "cognition", {}).__dict__ if self.state else {},
                thought={"last_response": trace_response, **trace_meta},
                outcome=trace_outcome,
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_kernel_degradation(
                e,
                action="completed tick without structured thought trace entry",
            )
            logger.debug("Tracer failed: %s", e)

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
            logger.warning(
                "Vault commit failed (pipe/connection): %s — state not persisted this tick.", e
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_kernel_degradation(
                e,
                action="returned tick without persisting state after vault commit failed",
                severity="error",
            )
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
            from core.governance_context import governed_scope

            # Offload blocking write to thread pool
            def _sync_write():
                from core.runtime.file_write_gateway import get_file_write_gateway

                source = "aura_kernel.process_intent_file_append"
                get_file_write_gateway().append_text(
                    path,
                    json.dumps(payload) + "\n",
                    encoding="utf-8",
                    source=source,
                )

            source = "aura_kernel.process_intent_file_append"
            async with governed_scope(_kernel_file_write_decision(source)):
                await asyncio.to_thread(_sync_write)

        for intent in intents:
            try:
                t = intent.get("type")
                if t == "db_write":
                    # Direct vault commitment for intentional state shifts.
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
            except (OSError, ConnectionError, TimeoutError) as e:
                _record_kernel_degradation(
                    e,
                    action="dropped deferred storage intent after side-effect processing failed",
                    severity="error",
                )
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
                version=getattr(self.state, "version", 0),
                vitality=getattr(self.state, "vitality", 1.0),
                mood=getattr(self.state, "mood", "neutral"),
                curiosity=getattr(getattr(self.state, "affect", None), "curiosity", 0.5),
                phi=getattr(self.state, "phi", 0.1),
                last_objective=(
                    getattr(getattr(self.state, "cognition", None), "current_objective", "Unknown")
                    or "Unknown"
                )[:80],
                timestamp=time.time(),
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
        except (RuntimeError, AttributeError, TypeError) as e:
            _record_kernel_degradation(
                e,
                action="skipped mirror snapshot update after projection failed",
            )
            logger.error("Mirror projection failed: %s", e)

    async def shutdown(self, *, finalize_process_runtime: bool = True):
        """Gracefully stop this kernel and, when owned here, process finalizers.

        The standalone kernel owns the process runtime.  An orchestrator-owned
        kernel does not: its service container must stop every service before
        the task tracker, runtime hygiene manager, and default executor are
        finalized.
        """
        async with self._shutdown_lock:
            if self._shutdown_complete:
                return
            if self._shutdown_process_runtime_owner is None:
                self._shutdown_process_runtime_owner = bool(finalize_process_runtime)
            elif not self._shutdown_process_runtime_owner and finalize_process_runtime:
                logger.info(
                    "Kernel ignored a duplicate request to escalate component shutdown "
                    "into process-root finalization."
                )

            await self._shutdown_impl(
                finalize_process_runtime=bool(self._shutdown_process_runtime_owner),
            )
            self._shutdown_complete = True

    async def on_stop_async(self) -> None:
        """Container hook that preserves process-root finalizer ownership."""
        await self.shutdown(finalize_process_runtime=False)

    async def _call_shutdown_hook(self, label: str, target: Any, *hook_names: str) -> None:
        """Call the first available lifecycle hook on a runtime singleton."""
        for hook_name in hook_names:
            hook = getattr(target, hook_name, None)
            if not callable(hook):
                continue
            try:
                result = hook()
                if inspect.isawaitable(result):
                    await asyncio.wait_for(result, timeout=5.0)
                return
            except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
                _record_kernel_degradation(
                    exc,
                    action=f"continued shutdown after {label}.{hook_name} failed",
                    severity="error",
                )
                logger.warning("Runtime shutdown hook failed for %s.%s: %s", label, hook_name, exc)
                return

    async def _close_kernel_vault(self) -> None:
        close = getattr(self.vault, "close", None)
        if not callable(close):
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await asyncio.wait_for(result, timeout=5.0)
        except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
            _record_kernel_degradation(
                exc,
                action="continued shutdown after kernel vault close failed",
                severity="error",
            )
            logger.warning("Kernel vault close failed: %s", exc)

    async def hot_reboot(self, *, changed_files: tuple[str, ...] = ()):
        """
        Bounded hot-reload trigger for the local runtime.
        Re-initializes the phase pipeline and reloads changed modules without stopping the process.
        """
        from core.runtime.backpressure import primary_inference_active

        changed = tuple(str(path or "").strip() for path in changed_files if str(path or "").strip())
        if not changed:
            logger.info("⚡ [HOT-RELOAD] No applied source files were receipted; runtime unchanged.")
            return {
                "ok": True,
                "reloaded": [],
                "skipped": [],
                "restart_required": False,
                "reason": "no_applied_source_changes",
            }
        if primary_inference_active():
            logger.info(
                "⚡ [HOT-RELOAD] Deferred %d applied file(s) while foreground "
                "or primary cognition owns the model lane.",
                len(changed),
            )
            return {
                "ok": False,
                "reloaded": [],
                "skipped": list(changed),
                "restart_required": True,
                "reason": "primary_inference_active",
            }

        logger.info("⚡ [HOT-RELOAD] Applying %d receipted source change(s).", len(changed))

        # Reload only the files the mutation receipt names, through the
        # canonical reloader. It rejects runtime-owned modules and inheritance
        # anchors; direct importlib.reload here used to bypass both protections
        # and mint duplicate class identities throughout the live process.
        from core.ops.hot_reload import get_hot_reloader

        reloaded: list[str] = []
        skipped: list[str] = []
        failures: list[dict[str, str]] = []
        restart_required = False
        reloader = get_hot_reloader()
        for filepath in changed:
            result = await asyncio.to_thread(reloader.reload_file, filepath)
            reloaded.extend(result.reloaded)
            skipped.extend(result.skipped)
            failures.extend(result.failed)
            restart_required = bool(
                restart_required or result.skipped or result.orphan_risks or result.failed
            )

        # Re-setup phases only when a phase implementation actually changed.
        # Rebuilding the pipeline for a protected/no-op change discards live
        # phase-local state while making no new code active.
        if any(name.startswith("core.phases.") for name in reloaded):
            self._setup_phases()

        logger.info(
            "✅ [HOT-RELOAD] Refresh complete. %d reloaded, %d require restart.",
            len(reloaded),
            len(skipped) + len(failures),
        )
        return {
            "ok": not failures,
            "reloaded": reloaded,
            "skipped": skipped,
            "failures": failures,
            "restart_required": restart_required,
            "reason": "restart_required" if restart_required else "applied",
        }

    async def _supervise_background_loops(self):
        """
        [CF-7] Actual supervision — detect and restart dead background tasks.
        Also runs the ResourceGovernor periodically for long-term stability.
        """
        _governor = None
        _governor_interval = 60  # seconds
        _last_govern = 0

        while not is_shutdown_requested():
            await asyncio.sleep(5)

            # ── Task supervision ──
            for task in list(self._background_tasks):
                if task.done():
                    exc = None
                    try:
                        exc = task.exception()
                    except (asyncio.CancelledError, Exception) as e:
                        logger.debug("Background supervisor: Task exception ignored: %s", e)

                    logger.error(
                        "⚠️ Background task '%s' died unexpectedly: %s", task.get_name(), exc
                    )
                    self._background_tasks.remove(task)

                    # Restart critical tasks
                    if task.get_name() == "vault_mutation_consumer":
                        logger.info("🔄 Restarting StateRepository mutation consumer...")
                        self._spawn_background_task(
                            self.vault._mutation_consumer(), name="vault_mutation_consumer"
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
                except (ImportError, AttributeError, RuntimeError) as e:
                    _record_kernel_degradation(
                        e,
                        action="continued background supervision without resource governor report",
                    )
                    logger.debug("ResourceGovernor cycle failed (non-critical): %s", e)

    async def stop(self):
        """Graceful shutdown of the kernel."""
        import inspect

        stack = inspect.stack()
        stack_str = "\n".join([f"  {s.filename}:{s.lineno} in {s.function}" for s in stack])
        logger.info("🛑 [KERNEL] Stop requested. Called by:\n%s", stack_str)
        self._running = False
        for task in self._background_tasks:
            task.cancel()

    def _extract_thought(self) -> str | None:
        """Extracts the resulting thought from the final state."""
        return self.state.cognition.last_response if self.state else None
